from __future__ import annotations

import asyncio
import ctypes
import functools
import itertools
import math
import operator
import sys
import threading
import time
import uuid
import weakref
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast, ParamSpec, TypeVar

import torch
from typing_extensions import Self

try:
    from torch._C._distributed_transfer import (  # type: ignore[attr-defined]
        _snapshot_and_analyze_int32_indices as _compiled_index_analyzer,
    )
except ImportError:
    # Source-tree Python can run against an older installed torch._C while this
    # private API is under development. Mixed/general sequences also never
    # depend on the compiled helper.
    _compiled_index_analyzer = None

from ._backend import (
    Backend,
    BackendIndexSelection,
    BackendRequestSlot,
    BackendWork,
    ImportedRawMemory,
    ImportedRegistration,
    RegisteredMemory,
    _get_backend_registration,
    _make_backend_index_selection,
)
from ._errors import (
    BackendFailureError,
    BackendMismatchError,
    BusyError,
    ClosedError,
    InvalidMetadataError,
    InvalidRegionError,
    TransferCancelledError,
    TransferError,
    TransferTimeoutError,
    UnsupportedError,
)
from ._types import (
    BackendNotification,
    BackendNotificationBatch,
    BackendRegion,
    Capabilities,
    CudaOrderingMode,
    EndpointIdentity,
    EndpointInfo,
    EndpointMetadata,
    Notification,
    NotificationBatch,
    ProgressMode,
    RegistrationMetadata,
    ThreadMode,
    TransferOp,
    WorkState,
    _notification_batch_from_backend,
)


__all__ = [
    "BoundTransfer",
    "Endpoint",
    "IndexSelection",
    "Peer",
    "RawSpan",
    "RequestLocalNotificationSlot",
    "Region",
    "Registration",
    "TransferPlan",
    "TransferSlot",
    "Work",
]

_TERMINAL_STATES = frozenset(
    {WorkState.COMPLETED, WorkState.FAILED, WorkState.CANCELLED}
)
_CONSTRUCTION_TOKEN = object()
_UNADOPTED_HANDLE = object()
_UNSET_NOTIFICATION = object()
_MAX_NOTIFICATION_BYTES = 1024 * 1024
_FUSED_REQUEST_SLOT_MAX_POLLS = 64
_MAX_FUSED_REQUEST_SLOT_TIMEOUT_NS = (1 << 63) - 1
_FUSED_REQUEST_SLOT_POLL_MAX_POLLS = 4096
_FUSED_REQUEST_SLOT_POLL_TIMEOUT_NS = 100_000
_FUSED_REQUEST_SLOT_POLL_TIMEOUT_CHECK_INTERVAL = 32
_GET_IDENT = threading.get_ident
_WORK_COMPLETED = WorkState.COMPLETED
_WORK_RUNNING = WorkState.RUNNING
_NATIVE_INT32_FORMATS = frozenset(
    {"i", "@i", "=i", "<i" if sys.byteorder == "little" else ">i"}
)
_INT32_MAX = (1 << 31) - 1
_INDEX_SELECTION_TORCH_THRESHOLD = 128
_P = ParamSpec("_P")
_R = TypeVar("_R")


class _BackendConstructionReceipt:
    """Core ownership record independent of an Endpoint object's lifetime."""

    __slots__ = (
        "_adoption_count",
        "_backend",
        "_closed",
        "_lock",
        "_poison_message",
        "_published",
        "_sealed",
        "backend_name",
        "endpoint_id",
    )

    def __init__(self, backend_name: str, endpoint_id: str) -> None:
        self.backend_name = backend_name
        self.endpoint_id = endpoint_id
        self._lock = threading.RLock()
        self._backend: Backend | None = None
        self._adoption_count = 0
        self._sealed = False
        self._published = False
        self._closed = False
        self._poison_message: str | None = None

    @property
    def backend(self) -> Backend | None:
        with self._lock:
            return self._backend

    @property
    def published(self) -> bool:
        with self._lock:
            return self._published

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def adopt(self, backend: Backend) -> None:
        if not isinstance(backend, Backend):
            message = "backend factory must adopt a Backend instance"
            with self._lock:
                if self._poison_message is None:
                    self._poison_message = message
            raise TypeError(message)
        with self._lock:
            if self._sealed:
                message = "backend factory attempted adoption after its call returned"
                if self._poison_message is None:
                    self._poison_message = message
                raise BackendFailureError(message)
            if self._backend is not None or self._adoption_count:
                message = "backend factory must call adopt_backend exactly once"
                if self._poison_message is None:
                    self._poison_message = message
                raise BackendFailureError(message)
            # Ownership commits before the diagnostic count. An interruption
            # between these stores still leaves the native owner closeable.
            self._backend = backend
            self._adoption_count = 1

    def seal_failed_call(self) -> None:
        with self._lock:
            self._sealed = True

    def seal_successful_call(self, result: object) -> Backend:
        with self._lock:
            self._sealed = True
            if self._poison_message is not None:
                raise BackendFailureError(self._poison_message)
            if result is not None:
                raise TypeError("version-2 backend factory must return None")
            if self._adoption_count != 1 or self._backend is None:
                raise BackendFailureError(
                    "version-2 backend factory must call adopt_backend exactly once"
                )
            return self._backend

    def mark_published(self) -> None:
        with self._lock:
            if self._closed or self._backend is None:
                raise BackendFailureError("cannot publish a closed backend receipt")
            self._published = True

    def close(self, close_backend: Callable[[], object] | None = None) -> None:
        """Close once; an interrupted/failed close remains retryable."""

        with self._lock:
            if self._closed:
                return
            backend = self._backend
            if backend is None:
                self._closed = True
                return
            callback = backend.close if close_backend is None else close_backend
            try:
                result = callback()
                if result is not None:
                    raise BackendFailureError("backend close must return None")
            except BaseException:
                raise
            self._closed = True


_ORPHANED_BACKEND_RECEIPTS: dict[tuple[str, str], set[_BackendConstructionReceipt]] = {}
_ORPHANED_BACKEND_RECEIPTS_LOCK = threading.Lock()


def _retain_orphaned_backend_receipt(receipt: _BackendConstructionReceipt) -> None:
    key = (receipt.backend_name, receipt.endpoint_id)
    with _ORPHANED_BACKEND_RECEIPTS_LOCK:
        _ORPHANED_BACKEND_RECEIPTS.setdefault(key, set()).add(receipt)


def _forget_orphaned_backend_receipt(receipt: _BackendConstructionReceipt) -> None:
    key = (receipt.backend_name, receipt.endpoint_id)
    with _ORPHANED_BACKEND_RECEIPTS_LOCK:
        receipts = _ORPHANED_BACKEND_RECEIPTS.get(key)
        if receipts is None:
            return
        receipts.discard(receipt)
        if not receipts:
            _ORPHANED_BACKEND_RECEIPTS.pop(key, None)


def _finalize_backend_receipt(receipt: _BackendConstructionReceipt) -> None:
    """Best-effort GC cleanup without retaining the Endpoint itself."""

    if receipt.backend is None or receipt.closed:
        return
    _retain_orphaned_backend_receipt(receipt)
    try:
        # No Endpoint remains to supply its SERIALIZED lock. The receipt lock
        # still serializes finalizer cleanup and all later orphan retries.
        receipt.close()
    except BaseException:
        return
    _forget_orphaned_backend_receipt(receipt)


def _retry_orphaned_backend_receipts(backend_name: str, endpoint_id: str) -> None:
    """Audit one provider's orphans; only an exact identity blocks creation."""

    with _ORPHANED_BACKEND_RECEIPTS_LOCK:
        snapshot = tuple(
            (key, tuple(receipts))
            for key, receipts in _ORPHANED_BACKEND_RECEIPTS.items()
            if key[0] == backend_name
        )

    exact_failures: list[tuple[_BackendConstructionReceipt, Exception]] = []
    for key, receipts in snapshot:
        for receipt in receipts:
            try:
                receipt.close()
            except BaseException as error:
                # Process-control interruptions are not converted or hidden;
                # the retained receipt makes the next attempt safe.
                if not isinstance(error, Exception):
                    raise
                if key[1] == endpoint_id:
                    exact_failures.append((receipt, error))
            else:
                _forget_orphaned_backend_receipt(receipt)

    if exact_failures:
        receipt, error = exact_failures[0]
        origin = (
            "an abandoned published endpoint (possibly with live provider children)"
            if receipt.published
            else "an unpublished endpoint construction"
        )
        failure = BackendFailureError(
            f"cannot reuse endpoint identity {endpoint_id!r} for backend "
            f"{backend_name!r}: cleanup of {origin} is still pending"
        )
        add_note = getattr(failure, "add_note", None)
        if callable(add_note):
            add_note(f"orphan cleanup failure type: {type(error).__name__}")
        raise failure from error


class _NoopLock:
    """Context-manager lock for SINGLE or externally serialized resources."""

    __slots__ = ()

    def __enter__(self) -> _NoopLock:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool:
        return False


_NOOP_LOCK = _NoopLock()


class _ReuseToken:
    """Weak-referenceable identity token with no provider-visible semantics."""

    __slots__ = ("__weakref__",)


class _SyncNotificationReservation:
    """Pooled strong token with a close-visible weak identity."""

    __slots__ = ("active", "reference", "__weakref__")

    def __init__(self) -> None:
        self.active = False
        self.reference = weakref.ref(self)


class _BackendCallReservation:
    """Reusable MULTIPLE-call token whose weak identity is visible to close."""

    __slots__ = ("active", "reference", "__weakref__")

    def __init__(self) -> None:
        self.active = False
        self.reference = weakref.ref(self)


@dataclass(slots=True)
class _LifetimeAnchor:
    owner: object
    storage: object | None = None
    buffer: memoryview | None = None
    allocation_address: int | None = None
    allocation_nbytes: int | None = None


@dataclass(frozen=True, slots=True)
class _RemoteRawDescriptor:
    address: int
    nbytes: int
    device: str
    memory_type: str


def _state_locked(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Serialize endpoint resource-graph mutations."""

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        endpoint = cast(Any, args[0])
        state_lock = endpoint._state_lock
        if state_lock is _NOOP_LOCK:
            return function(*args, **kwargs)
        with state_lock:
            return function(*args, **kwargs)

    return wrapper


@dataclass(frozen=True, slots=True)
class RawSpan:
    """Expert-only local address range with an explicit lifetime anchor."""

    address: int
    nbytes: int
    device: str
    memory_type: str
    owner: object = field(repr=False)

    def __post_init__(self) -> None:
        _validate_positive_int(self.address, "address")
        _validate_positive_int(self.nbytes, "nbytes")
        _validate_nonempty_str(self.device, "device", InvalidRegionError)
        _validate_nonempty_str(self.memory_type, "memory_type", InvalidRegionError)
        if self.owner is None:
            raise InvalidRegionError("a local RawSpan requires a lifetime owner")


class Region:
    """A bounded local or remote byte region.

    Regions are created by :meth:`Registration.region`, :meth:`Peer.region`,
    or :meth:`Peer.unsafe_region`; users do not construct them directly.
    """

    __slots__ = (
        "_backend_descriptor",
        "_device",
        "_memory_type",
        "_nbytes",
        "_offset",
        "_origin",
        "_remote",
        "_stride",
        "_count",
    )

    def __init__(
        self,
        origin: Registration | Peer,
        backend_descriptor: object,
        *,
        offset: int,
        nbytes: int,
        device: str,
        memory_type: str,
        remote: bool,
        stride: int | None,
        count: int | None,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("Region objects are created by Registration or Peer")
        self._origin = origin
        self._backend_descriptor = backend_descriptor
        self._offset = offset
        self._nbytes = nbytes
        self._device = device
        self._memory_type = memory_type
        self._remote = remote
        self._stride = stride
        self._count = count

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def nbytes(self) -> int:
        """Number of bytes in each strided segment."""

        return self._nbytes

    @property
    def device(self) -> str:
        return self._device

    @property
    def memory_type(self) -> str:
        return self._memory_type

    @property
    def stride(self) -> int | None:
        return self._stride

    @property
    def count(self) -> int | None:
        return self._count

    @property
    def total_nbytes(self) -> int:
        return self._nbytes * (self._count or 1)

    @property
    def is_remote(self) -> bool:
        return self._remote

    def _to_backend(self) -> BackendRegion:
        self._origin._check_open()
        descriptor = self._backend_descriptor
        if isinstance(self._origin, Registration):
            descriptor = self._origin._handle
        elif isinstance(descriptor, RegistrationMetadata):
            descriptor = ImportedRegistration(
                peer_handle=self._origin._handle,
                registration_id=descriptor.registration_id,
                name=descriptor.name,
                nbytes=descriptor.nbytes,
                device=descriptor.device,
                memory_type=descriptor.memory_type,
                descriptor=descriptor.backend_descriptor,
            )
        elif isinstance(descriptor, _RemoteRawDescriptor):
            descriptor = ImportedRawMemory(
                peer_handle=self._origin._handle,
                address=descriptor.address,
                nbytes=descriptor.nbytes,
                device=descriptor.device,
                memory_type=descriptor.memory_type,
            )
        return BackendRegion(
            owner=self._origin,
            offset=self._offset,
            nbytes=self._nbytes,
            device=self._device,
            memory_type=self._memory_type,
            backend_descriptor=descriptor,
            remote=self._remote,
            stride=self._stride,
            count=self._count,
        )


class Registration:
    """A backend registration that retains its source allocation.

    Retention does not freeze mutable Storage metadata. The caller must keep
    the allocation address and extent unchanged until this object is closed;
    in particular, resizing while a plan or Work exists is unsafe. Core's
    pointer/size checks are diagnostics and cannot close the mutation race.
    """

    __slots__ = (
        "_closed",
        "_creation_failed",
        "_endpoint",
        "_handle",
        "_memory",
    )

    def __init__(
        self,
        endpoint: Endpoint,
        memory: RegisteredMemory,
        handle: object,
        *,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("Registration objects are created by Endpoint")
        self._endpoint = endpoint
        self._memory = memory
        self._handle = handle
        self._creation_failed = False
        self._closed = False

    @property
    def registration_id(self) -> str:
        return self._memory.registration_id

    @property
    def name(self) -> str:
        return self._memory.name

    @property
    def nbytes(self) -> int:
        return self._memory.nbytes

    @property
    def device(self) -> str:
        return self._memory.device

    @property
    def memory_type(self) -> str:
        return self._memory.memory_type

    @property
    def closed(self) -> bool:
        return self._closed

    def region(
        self,
        offset: int = 0,
        nbytes: int | None = None,
        *,
        stride: int | None = None,
        count: int | None = None,
    ) -> Region:
        self._endpoint._validate_registration(self)
        offset, nbytes, stride, count = _validate_region_bounds(
            self.nbytes,
            offset,
            nbytes,
            stride,
            count,
        )
        if stride is not None and not self._endpoint.capabilities.strided_regions:
            raise UnsupportedError(
                f"backend {self._endpoint.backend!r} does not support strided regions"
            )
        return Region(
            self,
            None,
            offset=offset,
            nbytes=nbytes,
            device=self.device,
            memory_type=self.memory_type,
            remote=False,
            stride=stride,
            count=count,
            _token=_CONSTRUCTION_TOKEN,
        )

    def close(self) -> None:
        self._endpoint._close_registration(self)

    def _check_open(self) -> None:
        if self._closed:
            raise ClosedError(f"registration {self.name!r} is closed")
        if self._creation_failed:
            raise BackendFailureError(
                f"registration {self.name!r} is retained for cleanup only"
            )
        self._endpoint._check_open()

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class Peer:
    """An imported remote endpoint and its named registrations."""

    __slots__ = (
        "_active_sync_notification_sends",
        "_closed",
        "_creation_failed",
        "_endpoint",
        "_endpoint_id",
        "_handle",
        "_incarnation",
        "_name",
        "_registrations",
        "_sync_notification_reservation_pool",
    )

    def __init__(
        self,
        endpoint: Endpoint,
        metadata: EndpointMetadata,
        handle: object,
        *,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("Peer objects are created by Endpoint.import_peer")
        self._endpoint = endpoint
        self._endpoint_id = metadata.endpoint_id
        self._name = metadata.name
        self._incarnation = metadata.incarnation
        self._handle = handle
        self._creation_failed = False
        self._registrations = {
            registration.name: registration for registration in metadata.registrations
        }
        # MULTIPLE-mode synchronous sends have no Work object to retain this
        # peer. The send frame strongly owns a pooled token while this set keeps
        # only its weak identity. If asynchronous interruption prevents the
        # cleanup epilogue, the dead weak reference cannot wedge peer closure.
        self._active_sync_notification_sends: set[
            weakref.ReferenceType[_SyncNotificationReservation]
        ] = set()
        self._sync_notification_reservation_pool: list[_SyncNotificationReservation] = (
            []
        )
        self._closed = False

    @property
    def endpoint_id(self) -> str:
        return self._endpoint_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def incarnation(self) -> str:
        return self._incarnation

    @property
    def identity(self) -> EndpointIdentity:
        return EndpointIdentity(self.endpoint_id, self.incarnation)

    @property
    def closed(self) -> bool:
        return self._closed

    def region(
        self,
        name: str,
        offset: int = 0,
        nbytes: int | None = None,
        *,
        stride: int | None = None,
        count: int | None = None,
    ) -> Region:
        self._check_open()
        try:
            metadata = self._registrations[name]
        except KeyError as error:
            raise InvalidRegionError(
                f"peer {self.name!r} has no registration {name!r}"
            ) from error
        offset, nbytes, stride, count = _validate_region_bounds(
            metadata.nbytes,
            offset,
            nbytes,
            stride,
            count,
        )
        if stride is not None and not self._endpoint.capabilities.strided_regions:
            raise UnsupportedError(
                f"backend {self._endpoint.backend!r} does not support strided regions"
            )
        return Region(
            self,
            metadata,
            offset=offset,
            nbytes=nbytes,
            device=metadata.device,
            memory_type=metadata.memory_type,
            remote=True,
            stride=stride,
            count=count,
            _token=_CONSTRUCTION_TOKEN,
        )

    def unsafe_region(
        self,
        address: int,
        nbytes: int,
        *,
        device: str,
        memory_type: str,
        stride: int | None = None,
        count: int | None = None,
    ) -> Region:
        """Create a provider-validated remote raw-address region.

        This escape hatch exists for legacy protocols that exchange addresses.
        The backend must validate that the span is covered by imported metadata.
        """

        self._check_open()
        if not self._endpoint.capabilities.raw_spans:
            raise UnsupportedError(
                f"backend {self._endpoint.backend!r} does not support raw spans"
            )
        _validate_positive_int(address, "address")
        _validate_positive_int(nbytes, "nbytes")
        _validate_nonempty_str(device, "device", InvalidRegionError)
        _validate_nonempty_str(memory_type, "memory_type", InvalidRegionError)
        stride, count = _validate_stride(stride, count)
        if stride is not None and stride < nbytes:
            raise InvalidRegionError("stride must be at least nbytes")
        descriptor = _RemoteRawDescriptor(
            address=address,
            nbytes=_region_extent(nbytes, stride, count),
            device=device,
            memory_type=memory_type,
        )
        return Region(
            self,
            descriptor,
            offset=0,
            nbytes=nbytes,
            device=device,
            memory_type=memory_type,
            remote=True,
            stride=stride,
            count=count,
            _token=_CONSTRUCTION_TOKEN,
        )

    def close(self) -> None:
        self._endpoint._close_peer(self)

    def _check_open(self) -> None:
        if self._closed:
            raise ClosedError(f"peer {self.name!r} is closed")
        if self._creation_failed:
            raise BackendFailureError(
                f"peer {self.name!r} is retained for cleanup only"
            )
        self._endpoint._check_open()

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class TransferPlan:
    """Reusable provider descriptors for one peer and one region layout.

    Indexed plans expose flattened region-major, logical-block catalogs. Each
    Region contributes ``count`` blocks, or one block when ``count`` is absent.
    """

    __slots__ = (
        "_bindings",
        "_closed",
        "_creation_failed",
        "_endpoint",
        "_handle",
        "_indexed",
        "_local",
        "_local_block_count",
        "_local_full_nonoverlap",
        "_local_starts",
        "_local_total_nbytes",
        "_local_uniform_device",
        "_local_uniform_nbytes",
        "_origins",
        "_remote",
        "_remote_block_count",
        "_remote_full_nonoverlap",
        "_remote_starts",
        "_remote_total_nbytes",
        "_remote_uniform_nbytes",
        "_reuse_tokens",
        "_submitting",
        "_works",
    )

    def __init__(
        self,
        endpoint: Endpoint,
        handle: object,
        local: tuple[Region, ...],
        remote: tuple[Region, ...],
        indexed: bool,
        *,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("TransferPlan objects are created by Endpoint.prepare")
        self._endpoint = endpoint
        self._handle = handle
        self._local = local
        self._remote = remote
        self._indexed = indexed
        self._local_block_count = _logical_block_count(local)
        self._remote_block_count = _logical_block_count(remote)
        self._local_starts = _catalog_starts(local)
        self._remote_starts = _catalog_starts(remote)
        self._local_total_nbytes = sum(region.total_nbytes for region in local)
        self._remote_total_nbytes = sum(region.total_nbytes for region in remote)
        self._local_uniform_nbytes = _uniform_region_nbytes(local)
        self._remote_uniform_nbytes = _uniform_region_nbytes(remote)
        self._local_uniform_device = _uniform_region_device(local)
        self._local_full_nonoverlap = _regions_are_nonoverlapping(local)
        self._remote_full_nonoverlap = _regions_are_nonoverlapping(remote)
        self._origins = frozenset(region._origin for region in (*local, *remote))
        self._bindings: set[BoundTransfer] = set()
        self._reuse_tokens = {
            TransferOp.READ: _ReuseToken(),
            TransferOp.WRITE: _ReuseToken(),
        }
        self._submitting: set[object] = set()
        self._works: set[Work] = set()
        self._creation_failed = False
        self._closed = False

    @property
    def indexed(self) -> bool:
        return self._indexed

    @property
    def local_block_count(self) -> int:
        """Number of logical blocks in the local indexed catalog."""

        return self._local_block_count

    @property
    def remote_block_count(self) -> int:
        """Number of logical blocks in the remote indexed catalog."""

        return self._remote_block_count

    @property
    def closed(self) -> bool:
        return self._closed

    def bind(
        self,
        op: TransferOp | str,
        local_indices: Sequence[int] | None = None,
        remote_indices: Sequence[int] | None = None,
        notification: bytes | bytearray | memoryview | None = None,
        cuda_ordering: CudaOrderingMode | str = CudaOrderingMode.CALLER_READY,
        stream: object | None = None,
        ready_event: object | None = None,
    ) -> BoundTransfer:
        """Validate and retain one reusable submission shape.

        A binding amortizes operation, selection, notification, and CUDA-token
        validation across repeated submissions. Each :meth:`BoundTransfer.submit`
        still creates an independently owned :class:`Work`.
        """

        return self._endpoint._bind_transfer(
            self,
            op,
            local_indices=local_indices,
            remote_indices=remote_indices,
            notification=notification,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
        )

    def select_indices(
        self,
        op: TransferOp | str,
        *,
        local_indices: Sequence[int],
        remote_indices: Sequence[int],
    ) -> IndexSelection:
        """Snapshot and validate one reusable dynamic indexed selection.

        C-contiguous native-int32 buffers use the packed fast input. Mutable
        inputs are copied into one private Core-owned buffer, so later caller
        mutation cannot change this selection. General sequences remain
        supported through ``operator.index`` normalization.
        """

        return self._endpoint._select_plan_indices(
            self,
            op,
            local_indices=local_indices,
            remote_indices=remote_indices,
        )

    def submit_indices(
        self,
        op: TransferOp | str,
        *,
        local_indices: Sequence[int],
        remote_indices: Sequence[int],
        notification: bytes | bytearray | memoryview | None = None,
        cuda_ordering: CudaOrderingMode | str = CudaOrderingMode.CALLER_READY,
        stream: object | None = None,
        ready_event: object | None = None,
        recovery_token: object | None = None,
    ) -> Work:
        """Snapshot, validate, and submit one dynamic indexed transfer.

        This is the allocation-minimal one-shot form. It creates only Core's
        private authenticated backend selection and retains that snapshot in
        the returned Work. Use :meth:`select_indices` when the same validated
        selection will be bound or reused.
        """

        return self._endpoint._submit_indexed_selection(
            plan=self,
            op=op,
            local_indices=local_indices,
            remote_indices=remote_indices,
            selection=None,
            notification=notification,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
            recovery_token=recovery_token,
        )

    def submit_indices_and_test(
        self,
        op: TransferOp | str,
        *,
        local_indices: Sequence[int],
        remote_indices: Sequence[int],
        notification: bytes | bytearray | memoryview | None = None,
        cuda_ordering: CudaOrderingMode | str = CudaOrderingMode.CALLER_READY,
        stream: object | None = None,
        ready_event: object | None = None,
        recovery_token: object | None = None,
        max_polls: int = _FUSED_REQUEST_SLOT_MAX_POLLS,
        timeout_ns: int | None = None,
    ) -> tuple[Work, bool]:
        """Compatibility wrapper returning whether the first state is terminal.

        The Boolean means terminal, including failure or cancellation. A zero
        poll budget is post-only. Prefer :meth:`submit_indices_and_poll` when
        the caller also needs the exact state; the wrapper does not issue an
        additional provider observation.
        """

        work, state = self.submit_indices_and_poll(
            op,
            local_indices=local_indices,
            remote_indices=remote_indices,
            notification=notification,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
            recovery_token=recovery_token,
            max_polls=max_polls,
            timeout_ns=timeout_ns,
        )
        return work, state in _TERMINAL_STATES

    def submit_indices_and_poll(
        self,
        op: TransferOp | str,
        *,
        local_indices: Sequence[int],
        remote_indices: Sequence[int],
        notification: bytes | bytearray | memoryview | None = None,
        cuda_ordering: CudaOrderingMode | str = CudaOrderingMode.CALLER_READY,
        stream: object | None = None,
        ready_event: object | None = None,
        recovery_token: object | None = None,
        max_polls: int = _FUSED_REQUEST_SLOT_MAX_POLLS,
        timeout_ns: int | None = None,
    ) -> tuple[Work, WorkState]:
        """Submit one dynamic selection and return its exact bounded-poll state.

        A zero poll budget is post-only. Fused providers return the state from
        the post boundary; other providers conservatively return ``PENDING``
        until at least one poll is requested. The returned Work remains the
        authoritative owner in every state.
        """

        _validate_bounded_poll_arguments(
            max_polls=max_polls,
            timeout_ns=timeout_ns,
            timeout_check_interval=None,
            allow_zero=True,
        )
        fused = self._endpoint.capabilities.fused_prevalidated_submit_poll
        work = self._endpoint._submit_indexed_selection(
            plan=self,
            op=op,
            local_indices=local_indices,
            remote_indices=remote_indices,
            selection=None,
            notification=notification,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
            recovery_token=recovery_token,
            initial_max_polls=max_polls if fused else None,
            initial_timeout_ns=timeout_ns if fused else None,
        )
        if fused:
            return work, work._state
        if max_polls == 0:
            return work, WorkState.PENDING
        return work, work.poll_bounded(max_polls=max_polls, timeout_ns=timeout_ns)

    def close(self) -> None:
        self._endpoint._close_plan(self)

    def _check_open(self) -> None:
        if self._closed:
            raise ClosedError("transfer plan is closed")
        if self._creation_failed:
            raise BackendFailureError("transfer plan is retained for cleanup only")
        self._endpoint._check_open()

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class IndexSelection:
    """Immutable, exact-plan indexed selection with amortized validation.

    A selection is not an independently closeable resource: its exact plan may
    be closed, which invalidates later ``submit`` and ``bind`` calls. Open Work
    and BoundTransfer objects retain the selection (and therefore its packed
    index snapshot) for as long as a provider may access it.
    """

    __slots__ = (
        "_backend_selection",
        "_endpoint",
        "_local_devices",
        "_op",
        "_plan",
        "_reuse_token",
        "_sealed",
    )

    def __init__(
        self,
        endpoint: Endpoint,
        plan: TransferPlan,
        op: TransferOp,
        backend_selection: BackendIndexSelection,
        local_devices: tuple[str, ...],
        *,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("IndexSelection objects are created by TransferPlan")
        set_attribute = object.__setattr__
        set_attribute(self, "_endpoint", endpoint)
        set_attribute(self, "_plan", plan)
        set_attribute(self, "_op", op)
        set_attribute(self, "_backend_selection", backend_selection)
        set_attribute(self, "_local_devices", local_devices)
        # Allocate the provider-neutral plan/op/index-geometry token only if a
        # selection-backed BoundTransfer is actually submitted. A one-shot
        # selection and a binding used only to create a persistent slot do not
        # need a request-pool identity.
        set_attribute(self, "_reuse_token", None)
        set_attribute(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("IndexSelection is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("IndexSelection is immutable")

    @property
    def op(self) -> TransferOp:
        return self._op

    @property
    def local_count(self) -> int:
        return len(self._backend_selection.local_indices)

    @property
    def remote_count(self) -> int:
        return len(self._backend_selection.remote_indices)

    @property
    def selected_nbytes(self) -> int:
        return self._backend_selection.selected_nbytes

    def submit(
        self,
        *,
        notification: bytes | bytearray | memoryview | None = None,
        cuda_ordering: CudaOrderingMode | str = CudaOrderingMode.CALLER_READY,
        stream: object | None = None,
        ready_event: object | None = None,
        recovery_token: object | None = None,
    ) -> Work:
        """Submit this selection with request-local completion arguments."""

        return self._endpoint._submit_indexed_selection(
            plan=None,
            op=None,
            local_indices=None,
            remote_indices=None,
            selection=self,
            notification=notification,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
            recovery_token=recovery_token,
        )

    def submit_and_test(
        self,
        *,
        notification: bytes | bytearray | memoryview | None = None,
        cuda_ordering: CudaOrderingMode | str = CudaOrderingMode.CALLER_READY,
        stream: object | None = None,
        ready_event: object | None = None,
        recovery_token: object | None = None,
        max_polls: int = _FUSED_REQUEST_SLOT_MAX_POLLS,
        timeout_ns: int | None = None,
    ) -> tuple[Work, bool]:
        """Compatibility wrapper returning whether the first state is terminal.

        The Boolean has the same meaning as :meth:`Work.test`: ``True`` means
        terminal, including failure or cancellation, rather than successful.
        Prefer :meth:`submit_and_poll` when the exact state is needed.
        """

        work, state = self.submit_and_poll(
            notification=notification,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
            recovery_token=recovery_token,
            max_polls=max_polls,
            timeout_ns=timeout_ns,
        )
        return work, state in _TERMINAL_STATES

    def submit_and_poll(
        self,
        *,
        notification: bytes | bytearray | memoryview | None = None,
        cuda_ordering: CudaOrderingMode | str = CudaOrderingMode.CALLER_READY,
        stream: object | None = None,
        ready_event: object | None = None,
        recovery_token: object | None = None,
        max_polls: int = _FUSED_REQUEST_SLOT_MAX_POLLS,
        timeout_ns: int | None = None,
    ) -> tuple[Work, WorkState]:
        """Submit and return the exact state from a bounded initial poll.

        A zero poll budget is post-only. On a provider without fused support,
        Core conservatively returns ``PENDING`` because no post-derived state
        has been observed. The returned Work remains authoritative.
        """

        _validate_bounded_poll_arguments(
            max_polls=max_polls,
            timeout_ns=timeout_ns,
            timeout_check_interval=None,
            allow_zero=True,
        )
        if self._endpoint.capabilities.fused_prevalidated_submit_poll:
            work = self._endpoint._submit_indexed_selection(
                plan=None,
                op=None,
                local_indices=None,
                remote_indices=None,
                selection=self,
                notification=notification,
                cuda_ordering=cuda_ordering,
                stream=stream,
                ready_event=ready_event,
                recovery_token=recovery_token,
                initial_max_polls=max_polls,
                initial_timeout_ns=timeout_ns,
            )
            return work, work._state

        work = self.submit(
            notification=notification,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
            recovery_token=recovery_token,
        )
        if max_polls == 0:
            return work, WorkState.PENDING
        return work, work.poll_bounded(
            max_polls=max_polls,
            timeout_ns=timeout_ns,
        )

    def bind(
        self,
        *,
        cuda_ordering: CudaOrderingMode | str = CudaOrderingMode.CALLER_READY,
        stream: object | None = None,
        ready_event: object | None = None,
    ) -> BoundTransfer:
        """Bind completion arguments without revalidating this selection."""

        return self._endpoint._bind_index_selection(
            self,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
        )

    def _check_open(self) -> TransferPlan:
        if self._plan._closed:
            raise ClosedError("transfer plan is closed")
        self._plan._check_open()
        return self._plan


class BoundTransfer:
    """A prevalidated, reusable transfer submission.

    Bindings retain their plan and canonical submission arguments. They are
    Core-only resources: providers observe only the binding's opaque, stable
    reuse token on submission.
    """

    __slots__ = (
        "_closed",
        "_cuda_device",
        "_cuda_ordering",
        "_endpoint",
        "_local_indices",
        "_notification",
        "_op",
        "_plan",
        "_ready_event",
        "_remote_indices",
        "_reuse_token",
        "_selection",
        "_slots",
        "_stream",
        "_submitting",
        "_works",
    )

    def __init__(
        self,
        endpoint: Endpoint,
        plan: TransferPlan,
        op: TransferOp,
        *,
        local_indices: tuple[int, ...] | None,
        remote_indices: tuple[int, ...] | None,
        notification: bytes | None,
        cuda_device: torch.device | None,
        cuda_ordering: CudaOrderingMode,
        stream: object | None,
        ready_event: object | None,
        selection: IndexSelection | None = None,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("BoundTransfer objects are created by TransferPlan.bind")
        self._endpoint = endpoint
        self._plan: TransferPlan | None = plan
        self._op = op
        self._local_indices = local_indices
        self._remote_indices = remote_indices
        self._notification = notification
        self._cuda_device = cuda_device
        self._cuda_ordering = cuda_ordering
        self._stream = stream
        self._ready_event = ready_event
        self._selection = selection
        # Selection-backed bindings share the selection's lazily allocated
        # geometry token. Do not allocate a redundant binding token for them.
        self._reuse_token: object | None = _ReuseToken() if selection is None else None
        self._submitting: set[Work] = set()
        self._works: set[Work] = set()
        self._slots: set[TransferSlot] = set()
        self._closed = False

    @property
    def op(self) -> TransferOp:
        return self._op

    @property
    def closed(self) -> bool:
        return self._closed

    def submit(
        self,
        *,
        notification: bytes | bytearray | memoryview | None = cast(
            Any, _UNSET_NOTIFICATION
        ),
        recovery_token: object | None = None,
    ) -> Work:
        """Submit the binding.

        Selection-backed bindings accept a request-local notification override
        so one native indexed lowering can serve successive framework tags.
        Ordinary bindings retain their original fully immutable contract.
        """

        return self._endpoint._submit_bound(
            self,
            notification=notification,
            recovery_token=recovery_token,
        )

    def create_slot(
        self, *, request_local_notifications: bool = False
    ) -> TransferSlot | RequestLocalNotificationSlot:
        """Create a persistent, single-flight request for this binding.

        The default returns the fixed-notification :class:`TransferSlot` and
        preserves its minimal execute path. ``request_local_notifications=True``
        returns a :class:`RequestLocalNotificationSlot`, whose start operation
        requires an explicit payload (or ``None`` to clear it) for every
        generation. This opt-in mode is capability gated and requires a
        binding with no fixed notification.
        """

        if not isinstance(request_local_notifications, bool):
            raise TypeError("request_local_notifications must be a bool")
        return self._endpoint._create_transfer_slot(
            self,
            request_local_notifications=request_local_notifications,
        )

    def close(self) -> None:
        self._endpoint._close_bound_transfer(self)

    def _check_open(self) -> TransferPlan:
        if self._closed:
            raise ClosedError("bound transfer is closed")
        plan = self._plan
        if plan is None:
            raise ClosedError("bound transfer is closed")
        plan._check_open()
        return plan

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class TransferSlot:
    """Stable, mutable single-flight request for one ``BoundTransfer``.

    A slot owns at most one unrecycled generation. ``state`` is ``None`` while
    idle and a :class:`WorkState` after ``start`` until terminal ``recycle``.
    ``active`` deliberately remains true for a terminal generation: recycling
    is the explicit boundary that permits the next start and prevents ABA-style
    observations across generations. ``execute`` is the opt-in full-cycle form;
    it publishes a clean successful generation directly back to idle.
    """

    __slots__ = (
        "_active",
        "_backend_cancel",
        "_backend_close",
        "_backend_poll_bounded",
        "_backend_poll_state",
        "_backend_recycle",
        "_backend_deferred_successful_recycle",
        "_backend_execution_dispatch",
        "_backend_generation_active",
        "_caller_ready",
        "_ready_event_synchronize",
        "_fast_execute",
        "_backend_slot",
        "_backend_start",
        "_backend_start_and_poll",
        "_bound",
        "_closed",
        "_creation_failed",
        "_endpoint",
        "_error",
        "_generation",
        "_lock",
        "_owner_thread",
        "_start_error",
        "_state",
    )

    def __init__(
        self,
        endpoint: Endpoint,
        bound: BoundTransfer,
        *,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("TransferSlot objects are created by BoundTransfer")
        self._endpoint = endpoint
        self._bound: BoundTransfer | None = bound
        self._backend_slot: BackendRequestSlot | None = None
        self._backend_execution_dispatch: object | None = None
        self._backend_start: Callable[[], None] | None = None
        self._backend_start_and_poll: Callable[..., WorkState | None] | None = None
        self._backend_poll_bounded: Callable[..., WorkState | None] | None = None
        self._backend_poll_state: Callable[[], WorkState | None] | None = None
        self._backend_cancel: Callable[[], bool] | None = None
        self._backend_recycle: Callable[[], None] | None = None
        self._backend_close: Callable[[], None] | None = None
        self._backend_deferred_successful_recycle = (
            endpoint.capabilities.deferred_successful_request_slot_recycle
        )
        self._backend_generation_active = False
        self._caller_ready = bound._cuda_ordering is CudaOrderingMode.CALLER_READY
        # A BoundTransfer has already authenticated the token type and device.
        # Pin one complete readiness operation per persistent slot. An explicit
        # event contributes its bound synchronize method; a stream contributes
        # one reusable, lazily materialized CUDA event plus cached capture,
        # record, and synchronize callables. Every generation still performs
        # capture detection, records after prior producer work, and host-waits,
        # but it allocates neither a Python nor native CUDA event in steady state.
        if self._caller_ready:
            self._ready_event_synchronize = None
        elif bound._stream is not None:
            self._ready_event_synchronize = _make_reusable_cuda_stream_readiness(
                bound._stream,
                bound._cuda_device,
            )
        else:
            self._ready_event_synchronize = cast(
                Callable[[], None], bound._ready_event.synchronize
            )
        self._fast_execute: Callable[..., WorkState | None] | None = None
        # SINGLE and CALLER_SERIALIZED use a genuinely lock-free path. ``None``
        # avoids even the Python context-manager calls paid by Core's general
        # no-op lock. The latter contract requires whole-graph exclusion by its
        # caller; it is not an independently thread-safe slot.
        self._lock = (
            None
            if endpoint._thread_mode
            in (ThreadMode.SINGLE, ThreadMode.CALLER_SERIALIZED)
            else threading.RLock()
        )
        self._owner_thread = (
            endpoint._owner_thread
            if endpoint._thread_mode is ThreadMode.SINGLE
            else None
        )
        self._generation = 0
        self._active = False
        self._state: WorkState | None = None
        self._error: BaseException | None = None
        self._start_error: BaseException | None = None
        self._creation_failed = False
        self._closed = False

    @property
    def generation(self) -> int:
        """Number of starts owned by this slot, including retained failures."""

        return self._generation

    @property
    def active(self) -> bool:
        """Whether this slot owns a generation that has not been recycled."""

        return self._active

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def state(self) -> WorkState | None:
        return self._refresh()

    @property
    def error(self) -> BaseException | None:
        lock = self._lock
        if lock is None:
            self._check_access()
            state = self._refresh_unlocked()
            return None if state is None else self._error
        with lock:
            self._check_access()
            state = self._refresh_unlocked()
            return None if state is None else self._error

    @property
    def completion_event(self) -> object | None:
        lock = self._lock
        if lock is None:
            return self._completion_event_unlocked()
        with lock:
            return self._completion_event_unlocked()

    def start(self) -> None:
        """Start one generation after re-establishing CUDA readiness.

        Once Core enters the provider ``start`` call, the generation is owned
        even if an exception crosses that boundary. Such an exception is
        retained and becomes visible as terminal failure; the slot cannot be
        reused or closed until its provider state is terminal and recycled.
        """

        lock = self._lock
        if lock is None:
            self._start_unlocked()
            return
        with lock:
            self._start_unlocked()

    def start_and_test(self) -> bool:
        """Start and logically recycle a post-reported immediate success.

        ``True`` means the generation completed successfully and the public
        slot is already idle/repostable. ``False`` means it remains active and
        can be observed with :meth:`test_and_recycle`. Terminal failure is
        raised and deliberately remains active for explicit error recovery.
        """

        fast_start_and_test = self._fast_execute
        if fast_start_and_test is not None:
            next_generation = self._generation + 1
            try:
                if self._closed:
                    raise ClosedError("transfer slot is closed")
                if self._creation_failed:
                    raise BackendFailureError(
                        "transfer slot is retained for cleanup only"
                    )
                if (
                    self._owner_thread is not None
                    and _GET_IDENT() != self._owner_thread
                ):
                    raise UnsupportedError(
                        "single-thread endpoint used from a different thread"
                    )
                if self._active:
                    raise BusyError(
                        "transfer slot already owns an unrecycled generation"
                    )

                if not self._caller_ready:
                    ready_event_synchronize = self._ready_event_synchronize
                    if ready_event_synchronize is not None:
                        try:
                            ready_event_synchronize()
                        except TransferError:
                            raise
                        except Exception as error:
                            raise BackendFailureError(
                                "failed to wait for CUDA readiness"
                            ) from error
                    else:
                        bound = self._bound
                        _establish_cuda_ordering_prevalidated(
                            bound._cuda_ordering,
                            bound._stream,
                            bound._ready_event,
                            bound._cuda_device,
                        )

                try:
                    self._generation = next_generation
                    self._active = True
                except BaseException:
                    self._active = False
                    self._generation = next_generation - 1
                    self._state = None
                    self._error = None
                    self._start_error = None
                    self._backend_generation_active = False
                    raise

                state = fast_start_and_test(max_polls=0, timeout_ns=None)
            except BaseException as error:
                if self._generation != next_generation:
                    raise
                self._start_error = error
                refreshed = self._refresh_unlocked()
                assert refreshed is not None
                state = refreshed
            else:
                if state is _WORK_COMPLETED:
                    self._active = False
                    return True
                if state is _WORK_RUNNING:
                    # The fused provider has acknowledged this generation and
                    # reported the dominant QD-many nonterminal state. Avoid
                    # reclassifying the same enum through the generic observer;
                    # these two stores are exactly its RUNNING-state commit.
                    self._backend_generation_active = True
                    self._state = state
                    return False
                if state is None:
                    self._start_error = BackendFailureError(
                        "backend fused request-slot start returned without "
                        "acknowledging a generation"
                    )
                    refreshed = self._refresh_unlocked()
                    assert refreshed is not None
                    state = refreshed
                elif not isinstance(state, WorkState):
                    self._start_error = BackendFailureError(
                        "backend returned invalid fused request-slot state "
                        f"{state!r}"
                    )
                    refreshed = self._refresh_unlocked()
                    assert refreshed is not None
                    state = refreshed
                else:
                    state = self._observe_backend_state_unlocked(state)

            if state in _TERMINAL_STATES:
                self._raise_for_state()
            return False

        start_and_poll = self._backend_start_and_poll
        lock = self._lock
        if start_and_poll is None:
            if lock is None:
                return self._start_and_test_fallback_unlocked()
            with lock:
                return self._start_and_test_fallback_unlocked()
        if lock is None:
            state = self._start_and_poll_unlocked(start_and_poll, None, max_polls=0)
            expected_generation = self._generation
            if state is WorkState.COMPLETED:
                self._finish_execute_success_unlocked(expected_generation)
                return True
            if state in _TERMINAL_STATES:
                self._raise_for_state()
            return False
        with lock:
            state = self._start_and_poll_unlocked(start_and_poll, None, max_polls=0)
            expected_generation = self._generation
            if state is WorkState.COMPLETED:
                self._finish_execute_success_unlocked(expected_generation)
                return True
            if state in _TERMINAL_STATES:
                self._raise_for_state()
            return False

    def _start_and_test_fallback_unlocked(self) -> bool:
        """Keep fallback start/observe/finalize in one slot transaction."""

        self._start_unlocked()
        expected_generation = self._generation
        state = self._refresh_unlocked()
        if state is WorkState.COMPLETED:
            self._finish_execute_success_unlocked(expected_generation)
            return True
        if state in _TERMINAL_STATES:
            self._raise_for_state()
        return False

    def test_and_recycle(self) -> bool:
        """Observe once and logically recycle only successful completion.

        Returns ``False`` while the generation remains pending/running.
        Successful completion returns ``True`` with the public slot idle.
        Failure or cancellation is raised and remains active for explicit
        recovery. Deferred-success providers retain the clean native receipt
        for next-start/close without a provider recycle call.
        """

        lock = self._lock
        if lock is None and self._backend_deferred_successful_recycle:
            # Exact lock-free/deferred continuation path. Slot construction has
            # already authenticated the lifetime graph and cached the provider
            # dispatch. Keep the common PENDING/RUNNING/COMPLETED states on one
            # straight-line poll without the generic refresh/observe/finalize
            # frames; failure and malformed-provider states retain the full
            # diagnostic path below.
            if self._closed:
                raise ClosedError("transfer slot is closed")
            if self._creation_failed:
                raise BackendFailureError("transfer slot is retained for cleanup only")
            if self._owner_thread is not None and _GET_IDENT() != self._owner_thread:
                raise UnsupportedError(
                    "single-thread endpoint used from a different thread"
                )
            if not self._active:
                raise BusyError("transfer slot has no active generation")
            if self._state is WorkState.COMPLETED:
                # A preceding state/test observation may already have cached
                # successful completion. Commit Core-idle directly; polling
                # the provider again cannot add information and penalizes the
                # normal test-then-recycle composition.
                self._fast_execute = None
                self._backend_generation_active = False
                self._error = None
                self._start_error = None
                self._state = None
                # Deferred success leaves a durable provider receipt, so all
                # private Core state can be clean before the public idle commit.
                # This prevents an asynchronous exception from exposing stale
                # terminal state beside a subclass-specific fast dispatcher.
                self._active = False
                self._restore_fast_execute_unlocked()
                return True
            if self._state in _TERMINAL_STATES:
                self._raise_for_state()
            poll_state = self._backend_poll_state
            if poll_state is None:
                raise BackendFailureError(
                    "transfer slot has no backend state-poll operation"
                )
            try:
                state = poll_state()
            except TransferError:
                raise
            except Exception as error:
                raise BackendFailureError(
                    f"backend {self._endpoint.backend!r} failed to query "
                    "persistent request state"
                ) from error
            if state is WorkState.COMPLETED and self._start_error is None:
                self._state = WorkState.COMPLETED
                self._backend_generation_active = True
                self._fast_execute = None
                self._backend_generation_active = False
                self._error = None
                self._start_error = None
                self._state = None
                # Keep the deferred-success idle commit last for the same
                # interruption-safety reason as the cached-completion branch.
                self._active = False
                self._restore_fast_execute_unlocked()
                return True
            if state is None:
                if self._start_error is None:
                    self._start_error = BackendFailureError(
                        "backend request-slot start returned without "
                        "acknowledging a generation"
                    )
                self._error = self._start_error
                self._state = WorkState.FAILED
                self._raise_for_state()
            if not isinstance(state, WorkState):
                raise BackendFailureError(
                    f"backend returned invalid transfer slot state {state!r}"
                )
            if self._state is WorkState.RUNNING and state is WorkState.PENDING:
                raise BackendFailureError(
                    "backend transfer slot state regressed from running to pending"
                )
            self._backend_generation_active = True
            if state is WorkState.PENDING or state is WorkState.RUNNING:
                self._state = state
                return False
            self._observe_backend_state_unlocked(state)
            self._raise_for_state()

        if lock is None:
            self._check_access()
            if not self._active:
                raise BusyError("transfer slot has no active generation")
            expected_generation = self._generation
            state = self._refresh_unlocked()
            if state is WorkState.COMPLETED:
                self._finish_execute_success_unlocked(expected_generation)
                return True
            if state in _TERMINAL_STATES:
                self._raise_for_state()
            return False
        with lock:
            self._check_access()
            if not self._active:
                raise BusyError("transfer slot has no active generation")
            expected_generation = self._generation
            state = self._refresh_unlocked()
            if state is WorkState.COMPLETED:
                self._finish_execute_success_unlocked(expected_generation)
                return True
            if state in _TERMINAL_STATES:
                self._raise_for_state()
            return False

    def test(self) -> bool:
        state = self._refresh()
        return state in _TERMINAL_STATES if state is not None else False

    def wait(self, timeout: float | None = None) -> TransferSlot:
        deadline_ns = _transfer_slot_deadline_ns(timeout)
        lock = self._lock
        if lock is None:
            expected_generation = self._generation_for_wait_unlocked()
        else:
            with lock:
                expected_generation = self._generation_for_wait_unlocked()
        return self._wait_until(
            deadline_ns,
            spin=False,
            expected_generation=expected_generation,
        )

    def start_and_wait(self, timeout: float | None = None) -> TransferSlot:
        """Start one generation and wait, using bounded provider fusion.

        The optional timeout is a total host-side budget beginning at method
        entry. It cannot preempt a blocking CUDA-readiness or provider call.
        Timeout leaves the generation active so it can be observed, cancelled,
        and explicitly recycled after reaching terminal state.
        """

        deadline_ns = _transfer_slot_deadline_ns(timeout)
        start_and_poll = self._backend_start_and_poll
        poll_bounded = self._backend_poll_bounded
        if start_and_poll is None:
            lock = self._lock
            if lock is None:
                self._start_unlocked()
                expected_generation = self._generation
            else:
                with lock:
                    self._start_unlocked()
                    expected_generation = self._generation
            if poll_bounded is not None:
                return self._wait_bounded_until(
                    deadline_ns,
                    expected_generation=expected_generation,
                    poll_bounded=poll_bounded,
                )
            return self._wait_until(
                deadline_ns, spin=True, expected_generation=expected_generation
            )

        lock = self._lock
        if lock is None:
            state = self._start_and_poll_unlocked(start_and_poll, deadline_ns)
            expected_generation = self._generation
            if state is WorkState.COMPLETED:
                return self
            if state in _TERMINAL_STATES:
                return self._raise_for_state()
        else:
            with lock:
                state = self._start_and_poll_unlocked(start_and_poll, deadline_ns)
                expected_generation = self._generation
                if state is WorkState.COMPLETED:
                    return self
                if state in _TERMINAL_STATES:
                    return self._raise_for_state()

        # A provider is allowed to exhaust its small bounded poll budget while
        # work remains active. Do not retain the per-slot lock across the
        # potentially long ordinary wait.
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            # Preserve the zero-timeout no-extra-poll contract without letting
            # a concurrent recycle/restart be misreported as a timeout.
            self._validate_wait_generation(expected_generation)
            raise TransferTimeoutError("transfer slot timed out")
        if poll_bounded is not None:
            return self._wait_bounded_until(
                deadline_ns,
                expected_generation=expected_generation,
                poll_bounded=poll_bounded,
            )
        return self._wait_until(
            deadline_ns,
            spin=True,
            expected_generation=expected_generation,
        )

    def execute(self, timeout: float | None = None) -> int:
        """Start, wait, and logically recycle one successful generation.

        The returned integer identifies the completed generation. Success makes
        the public slot idle before returning, so ``state`` is ``None`` and a
        later generation may start immediately. Failure, cancellation, and
        timeout deliberately retain the active generation for observation,
        cancellation, and explicit recycle.

        Providers advertising deferred successful recycle may fold their
        successful provider-side recycle into the next start or close boundary.
        Other providers are recycled before Core publishes the slot as idle.
        """

        fast_execute = self._fast_execute

        # Specialize the dominant small-transfer lifecycle as one straight-line
        # Core transaction: lock-free ownership, an unbounded fused provider
        # call, and a provider that retains clean successful receipts. This
        # preserves the generic transaction's checks and interruption
        # boundaries while avoiding four Python helper frames on immediate
        # completion.
        if timeout is None and fast_execute is not None:
            next_generation = self._generation + 1
            try:
                if self._closed:
                    raise ClosedError("transfer slot is closed")
                if self._creation_failed:
                    raise BackendFailureError(
                        "transfer slot is retained for cleanup only"
                    )
                if (
                    self._owner_thread is not None
                    and _GET_IDENT() != self._owner_thread
                ):
                    raise UnsupportedError(
                        "single-thread endpoint used from a different thread"
                    )
                if self._active:
                    raise BusyError(
                        "transfer slot already owns an unrecycled generation"
                    )

                if not self._caller_ready:
                    ready_event_synchronize = self._ready_event_synchronize
                    if ready_event_synchronize is not None:
                        try:
                            ready_event_synchronize()
                        except TransferError:
                            raise
                        except Exception as error:
                            raise BackendFailureError(
                                "failed to wait for CUDA readiness"
                            ) from error
                    else:
                        bound = self._bound
                        _establish_cuda_ordering_prevalidated(
                            bound._cuda_ordering,
                            bound._stream,
                            bound._ready_event,
                            bound._cuda_device,
                        )

                try:
                    # Idle already guarantees clear diagnostics and no active
                    # provider generation. `_active` is the sole publication
                    # commit; the provider dispatch owns durable request
                    # state after entry, so this synchronous SINGLE path needs
                    # no transient Core PENDING store.
                    self._generation = next_generation
                    self._active = True
                except BaseException:
                    self._active = False
                    self._generation = next_generation - 1
                    self._state = None
                    self._error = None
                    self._start_error = None
                    self._backend_generation_active = False
                    raise

                state = fast_execute(
                    max_polls=_FUSED_REQUEST_SLOT_MAX_POLLS,
                    timeout_ns=None,
                )
            except BaseException as error:
                if self._generation != next_generation:
                    raise
                # Once Core publishes the generation, a provider call may have
                # launched native work. Preserve the same ambiguous-start
                # recovery used by the generic fused path.
                self._start_error = error
                refreshed = self._refresh_unlocked()
                assert refreshed is not None
                state = refreshed
            else:
                if state is _WORK_COMPLETED:
                    # The provider completion receipt is already durable. Before
                    # idle commit, an active Core generation rediscovers it;
                    # after commit, Core is already clean-idle. No transient
                    # Core state stores are needed on this exact path.
                    self._active = False
                    return next_generation
                if state is _WORK_RUNNING:
                    # The exact RUNNING singleton is already a valid provider
                    # acknowledgement. Publish provider ownership before the
                    # observable state, then enter the cached continuation
                    # directly instead of reclassifying the initial result.
                    self._backend_generation_active = True
                    self._state = state
                    poll_bounded = self._backend_poll_bounded
                    if poll_bounded is not None:
                        return self._wait_execute_bounded_until(
                            None,
                            expected_generation=next_generation,
                            poll_bounded=poll_bounded,
                        )
                    return self._wait_execute_until(
                        None, expected_generation=next_generation
                    )
                if state is None:
                    self._start_error = BackendFailureError(
                        "backend fused request-slot start returned without "
                        "acknowledging a generation"
                    )
                    refreshed = self._refresh_unlocked()
                    assert refreshed is not None
                    state = refreshed
                elif not isinstance(state, WorkState):
                    self._start_error = BackendFailureError(
                        "backend returned invalid fused request-slot state "
                        f"{state!r}"
                    )
                    refreshed = self._refresh_unlocked()
                    assert refreshed is not None
                    state = refreshed
                else:
                    state = self._observe_backend_state_unlocked(state)

            expected_generation = self._generation
            if state in _TERMINAL_STATES:
                self._raise_for_state()
            poll_bounded = self._backend_poll_bounded
            if poll_bounded is not None:
                return self._wait_execute_bounded_until(
                    None,
                    expected_generation=expected_generation,
                    poll_bounded=poll_bounded,
                )
            return self._wait_execute_until(
                None, expected_generation=expected_generation
            )

        # Non-fast paths preserve the shared finite-timeout validator. Keeping
        # this below the fast return avoids a deadline local on the dominant
        # unbounded deferred SINGLE completion.
        deadline_ns = None if timeout is None else _transfer_slot_deadline_ns(timeout)
        start_and_poll = self._backend_start_and_poll
        lock = self._lock
        poll_bounded = self._backend_poll_bounded
        if start_and_poll is None:
            if lock is None:
                self._start_unlocked()
                expected_generation = self._generation
            else:
                with lock:
                    self._start_unlocked()
                    expected_generation = self._generation
            if poll_bounded is not None:
                return self._wait_execute_bounded_until(
                    deadline_ns,
                    expected_generation=expected_generation,
                    poll_bounded=poll_bounded,
                )
            return self._wait_execute_until(
                deadline_ns, expected_generation=expected_generation
            )

        if lock is None:
            state = self._start_and_poll_unlocked(start_and_poll, deadline_ns)
            if state is WorkState.COMPLETED:
                return self._finish_execute_success_unlocked(self._generation)
            expected_generation = self._generation
            if state in _TERMINAL_STATES:
                self._raise_for_state()
        else:
            with lock:
                state = self._start_and_poll_unlocked(start_and_poll, deadline_ns)
                expected_generation = self._generation
                if state is WorkState.COMPLETED:
                    return self._finish_execute_success_unlocked(expected_generation)
                if state in _TERMINAL_STATES:
                    self._raise_for_state()

        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            self._validate_wait_generation(expected_generation)
            raise TransferTimeoutError("transfer slot timed out")
        if poll_bounded is not None:
            return self._wait_execute_bounded_until(
                deadline_ns,
                expected_generation=expected_generation,
                poll_bounded=poll_bounded,
            )
        return self._wait_execute_until(
            deadline_ns, expected_generation=expected_generation
        )

    def _wait_until(
        self,
        deadline_ns: int | None,
        *,
        spin: bool,
        expected_generation: int,
    ) -> TransferSlot:
        polls = 0
        while True:
            if self._observe_wait_generation(expected_generation):
                return self
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TransferTimeoutError("transfer slot timed out")
            if self._endpoint._progress_mode is ProgressMode.MANUAL:
                self._endpoint.progress()
            polls += 1
            if not spin and polls >= 16:
                time.sleep(0 if polls < 128 else 0.0001)

    def _wait_execute_until(
        self,
        deadline_ns: int | None,
        *,
        expected_generation: int,
    ) -> int:
        """Spin until success and publish Core-idle in the observation lock."""

        while True:
            lock = self._lock
            if lock is None:
                completed_generation = self._observe_execute_generation_unlocked(
                    expected_generation
                )
            else:
                with lock:
                    completed_generation = self._observe_execute_generation_unlocked(
                        expected_generation
                    )
            if completed_generation is not None:
                return completed_generation
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                self._validate_wait_generation(expected_generation)
                raise TransferTimeoutError("transfer slot timed out")
            if self._endpoint._progress_mode is ProgressMode.MANUAL:
                self._endpoint.progress()

    def _wait_bounded_until(
        self,
        deadline_ns: int | None,
        *,
        expected_generation: int,
        poll_bounded: Callable[..., WorkState | None],
    ) -> TransferSlot:
        lock = self._lock
        while True:
            if lock is None:
                terminal = self._poll_bounded_generation_unlocked(
                    expected_generation, deadline_ns, poll_bounded
                )
            else:
                with lock:
                    terminal = self._poll_bounded_generation_unlocked(
                        expected_generation, deadline_ns, poll_bounded
                    )
            if terminal:
                return self
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TransferTimeoutError("transfer slot timed out")
            if self._endpoint._progress_mode is ProgressMode.MANUAL:
                self._endpoint.progress()

    def _wait_execute_bounded_until(
        self,
        deadline_ns: int | None,
        *,
        expected_generation: int,
        poll_bounded: Callable[..., WorkState | None],
    ) -> int:
        """Run bounded polls and publish successful recycle under the slot lock."""

        lock = self._lock
        while True:
            if lock is None:
                terminal = self._poll_bounded_generation_unlocked(
                    expected_generation, deadline_ns, poll_bounded
                )
                if terminal:
                    return self._finish_execute_success_unlocked(expected_generation)
            else:
                with lock:
                    terminal = self._poll_bounded_generation_unlocked(
                        expected_generation, deadline_ns, poll_bounded
                    )
                    if terminal:
                        return self._finish_execute_success_unlocked(
                            expected_generation
                        )
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                self._validate_wait_generation(expected_generation)
                raise TransferTimeoutError("transfer slot timed out")
            if self._endpoint._progress_mode is ProgressMode.MANUAL:
                self._endpoint.progress()

    def _poll_bounded_generation_unlocked(
        self,
        expected_generation: int,
        deadline_ns: int | None,
        poll_bounded: Callable[..., WorkState | None],
    ) -> bool:
        self._validate_wait_generation_unlocked(expected_generation)
        if self._state is WorkState.COMPLETED:
            return True
        if self._state in _TERMINAL_STATES:
            self._raise_for_state()
        configured_timeout_ns = _FUSED_REQUEST_SLOT_POLL_TIMEOUT_NS
        if deadline_ns is None:
            timeout_ns = configured_timeout_ns
        else:
            remaining_ns = max(0, deadline_ns - time.monotonic_ns())
            timeout_ns = (
                remaining_ns
                if configured_timeout_ns is None
                else min(configured_timeout_ns, remaining_ns)
            )
        if self._lock is None:
            try:
                state = poll_bounded(
                    max_polls=_FUSED_REQUEST_SLOT_POLL_MAX_POLLS,
                    timeout_ns=timeout_ns,
                    timeout_check_interval=(
                        _FUSED_REQUEST_SLOT_POLL_TIMEOUT_CHECK_INTERVAL
                    ),
                )
            except TransferError:
                raise
            except Exception as error:
                raise BackendFailureError(
                    f"backend {self._endpoint.backend!r} failed during bounded "
                    "persistent request polling"
                ) from error
        else:
            state = self._endpoint._backend_request_slot_poll_bounded(
                poll_bounded,
                max_polls=_FUSED_REQUEST_SLOT_POLL_MAX_POLLS,
                timeout_ns=timeout_ns,
                timeout_check_interval=(
                    _FUSED_REQUEST_SLOT_POLL_TIMEOUT_CHECK_INTERVAL
                ),
            )
        state = self._observe_backend_state_unlocked(state)
        if state is WorkState.COMPLETED:
            return True
        if state in _TERMINAL_STATES:
            self._raise_for_state()
        return False

    def cancel(self) -> bool:
        lock = self._lock
        if lock is None:
            return self._cancel_unlocked()
        with lock:
            return self._cancel_unlocked()

    def recycle(self) -> None:
        """Return a terminal generation to idle without changing its shape."""

        lock = self._lock
        if lock is None:
            self._recycle_unlocked()
            return
        with lock:
            self._recycle_unlocked()

    def close(self) -> None:
        lock = self._lock
        if lock is None:
            self._endpoint._close_transfer_slot(self)
            return
        with lock:
            self._endpoint._close_transfer_slot(self)

    def _adopt_backend_slot(
        self,
        backend_slot: BackendRequestSlot,
        *,
        execution_dispatch: object | None = None,
    ) -> None:
        if not isinstance(backend_slot, BackendRequestSlot):
            raise BackendFailureError(
                "backend must adopt a BackendRequestSlot instance"
            )
        lock = self._lock
        if lock is None:
            self._adopt_backend_slot_unlocked(
                backend_slot, execution_dispatch=execution_dispatch
            )
            return
        with lock:
            self._adopt_backend_slot_unlocked(
                backend_slot, execution_dispatch=execution_dispatch
            )

    def _refresh(self) -> WorkState | None:
        lock = self._lock
        if lock is None:
            self._check_access()
            return self._refresh_unlocked()
        with lock:
            self._check_access()
            return self._refresh_unlocked()

    def _start_unlocked(self) -> None:
        previous_generation = self._generation
        try:
            self._begin_generation_unlocked()
            start = self._backend_start
            if start is None:
                self._start_error = BackendFailureError(
                    "transfer slot has no adopted backend request"
                )
                return
            if self._lock is None:
                result = start()
            else:
                result = self._endpoint._backend_request_slot_start(start)
            # A normal return acknowledges that provider start was entered,
            # even when the provider then violates the required None return.
            self._backend_generation_active = True
            if result is not None:
                raise BackendFailureError("backend request-slot start must return None")
        except BaseException as error:
            if self._generation == previous_generation:
                # CUDA readiness or an interrupted pre-commit publication did
                # not create a generation and must remain a direct exception.
                raise
            # Native launch is ambiguous once the provider call has begun.
            # Preserve ownership; a conforming provider exposes terminal state
            # or permits cancellation rather than stranding this generation.
            self._start_error = error

    def _begin_generation_unlocked(self) -> None:
        bound = self._check_access()
        if self._active:
            raise BusyError("transfer slot already owns an unrecycled generation")

        if self._state is not None:
            self._backend_generation_active = False
            self._error = None
            self._start_error = None
            self._state = None

        # CUDA readiness failure is unambiguously before provider activity, so
        # it does not consume a generation. This runs outside endpoint state
        # and provider locks and is repeated for every start.
        if not self._caller_ready:
            ready_event_synchronize = self._ready_event_synchronize
            if ready_event_synchronize is not None:
                try:
                    ready_event_synchronize()
                except TransferError:
                    raise
                except Exception as error:
                    raise BackendFailureError(
                        "failed to wait for CUDA readiness"
                    ) from error
            else:
                _establish_cuda_ordering_prevalidated(
                    bound._cuda_ordering,
                    bound._stream,
                    bound._ready_event,
                    bound._cuda_device,
                )

        previous_generation = self._generation
        try:
            # A generic/mixed generation temporarily disables the exact execute
            # path. Its idle transition restores the cached callable only after
            # all private state is clean. CUDA readiness precedes this store, so
            # an unambiguous readiness failure preserves fast eligibility.
            self._fast_execute = None
            # ``_active`` is the publication commit and is deliberately last.
            # An interrupt before it leaves an idle slot; an interrupt after it
            # leaves a complete, self-consistent Core generation that the
            # caller can observe and recycle without touching an idle provider.
            self._state = WorkState.PENDING
            self._error = None
            self._start_error = None
            self._backend_generation_active = False
            self._generation = previous_generation + 1
            self._active = True
        except BaseException:
            self._active = False
            self._generation = previous_generation
            self._state = None
            self._error = None
            self._start_error = None
            self._backend_generation_active = False
            self._restore_fast_execute_unlocked()
            raise

    def _start_and_poll_unlocked(
        self,
        start_and_poll: Callable[..., WorkState | None],
        deadline_ns: int | None,
        *,
        max_polls: int = _FUSED_REQUEST_SLOT_MAX_POLLS,
    ) -> WorkState:
        previous_generation = self._generation
        try:
            self._begin_generation_unlocked()
            timeout_ns = (
                None
                if deadline_ns is None
                else min(
                    _MAX_FUSED_REQUEST_SLOT_TIMEOUT_NS,
                    max(0, deadline_ns - time.monotonic_ns()),
                )
            )
            if self._lock is None:
                state = start_and_poll(
                    max_polls=max_polls,
                    timeout_ns=timeout_ns,
                )
            else:
                state = self._endpoint._backend_request_slot_start_and_poll(
                    start_and_poll,
                    max_polls=max_polls,
                    timeout_ns=timeout_ns,
                )
        except BaseException as error:
            if self._generation == previous_generation:
                raise
            self._start_error = error
            refreshed = self._refresh_unlocked()
            assert refreshed is not None
            return refreshed
        if state is None:
            self._start_error = BackendFailureError(
                "backend fused request-slot start returned without acknowledging "
                "a generation"
            )
            refreshed = self._refresh_unlocked()
            assert refreshed is not None
            return refreshed
        if not isinstance(state, WorkState):
            self._start_error = BackendFailureError(
                f"backend returned invalid fused request-slot state {state!r}"
            )
            refreshed = self._refresh_unlocked()
            assert refreshed is not None
            return refreshed
        return self._observe_backend_state_unlocked(state)

    def _completion_event_unlocked(self) -> object | None:
        self._check_access()
        if not self._active:
            raise BusyError("transfer slot has no active generation")
        if not self._endpoint.capabilities.cuda_completion_event:
            return None
        backend_slot = self._require_backend_slot()
        return self._endpoint._backend_request_slot_completion_event(backend_slot)

    def _generation_for_wait_unlocked(self) -> int:
        self._check_access()
        if not self._active:
            raise BusyError("transfer slot has no active generation")
        return self._generation

    def _observe_wait_generation(self, expected_generation: int) -> bool:
        lock = self._lock
        if lock is None:
            return self._observe_wait_generation_unlocked(expected_generation)
        with lock:
            return self._observe_wait_generation_unlocked(expected_generation)

    def _validate_wait_generation(self, expected_generation: int) -> None:
        lock = self._lock
        if lock is None:
            self._validate_wait_generation_unlocked(expected_generation)
            return
        with lock:
            self._validate_wait_generation_unlocked(expected_generation)

    def _validate_wait_generation_unlocked(self, expected_generation: int) -> None:
        self._check_access()
        if not self._active or self._generation != expected_generation:
            raise BusyError(
                f"transfer slot generation {expected_generation} changed while "
                "waiting"
            )

    def _observe_wait_generation_unlocked(self, expected_generation: int) -> bool:
        self._check_access()
        if not self._active or self._generation != expected_generation:
            raise BusyError(
                f"transfer slot generation {expected_generation} changed while "
                "waiting"
            )
        state = self._refresh_unlocked()
        if state in _TERMINAL_STATES:
            self._raise_for_state()
            return True
        return False

    def _observe_execute_generation_unlocked(
        self, expected_generation: int
    ) -> int | None:
        self._validate_wait_generation_unlocked(expected_generation)
        state = self._refresh_unlocked()
        if state is WorkState.COMPLETED:
            return self._finish_execute_success_unlocked(expected_generation)
        if state in _TERMINAL_STATES:
            self._raise_for_state()
        return None

    def _cancel_unlocked(self) -> bool:
        self._check_access()
        if not self._active:
            return False
        if self._refresh_unlocked() in _TERMINAL_STATES:
            return self._state is WorkState.CANCELLED
        cancel = self._backend_cancel
        if cancel is None:
            raise BackendFailureError("transfer slot has no adopted backend request")
        cancelled = self._endpoint._backend_request_slot_cancel(cancel)
        if not isinstance(cancelled, bool):
            raise BackendFailureError("backend slot cancellation must return a bool")
        self._refresh_unlocked()
        return cancelled and self._state is WorkState.CANCELLED

    def _recycle_unlocked(self) -> None:
        self._check_access()
        if not self._active:
            raise BusyError("transfer slot has no generation to recycle")
        self._fast_execute = None
        if self._refresh_unlocked() not in _TERMINAL_STATES:
            raise BusyError("cannot recycle a non-terminal transfer slot")
        if self._backend_generation_active:
            recycle = self._backend_recycle
            if recycle is None:
                raise BackendFailureError(
                    "transfer slot has no adopted backend request"
                )
            if self._lock is None:
                try:
                    result = recycle()
                    if result is not None:
                        raise BackendFailureError(
                            "backend request-slot recycle must return None"
                        )
                except TransferError:
                    raise
                except Exception as error:
                    raise BackendFailureError(
                        f"backend {self._endpoint.backend!r} failed to recycle a "
                        "persistent request"
                    ) from error
            else:
                self._endpoint._backend_request_slot_recycle(recycle)
        # ``_active = False`` is the sole public idle-state commit. Private
        # terminal diagnostics may remain briefly if an asynchronous exception
        # lands after it; public state/error ignore them while idle, and the
        # next generation overwrites them before its own active commit.
        self._active = False
        self._backend_generation_active = False
        self._error = None
        self._start_error = None
        # Clear the dirty-idle marker only after every other private field.
        self._state = None
        self._restore_fast_execute_unlocked()

    def _finish_execute_success_unlocked(self, expected_generation: int) -> int:
        """Commit a successful full-cycle generation while holding its lock."""

        # Every caller has already validated access and generation while holding
        # this slot's serialization boundary. Retain the cheap inline ABA guard
        # without repeating `_check_access()` on every successful generation.
        if not self._active or self._generation != expected_generation:
            raise BusyError(
                f"transfer slot generation {expected_generation} changed while "
                "waiting"
            )
        if self._state is not WorkState.COMPLETED:
            raise BackendFailureError(
                "transfer slot execute success requires completed provider state"
            )
        self._fast_execute = None
        if (
            self._backend_generation_active
            and not self._backend_deferred_successful_recycle
        ):
            recycle = self._backend_recycle
            if recycle is None:
                raise BackendFailureError(
                    "transfer slot has no adopted backend request"
                )
            if self._lock is None:
                try:
                    result = recycle()
                    if result is not None:
                        raise BackendFailureError(
                            "backend request-slot recycle must return None"
                        )
                except TransferError:
                    raise
                except Exception as error:
                    raise BackendFailureError(
                        f"backend {self._endpoint.backend!r} failed to recycle a "
                        "persistent request"
                    ) from error
            else:
                self._endpoint._backend_request_slot_recycle(recycle)

        completed_generation = self._generation
        # The provider retains an observable COMPLETED receipt when its recycle
        # is deferred. ``_active = False`` is the sole public idle commit; the
        # private terminal state is cleared afterward. Restore the exact fast
        # callable only after every private field is clean.
        self._active = False
        self._backend_generation_active = False
        self._error = None
        self._start_error = None
        self._state = None
        self._restore_fast_execute_unlocked()
        return completed_generation

    def _restore_fast_execute_unlocked(self) -> None:
        """Publish exact-execute eligibility after a completely clean idle."""

        start_and_poll = self._backend_start_and_poll
        self._fast_execute = (
            start_and_poll
            if (
                self._lock is None
                and start_and_poll is not None
                and self._backend_deferred_successful_recycle
            )
            else None
        )

    def _adopt_backend_slot_unlocked(
        self,
        backend_slot: BackendRequestSlot,
        *,
        execution_dispatch: object | None,
    ) -> None:
        if self._closed:
            raise BackendFailureError(
                "backend attempted to adopt an expired transfer slot"
            )
        if self._backend_slot is not None:
            raise BackendFailureError(
                "backend attempted to adopt transfer slot more than once"
            )
        # Allocate every bound method before publishing `_backend_slot` as the
        # adoption commit marker. An asynchronous interruption therefore leaves
        # either a wholly unadopted slot or a fully usable one, never a public
        # slot with only part of its steady-state dispatch table installed.
        dispatch = backend_slot if execution_dispatch is None else execution_dispatch
        start = dispatch.start
        poll_state = dispatch.poll_state
        start_and_poll = (
            dispatch.start_and_poll
            if self._endpoint.capabilities.fused_request_slot_start_poll
            else None
        )
        poll_bounded = (
            dispatch.poll_bounded
            if self._endpoint.capabilities.fused_request_slot_poll
            else None
        )
        cancel = dispatch.cancel
        recycle = dispatch.recycle
        close = backend_slot.close
        operations = (
            ("start", start),
            ("poll_state", poll_state),
            ("cancel", cancel),
            ("recycle", recycle),
            ("close", close),
        )
        if self._endpoint.capabilities.fused_request_slot_start_poll:
            operations += (("start_and_poll", start_and_poll),)
        if self._endpoint.capabilities.fused_request_slot_poll:
            operations += (("poll_bounded", poll_bounded),)
        for name, operation in operations:
            if not callable(operation):
                raise BackendFailureError(
                    f"backend request-slot {name} operation must be callable"
                )
        self._backend_start = start
        self._backend_poll_state = poll_state
        self._backend_start_and_poll = start_and_poll
        self._backend_poll_bounded = poll_bounded
        self._backend_cancel = cancel
        self._backend_recycle = recycle
        self._backend_close = close
        self._backend_execution_dispatch = dispatch
        self._fast_execute = (
            start_and_poll
            if (
                self._lock is None
                and start_and_poll is not None
                and self._backend_deferred_successful_recycle
            )
            else None
        )
        self._backend_slot = backend_slot

    def _refresh_unlocked(self) -> WorkState | None:
        if not self._active:
            return None
        if self._state is WorkState.COMPLETED:
            return self._state
        if self._state in _TERMINAL_STATES:
            return self._state
        poll_state = self._backend_poll_state
        if poll_state is None:
            raise BackendFailureError(
                "transfer slot has no backend state-poll operation"
            )
        if self._lock is None:
            try:
                state = poll_state()
            except TransferError:
                raise
            except Exception as error:
                raise BackendFailureError(
                    f"backend {self._endpoint.backend!r} failed to query "
                    "persistent request state"
                ) from error
        else:
            state = self._endpoint._backend_request_slot_state(poll_state)
        if state is None:
            if self._start_error is None:
                self._start_error = BackendFailureError(
                    "backend request-slot start returned without acknowledging "
                    "a generation"
                )
            self._error = self._start_error
            self._state = WorkState.FAILED
            return self._state
        return self._observe_backend_state_unlocked(state)

    def _observe_backend_state_unlocked(self, state: object) -> WorkState:
        if not isinstance(state, WorkState):
            raise BackendFailureError(
                f"backend returned invalid transfer slot state {state!r}"
            )
        if self._state is WorkState.RUNNING and state is WorkState.PENDING:
            raise BackendFailureError(
                "backend transfer slot state regressed from running to pending"
            )
        if not self._backend_generation_active:
            self._backend_generation_active = True
        if state is WorkState.COMPLETED and self._start_error is None:
            self._state = state
            return state
        observed_state = state
        observed_error: BaseException | None = None
        if state in _TERMINAL_STATES and self._start_error is not None:
            observed_state = WorkState.FAILED
            observed_error = self._start_error
        elif state is WorkState.FAILED:
            try:
                backend_slot = self._require_backend_slot()
                error = self._endpoint._backend_request_slot_error(backend_slot)
            except Exception as observation_error:
                observed_error = observation_error
            else:
                if error is None:
                    error = BackendFailureError(
                        "transfer backend reported slot failure without an error"
                    )
                elif not isinstance(error, BaseException):
                    error = BackendFailureError(
                        "transfer backend returned a non-exception slot error"
                    )
                observed_error = error
        elif state is WorkState.CANCELLED:
            observed_error = TransferCancelledError("transfer slot was cancelled")

        # Error construction/provider inspection happens first; terminal state
        # is the commit marker. A BaseException during error discovery leaves
        # the prior nonterminal state, so the next observation retries instead
        # of permanently caching a terminal state with no diagnostic.
        if observed_state in _TERMINAL_STATES:
            self._error = observed_error
        self._state = observed_state
        return observed_state

    def _raise_for_state(self) -> TransferSlot:
        if self._state is WorkState.COMPLETED:
            return self
        if self._error is None:
            self._error = BackendFailureError(
                f"transfer slot reached {self._state.value} without an error"
            )
        if isinstance(self._error, TransferError):
            raise self._error
        raise BackendFailureError("transfer slot failed") from self._error

    def _require_backend_slot(self) -> BackendRequestSlot:
        backend_slot = self._backend_slot
        if backend_slot is None:
            raise BackendFailureError("transfer slot has no adopted backend request")
        return backend_slot

    def _check_access(self) -> BoundTransfer:
        if self._closed:
            raise ClosedError("transfer slot is closed")
        if self._creation_failed:
            raise BackendFailureError("transfer slot is retained for cleanup only")
        bound = self._bound
        if bound is None:
            raise ClosedError("transfer slot is closed")
        owner_thread = self._owner_thread
        if owner_thread is not None and threading.get_ident() != owner_thread:
            raise UnsupportedError(
                "single-thread endpoint used from a different thread"
            )
        # An open slot prevents its BoundTransfer, plan, and endpoint from
        # closing. Creation validates that graph once, so polling need not walk
        # it again on every request-state query.
        return bound

    def __enter__(self) -> Self:
        self._check_access()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class RequestLocalNotificationSlot(TransferSlot):
    """Persistent request with an explicit notification per generation.

    The immutable operation, descriptors, index selection, CUDA-ordering
    contract, and native request storage are inherited from ``TransferSlot``.
    Only the attached notification varies. Every start must state either a
    non-empty bytes-like payload or ``None``; the latter is an explicit native
    clear, not "reuse the previous value". The distinct type and methods keep
    all validation and dispatch branches out of fixed ``TransferSlot.execute``.
    """

    __slots__ = (
        "_backend_start_with_notification",
        "_backend_start_and_poll_with_notification",
        "_notification_fast_execute",
    )

    def __init__(
        self,
        endpoint: Endpoint,
        bound: BoundTransfer,
        *,
        _token: object,
    ) -> None:
        super().__init__(endpoint, bound, _token=_token)
        self._backend_start_with_notification: Callable[[bytes | None], None] | None = (
            None
        )
        self._backend_start_and_poll_with_notification: (
            Callable[..., WorkState | None] | None
        ) = None
        self._notification_fast_execute: Callable[..., WorkState | None] | None = None

    def start(
        self,
        *,
        notification: bytes | bytearray | memoryview | None,
    ) -> None:
        """Start one generation with an explicit notification or clear."""

        if notification is not None:
            if type(notification) is bytes:
                if not notification:
                    raise ValueError(
                        "an attached notification payload must be non-empty"
                    )
                if len(notification) > _MAX_NOTIFICATION_BYTES:
                    raise ValueError(
                        "notification exceeds the "
                        f"{_MAX_NOTIFICATION_BYTES}-byte size limit"
                    )
            else:
                notification = _validate_request_slot_notification(notification)
        lock = self._lock
        if lock is None:
            self._start_with_notification_unlocked(notification)
            return
        with lock:
            self._start_with_notification_unlocked(notification)

    def start_and_wait(
        self,
        *,
        notification: bytes | bytearray | memoryview | None,
        timeout: float | None = None,
    ) -> RequestLocalNotificationSlot:
        """Start with an explicit notification and wait for terminal state."""

        if notification is not None:
            if type(notification) is bytes:
                if not notification:
                    raise ValueError(
                        "an attached notification payload must be non-empty"
                    )
                if len(notification) > _MAX_NOTIFICATION_BYTES:
                    raise ValueError(
                        "notification exceeds the "
                        f"{_MAX_NOTIFICATION_BYTES}-byte size limit"
                    )
            else:
                notification = _validate_request_slot_notification(notification)
        deadline_ns = _transfer_slot_deadline_ns(timeout)
        start_and_poll = self._backend_start_and_poll_with_notification
        poll_bounded = self._backend_poll_bounded
        if start_and_poll is None:
            lock = self._lock
            if lock is None:
                self._start_with_notification_unlocked(notification)
                expected_generation = self._generation
            else:
                with lock:
                    self._start_with_notification_unlocked(notification)
                    expected_generation = self._generation
            if poll_bounded is not None:
                self._wait_bounded_until(
                    deadline_ns,
                    expected_generation=expected_generation,
                    poll_bounded=poll_bounded,
                )
            else:
                self._wait_until(
                    deadline_ns,
                    spin=True,
                    expected_generation=expected_generation,
                )
            return self

        lock = self._lock
        if lock is None:
            state = self._start_and_poll_with_notification_unlocked(
                start_and_poll, notification, deadline_ns
            )
            expected_generation = self._generation
            if state is WorkState.COMPLETED:
                return self
            if state in _TERMINAL_STATES:
                return cast(RequestLocalNotificationSlot, self._raise_for_state())
        else:
            with lock:
                state = self._start_and_poll_with_notification_unlocked(
                    start_and_poll, notification, deadline_ns
                )
                expected_generation = self._generation
                if state is WorkState.COMPLETED:
                    return self
                if state in _TERMINAL_STATES:
                    return cast(RequestLocalNotificationSlot, self._raise_for_state())

        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            self._validate_wait_generation(expected_generation)
            raise TransferTimeoutError("transfer slot timed out")
        if poll_bounded is not None:
            self._wait_bounded_until(
                deadline_ns,
                expected_generation=expected_generation,
                poll_bounded=poll_bounded,
            )
        else:
            self._wait_until(
                deadline_ns,
                spin=True,
                expected_generation=expected_generation,
            )
        return self

    def start_and_test(
        self,
        *,
        notification: bytes | bytearray | memoryview | None,
    ) -> bool:
        """Start with an explicit tag/clear and consume immediate success."""

        if notification is not None:
            if type(notification) is bytes:
                if not notification:
                    raise ValueError(
                        "an attached notification payload must be non-empty"
                    )
                if len(notification) > _MAX_NOTIFICATION_BYTES:
                    raise ValueError(
                        "notification exceeds the "
                        f"{_MAX_NOTIFICATION_BYTES}-byte size limit"
                    )
            else:
                notification = _validate_request_slot_notification(notification)

        fast_start_and_test = self._notification_fast_execute
        if fast_start_and_test is not None:
            next_generation = self._generation + 1
            try:
                if self._closed:
                    raise ClosedError("transfer slot is closed")
                if self._creation_failed:
                    raise BackendFailureError(
                        "transfer slot is retained for cleanup only"
                    )
                if (
                    self._owner_thread is not None
                    and _GET_IDENT() != self._owner_thread
                ):
                    raise UnsupportedError(
                        "single-thread endpoint used from a different thread"
                    )
                if self._active:
                    raise BusyError(
                        "transfer slot already owns an unrecycled generation"
                    )

                if not self._caller_ready:
                    ready_event_synchronize = self._ready_event_synchronize
                    if ready_event_synchronize is not None:
                        try:
                            ready_event_synchronize()
                        except TransferError:
                            raise
                        except Exception as error:
                            raise BackendFailureError(
                                "failed to wait for CUDA readiness"
                            ) from error
                    else:
                        bound = self._bound
                        _establish_cuda_ordering_prevalidated(
                            bound._cuda_ordering,
                            bound._stream,
                            bound._ready_event,
                            bound._cuda_device,
                        )

                try:
                    self._generation = next_generation
                    self._active = True
                except BaseException:
                    self._active = False
                    self._generation = next_generation - 1
                    self._state = None
                    self._error = None
                    self._start_error = None
                    self._backend_generation_active = False
                    raise

                state = fast_start_and_test(
                    notification,
                    max_polls=0,
                    timeout_ns=None,
                )
            except BaseException as error:
                if self._generation != next_generation:
                    raise
                self._start_error = error
                refreshed = self._refresh_unlocked()
                assert refreshed is not None
                state = refreshed
            else:
                if state is _WORK_COMPLETED:
                    self._active = False
                    return True
                if state is _WORK_RUNNING:
                    # Match the fixed-notification slot's exact QD-many path:
                    # the provider already authenticated the enum and Core
                    # needs only to publish the acknowledged RUNNING state.
                    self._backend_generation_active = True
                    self._state = state
                    return False
                if state is None:
                    self._start_error = BackendFailureError(
                        "backend fused request-slot notification start returned "
                        "without acknowledging a generation"
                    )
                    refreshed = self._refresh_unlocked()
                    assert refreshed is not None
                    state = refreshed
                elif not isinstance(state, WorkState):
                    self._start_error = BackendFailureError(
                        "backend returned invalid fused request-slot state "
                        f"{state!r}"
                    )
                    refreshed = self._refresh_unlocked()
                    assert refreshed is not None
                    state = refreshed
                else:
                    state = self._observe_backend_state_unlocked(state)

            if state in _TERMINAL_STATES:
                self._raise_for_state()
            return False

        start_and_poll = self._backend_start_and_poll_with_notification
        lock = self._lock
        if start_and_poll is None:
            if lock is None:
                return self._start_and_test_with_notification_fallback_unlocked(
                    notification
                )
            with lock:
                return self._start_and_test_with_notification_fallback_unlocked(
                    notification
                )
        if lock is None:
            state = self._start_and_poll_with_notification_unlocked(
                start_and_poll,
                notification,
                None,
                max_polls=0,
            )
            expected_generation = self._generation
            if state is WorkState.COMPLETED:
                self._finish_execute_success_unlocked(expected_generation)
                return True
            if state in _TERMINAL_STATES:
                self._raise_for_state()
            return False
        with lock:
            state = self._start_and_poll_with_notification_unlocked(
                start_and_poll,
                notification,
                None,
                max_polls=0,
            )
            expected_generation = self._generation
            if state is WorkState.COMPLETED:
                self._finish_execute_success_unlocked(expected_generation)
                return True
            if state in _TERMINAL_STATES:
                self._raise_for_state()
            return False

    def _start_and_test_with_notification_fallback_unlocked(
        self, notification: bytes | None
    ) -> bool:
        """Keep tagged fallback start/observe/finalize under one slot lock."""

        self._start_with_notification_unlocked(notification)
        expected_generation = self._generation
        state = self._refresh_unlocked()
        if state is WorkState.COMPLETED:
            self._finish_execute_success_unlocked(expected_generation)
            return True
        if state in _TERMINAL_STATES:
            self._raise_for_state()
        return False

    def execute(
        self,
        *,
        notification: bytes | bytearray | memoryview | None,
        timeout: float | None = None,
    ) -> int:
        """Execute and logically recycle one explicitly tagged generation."""

        # Keep the common bytes/None cases inline: unlike the fixed slot this
        # mode necessarily examines a request-local value, but it should not
        # add a helper frame before the one fused pybind call.
        if notification is not None:
            if type(notification) is bytes:
                if not notification:
                    raise ValueError(
                        "an attached notification payload must be non-empty"
                    )
                if len(notification) > _MAX_NOTIFICATION_BYTES:
                    raise ValueError(
                        "notification exceeds the "
                        f"{_MAX_NOTIFICATION_BYTES}-byte size limit"
                    )
            else:
                notification = _validate_request_slot_notification(notification)

        fast_execute = self._notification_fast_execute
        if timeout is None and fast_execute is not None:
            next_generation = self._generation + 1
            try:
                if self._closed:
                    raise ClosedError("transfer slot is closed")
                if self._creation_failed:
                    raise BackendFailureError(
                        "transfer slot is retained for cleanup only"
                    )
                if (
                    self._owner_thread is not None
                    and _GET_IDENT() != self._owner_thread
                ):
                    raise UnsupportedError(
                        "single-thread endpoint used from a different thread"
                    )
                if self._active:
                    raise BusyError(
                        "transfer slot already owns an unrecycled generation"
                    )

                if not self._caller_ready:
                    ready_event_synchronize = self._ready_event_synchronize
                    if ready_event_synchronize is not None:
                        try:
                            ready_event_synchronize()
                        except TransferError:
                            raise
                        except Exception as error:
                            raise BackendFailureError(
                                "failed to wait for CUDA readiness"
                            ) from error
                    else:
                        bound = self._bound
                        _establish_cuda_ordering_prevalidated(
                            bound._cuda_ordering,
                            bound._stream,
                            bound._ready_event,
                            bound._cuda_device,
                        )

                try:
                    self._generation = next_generation
                    self._active = True
                except BaseException:
                    self._active = False
                    self._generation = next_generation - 1
                    self._state = None
                    self._error = None
                    self._start_error = None
                    self._backend_generation_active = False
                    raise

                state = fast_execute(
                    notification,
                    max_polls=_FUSED_REQUEST_SLOT_MAX_POLLS,
                    timeout_ns=None,
                )
            except BaseException as error:
                if self._generation != next_generation:
                    raise
                self._start_error = error
                refreshed = self._refresh_unlocked()
                assert refreshed is not None
                state = refreshed
            else:
                if state is _WORK_COMPLETED:
                    self._active = False
                    return next_generation
                if state is _WORK_RUNNING:
                    # Preserve the same ownership-before-state publication as
                    # the fixed slot. Disable the request-local dispatcher
                    # before the inherited continuation: an interruption after
                    # its idle commit must not leave a stale fast dispatcher.
                    self._backend_generation_active = True
                    self._state = state
                    self._notification_fast_execute = None
                    poll_bounded = self._backend_poll_bounded
                    if poll_bounded is not None:
                        return self._wait_execute_bounded_until(
                            None,
                            expected_generation=next_generation,
                            poll_bounded=poll_bounded,
                        )
                    return self._wait_execute_until(
                        None, expected_generation=next_generation
                    )
                if state is None:
                    self._start_error = BackendFailureError(
                        "backend fused request-slot notification start returned "
                        "without acknowledging a generation"
                    )
                    refreshed = self._refresh_unlocked()
                    assert refreshed is not None
                    state = refreshed
                elif not isinstance(state, WorkState):
                    self._start_error = BackendFailureError(
                        "backend returned invalid fused request-slot state "
                        f"{state!r}"
                    )
                    refreshed = self._refresh_unlocked()
                    assert refreshed is not None
                    state = refreshed
                else:
                    state = self._observe_backend_state_unlocked(state)

            expected_generation = self._generation
            if state in _TERMINAL_STATES:
                self._raise_for_state()
            # A bounded continuation finalizes through the inherited idle
            # transaction. Disable this subclass's exact dispatcher first so
            # an interruption after that transaction's public idle commit
            # cannot pair it with a stale terminal state.
            self._notification_fast_execute = None
            poll_bounded = self._backend_poll_bounded
            if poll_bounded is not None:
                return self._wait_execute_bounded_until(
                    None,
                    expected_generation=expected_generation,
                    poll_bounded=poll_bounded,
                )
            return self._wait_execute_until(
                None,
                expected_generation=expected_generation,
            )

        deadline_ns = None if timeout is None else _transfer_slot_deadline_ns(timeout)
        start_and_poll = self._backend_start_and_poll_with_notification
        lock = self._lock
        poll_bounded = self._backend_poll_bounded
        if start_and_poll is None:
            if lock is None:
                self._start_with_notification_unlocked(notification)
                expected_generation = self._generation
            else:
                with lock:
                    self._start_with_notification_unlocked(notification)
                    expected_generation = self._generation
            if poll_bounded is not None:
                return self._wait_execute_bounded_until(
                    deadline_ns,
                    expected_generation=expected_generation,
                    poll_bounded=poll_bounded,
                )
            return self._wait_execute_until(
                deadline_ns,
                expected_generation=expected_generation,
            )

        if lock is None:
            state = self._start_and_poll_with_notification_unlocked(
                start_and_poll,
                notification,
                deadline_ns,
            )
            expected_generation = self._generation
            if state is WorkState.COMPLETED:
                return self._finish_execute_success_unlocked(expected_generation)
            if state in _TERMINAL_STATES:
                self._raise_for_state()
        else:
            with lock:
                state = self._start_and_poll_with_notification_unlocked(
                    start_and_poll,
                    notification,
                    deadline_ns,
                )
                expected_generation = self._generation
                if state is WorkState.COMPLETED:
                    return self._finish_execute_success_unlocked(expected_generation)
                if state in _TERMINAL_STATES:
                    self._raise_for_state()

        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            self._validate_wait_generation(expected_generation)
            raise TransferTimeoutError("transfer slot timed out")
        if poll_bounded is not None:
            return self._wait_execute_bounded_until(
                deadline_ns,
                expected_generation=expected_generation,
                poll_bounded=poll_bounded,
            )
        return self._wait_execute_until(
            deadline_ns,
            expected_generation=expected_generation,
        )

    def _start_with_notification_unlocked(self, notification: bytes | None) -> None:
        previous_generation = self._generation
        try:
            self._begin_generation_unlocked()
            self._notification_fast_execute = None
            start = self._backend_start_with_notification
            if start is None:
                self._start_error = BackendFailureError(
                    "transfer slot has no notification-override operation"
                )
                return
            if self._lock is None:
                result = start(notification)
            else:
                result = self._endpoint._backend_request_slot_start_with_notification(
                    start, notification
                )
            self._backend_generation_active = True
            if result is not None:
                raise BackendFailureError(
                    "backend request-slot start_with_notification must return None"
                )
        except BaseException as error:
            if self._generation == previous_generation:
                raise
            self._start_error = error

    def _start_and_poll_with_notification_unlocked(
        self,
        start_and_poll: Callable[..., WorkState | None],
        notification: bytes | None,
        deadline_ns: int | None,
        *,
        max_polls: int = _FUSED_REQUEST_SLOT_MAX_POLLS,
    ) -> WorkState:
        previous_generation = self._generation
        try:
            self._begin_generation_unlocked()
            self._notification_fast_execute = None
            timeout_ns = (
                None
                if deadline_ns is None
                else min(
                    _MAX_FUSED_REQUEST_SLOT_TIMEOUT_NS,
                    max(0, deadline_ns - time.monotonic_ns()),
                )
            )
            if self._lock is None:
                state = start_and_poll(
                    notification,
                    max_polls=max_polls,
                    timeout_ns=timeout_ns,
                )
            else:
                state = self._endpoint._backend_request_slot_start_and_poll_with_notification(
                    start_and_poll,
                    notification,
                    max_polls=max_polls,
                    timeout_ns=timeout_ns,
                )
        except BaseException as error:
            if self._generation == previous_generation:
                raise
            self._start_error = error
            refreshed = self._refresh_unlocked()
            assert refreshed is not None
            return refreshed
        if state is None:
            self._start_error = BackendFailureError(
                "backend fused request-slot notification start returned without "
                "acknowledging a generation"
            )
            refreshed = self._refresh_unlocked()
            assert refreshed is not None
            return refreshed
        if not isinstance(state, WorkState):
            self._start_error = BackendFailureError(
                f"backend returned invalid fused request-slot state {state!r}"
            )
            refreshed = self._refresh_unlocked()
            assert refreshed is not None
            return refreshed
        return self._observe_backend_state_unlocked(state)

    def _recycle_unlocked(self) -> None:
        """Disable notification fast dispatch before an exceptional recycle."""

        # Do not penalize or mutate an erroneous idle recycle. Active failure
        # and cancellation recovery, however, may publish public idle before
        # clearing retained diagnostics in the base transaction. Keeping the
        # notification dispatcher disabled makes that dirty-idle boundary a
        # safe generic fallback.
        if self._active:
            self._notification_fast_execute = None
        super()._recycle_unlocked()

    def _restore_fast_execute_unlocked(self) -> None:
        """Restore only the notification-aware exact-execute callable."""

        self._fast_execute = None
        start_and_poll = self._backend_start_and_poll_with_notification
        self._notification_fast_execute = (
            start_and_poll
            if (
                self._lock is None
                and start_and_poll is not None
                and self._backend_deferred_successful_recycle
            )
            else None
        )

    def _adopt_backend_slot_unlocked(
        self,
        backend_slot: BackendRequestSlot,
        *,
        execution_dispatch: object | None,
    ) -> None:
        dispatch = backend_slot if execution_dispatch is None else execution_dispatch
        start = getattr(dispatch, "start_with_notification", None)
        start_and_poll = (
            getattr(dispatch, "start_and_poll_with_notification", None)
            if self._endpoint.capabilities.fused_request_slot_start_poll
            else None
        )
        if not callable(start):
            raise BackendFailureError(
                "backend request-slot start_with_notification operation must be "
                "callable"
            )
        if (
            getattr(start, "__func__", None)
            is BackendRequestSlot.start_with_notification
        ):
            raise BackendFailureError(
                "backend advertised request-slot notification overrides without "
                "implementing start_with_notification"
            )
        if self._endpoint.capabilities.fused_request_slot_start_poll and not callable(
            start_and_poll
        ):
            raise BackendFailureError(
                "backend request-slot start_and_poll_with_notification operation "
                "must be callable"
            )
        if (
            start_and_poll is not None
            and getattr(start_and_poll, "__func__", None)
            is BackendRequestSlot.start_and_poll_with_notification
        ):
            raise BackendFailureError(
                "backend advertised fused request-slot notification overrides "
                "without implementing start_and_poll_with_notification"
            )

        # Publish the extra dispatch table before the base adoption commit.
        # An interruption can therefore expose either no backend slot or one
        # with every method required by its public dynamic type.
        self._backend_start_with_notification = start
        self._backend_start_and_poll_with_notification = start_and_poll
        super()._adopt_backend_slot_unlocked(
            backend_slot,
            execution_dispatch=execution_dispatch,
        )
        # Even an explicit unbound TransferSlot.start(dynamic_slot) must not
        # enter the fixed provider family: NIXL's legacy nullptr options mean
        # "retain the last tag", not "clear". Leave only observation,
        # recycle, cancellation, and close shared between the two families.
        self._backend_start = None
        self._backend_start_and_poll = None
        self._fast_execute = None
        self._restore_fast_execute_unlocked()


class Work:
    """Request-local asynchronous transfer state."""

    __slots__ = (
        "_backend_work",
        "_backend_released",
        "_bound",
        "_closed",
        "_dependencies",
        "_endpoint",
        "_error",
        "_owned_plan",
        "_plan",
        "_recovery_token",
        "_lock",
        "_owner_thread",
        "_submission_error",
        "_submitting",
        "_state",
    )

    def __init__(
        self,
        endpoint: Endpoint,
        backend_work: BackendWork | None,
        *,
        plan: TransferPlan | None = None,
        bound: BoundTransfer | None = None,
        dependencies: tuple[object, ...] = (),
        owned_plan: bool = False,
        recovery_token: object | None = None,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("Work objects are created by Endpoint operations")
        self._endpoint = endpoint
        self._backend_work = backend_work
        self._backend_released = False
        self._plan = plan
        self._bound = bound
        self._dependencies = dependencies
        self._owned_plan = owned_plan
        # Passive caller identity: never interpret or provider-forward it.
        self._recovery_token = recovery_token
        self._lock = (
            None
            if endpoint._thread_mode
            in (ThreadMode.SINGLE, ThreadMode.CALLER_SERIALIZED)
            else threading.RLock()
        )
        self._owner_thread = (
            endpoint._owner_thread
            if endpoint._thread_mode is ThreadMode.SINGLE
            else None
        )
        self._submission_error: BaseException | None = None
        # A Work is published before provider entry so an interrupted caller can
        # recover its identity. Observation is unsafe until submission return or
        # exception reconciliation has completed.
        self._submitting = True
        self._state = WorkState.PENDING
        self._error: BaseException | None = None
        self._closed = False

    @property
    def state(self) -> WorkState:
        return self._refresh()

    @property
    def recovery_token(self) -> object | None:
        """Return the opaque identity retained for lost-return recovery.

        Match this object with ``is`` in :attr:`Endpoint.open_works`. Core
        never hashes, compares, invokes, serializes, or passes it to a provider.
        """

        return self._recovery_token

    @property
    def error(self) -> BaseException | None:
        self._refresh()
        return self._error

    def exception(self) -> BaseException | None:
        return self.error

    @property
    def completion_event(self) -> object | None:
        lock = self._lock
        if lock is None:
            return self._completion_event_unlocked()
        with lock:
            return self._completion_event_unlocked()

    def _completion_event_unlocked(self) -> object | None:
        self._check_not_closed()
        self._check_submission_complete_unlocked()
        if not self._endpoint.capabilities.cuda_completion_event:
            return None
        if self._backend_work is None:
            raise BusyError("transfer work has not been adopted by its backend")
        return self._endpoint._backend_call(
            "query work completion event",
            lambda: self._backend_work.completion_event,
        )

    def _adopt_backend_work(self, backend_work: BackendWork) -> None:
        """Adopt provider state without allocating a submission closure."""

        if not isinstance(backend_work, BackendWork):
            raise BackendFailureError("backend must adopt a BackendWork instance")
        lock = self._lock
        if lock is None:
            self._adopt_backend_work_unlocked(backend_work)
            return
        with lock:
            self._adopt_backend_work_unlocked(backend_work)

    def _adopt_backend_work_unlocked(self, backend_work: BackendWork) -> None:
        if self._closed:
            raise BackendFailureError(
                "backend attempted to adopt an expired transfer work"
            )
        if self._backend_work is not None:
            raise BackendFailureError(
                "backend attempted to adopt transfer work more than once"
            )
        self._backend_work = backend_work

    @property
    def closed(self) -> bool:
        return self._closed

    def test(self) -> bool:
        return self._refresh() in _TERMINAL_STATES

    def poll_bounded(
        self,
        *,
        max_polls: int = _FUSED_REQUEST_SLOT_POLL_MAX_POLLS,
        timeout_ns: int | None = _FUSED_REQUEST_SLOT_POLL_TIMEOUT_NS,
        timeout_check_interval: int = (_FUSED_REQUEST_SLOT_POLL_TIMEOUT_CHECK_INTERVAL),
    ) -> WorkState:
        """Perform a bounded poll and return the exact observed state.

        A provider advertising ``fused_work_poll`` performs the entire loop in
        one provider call. Other providers retain identical semantics through
        repeated ordinary state observations. The returned state may be
        nonterminal; this Work remains the authoritative owner.
        """

        _validate_bounded_poll_arguments(
            max_polls=max_polls,
            timeout_ns=timeout_ns,
            timeout_check_interval=timeout_check_interval,
            allow_zero=False,
        )
        if self._endpoint.capabilities.fused_work_poll:
            lock = self._lock
            if lock is None:
                return self._poll_bounded_unlocked(
                    max_polls=max_polls,
                    timeout_ns=timeout_ns,
                    timeout_check_interval=timeout_check_interval,
                )
            with lock:
                return self._poll_bounded_unlocked(
                    max_polls=max_polls,
                    timeout_ns=timeout_ns,
                    timeout_check_interval=timeout_check_interval,
                )

        deadline_ns = None if timeout_ns is None else time.monotonic_ns() + timeout_ns
        state = self._state
        for poll_index in range(max_polls):
            state = self._refresh()
            if state in _TERMINAL_STATES:
                return state
            if (
                deadline_ns is not None
                and (poll_index == 0 or (poll_index + 1) % timeout_check_interval == 0)
                and time.monotonic_ns() >= deadline_ns
            ):
                break
        return state

    def _poll_bounded_unlocked(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
        timeout_check_interval: int,
    ) -> WorkState:
        self._check_not_closed()
        self._check_thread_unlocked()
        self._check_submission_complete_unlocked()
        if self._state in _TERMINAL_STATES:
            return self._state
        backend_work = self._backend_work
        if backend_work is None:
            return self._state
        state = self._endpoint._backend_work_poll_bounded(
            backend_work,
            max_polls=max_polls,
            timeout_ns=timeout_ns,
            timeout_check_interval=timeout_check_interval,
        )
        self._ingest_backend_state_unlocked(state)
        return self._state

    def test_bounded(
        self,
        *,
        max_polls: int = _FUSED_REQUEST_SLOT_POLL_MAX_POLLS,
        timeout_ns: int | None = _FUSED_REQUEST_SLOT_POLL_TIMEOUT_NS,
        timeout_check_interval: int = (_FUSED_REQUEST_SLOT_POLL_TIMEOUT_CHECK_INTERVAL),
    ) -> bool:
        """Compatibility wrapper returning whether a bounded poll is terminal."""

        return (
            self.poll_bounded(
                max_polls=max_polls,
                timeout_ns=timeout_ns,
                timeout_check_interval=timeout_check_interval,
            )
            in _TERMINAL_STATES
        )

    def wait(self, timeout: float | None = None) -> Work:
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout must be a non-negative number or None")
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError("timeout must be finite and non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        polls = 0
        while True:
            state = self._refresh()
            if state in _TERMINAL_STATES:
                return self._raise_for_state()
            if deadline is not None and time.monotonic() >= deadline:
                raise TransferTimeoutError("transfer work timed out")
            if self._endpoint.progress_mode is ProgressMode.MANUAL:
                self._endpoint.progress()
            polls += 1
            if polls >= 16:
                # Yield without imposing a fixed millisecond latency floor.
                # Long waits back off to cap host polling overhead.
                time.sleep(0 if polls < 128 else 0.0001)

    def result(self, timeout: float | None = None) -> None:
        self.wait(timeout)
        return None

    def cancel(self) -> bool:
        lock = self._lock
        if lock is None:
            return self._cancel_unlocked()
        with lock:
            return self._cancel_unlocked()

    def _cancel_unlocked(self) -> bool:
        self._check_not_closed()
        self._check_submission_complete_unlocked()
        if self._refresh_unlocked() in _TERMINAL_STATES:
            return self._state is WorkState.CANCELLED
        if self._backend_work is None:
            return False
        cancelled = self._endpoint._backend_call(
            "cancel work", self._backend_work.cancel
        )
        if not isinstance(cancelled, bool):
            raise BackendFailureError("backend work cancellation must return a bool")
        self._refresh_unlocked()
        return cancelled and self._state is WorkState.CANCELLED

    def close(self) -> None:
        state_lock = self._endpoint._state_lock
        if state_lock is _NOOP_LOCK:
            work_lock = self._lock
            if work_lock is None:
                self._close_unlocked()
                return
            with work_lock:
                self._close_unlocked()
            return
        with state_lock:
            self._close_with_work_lock()

    def _close_with_work_lock(self) -> None:
        lock = self._lock
        if lock is None:
            self._close_unlocked()
            return
        with lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        if self._closed:
            # CLOSED is the provider-release commit, not proof that every
            # Core-only graph edge and retained reference was cleared. A
            # retry must finish an epilogue interrupted after that commit.
            self._finish_close_unlocked()
            return
        self._check_submission_complete_unlocked()
        self._check_thread_unlocked()
        if (
            self._state not in _TERMINAL_STATES
            and self._refresh_unlocked() not in _TERMINAL_STATES
        ):
            raise BusyError("cannot close non-terminal transfer work")
        if not self._backend_released:
            if self._backend_work is None:
                raise BackendFailureError(
                    "terminal transfer work has no backend request"
                )
            self._endpoint._backend_work_close(self._backend_work)
            self._backend_released = True
        if self._plan is not None:
            self._plan._works.discard(self)
        if self._bound is not None:
            self._bound._works.discard(self)
        try:
            if self._owned_plan and self._plan is not None:
                self._plan.close()
        except BaseException:
            if self._plan is not None:
                self._plan._works.add(self)
            raise
        self._closed = True
        self._finish_close_unlocked()

    def _finish_close_unlocked(self) -> None:
        """Idempotently unlink a closed Work from the Core lifetime graph."""

        plan = self._plan
        bound = self._bound
        # Retain both parent references until every blocking edge is gone. An
        # interrupt during any discard therefore leaves enough state to retry.
        if plan is not None:
            plan._works.discard(self)
        if bound is not None:
            bound._works.discard(self)
        self._endpoint._works.discard(self)
        self._dependencies = ()
        self._plan = None
        self._bound = None
        self._backend_work = None
        self._error = None
        self._submission_error = None
        # Commit last: absence means every provider/Core lifetime edge is gone.
        self._recovery_token = None

    async def _wait_async(self) -> Work:
        while True:
            state = self._refresh()
            if state in _TERMINAL_STATES:
                return self._raise_for_state()
            if self._endpoint.progress_mode is ProgressMode.MANUAL:
                self._endpoint.progress()
            await asyncio.sleep(0)

    def __await__(self):  # type: ignore[no-untyped-def]
        return self._wait_async().__await__()

    def _refresh(self) -> WorkState:
        lock = self._lock
        if lock is None:
            return self._refresh_unlocked()
        with lock:
            return self._refresh_unlocked()

    def _refresh_unlocked(self) -> WorkState:
        self._check_not_closed()
        self._check_thread_unlocked()
        self._check_submission_complete_unlocked()
        if self._state in _TERMINAL_STATES:
            return self._state
        if self._backend_work is None:
            return self._state
        state = self._endpoint._backend_work_state(self._backend_work)
        return self._ingest_backend_state_unlocked(state)

    def _ingest_backend_state_unlocked(self, state: WorkState) -> WorkState:
        """Validate and cache a provider state already observed this call."""

        if not isinstance(state, WorkState):
            raise BackendFailureError(f"backend returned invalid work state {state!r}")
        if self._state is WorkState.RUNNING and state is WorkState.PENDING:
            raise BackendFailureError(
                "backend work state regressed from running to pending"
            )
        observed_state = state
        observed_error = self._error
        if state in _TERMINAL_STATES and self._submission_error is not None:
            observed_state = WorkState.FAILED
            observed_error = self._submission_error
        elif state is WorkState.FAILED:
            backend_work = self._backend_work
            if backend_work is None:
                observed_error = BackendFailureError(
                    "failed transfer work has no backend request"
                )
            else:
                try:
                    error = self._endpoint._backend_call(
                        "query work error", lambda: backend_work.error
                    )
                except Exception as observation_error:
                    observed_error = observation_error
                else:
                    if error is None:
                        error = BackendFailureError(
                            "transfer backend reported failure without an error"
                        )
                    elif not isinstance(error, BaseException):
                        error = BackendFailureError(
                            "transfer backend returned a non-exception error"
                        )
                    observed_error = error
        elif state is WorkState.CANCELLED:
            observed_error = TransferCancelledError("transfer work was cancelled")

        # Terminal state is the public commit. Derive its diagnostic first so
        # an asynchronous BaseException during provider error observation leaves
        # the prior nonterminal state retryable.
        if observed_state in _TERMINAL_STATES:
            self._error = observed_error
        self._state = observed_state
        return self._state

    def _raise_for_state(self) -> Work:
        if self._state is WorkState.COMPLETED:
            return self
        if self._error is None:
            self._error = BackendFailureError(
                f"transfer work reached {self._state.value} without an error"
            )
        if isinstance(self._error, TransferError):
            raise self._error
        raise BackendFailureError("transfer work failed") from self._error

    def _check_not_closed(self) -> None:
        if self._closed:
            raise ClosedError("transfer work is closed")

    def _check_submission_complete_unlocked(self) -> None:
        if self._submitting:
            raise BusyError("transfer work submission is in progress")

    def _check_thread_unlocked(self) -> None:
        owner_thread = self._owner_thread
        if owner_thread is not None and _GET_IDENT() != owner_thread:
            raise UnsupportedError(
                "single-thread endpoint used from a different thread"
            )


class Endpoint:
    """Backend-neutral endpoint for asymmetric one-sided data transfer."""

    def __init__(
        self,
        name: str,
        *,
        backend: str = "reference",
        endpoint_id: str | None = None,
        progress_mode: ProgressMode | str = ProgressMode.MANUAL,
        thread_mode: ThreadMode | str = ThreadMode.SERIALIZED,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        _validate_nonempty_str(name, "name")
        _validate_nonempty_str(backend, "backend")
        if endpoint_id is not None:
            _validate_nonempty_str(endpoint_id, "endpoint_id")
        progress_mode = _coerce_enum(progress_mode, ProgressMode, "progress_mode")
        thread_mode = _coerce_enum(thread_mode, ThreadMode, "thread_mode")
        if options is None:
            options = {}
        elif not isinstance(options, Mapping):
            raise TypeError("options must be a mapping or None")

        self._name = name
        self._backend_name = backend
        self._endpoint_id = endpoint_id or str(uuid.uuid4())
        self._incarnation = str(uuid.uuid4())
        self._progress_mode = progress_mode
        self._thread_mode = thread_mode
        self._owner_thread = threading.get_ident()
        self._backend_lock = (
            threading.RLock() if thread_mode is ThreadMode.SERIALIZED else _NOOP_LOCK
        )
        self._state_lock = (
            _NOOP_LOCK
            if thread_mode in (ThreadMode.SINGLE, ThreadMode.CALLER_SERIALIZED)
            else threading.RLock()
        )
        self._notification_consumer_lock = (
            threading.RLock() if thread_mode is ThreadMode.MULTIPLE else _NOOP_LOCK
        )
        self._active_multiple_backend_calls: (
            set[weakref.ReferenceType[_BackendCallReservation]] | None
        ) = (set() if thread_mode is ThreadMode.MULTIPLE else None)
        self._multiple_backend_call_reservation_pool: (
            list[_BackendCallReservation] | None
        ) = ([] if thread_mode is ThreadMode.MULTIPLE else None)
        self._registrations: dict[str, Registration] = {}
        self._registration_names: dict[str, Registration] = {}
        self._peers: dict[str, Peer] = {}
        self._plans: set[TransferPlan] = set()
        self._orphaned_plans: set[TransferPlan] = set()
        self._bound_transfers: set[BoundTransfer] = set()
        self._transfer_slots: set[TransferSlot] = set()
        self._works: set[Work] = set()
        self._closing = False
        self._creation_failed = False
        self._closed = False

        _retry_orphaned_backend_receipts(backend, self._endpoint_id)
        registration = _get_backend_registration(backend)
        backend_options = MappingProxyType(dict(options))
        receipt = _BackendConstructionReceipt(backend, self._endpoint_id)
        self._backend_receipt = receipt
        self._backend_finalizer = weakref.finalize(
            self, _finalize_backend_receipt, receipt
        )
        self._backend_finalizer.atexit = False
        endpoint_ref = weakref.ref(self)

        def adopt_backend(implementation: Backend) -> None:
            # The receipt owns the backend before an Endpoint attribute becomes
            # observable. The outer transaction repairs either STORE_ATTR cut.
            receipt.adopt(implementation)
            endpoint = endpoint_ref()
            if endpoint is None:
                raise BackendFailureError(
                    "endpoint disappeared while adopting its backend"
                )
            endpoint._backend = implementation
            endpoint._creation_failed = True

        factory = registration.factory
        unowned_v1_result = False
        try:
            if registration.factory_api_version == 1:
                candidate = cast(Callable[..., object], factory)(
                    self._endpoint_id,
                    name,
                    self._incarnation,
                    progress_mode,
                    thread_mode,
                    backend_options,
                )
                if not isinstance(candidate, Backend):
                    # A wrong-type legacy return was never adopted by Core.  It
                    # may be an arbitrary foreign object, so do not call close
                    # or inspect any provider-shaped attribute on it.
                    unowned_v1_result = True
                    raise TypeError("backend factory must return a Backend instance")
                # This compatibility path necessarily retains the legacy
                # CALL-to-adoption gap. Version-2 providers close it.
                adopt_backend(candidate)
                implementation = receipt.seal_successful_call(None)
            else:
                result = cast(Callable[..., object], factory)(
                    self._endpoint_id,
                    name,
                    self._incarnation,
                    progress_mode,
                    thread_mode,
                    backend_options,
                    adopt_backend=adopt_backend,
                )
                implementation = receipt.seal_successful_call(result)
        except BaseException as error:
            receipt.seal_failed_call()
            try:
                self._cleanup_adopted_backend_after_creation_error(error)
            except BaseException:
                raise
            if unowned_v1_result:
                raise
            if isinstance(error, TransferError) or not isinstance(error, Exception):
                raise
            raise BackendFailureError(
                f"failed to create transfer backend {backend!r}"
            ) from error

        try:
            # From this assignment onward Core owns the returned backend. Keep
            # capability discovery, dispatch caching, and EndpointInfo
            # publication in one cleanup transaction.
            self._backend = implementation
            self._creation_failed = True
            capabilities = implementation.capabilities
            if not isinstance(capabilities, Capabilities):
                raise TypeError("backend capabilities must be Capabilities")
            if (
                progress_mode is ProgressMode.MANUAL
                and not capabilities.manual_progress
            ):
                raise UnsupportedError(
                    f"backend {backend!r} does not support manual progress"
                )
            if (
                progress_mode is ProgressMode.BACKGROUND
                and not capabilities.background_progress
            ):
                raise UnsupportedError(
                    f"backend {backend!r} does not support background progress"
                )
            supported_thread_modes = implementation.supported_thread_modes
            if not isinstance(supported_thread_modes, frozenset) or not all(
                isinstance(mode, ThreadMode) for mode in supported_thread_modes
            ):
                raise TypeError(
                    "backend supported_thread_modes must be a frozenset of "
                    "ThreadMode values"
                )
            if thread_mode not in supported_thread_modes:
                raise UnsupportedError(
                    f"backend {backend!r} does not support thread mode "
                    f"{thread_mode.value!r}"
                )
            send_notification_sync_dispatch = (
                cast(
                    Callable[[object, bytes], object | None],
                    implementation.send_notification_sync,
                )
                if capabilities.synchronous_notification_send
                else None
            )
            info = EndpointInfo(
                backend=backend,
                endpoint_id=self._endpoint_id,
                name=name,
                incarnation=self._incarnation,
                capabilities=capabilities,
            )
            self._capabilities = capabilities
            self._send_notification_sync_dispatch = send_notification_sync_dispatch
            self._info = info
            self._creation_failed = False
            receipt.mark_published()
        except BaseException as error:
            # Re-establish both fields if interruption followed either
            # STORE_ATTR. Failed cleanup is exposed through a recovery endpoint.
            self._backend = implementation
            self._creation_failed = True
            try:
                self._close_failed_endpoint_backend()
            except BaseException as cleanup_error:
                raise _endpoint_creation_cleanup_failure(
                    self, error, cleanup_error
                ) from cleanup_error
            raise

    @property
    def name(self) -> str:
        return self._name

    @property
    def backend(self) -> str:
        return self._backend_name

    @property
    def endpoint_id(self) -> str:
        return self._endpoint_id

    @property
    def incarnation(self) -> str:
        return self._incarnation

    @property
    def identity(self) -> EndpointIdentity:
        return EndpointIdentity(self.endpoint_id, self.incarnation)

    @property
    def progress_mode(self) -> ProgressMode:
        return self._progress_mode

    @property
    def thread_mode(self) -> ThreadMode:
        return self._thread_mode

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def info(self) -> EndpointInfo:
        return self._info

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def open_registrations(self) -> tuple[Registration, ...]:
        """Return registrations owned by this endpoint for recovery/inspection."""

        with self._state_lock:
            self._check_open()
            return tuple(self._registrations.values())

    @property
    def open_peers(self) -> tuple[Peer, ...]:
        """Return imported peers owned by this endpoint for recovery/inspection."""

        with self._state_lock:
            self._check_open()
            return tuple(self._peers.values())

    @property
    def open_plans(self) -> tuple[TransferPlan, ...]:
        """Return plans owned by this endpoint for recovery/inspection."""

        with self._state_lock:
            self._check_open()
            return tuple(self._plans)

    @property
    def open_works(self) -> tuple[Work, ...]:
        """Return open Work objects, including interrupted submissions.

        The snapshot lets an application recover and drain a Work if an
        asynchronous exception lands after submission but before the caller
        stores the returned object.
        """

        with self._state_lock:
            self._check_open()
            return tuple(self._works)

    @property
    def open_bound_transfers(self) -> tuple[BoundTransfer, ...]:
        """Return reusable bindings owned by this endpoint for recovery."""

        with self._state_lock:
            self._check_open()
            return tuple(self._bound_transfers)

    @property
    def open_transfer_slots(self) -> tuple[TransferSlot, ...]:
        """Return persistent request slots owned by this endpoint for recovery."""

        with self._state_lock:
            self._check_open()
            return tuple(self._transfer_slots)

    @_state_locked
    def register(
        self,
        tensor_or_storage: object,
        *,
        name: str | None = None,
    ) -> Registration:
        self._check_open()
        if name is None:
            name = f"registration-{uuid.uuid4()}"
        _validate_nonempty_str(name, "name")
        if name in self._registration_names:
            raise InvalidRegionError(f"registration name {name!r} is not unique")
        registration_id = str(uuid.uuid4())
        memory = _registered_memory(tensor_or_storage, registration_id, name)
        for existing in self._registrations.values():
            _validate_memory_stable(existing._memory)
            if _memory_ranges_overlap(memory, existing._memory):
                raise InvalidRegionError(
                    f"registration {name!r} overlaps existing registration "
                    f"{existing.name!r}"
                )
        if isinstance(tensor_or_storage, RawSpan) and not self.capabilities.raw_spans:
            raise UnsupportedError(
                f"backend {self.backend!r} does not support raw spans"
            )
        registration = Registration(
            self,
            memory,
            _UNADOPTED_HANDLE,
            _token=_CONSTRUCTION_TOKEN,
        )
        try:
            self._registrations[registration_id] = registration
            self._registration_names[name] = registration
            result = self._backend_call(
                "register memory",
                lambda: self._backend.register(
                    memory,
                    adopt_handle=lambda handle: _adopt_resource_handle(
                        registration, handle, "registration"
                    ),
                ),
            )
            if result is not None:
                raise BackendFailureError(
                    "backend register must return None after adopting its handle"
                )
            if registration._handle is _UNADOPTED_HANDLE:
                raise BackendFailureError(
                    "backend register returned without adopting a handle"
                )
            return registration
        except BaseException as error:
            if registration._handle is not _UNADOPTED_HANDLE:
                try:
                    registration.close()
                except BaseException as cleanup_error:
                    registration._creation_failed = True
                    raise _creation_cleanup_failure(
                        "registration", error, cleanup_error
                    ) from cleanup_error
            else:
                self._registrations.pop(registration_id, None)
                self._registration_names.pop(name, None)
                registration._closed = True
                registration._handle = None
                registration._memory = _released_memory(memory)
            raise

    @_state_locked
    def export_metadata(
        self,
        registrations: Sequence[Registration] | None = None,
    ) -> EndpointMetadata:
        self._check_open()
        selected = (
            tuple(self._registrations.values())
            if registrations is None
            else tuple(registrations)
        )
        if len(selected) != len(set(selected)):
            raise InvalidMetadataError("registrations must not contain duplicates")
        values: list[RegistrationMetadata] = []
        for registration in selected:
            self._validate_registration(registration)
            descriptor = self._backend_call(
                "describe registration",
                lambda registration=registration: (
                    self._backend.describe_registration(registration._handle)
                ),
            )
            if not isinstance(descriptor, (bytes, bytearray, memoryview)):
                raise BackendFailureError(
                    "backend registration descriptor must be bytes-like"
                )
            values.append(
                RegistrationMetadata(
                    registration_id=registration.registration_id,
                    name=registration.name,
                    nbytes=registration.nbytes,
                    device=registration.device,
                    memory_type=registration.memory_type,
                    backend_descriptor=bytes(descriptor),
                )
            )
        payload = self._backend_call(
            "export endpoint metadata",
            lambda: self._backend.endpoint_payload(
                tuple(registration._handle for registration in selected)
            ),
        )
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise BackendFailureError("backend endpoint payload must be bytes-like")
        return EndpointMetadata(
            wire_major=EndpointMetadata.WIRE_MAJOR,
            wire_minor=EndpointMetadata.WIRE_MINOR,
            required_features=(),
            backend=self.backend,
            endpoint_id=self.endpoint_id,
            name=self.name,
            incarnation=self.incarnation,
            backend_payload=bytes(payload),
            registrations=tuple(values),
        )

    @_state_locked
    def import_peer(
        self,
        metadata: EndpointMetadata | bytes | bytearray | memoryview,
        *,
        expected_identity: EndpointIdentity | None = None,
        expected_endpoint_id: str | None = None,
        expected_name: str | None = None,
        expected_incarnation: str | None = None,
    ) -> Peer:
        self._check_open()
        if not isinstance(metadata, EndpointMetadata):
            metadata = EndpointMetadata.from_bytes(metadata)
        if expected_identity is not None:
            if not isinstance(expected_identity, EndpointIdentity):
                raise TypeError("expected_identity must be an EndpointIdentity")
            if expected_endpoint_id is not None or expected_incarnation is not None:
                raise ValueError(
                    "expected_identity is mutually exclusive with separate expected "
                    "endpoint_id/incarnation values"
                )
            expected_endpoint_id = expected_identity.endpoint_id
            expected_incarnation = expected_identity.incarnation
        expected_fields = {
            "endpoint_id": expected_endpoint_id,
            "name": expected_name,
            "incarnation": expected_incarnation,
        }
        for field_name, expected in expected_fields.items():
            if expected is None:
                continue
            _validate_nonempty_str(expected, f"expected_{field_name}")
            actual = getattr(metadata, field_name)
            if actual != expected:
                raise InvalidMetadataError(
                    f"peer {field_name} {actual!r} does not match expected "
                    f"value {expected!r}"
                )
        if metadata.backend != self.backend:
            raise BackendMismatchError(
                f"peer backend {metadata.backend!r} does not match "
                f"local backend {self.backend!r}"
            )
        existing = self._peers.get(metadata.endpoint_id)
        if existing is not None:
            if existing.incarnation != metadata.incarnation:
                raise BusyError(
                    f"peer {metadata.endpoint_id!r} changed incarnation; "
                    "close the old peer before importing the replacement"
                )
            raise BusyError(f"peer {metadata.endpoint_id!r} is already imported")
        peer = Peer(
            self,
            metadata,
            _UNADOPTED_HANDLE,
            _token=_CONSTRUCTION_TOKEN,
        )
        try:
            self._peers[peer.endpoint_id] = peer
            result = self._backend_call(
                "import peer",
                lambda: self._backend.import_peer(
                    metadata,
                    adopt_handle=lambda handle: _adopt_resource_handle(
                        peer, handle, "peer"
                    ),
                ),
            )
            if result is not None:
                raise BackendFailureError(
                    "backend peer import must return None after adopting its handle"
                )
            if peer._handle is _UNADOPTED_HANDLE:
                raise BackendFailureError(
                    "backend peer import returned without adopting a handle"
                )
            return peer
        except BaseException as error:
            if peer._handle is not _UNADOPTED_HANDLE:
                try:
                    peer.close()
                except BaseException as cleanup_error:
                    peer._creation_failed = True
                    raise _creation_cleanup_failure(
                        "peer", error, cleanup_error
                    ) from cleanup_error
            else:
                self._peers.pop(peer.endpoint_id, None)
                peer._closed = True
                peer._handle = None
                peer._registrations = {}
            raise

    @_state_locked
    def prepare(
        self,
        *,
        local: Sequence[Region],
        remote: Sequence[Region],
        indexed: bool = False,
    ) -> TransferPlan:
        self._check_open()
        if not isinstance(indexed, bool):
            raise TypeError("indexed must be a bool")
        local = tuple(local)
        remote = tuple(remote)
        self._validate_regions(local, remote, indexed=indexed)
        backend_local = tuple(region._to_backend() for region in local)
        backend_remote = tuple(region._to_backend() for region in remote)
        # Finish every Python allocation and publish the local lifetime guard
        # before asking a provider to create native state.
        plan = TransferPlan(
            self,
            _UNADOPTED_HANDLE,
            local,
            remote,
            indexed,
            _token=_CONSTRUCTION_TOKEN,
        )
        try:
            self._plans.add(plan)
            result = self._backend_call(
                "prepare transfer plan",
                lambda: self._backend.prepare(
                    backend_local,
                    backend_remote,
                    indexed=indexed,
                    adopt_handle=lambda handle: _adopt_resource_handle(
                        plan, handle, "transfer plan"
                    ),
                ),
            )
            if result is not None:
                raise BackendFailureError(
                    "backend prepare must return None after adopting its handle"
                )
            if plan._handle is _UNADOPTED_HANDLE:
                raise BackendFailureError(
                    "backend prepare returned without adopting a handle"
                )
            return plan
        except BaseException as error:
            if plan._handle is not _UNADOPTED_HANDLE:
                try:
                    plan.close()
                except BaseException as cleanup_error:
                    plan._creation_failed = True
                    self._orphaned_plans.add(plan)
                    raise _creation_cleanup_failure(
                        "transfer plan", error, cleanup_error
                    ) from cleanup_error
            else:
                self._plans.discard(plan)
                plan._closed = True
                plan._handle = None
                plan._local = ()
                plan._remote = ()
                plan._origins = frozenset()
                plan._reuse_tokens = {}
            raise

    def read(
        self,
        plan: TransferPlan | None = None,
        *,
        local: Sequence[Region] | None = None,
        remote: Sequence[Region] | None = None,
        local_indices: Sequence[int] | None = None,
        remote_indices: Sequence[int] | None = None,
        notification: bytes | bytearray | memoryview | None = None,
        cuda_ordering: CudaOrderingMode | str = CudaOrderingMode.CALLER_READY,
        stream: object | None = None,
        ready_event: object | None = None,
        recovery_token: object | None = None,
    ) -> Work:
        return self._submit(
            TransferOp.READ,
            plan,
            local=local,
            remote=remote,
            local_indices=local_indices,
            remote_indices=remote_indices,
            notification=notification,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
            recovery_token=recovery_token,
        )

    def write(
        self,
        plan: TransferPlan | None = None,
        *,
        local: Sequence[Region] | None = None,
        remote: Sequence[Region] | None = None,
        local_indices: Sequence[int] | None = None,
        remote_indices: Sequence[int] | None = None,
        notification: bytes | bytearray | memoryview | None = None,
        cuda_ordering: CudaOrderingMode | str = CudaOrderingMode.CALLER_READY,
        stream: object | None = None,
        ready_event: object | None = None,
        recovery_token: object | None = None,
    ) -> Work:
        return self._submit(
            TransferOp.WRITE,
            plan,
            local=local,
            remote=remote,
            local_indices=local_indices,
            remote_indices=remote_indices,
            notification=notification,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
            recovery_token=recovery_token,
        )

    @_state_locked
    def _select_plan_indices(
        self,
        plan: TransferPlan,
        op: TransferOp | str,
        *,
        local_indices: Sequence[int],
        remote_indices: Sequence[int],
    ) -> IndexSelection:
        self._check_open()
        self._validate_plan_locked(plan)
        if not plan.indexed:
            raise InvalidRegionError(
                "indices require a plan prepared with indexed=True"
            )
        op = _coerce_enum(op, TransferOp, "op")
        supported = (
            self.capabilities.read if op is TransferOp.READ else self.capabilities.write
        )
        if not supported:
            raise UnsupportedError(
                f"backend {self.backend!r} does not support {op.value}"
            )
        backend_selection, local_devices = _snapshot_backend_index_selection(
            plan, op, local_indices, remote_indices
        )
        return IndexSelection(
            self,
            plan,
            op,
            backend_selection,
            local_devices,
            _token=_CONSTRUCTION_TOKEN,
        )

    @_state_locked
    def _bind_transfer(
        self,
        plan: TransferPlan,
        op: TransferOp | str,
        *,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        notification: bytes | bytearray | memoryview | None,
        cuda_ordering: CudaOrderingMode | str,
        stream: object | None,
        ready_event: object | None,
    ) -> BoundTransfer:
        self._check_open()
        self._validate_plan_locked(plan)
        op = _coerce_enum(op, TransferOp, "op")
        supported = (
            self.capabilities.read if op is TransferOp.READ else self.capabilities.write
        )
        if not supported:
            raise UnsupportedError(
                f"backend {self.backend!r} does not support {op.value}"
            )
        cuda_ordering = _coerce_enum(cuda_ordering, CudaOrderingMode, "cuda_ordering")
        _validate_cuda_ordering_arguments(
            self.capabilities,
            self.backend,
            cuda_ordering,
            stream,
            ready_event,
        )
        local_indices, remote_indices = _validate_plan_indices(
            plan, local_indices, remote_indices, op
        )
        notification = _validate_attached_notification(
            self.capabilities, self.backend, notification
        )
        cuda_device = _validate_cuda_ordering_selection(
            plan,
            local_indices,
            cuda_ordering,
            stream,
            ready_event,
        )
        binding = BoundTransfer(
            self,
            plan,
            op,
            local_indices=local_indices,
            remote_indices=remote_indices,
            notification=notification,
            cuda_device=cuda_device,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
            selection=None,
            _token=_CONSTRUCTION_TOKEN,
        )
        try:
            self._bound_transfers.add(binding)
            plan._bindings.add(binding)
        except BaseException:
            plan._bindings.discard(binding)
            self._bound_transfers.discard(binding)
            binding._closed = True
            binding._plan = None
            raise
        return binding

    @_state_locked
    def _bind_index_selection(
        self,
        selection: IndexSelection,
        *,
        cuda_ordering: CudaOrderingMode | str,
        stream: object | None,
        ready_event: object | None,
    ) -> BoundTransfer:
        self._check_open()
        if type(selection) is not IndexSelection:
            raise TypeError("expected an IndexSelection")
        if selection._endpoint is not self:
            raise InvalidRegionError("index selection belongs to another endpoint")
        plan = selection._check_open()
        cuda_ordering = _coerce_enum(cuda_ordering, CudaOrderingMode, "cuda_ordering")
        _validate_cuda_ordering_arguments(
            self.capabilities,
            self.backend,
            cuda_ordering,
            stream,
            ready_event,
        )
        cuda_device = _validate_cuda_ordering_devices(
            selection._local_devices,
            cuda_ordering,
            stream,
            ready_event,
        )
        binding = BoundTransfer(
            self,
            plan,
            selection._op,
            local_indices=selection._backend_selection.local_indices,
            remote_indices=selection._backend_selection.remote_indices,
            notification=None,
            cuda_device=cuda_device,
            cuda_ordering=cuda_ordering,
            stream=stream,
            ready_event=ready_event,
            selection=selection,
            _token=_CONSTRUCTION_TOKEN,
        )
        try:
            self._bound_transfers.add(binding)
            plan._bindings.add(binding)
        except BaseException:
            plan._bindings.discard(binding)
            self._bound_transfers.discard(binding)
            binding._closed = True
            binding._plan = None
            binding._selection = None
            raise
        return binding

    @_state_locked
    def _create_transfer_slot(
        self,
        bound: BoundTransfer,
        *,
        request_local_notifications: bool,
    ) -> TransferSlot | RequestLocalNotificationSlot:
        """Publish Core ownership before asking a provider for native state."""

        self._check_open()
        if not self._capabilities.persistent_request_slots:
            raise UnsupportedError(
                f"backend {self.backend!r} does not support persistent request slots"
            )
        if not isinstance(bound, BoundTransfer):
            raise TypeError("expected a BoundTransfer")
        if bound._endpoint is not self:
            raise InvalidRegionError("bound transfer belongs to another endpoint")
        if request_local_notifications:
            if not self._capabilities.request_slot_notification_overrides:
                raise UnsupportedError(
                    f"backend {self.backend!r} does not support request-local "
                    "persistent-slot notifications"
                )
            if bound._notification is not None:
                raise ValueError(
                    "request-local notification slots require a binding with "
                    "notification=None"
                )
        plan = bound._check_open()
        slot: TransferSlot
        if request_local_notifications:
            slot = RequestLocalNotificationSlot(self, bound, _token=_CONSTRUCTION_TOKEN)
        else:
            slot = TransferSlot(self, bound, _token=_CONSTRUCTION_TOKEN)
        slot._creation_failed = True
        try:
            # Publish the recovery registry first. At every subsequent
            # interruption point either normal rollback succeeds or the slot
            # remains discoverable through ``open_transfer_slots``.
            self._transfer_slots.add(slot)
            bound._slots.add(slot)
            result = self._backend_create_request_slot(
                bound._op,
                plan._handle,
                local_indices=bound._local_indices,
                remote_indices=bound._remote_indices,
                notification=bound._notification,
                adopt_slot=slot._adopt_backend_slot,
            )
            if result is not None:
                raise BackendFailureError(
                    "backend request-slot creation must return None after adoption"
                )
            if slot._backend_slot is None:
                raise BackendFailureError(
                    "backend request-slot creation returned without adoption"
                )
            slot._creation_failed = False
            return slot
        except BaseException as error:
            if slot._backend_slot is not None:
                try:
                    self._close_transfer_slot(slot)
                except BaseException as cleanup_error:
                    slot._creation_failed = True
                    raise _creation_cleanup_failure(
                        "transfer slot", error, cleanup_error
                    ) from cleanup_error
            else:
                # Keep the endpoint recovery edge until the bound blocker has
                # been removed. If cleanup itself is interrupted, the slot is
                # still public through the endpoint snapshot and closeable.
                bound._slots.discard(slot)
                slot._closed = True
                self._finish_transfer_slot_close(slot)
            raise

    def _reserve_indexed_selection_submission(
        self,
        *,
        plan: TransferPlan | None,
        op: TransferOp | str | None,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        selection: IndexSelection | None,
        notification: bytes | bytearray | memoryview | None,
        cuda_ordering: CudaOrderingMode | str,
        stream: object | None,
        ready_event: object | None,
        recovery_token: object | None,
    ) -> tuple[
        TransferPlan,
        TransferOp,
        BackendIndexSelection,
        bytes | None,
        CudaOrderingMode,
        object | None,
        Work,
    ]:
        """Validate and construct an unpublished indexed Work under state lock.

        The caller holds the lock through publication. Returning a published
        Work through a decorated helper would expose a lock-exit ownership gap.
        """

        active_plan: TransferPlan | None = None
        work: Work | None = None
        try:
            self._check_open()
            if selection is None:
                if plan is None:
                    raise RuntimeError(
                        "one-shot indexed submission is missing its internal plan"
                    )
                self._validate_plan_locked(plan)
                if not plan.indexed:
                    raise InvalidRegionError(
                        "indices require a plan prepared with indexed=True"
                    )
                active_plan = plan
                active_op = _coerce_enum(op, TransferOp, "op")
                supported = (
                    self.capabilities.read
                    if active_op is TransferOp.READ
                    else self.capabilities.write
                )
                if not supported:
                    raise UnsupportedError(
                        f"backend {self.backend!r} does not support {active_op.value}"
                    )
                caller_ready_hint = bool(
                    cuda_ordering is CudaOrderingMode.CALLER_READY
                    or (
                        type(cuda_ordering) is str
                        and cuda_ordering == CudaOrderingMode.CALLER_READY.value
                    )
                )
                backend_selection, local_devices = _snapshot_backend_index_selection(
                    active_plan,
                    active_op,
                    local_indices,
                    remote_indices,
                    collect_local_devices=not caller_ready_hint,
                )
                selection_dependency: object = backend_selection
            else:
                if type(selection) is not IndexSelection:
                    raise TypeError("expected an IndexSelection")
                if selection._endpoint is not self:
                    raise InvalidRegionError(
                        "index selection belongs to another endpoint"
                    )
                active_plan = selection._plan
                self._validate_plan_locked(active_plan)
                active_op = selection._op
                backend_selection = selection._backend_selection
                local_devices = selection._local_devices
                selection_dependency = selection
            cuda_ordering = _coerce_enum(
                cuda_ordering, CudaOrderingMode, "cuda_ordering"
            )
            _validate_cuda_ordering_arguments(
                self.capabilities,
                self.backend,
                cuda_ordering,
                stream,
                ready_event,
            )
            notification = _validate_attached_notification(
                self.capabilities, self.backend, notification
            )
            cuda_device = (
                None
                if cuda_ordering is CudaOrderingMode.CALLER_READY
                else _validate_cuda_ordering_devices(
                    local_devices,
                    cuda_ordering,
                    stream,
                    ready_event,
                )
            )
            if stream is None and ready_event is None:
                dependencies = (
                    (selection_dependency,)
                    if notification is None
                    else (selection_dependency, notification)
                )
            else:
                dependencies = tuple(
                    value
                    for value in (
                        selection_dependency,
                        stream,
                        ready_event,
                        notification,
                    )
                    if value is not None
                )
            work = Work(
                self,
                None,
                plan=active_plan,
                dependencies=dependencies,
                recovery_token=recovery_token,
                _token=_CONSTRUCTION_TOKEN,
            )
            return (
                active_plan,
                active_op,
                backend_selection,
                notification,
                cuda_ordering,
                cuda_device,
                work,
            )
        except BaseException:
            if work is not None:
                work_lock = work._lock
                if work_lock is None:
                    work._closed = True
                    work._submitting = False
                    work._dependencies = ()
                    work._plan = None
                    work._recovery_token = None
                else:
                    with work_lock:
                        work._closed = True
                        work._submitting = False
                        work._dependencies = ()
                        work._plan = None
                        work._recovery_token = None
            raise

    def _submit_indexed_selection(
        self,
        *,
        plan: TransferPlan | None,
        op: TransferOp | str | None,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        selection: IndexSelection | None,
        notification: bytes | bytearray | memoryview | None,
        cuda_ordering: CudaOrderingMode | str,
        stream: object | None,
        ready_event: object | None,
        recovery_token: object | None,
        initial_max_polls: int | None = None,
        initial_timeout_ns: int | None = None,
    ) -> Work:
        """Submit a reusable public or one-shot private indexed selection."""

        active_plan: TransferPlan | None = None
        active_op: TransferOp | None = None
        backend_selection: BackendIndexSelection | None = None
        cuda_device: object | None = None
        submission_token: object | None = None
        work: Work | None = None
        try:
            state_lock = self._state_lock
            if state_lock is _NOOP_LOCK:
                (
                    active_plan,
                    active_op,
                    backend_selection,
                    notification,
                    cuda_ordering,
                    cuda_device,
                    work,
                ) = self._reserve_indexed_selection_submission(
                    plan=plan,
                    op=op,
                    local_indices=local_indices,
                    remote_indices=remote_indices,
                    selection=selection,
                    notification=notification,
                    cuda_ordering=cuda_ordering,
                    stream=stream,
                    ready_event=ready_event,
                    recovery_token=recovery_token,
                )
                submission_token = work
                self._works.add(work)
                active_plan._submitting.add(work)
                active_plan._works.add(work)
            else:
                with state_lock:
                    (
                        active_plan,
                        active_op,
                        backend_selection,
                        notification,
                        cuda_ordering,
                        cuda_device,
                        work,
                    ) = self._reserve_indexed_selection_submission(
                        plan=plan,
                        op=op,
                        local_indices=local_indices,
                        remote_indices=remote_indices,
                        selection=selection,
                        notification=notification,
                        cuda_ordering=cuda_ordering,
                        stream=stream,
                        ready_event=ready_event,
                        recovery_token=recovery_token,
                    )
                    submission_token = work
                    self._works.add(work)
                    active_plan._submitting.add(work)
                    active_plan._works.add(work)

            if cuda_ordering is not CudaOrderingMode.CALLER_READY:
                _establish_cuda_ordering_prevalidated(
                    cuda_ordering,
                    stream,
                    ready_event,
                    cuda_device,
                )
            if initial_max_polls is None:
                result = self._backend_submit_prevalidated(
                    active_op,
                    active_plan._handle,
                    selection=backend_selection,
                    notification=notification,
                    # A direct selection submit is the allocation-minimal
                    # dynamic form used once by framework requests. Bind
                    # explicitly for native reuse across generations.
                    reuse_token=None,
                    adopt_work=work._adopt_backend_work,
                )
                if result is not None:
                    raise BackendFailureError(
                        "backend prevalidated submit must return None after "
                        "adopting work"
                    )
            else:
                result = self._backend_submit_prevalidated_and_poll(
                    active_op,
                    active_plan._handle,
                    selection=backend_selection,
                    notification=notification,
                    max_polls=initial_max_polls,
                    timeout_ns=initial_timeout_ns,
                    adopt_work=work._adopt_backend_work,
                )
            current_work = work
            work_lock = current_work._lock
            if work_lock is None:
                if current_work._backend_work is None:
                    raise BackendFailureError(
                        "backend prevalidated submit returned without adopting work"
                    )
                if initial_max_polls is not None:
                    current_work._ingest_backend_state_unlocked(result)
            else:
                with work_lock:
                    if current_work._backend_work is None:
                        raise BackendFailureError(
                            "backend prevalidated submit returned without adopting work"
                        )
                    if initial_max_polls is not None:
                        current_work._ingest_backend_state_unlocked(result)
            if state_lock is _NOOP_LOCK:
                active_plan._submitting.discard(submission_token)
                current_work._submitting = False
            else:
                with state_lock:
                    active_plan._submitting.discard(submission_token)
                    with current_work._lock or _NOOP_LOCK:
                        current_work._submitting = False
            return current_work
        except BaseException as error:
            state_lock = self._state_lock
            if state_lock is _NOOP_LOCK:
                adopted = self._reconcile_failed_indexed_submission(
                    active_plan, work, submission_token, error
                )
            else:
                with state_lock:
                    adopted = self._reconcile_failed_indexed_submission(
                        active_plan, work, submission_token, error
                    )
            if adopted:
                return cast(Work, work)
            raise

    def _reconcile_failed_indexed_submission(
        self,
        active_plan: TransferPlan | None,
        work: Work | None,
        submission_token: object | None,
        error: BaseException,
    ) -> bool:
        """Repair indexed submission while the caller holds state lock."""

        if active_plan is not None and work is not None:
            active_plan._submitting.discard(
                work if submission_token is None else submission_token
            )
        if work is None:
            return False
        work_lock = work._lock
        if work_lock is None:
            return self._reconcile_failed_indexed_work_unlocked(
                active_plan, work, error
            )
        with work_lock:
            return self._reconcile_failed_indexed_work_unlocked(
                active_plan, work, error
            )

    def _reconcile_failed_indexed_work_unlocked(
        self,
        active_plan: TransferPlan | None,
        work: Work,
        error: BaseException,
    ) -> bool:
        if work._backend_work is not None:
            work._submission_error = error
            if work._state in _TERMINAL_STATES:
                work._state = WorkState.FAILED
                work._error = error
            work._submitting = False
            return True
        if active_plan is not None:
            active_plan._works.discard(work)
        self._works.discard(work)
        work._closed = True
        work._submitting = False
        work._dependencies = ()
        work._plan = None
        work._recovery_token = None
        return False

    def _submit_bound(
        self,
        bound: BoundTransfer,
        *,
        notification: object = _UNSET_NOTIFICATION,
        recovery_token: object | None,
    ) -> Work:
        """Submit a binding without repeating layout or storage validation."""

        plan: TransferPlan | None = None
        selection: IndexSelection | None = None
        active_notification: bytes | None = None
        reuse_token: object | None = None
        work: Work | None = None
        try:
            state_lock = self._state_lock
            with state_lock:
                self._check_open()
                if not isinstance(bound, BoundTransfer):
                    raise TypeError("expected a BoundTransfer")
                if bound._endpoint is not self:
                    raise InvalidRegionError(
                        "bound transfer belongs to another endpoint"
                    )
                if bound._closed or bound._plan is None:
                    raise ClosedError("bound transfer is closed")
                plan = bound._plan
                if plan._closed:
                    raise ClosedError("transfer plan is closed")
                if plan._creation_failed:
                    raise BackendFailureError(
                        "transfer plan is retained for cleanup only"
                    )
                selection = bound._selection
                if selection is None:
                    if notification is not _UNSET_NOTIFICATION:
                        raise TypeError(
                            "notification override requires an IndexSelection binding"
                        )
                    active_notification = bound._notification
                else:
                    active_notification = (
                        None
                        if notification is _UNSET_NOTIFICATION
                        else _validate_attached_notification(
                            self.capabilities,
                            self.backend,
                            notification,
                        )
                    )
                    reuse_token = selection._reuse_token
                    if reuse_token is None:
                        reuse_token = _ReuseToken()
                        object.__setattr__(selection, "_reuse_token", reuse_token)
                work = Work(
                    self,
                    None,
                    plan=plan,
                    bound=bound,
                    dependencies=(
                        () if active_notification is None else (active_notification,)
                    ),
                    recovery_token=recovery_token,
                    _token=_CONSTRUCTION_TOKEN,
                )
                # Use Work itself as the in-progress token. This publishes all
                # lifetime reservations before CUDA or provider activity while
                # avoiding a second per-submission allocation.
                self._works.add(work)
                plan._submitting.add(work)
                bound._submitting.add(work)
                plan._works.add(work)
                bound._works.add(work)

            if bound._cuda_ordering is not CudaOrderingMode.CALLER_READY:
                _establish_cuda_ordering_prevalidated(
                    bound._cuda_ordering,
                    bound._stream,
                    bound._ready_event,
                    bound._cuda_device,
                )

            if selection is None:
                result = self._backend_submit(
                    bound._op,
                    plan._handle,
                    local_indices=bound._local_indices,
                    remote_indices=bound._remote_indices,
                    notification=active_notification,
                    reuse_token=bound._reuse_token,
                    adopt_work=work._adopt_backend_work,
                )
            else:
                result = self._backend_submit_prevalidated(
                    bound._op,
                    plan._handle,
                    selection=selection._backend_selection,
                    notification=active_notification,
                    reuse_token=reuse_token,
                    adopt_work=work._adopt_backend_work,
                )
            if result is not None:
                raise BackendFailureError(
                    "backend submit must return None after adopting its work"
                )
            current_work = work
            work_lock = current_work._lock
            if work_lock is None:
                if current_work._backend_work is None:
                    raise BackendFailureError(
                        "backend submit returned without adopting transfer work"
                    )
            else:
                with work_lock:
                    if current_work._backend_work is None:
                        raise BackendFailureError(
                            "backend submit returned without adopting transfer work"
                        )
            if state_lock is _NOOP_LOCK:
                plan._submitting.discard(current_work)
                bound._submitting.discard(current_work)
                current_work._submitting = False
            else:
                with state_lock:
                    plan._submitting.discard(current_work)
                    bound._submitting.discard(current_work)
                    with current_work._lock or _NOOP_LOCK:
                        current_work._submitting = False
            return current_work
        except BaseException as error:
            with self._state_lock:
                if plan is not None and work is not None:
                    plan._submitting.discard(work)
                    bound._submitting.discard(work)
                adopted = False
                if work is not None:
                    with work._lock or _NOOP_LOCK:
                        adopted = work._backend_work is not None
                if adopted:
                    # Preserve the generic submission recovery contract: once
                    # adopted, an interruption is reported by the retained Work.
                    work = cast(Work, work)
                    with work._lock or _NOOP_LOCK:
                        work._submission_error = error
                        work._submitting = False
                    return work
                if work is not None:
                    if plan is not None:
                        plan._works.discard(work)
                    bound._works.discard(work)
                    self._works.discard(work)
                    with work._lock or _NOOP_LOCK:
                        work._closed = True
                        work._submitting = False
                        work._dependencies = ()
                        work._plan = None
                        work._bound = None
                        work._recovery_token = None
            raise

    def send_notification(
        self,
        peer: Peer,
        payload: bytes | bytearray | memoryview,
        *,
        recovery_token: object | None = None,
    ) -> Work:
        if self._thread_mode is ThreadMode.MULTIPLE:
            return self._send_notification_multiple(
                peer, payload, recovery_token=recovery_token
            )

        state_lock = self._state_lock
        if state_lock is _NOOP_LOCK:
            return self._send_notification_locked(
                peer, payload, recovery_token=recovery_token
            )
        with state_lock:
            return self._send_notification_locked(
                peer, payload, recovery_token=recovery_token
            )

    def _send_notification_locked(
        self,
        peer: Peer,
        payload: bytes | bytearray | memoryview,
        *,
        recovery_token: object | None,
    ) -> Work:
        """Send while the selected non-MULTIPLE lifecycle boundary is held."""

        self._check_open()
        self._validate_peer(peer)
        if not self.capabilities.standalone_notifications:
            raise UnsupportedError(
                f"backend {self.backend!r} does not support notifications"
            )
        payload = _as_bytes(payload, "payload", max_nbytes=_MAX_NOTIFICATION_BYTES)
        work = Work(
            self,
            None,
            dependencies=(peer, payload),
            recovery_token=recovery_token,
            _token=_CONSTRUCTION_TOKEN,
        )
        try:
            self._works.add(work)
            result = self._backend_send_notification(
                peer._handle,
                payload,
                adopt_work=work._adopt_backend_work,
            )
            if result is not None:
                raise BackendFailureError(
                    "backend notification send must return None after adopting its work"
                )
            lock = work._lock
            if lock is None:
                if work._backend_work is None:
                    raise BackendFailureError(
                        "backend notification send returned without adopting work"
                    )
            else:
                with lock:
                    if work._backend_work is None:
                        raise BackendFailureError(
                            "backend notification send returned without adopting work"
                        )
            work._submitting = False
            return work
        except BaseException as error:
            with work._lock or _NOOP_LOCK:
                adopted = work._backend_work is not None
                if adopted:
                    work._submission_error = error
                    work._submitting = False
                    return work
                work._closed = True
                work._submitting = False
                work._dependencies = ()
                work._recovery_token = None
            self._works.discard(work)
            raise

    def _send_notification_multiple(
        self,
        peer: Peer,
        payload: bytes | bytearray | memoryview,
        *,
        recovery_token: object | None,
    ) -> Work:
        """Publish Work ownership, then preserve provider MULTIPLE concurrency."""

        work: Work | None = None
        try:
            with self._state_lock:
                self._check_open()
                self._validate_peer(peer)
                if not self._capabilities.standalone_notifications:
                    raise UnsupportedError(
                        f"backend {self._backend_name!r} does not support notifications"
                    )
                payload = _as_bytes(
                    payload, "payload", max_nbytes=_MAX_NOTIFICATION_BYTES
                )
                work = Work(
                    self,
                    None,
                    dependencies=(peer, payload),
                    recovery_token=recovery_token,
                    _token=_CONSTRUCTION_TOKEN,
                )
                # Work itself is the allocation already required by this API and
                # becomes the close-visible lifetime reservation before native
                # entry. No second per-send token is needed.
                self._works.add(work)

            result = self._backend_send_notification(
                peer._handle,
                payload,
                adopt_work=work._adopt_backend_work,
            )
            if result is not None:
                raise BackendFailureError(
                    "backend notification send must return None after adopting its work"
                )
            with work._lock or _NOOP_LOCK:
                if work._backend_work is None:
                    raise BackendFailureError(
                        "backend notification send returned without adopting work"
                    )
            with self._state_lock:
                with work._lock or _NOOP_LOCK:
                    work._submitting = False
            return work
        except BaseException as error:
            with self._state_lock:
                if work is None:
                    raise
                with work._lock or _NOOP_LOCK:
                    if work._backend_work is not None:
                        work._submission_error = error
                        work._submitting = False
                        return work
                    work._closed = True
                    work._submitting = False
                    work._dependencies = ()
                    work._recovery_token = None
                self._works.discard(work)
            raise

    def send_notification_sync(
        self,
        peer: Peer,
        payload: bytes | bytearray | memoryview,
    ) -> None:
        """Send without allocating a :class:`Work`.

        Return means exactly ``standalone_notification_completion``. In
        particular, NIXL's ``LOCAL_ACCEPTED`` does not mean the peer observed
        the payload. Once backend entry begins, an exception can be
        outcome-ambiguous; protocols that need certainty must use sequence
        numbers and acknowledgement, or retain :meth:`send_notification` and
        its inspectable Work ownership.
        """

        if self._thread_mode is ThreadMode.MULTIPLE:
            self._send_notification_sync_multiple(peer, payload)
            return

        state_lock = self._state_lock
        if state_lock is _NOOP_LOCK:
            self._send_notification_sync_locked(peer, payload)
            return
        with state_lock:
            self._send_notification_sync_locked(peer, payload)

    def _send_notification_sync_locked(
        self,
        peer: Peer,
        payload: bytes | bytearray | memoryview,
    ) -> None:
        """Validate and send while the caller supplies lifecycle exclusion."""

        self._check_open()
        self._validate_peer(peer)
        if not self._capabilities.synchronous_notification_send:
            raise UnsupportedError(
                f"backend {self._backend_name!r} does not support synchronous "
                "notification send"
            )
        payload = _as_bytes(payload, "payload", max_nbytes=_MAX_NOTIFICATION_BYTES)
        send = self._send_notification_sync_dispatch
        if send is None:
            raise BackendFailureError(
                "backend advertised synchronous notification send without a "
                "dispatch method"
            )
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    result = send(peer._handle, payload)
            else:
                result = send(peer._handle, payload)
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self._backend_name!r} failed to send notification "
                "synchronously; delivery outcome may be ambiguous"
            ) from error
        if result is not None:
            raise BackendFailureError(
                "backend synchronous notification send must return None; "
                "delivery outcome may be ambiguous"
            )

    def _send_notification_sync_multiple(
        self,
        peer: Peer,
        payload: bytes | bytearray | memoryview,
    ) -> None:
        """Reserve peer lifetime briefly, then preserve native RW concurrency."""

        reserved_peer: Peer | None = None
        reservation: _SyncNotificationReservation | None = None
        try:
            with self._state_lock:
                self._check_open()
                self._validate_peer(peer)
                if not self._capabilities.synchronous_notification_send:
                    raise UnsupportedError(
                        f"backend {self._backend_name!r} does not support synchronous "
                        "notification send"
                    )
                payload = _as_bytes(
                    payload, "payload", max_nbytes=_MAX_NOTIFICATION_BYTES
                )
                send = self._send_notification_sync_dispatch
                if send is None:
                    raise BackendFailureError(
                        "backend advertised synchronous notification send without a "
                        "dispatch method"
                    )
                peer_handle = peer._handle
                pool = peer._sync_notification_reservation_pool
                reservation = pool.pop() if pool else _SyncNotificationReservation()
                reserved_peer = peer
                reservation.active = True
                peer._active_sync_notification_sends.add(reservation.reference)

            try:
                result = send(peer_handle, payload)
            except TransferError:
                raise
            except Exception as error:
                raise BackendFailureError(
                    f"backend {self._backend_name!r} failed to send notification "
                    "synchronously; delivery outcome may be ambiguous"
                ) from error
            if result is not None:
                raise BackendFailureError(
                    "backend synchronous notification send must return None; "
                    "delivery outcome may be ambiguous"
                )
        finally:
            if reserved_peer is not None and reservation is not None:
                # Publish native exit before taking an interruptible cleanup
                # lock. A retained exception traceback may keep this frame and
                # token alive, so liveness of the weakref alone is insufficient.
                reservation.active = False
                with self._state_lock:
                    reserved_peer._active_sync_notification_sends.discard(
                        reservation.reference
                    )
                    if not reserved_peer._closed:
                        reserved_peer._sync_notification_reservation_pool.append(
                            reservation
                        )

    def poll_notifications(self, max_items: int | None = None) -> list[Notification]:
        if self._thread_mode is ThreadMode.MULTIPLE:
            notifications = self._poll_backend_notifications_multiple(
                max_items, grouped=False
            )
        else:
            state_lock = self._state_lock
            with state_lock:
                self._check_open()
                if not self.capabilities.notifications:
                    raise UnsupportedError(
                        f"backend {self.backend!r} does not support notifications"
                    )
                if max_items is not None:
                    if isinstance(max_items, bool) or not isinstance(max_items, int):
                        raise TypeError(
                            "max_items must be a non-negative integer or None"
                        )
                    if max_items < 0:
                        raise ValueError("max_items must be non-negative")
                if max_items == 0:
                    return []
                try:
                    if self._thread_mode is ThreadMode.SERIALIZED:
                        with self._backend_lock:
                            notifications = self._backend.poll_notifications(max_items)
                    else:
                        notifications = self._backend.poll_notifications(max_items)
                except TransferError:
                    raise
                except Exception as error:
                    raise BackendFailureError(
                        f"backend {self.backend!r} failed to poll notifications"
                    ) from error
        if not isinstance(notifications, list) or not all(
            isinstance(item, BackendNotification) for item in notifications
        ):
            raise BackendFailureError(
                "backend notifications must be a list of BackendNotification"
            )
        if max_items is not None and len(notifications) > max_items:
            raise BackendFailureError(
                "backend returned more notifications than max_items"
            )
        return [
            Notification(
                payload=notification.payload,
                source_endpoint_id=notification.source_endpoint_id,
                source_incarnation=notification.source_incarnation,
            )
            for notification in notifications
        ]

    def poll_notification_batches(
        self, max_items: int | None = None
    ) -> list[NotificationBatch]:
        """Destructively receive source-grouped notifications.

        ``max_items`` counts payloads rather than returned batches. The v0
        receive contract remains at-most-once: there is no acknowledgement or
        rollback token. Per-source payload order is preserved, but providers
        need not define a total order across different sources and may return
        repeated batches for one source.
        """

        if self._thread_mode is ThreadMode.MULTIPLE:
            batches = self._poll_backend_notifications_multiple(max_items, grouped=True)
        else:
            state_lock = self._state_lock
            with state_lock:
                self._check_open()
                if not self.capabilities.notifications:
                    raise UnsupportedError(
                        f"backend {self.backend!r} does not support notifications"
                    )
                if max_items is not None:
                    if isinstance(max_items, bool) or not isinstance(max_items, int):
                        raise TypeError(
                            "max_items must be a non-negative integer or None"
                        )
                    if max_items < 0:
                        raise ValueError("max_items must be non-negative")
                if max_items == 0:
                    return []
                try:
                    if self._thread_mode is ThreadMode.SERIALIZED:
                        with self._backend_lock:
                            batches = self._backend.poll_notification_batches(max_items)
                    else:
                        batches = self._backend.poll_notification_batches(max_items)
                except TransferError:
                    raise
                except Exception as error:
                    raise BackendFailureError(
                        f"backend {self.backend!r} failed to poll notification batches"
                    ) from error
        if not isinstance(batches, list) or not all(
            type(item) is BackendNotificationBatch for item in batches
        ):
            raise BackendFailureError(
                "backend notification batches must be a list of "
                "BackendNotificationBatch"
            )
        if max_items is not None:
            payload_count = sum(len(batch.payloads) for batch in batches)
            if payload_count > max_items:
                raise BackendFailureError(
                    "backend returned more notification payloads than max_items"
                )
        return [_notification_batch_from_backend(batch) for batch in batches]

    def _poll_backend_notifications_multiple(
        self,
        max_items: int | None,
        *,
        grouped: bool,
    ) -> object:
        """Run one destructive receive with narrow, close-visible admission."""

        reservation: _BackendCallReservation | None = None
        # Serialize only destructive consumers. Acquire this lock before call
        # admission so queued consumers need no reservation and can safely
        # discover a concurrently completed close before backend entry.
        with self._notification_consumer_lock:
            try:
                with self._state_lock:
                    self._check_open()
                    if not self._capabilities.notifications:
                        raise UnsupportedError(
                            f"backend {self._backend_name!r} does not support "
                            "notifications"
                        )
                    if max_items is not None:
                        if isinstance(max_items, bool) or not isinstance(
                            max_items, int
                        ):
                            raise TypeError(
                                "max_items must be a non-negative integer or None"
                            )
                        if max_items < 0:
                            raise ValueError("max_items must be non-negative")
                    if max_items == 0:
                        return []
                    poll = (
                        self._backend.poll_notification_batches
                        if grouped
                        else self._backend.poll_notifications
                    )
                    reservation = self._reserve_multiple_backend_call_locked()
                try:
                    return poll(max_items)
                except TransferError:
                    raise
                except Exception as error:
                    action = (
                        "poll notification batches" if grouped else "poll notifications"
                    )
                    raise BackendFailureError(
                        f"backend {self._backend_name!r} failed to {action}"
                    ) from error
            finally:
                if reservation is not None:
                    self._finish_multiple_backend_call(reservation)

    def progress(self) -> None:
        if self._thread_mode is ThreadMode.MULTIPLE:
            self._progress_multiple()
            return

        state_lock = self._state_lock
        if state_lock is _NOOP_LOCK:
            self._progress_locked()
            return
        with state_lock:
            self._progress_locked()

    def _progress_locked(self) -> None:
        self._check_open()
        if not self.capabilities.manual_progress:
            raise UnsupportedError(
                f"backend {self.backend!r} does not support manual progress"
            )
        self._backend_call("progress backend", self._backend.progress)

    def _progress_multiple(self) -> None:
        reservation: _BackendCallReservation | None = None
        try:
            with self._state_lock:
                self._check_open()
                if not self._capabilities.manual_progress:
                    raise UnsupportedError(
                        f"backend {self._backend_name!r} does not support manual progress"
                    )
                progress = self._backend.progress
                reservation = self._reserve_multiple_backend_call_locked()
            self._backend_call("progress backend", progress)
        finally:
            if reservation is not None:
                self._finish_multiple_backend_call(reservation)

    @_state_locked
    def close(self) -> None:
        if self._creation_failed:
            self._close_failed_endpoint_backend()
            return
        if self._closed:
            return
        if not self._closing:
            if self._multiple_backend_calls_active_locked():
                raise BusyError(
                    "cannot close endpoint while backend calls are in progress"
                )
            self._retry_orphaned_plans()
            active = [work for work in self._works if not work.test()]
            if active:
                raise BusyError(
                    f"cannot close endpoint with {len(active)} active work item(s)"
                )
            counts = {
                "work": len(self._works),
                "transfer slot": len(self._transfer_slots),
                "bound transfer": len(self._bound_transfers),
                "plan": len(self._plans),
                "peer": len(self._peers),
                "registration": len(self._registrations),
            }
            if any(counts.values()):
                detail = ", ".join(
                    f"{name}={count}" for name, count in counts.items() if count
                )
                raise BusyError(
                    f"close child resources before endpoint close ({detail})"
                )
            # From this point only close() may be retried. This makes a signal
            # after successful provider teardown converge without calling
            # progress on an already closed provider.
            self._closing = True
        self._backend_call(
            "close backend",
            lambda: self._backend_receipt.close(self._backend.close),
        )
        self._detach_backend_finalizer()
        self._closed = True
        active_calls = self._active_multiple_backend_calls
        if active_calls is not None:
            active_calls.clear()
        reservation_pool = self._multiple_backend_call_reservation_pool
        if reservation_pool is not None:
            reservation_pool.clear()

    def _close_failed_endpoint_backend(self) -> None:
        """Retry backend-only cleanup after Endpoint initialization failed."""

        if self._closed:
            self._creation_failed = False
            self._detach_backend_finalizer()
            return
        self._backend_call(
            "close backend after failed endpoint initialization",
            lambda: self._backend_receipt.close(self._backend.close),
        )
        self._detach_backend_finalizer()
        self._closed = True
        self._creation_failed = False

    def _cleanup_adopted_backend_after_creation_error(
        self, original_error: BaseException
    ) -> None:
        implementation = self._backend_receipt.backend
        if implementation is None:
            self._detach_backend_finalizer()
            return
        self._backend = implementation
        self._creation_failed = True
        try:
            self._close_failed_endpoint_backend()
        except BaseException as cleanup_error:
            raise _endpoint_creation_cleanup_failure(
                self, original_error, cleanup_error
            ) from cleanup_error

    def _detach_backend_finalizer(self) -> None:
        self._backend_finalizer.detach()

    def _submit(
        self,
        op: TransferOp,
        plan: TransferPlan | None,
        *,
        local: Sequence[Region] | None,
        remote: Sequence[Region] | None,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        notification: bytes | bytearray | memoryview | None,
        cuda_ordering: CudaOrderingMode | str,
        stream: object | None,
        ready_event: object | None,
        recovery_token: object | None,
    ) -> Work:
        owned_plan = False
        plan_to_release: TransferPlan | None = None
        reserved_plan: TransferPlan | None = None
        submission_token: object | None = None
        work: Work | None = None
        try:
            state_lock = self._state_lock
            with state_lock:
                self._check_open()
                supported = (
                    self.capabilities.read
                    if op is TransferOp.READ
                    else self.capabilities.write
                )
                if not supported:
                    raise UnsupportedError(
                        f"backend {self.backend!r} does not support {op.value}"
                    )
                cuda_ordering = _coerce_enum(
                    cuda_ordering, CudaOrderingMode, "cuda_ordering"
                )
                _validate_cuda_ordering_arguments(
                    self.capabilities,
                    self.backend,
                    cuda_ordering,
                    stream,
                    ready_event,
                )
                if plan is None:
                    if local is None or remote is None:
                        raise TypeError(
                            "one-shot transfer requires both local and remote regions"
                        )
                    plan = self.prepare(local=local, remote=remote, indexed=False)
                    plan_to_release = plan
                    owned_plan = True
                elif local is not None or remote is not None:
                    raise TypeError("pass a plan or one-shot regions, not both")
                active_plan = plan
                self._validate_plan_locked(active_plan)
                local_indices, remote_indices = _validate_plan_indices(
                    active_plan, local_indices, remote_indices, op
                )
                notification = _validate_attached_notification(
                    self.capabilities, self.backend, notification
                )
                cuda_device = _validate_cuda_ordering_selection(
                    active_plan,
                    local_indices,
                    cuda_ordering,
                    stream,
                    ready_event,
                )
                dependencies = tuple(
                    value
                    for value in (
                        local_indices,
                        remote_indices,
                        stream,
                        ready_event,
                        notification,
                    )
                    if value is not None
                )
                # Publish the complete Core lifetime graph before entering the
                # provider. The provider adopts its request into this wrapper
                # before it is allowed to launch transport work.
                work = Work(
                    self,
                    None,
                    plan=active_plan,
                    dependencies=dependencies,
                    owned_plan=owned_plan,
                    recovery_token=recovery_token,
                    _token=_CONSTRUCTION_TOKEN,
                )
                submission_token = work
                reserved_plan = active_plan
                self._works.add(work)
                active_plan._submitting.add(submission_token)
                active_plan._works.add(work)
            _establish_cuda_ordering_prevalidated(
                cuda_ordering,
                stream,
                ready_event,
                cuda_device,
            )

            reuse_token = (
                active_plan._reuse_tokens[op]
                if not owned_plan
                and local_indices is None
                and remote_indices is None
                and notification is None
                and cuda_ordering is CudaOrderingMode.CALLER_READY
                else None
            )
            result = self._backend_submit(
                op,
                active_plan._handle,
                local_indices=local_indices,
                remote_indices=remote_indices,
                notification=notification,
                reuse_token=reuse_token,
                adopt_work=work._adopt_backend_work,
            )
            if result is not None:
                raise BackendFailureError(
                    "backend submit must return None after adopting its work"
                )
            current_work = work
            work_lock = current_work._lock
            if work_lock is None:
                if current_work._backend_work is None:
                    raise BackendFailureError(
                        "backend submit returned without adopting transfer work"
                    )
            else:
                with work_lock:
                    if current_work._backend_work is None:
                        raise BackendFailureError(
                            "backend submit returned without adopting transfer work"
                        )
            if state_lock is _NOOP_LOCK:
                reserved_plan._submitting.discard(submission_token)
                current_work._submitting = False
            else:
                with state_lock:
                    reserved_plan._submitting.discard(submission_token)
                    with current_work._lock or _NOOP_LOCK:
                        current_work._submitting = False
            return current_work
        except BaseException as error:
            with self._state_lock:
                if reserved_plan is not None and submission_token is not None:
                    reserved_plan._submitting.discard(submission_token)
                adopted = False
                if work is not None:
                    with work._lock or _NOOP_LOCK:
                        adopted = work._backend_work is not None
                if adopted:
                    # A provider must not raise after adoption, but retaining
                    # the Work is the only safe response if it does: native
                    # access may already be in flight. Surface the protocol
                    # violation only after the backend request is terminal.
                    work = cast(Work, work)
                    with work._lock or _NOOP_LOCK:
                        work._submission_error = error
                        work._submitting = False
                    return work
                if work is not None:
                    if work._plan is not None:
                        work._plan._works.discard(work)
                    self._works.discard(work)
                    with work._lock or _NOOP_LOCK:
                        work._closed = True
                        work._submitting = False
                        work._dependencies = ()
                        work._plan = None
                        work._recovery_token = None
                if plan_to_release is not None:
                    try:
                        plan_to_release.close()
                    except BaseException:
                        self._orphaned_plans.add(plan_to_release)
            raise

    def _validate_regions(
        self,
        local: tuple[Region, ...],
        remote: tuple[Region, ...],
        *,
        indexed: bool,
    ) -> None:
        if not local or not remote:
            raise InvalidRegionError("local and remote regions must be non-empty")
        for region in local:
            if not isinstance(region, Region) or region.is_remote:
                raise InvalidRegionError(
                    "local must contain local Registration regions"
                )
            if not isinstance(region._origin, Registration):
                raise InvalidRegionError("local region has an invalid owner")
            self._validate_registration(region._origin)
        peers: set[Peer] = set()
        for region in remote:
            if not isinstance(region, Region) or not region.is_remote:
                raise InvalidRegionError("remote must contain Peer regions")
            if not isinstance(region._origin, Peer):
                raise InvalidRegionError("remote region has an invalid owner")
            self._validate_peer(region._origin)
            peers.add(region._origin)
        if len(peers) != 1:
            raise InvalidRegionError("one transfer plan may target exactly one peer")
        if not indexed:
            local_size = sum(region.total_nbytes for region in local)
            remote_size = sum(region.total_nbytes for region in remote)
            if local_size != remote_size:
                raise InvalidRegionError(
                    f"local and remote byte totals differ: "
                    f"{local_size} != {remote_size}"
                )

    def _validate_registration(self, registration: Registration) -> None:
        if not isinstance(registration, Registration):
            raise TypeError("expected a Registration")
        if registration._endpoint is not self:
            raise InvalidRegionError("registration belongs to another endpoint")
        registration._check_open()
        _validate_memory_stable(registration._memory)

    def _validate_peer(self, peer: Peer) -> None:
        if not isinstance(peer, Peer):
            raise TypeError("expected a Peer")
        if peer._endpoint is not self:
            raise InvalidRegionError("peer belongs to another endpoint")
        peer._check_open()

    def _validate_plan(self, plan: TransferPlan) -> None:
        self._check_open()
        self._validate_plan_locked(plan)

    def _validate_plan_locked(self, plan: TransferPlan) -> None:
        """Validate a plan after this endpoint has entered its state lock."""

        if not isinstance(plan, TransferPlan):
            raise TypeError("expected a TransferPlan")
        if plan._endpoint is not self:
            raise InvalidRegionError("transfer plan belongs to another endpoint")
        if plan._closed:
            raise ClosedError("transfer plan is closed")
        if plan._creation_failed:
            raise BackendFailureError("transfer plan is retained for cleanup only")
        for origin in plan._origins:
            if isinstance(origin, Registration):
                if origin._endpoint is not self:
                    raise InvalidRegionError("registration belongs to another endpoint")
                if origin._closed:
                    raise ClosedError("registration is closed")
                if origin._creation_failed:
                    raise BackendFailureError(
                        "registration is retained for cleanup only"
                    )
                _validate_memory_stable(origin._memory)

    @_state_locked
    def _close_registration(self, registration: Registration) -> None:
        if registration._closed:
            self._finish_registration_close(registration)
            return
        self._check_open()
        self._retry_orphaned_plans()
        if any(
            not plan._closed and registration in plan._origins for plan in self._plans
        ):
            raise BusyError(
                f"registration {registration.name!r} is used by an open plan"
            )
        result = self._backend_call(
            "deregister memory",
            lambda: self._backend.deregister(registration._handle),
        )
        if result is not None:
            raise BackendFailureError("backend deregister must return None")
        registration._closed = True
        self._finish_registration_close(registration)

    def _finish_registration_close(self, registration: Registration) -> None:
        """Idempotently finish Core-only registration teardown."""

        registration_id = registration.registration_id
        name = registration.name
        if self._registration_names.get(name) is registration:
            self._registration_names.pop(name, None)
        registration._handle = None
        registration._memory = _released_memory(registration._memory)
        # Authoritative recovery registry removal is the Core teardown commit.
        if self._registrations.get(registration_id) is registration:
            self._registrations.pop(registration_id, None)

    @_state_locked
    def _close_peer(self, peer: Peer) -> None:
        if peer._closed:
            self._finish_peer_close(peer)
            return
        self._check_open()
        self._retry_orphaned_plans()
        active_sends = peer._active_sync_notification_sends
        for reference in tuple(active_sends):
            reservation = reference()
            if reservation is None or not reservation.active:
                active_sends.discard(reference)
        if active_sends:
            raise BusyError(
                f"peer {peer.name!r} has synchronous notification sends in progress"
            )
        if any(not plan._closed and peer in plan._origins for plan in self._plans):
            raise BusyError(f"peer {peer.name!r} is used by an open plan")
        if any(peer in work._dependencies for work in self._works):
            raise BusyError(f"peer {peer.name!r} is used by an open work item")
        result = self._backend_call(
            "release peer", lambda: self._backend.release_peer(peer._handle)
        )
        if result is not None:
            raise BackendFailureError("backend release_peer must return None")
        peer._closed = True
        self._finish_peer_close(peer)

    def _finish_peer_close(self, peer: Peer) -> None:
        """Idempotently finish Core-only peer teardown."""

        if self._peers.get(peer.endpoint_id) is peer:
            self._peers.pop(peer.endpoint_id, None)
        peer._handle = None
        peer._registrations = {}
        peer._active_sync_notification_sends.clear()
        peer._sync_notification_reservation_pool.clear()

    @_state_locked
    def _close_plan(self, plan: TransferPlan) -> None:
        if plan._closed:
            self._finish_plan_close(plan)
            return
        self._check_open()
        if plan._submitting:
            raise BusyError("transfer plan has submissions in progress")
        if plan._works:
            raise BusyError("transfer plan has open work items")
        if plan._bindings:
            raise BusyError("transfer plan has open bound transfers")
        result = self._backend_call(
            "release transfer plan",
            lambda: self._backend.release_plan(plan._handle),
        )
        if result is not None:
            raise BackendFailureError("backend release_plan must return None")
        plan._closed = True
        self._finish_plan_close(plan)

    def _finish_plan_close(self, plan: TransferPlan) -> None:
        """Idempotently finish Core-only transfer-plan teardown."""

        self._plans.discard(plan)
        self._orphaned_plans.discard(plan)
        plan._handle = None
        plan._local = ()
        plan._remote = ()
        plan._origins = frozenset()
        plan._reuse_tokens = {}

    @_state_locked
    def _close_bound_transfer(self, bound: BoundTransfer) -> None:
        if bound._closed:
            # Closure is the commit marker, not an early-return condition. An
            # interrupt after it may leave a parent edge or retained field;
            # retrying completes the idempotent Core-only epilogue.
            self._finish_bound_transfer_close(bound)
            return
        self._check_open()
        if bound._endpoint is not self:
            raise InvalidRegionError("bound transfer belongs to another endpoint")
        plan = bound._check_open()
        if bound._slots:
            raise BusyError("bound transfer has open persistent request slots")
        if bound._works:
            raise BusyError("bound transfer has open work items")
        if bound._submitting:
            raise BusyError("bound transfer has submissions in progress")
        # Unlink blocking graph edges before publishing CLOSED, while retaining
        # `_plan` so a retry after that commit can repair either edge.
        plan._bindings.discard(bound)
        self._bound_transfers.discard(bound)
        bound._closed = True
        self._finish_bound_transfer_close(bound)

    def _finish_bound_transfer_close(self, bound: BoundTransfer) -> None:
        """Idempotently finish Core-only bound-transfer unlink."""

        plan = bound._plan
        if plan is not None:
            plan._bindings.discard(bound)
        self._bound_transfers.discard(bound)
        bound._plan = None
        bound._local_indices = None
        bound._remote_indices = None
        bound._notification = None
        bound._cuda_device = None
        bound._stream = None
        bound._ready_event = None
        bound._reuse_token = None
        bound._selection = None

    @_state_locked
    def _close_transfer_slot(self, slot: TransferSlot) -> None:
        """Release one idle persistent request and unlink its lifetime graph."""

        if not isinstance(slot, TransferSlot):
            raise TypeError("expected a TransferSlot")
        if slot._endpoint is not self:
            raise InvalidRegionError("transfer slot belongs to another endpoint")
        if slot._closed:
            # Closure is the commit marker, not an early-return condition. An
            # interrupt after it may still leave one parent edge or retained
            # callback, so every retry finishes the idempotent Core epilogue.
            self._finish_transfer_slot_close(slot)
            return
        self._check_open()
        bound = slot._bound
        if bound is None or bound._closed:
            raise ClosedError("transfer slot's bound transfer is closed")
        if slot._active:
            raise BusyError(
                "transfer slot owns an unrecycled generation; recycle it before close"
            )
        close = slot._backend_close
        if close is None:
            if slot._backend_slot is not None:
                raise BackendFailureError(
                    "transfer slot has no backend close operation"
                )
        else:
            self._backend_request_slot_close(close)

        # Closure commits provider release while both recovery edges still
        # exist. The retry path can therefore remove the bound blocker first
        # and the endpoint's public recovery root last.
        slot._closed = True
        self._finish_transfer_slot_close(slot)

    def _finish_transfer_slot_close(self, slot: TransferSlot) -> None:
        """Idempotently finish Core-only unlink after provider release."""

        bound = slot._bound
        if bound is not None:
            bound._slots.discard(slot)
        self._transfer_slots.discard(slot)
        slot._bound = None
        slot._backend_slot = None
        slot._backend_execution_dispatch = None
        slot._backend_start = None
        slot._backend_start_and_poll = None
        if isinstance(slot, RequestLocalNotificationSlot):
            slot._backend_start_with_notification = None
            slot._backend_start_and_poll_with_notification = None
            slot._notification_fast_execute = None
        slot._backend_poll_bounded = None
        slot._backend_poll_state = None
        slot._backend_cancel = None
        slot._backend_recycle = None
        slot._backend_close = None
        slot._ready_event_synchronize = None
        slot._fast_execute = None
        slot._backend_generation_active = False
        slot._error = None
        slot._start_error = None

    def _retry_orphaned_plans(self) -> None:
        for plan in tuple(self._orphaned_plans):
            plan.close()

    def _check_open(self) -> None:
        self._check_thread()
        if self._creation_failed:
            raise BackendFailureError(
                f"endpoint {self.name!r} is retained for creation cleanup only; "
                "retry close()"
            )
        if self._closed:
            raise ClosedError(f"endpoint {self.name!r} is closed")
        if self._closing:
            raise ClosedError(
                f"endpoint {self.name!r} is closing; only close() may be retried"
            )

    def _reserve_multiple_backend_call_locked(self) -> _BackendCallReservation:
        active_calls = self._active_multiple_backend_calls
        pool = self._multiple_backend_call_reservation_pool
        assert active_calls is not None and pool is not None
        reservation = pool.pop() if pool else _BackendCallReservation()
        try:
            reservation.active = True
            active_calls.add(reservation.reference)
        except BaseException:
            reservation.active = False
            active_calls.discard(reservation.reference)
            pool.append(reservation)
            raise
        return reservation

    def _finish_multiple_backend_call(
        self, reservation: _BackendCallReservation
    ) -> None:
        # Publish native exit before interruptible cleanup. A traceback can keep
        # this frame alive; close therefore also checks the explicit active bit.
        reservation.active = False
        with self._state_lock:
            active_calls = self._active_multiple_backend_calls
            pool = self._multiple_backend_call_reservation_pool
            assert active_calls is not None and pool is not None
            active_calls.discard(reservation.reference)
            if not self._closing and not self._closed:
                pool.append(reservation)

    def _multiple_backend_calls_active_locked(self) -> bool:
        active_calls = self._active_multiple_backend_calls
        if not active_calls:
            return False
        for reference in tuple(active_calls):
            reservation = reference()
            if reservation is None or not reservation.active:
                active_calls.discard(reference)
        return bool(active_calls)

    def _backend_call(self, action: str, function):  # type: ignore[no-untyped-def]
        self._check_thread()
        try:
            if self.thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return function()
            return function()
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to {action}"
            ) from error

    def _backend_submit(
        self,
        op: TransferOp,
        plan_handle: object,
        *,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        notification: bytes | None,
        reuse_token: object | None,
        adopt_work: Callable[[BackendWork], None],
    ) -> object | None:
        """Call the submission SPI without allocating an extra lambda."""

        # Both callers validate endpoint/thread state before reaching this
        # same-stack helper; avoid repeating the thread-ident lookup here.
        submit = cast(Callable[..., object | None], self._backend.submit)
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return submit(
                        op,
                        plan_handle,
                        local_indices=local_indices,
                        remote_indices=remote_indices,
                        notification=notification,
                        reuse_token=reuse_token,
                        adopt_work=adopt_work,
                    )
            return submit(
                op,
                plan_handle,
                local_indices=local_indices,
                remote_indices=remote_indices,
                notification=notification,
                reuse_token=reuse_token,
                adopt_work=adopt_work,
            )
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to submit {op.value}"
            ) from error

    def _backend_send_notification(
        self,
        peer_handle: object,
        payload: bytes,
        *,
        adopt_work: Callable[[BackendWork], None],
    ) -> object | None:
        """Call standalone notification send without per-call closures."""

        # send_notification() has already validated endpoint and thread state.
        send = cast(Callable[..., object | None], self._backend.send_notification)
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return send(peer_handle, payload, adopt_work=adopt_work)
            return send(peer_handle, payload, adopt_work=adopt_work)
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to send notification"
            ) from error

    def _backend_submit_prevalidated(
        self,
        op: TransferOp,
        plan_handle: object,
        *,
        selection: BackendIndexSelection,
        notification: bytes | None,
        reuse_token: object | None,
        adopt_work: Callable[[BackendWork], None],
    ) -> object | None:
        """Call the authenticated indexed-selection SPI without a lambda."""

        submit = cast(Callable[..., object | None], self._backend.submit_prevalidated)
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return submit(
                        op,
                        plan_handle,
                        selection=selection,
                        notification=notification,
                        reuse_token=reuse_token,
                        adopt_work=adopt_work,
                    )
            return submit(
                op,
                plan_handle,
                selection=selection,
                notification=notification,
                reuse_token=reuse_token,
                adopt_work=adopt_work,
            )
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to submit prevalidated {op.value}"
            ) from error

    def _backend_submit_prevalidated_and_poll(
        self,
        op: TransferOp,
        plan_handle: object,
        *,
        selection: BackendIndexSelection,
        notification: bytes | None,
        max_polls: int,
        timeout_ns: int | None,
        adopt_work: Callable[[BackendWork], None],
    ) -> WorkState:
        """Call the optional fused one-shot selection SPI without a lambda."""

        submit = cast(
            Callable[..., WorkState],
            self._backend.submit_prevalidated_and_poll,
        )
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return submit(
                        op,
                        plan_handle,
                        selection=selection,
                        notification=notification,
                        max_polls=max_polls,
                        timeout_ns=timeout_ns,
                        adopt_work=adopt_work,
                    )
            return submit(
                op,
                plan_handle,
                selection=selection,
                notification=notification,
                max_polls=max_polls,
                timeout_ns=timeout_ns,
                adopt_work=adopt_work,
            )
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to submit and poll "
                f"prevalidated {op.value}"
            ) from error

    def _backend_create_request_slot(
        self,
        op: TransferOp,
        plan_handle: object,
        *,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        notification: bytes | None,
        adopt_slot: Callable[..., None],
    ) -> object | None:
        """Call the optional persistent-request SPI without a lambda."""

        create = cast(Callable[..., object | None], self._backend.create_request_slot)
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return create(
                        op,
                        plan_handle,
                        local_indices=local_indices,
                        remote_indices=remote_indices,
                        notification=notification,
                        adopt_slot=adopt_slot,
                    )
            return create(
                op,
                plan_handle,
                local_indices=local_indices,
                remote_indices=remote_indices,
                notification=notification,
                adopt_slot=adopt_slot,
            )
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to create a persistent "
                f"{op.value} request slot"
            ) from error

    def _backend_request_slot_start(self, start: Callable[[], None]) -> None:
        """Start a cached backend slot operation on the lowest-overhead path."""

        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    result = start()
            else:
                result = start()
            if result is not None:
                raise BackendFailureError("backend request-slot start must return None")
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to start a persistent request"
            ) from error

    def _backend_request_slot_start_with_notification(
        self,
        start: Callable[[bytes | None], None],
        notification: bytes | None,
    ) -> None:
        """Start a cached request with an explicit notification override."""

        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    result = start(notification)
            else:
                result = start(notification)
            if result is not None:
                raise BackendFailureError(
                    "backend request-slot start_with_notification must return None"
                )
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to start a persistent request "
                "with a notification override"
            ) from error

    def _backend_request_slot_start_and_poll(
        self,
        start_and_poll: Callable[..., WorkState | None],
        *,
        max_polls: int,
        timeout_ns: int | None,
    ) -> WorkState | None:
        """Enter the optional bounded persistent-request fused operation."""

        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return start_and_poll(
                        max_polls=max_polls,
                        timeout_ns=timeout_ns,
                    )
            return start_and_poll(max_polls=max_polls, timeout_ns=timeout_ns)
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed during fused persistent "
                "request start/poll"
            ) from error

    def _backend_request_slot_start_and_poll_with_notification(
        self,
        start_and_poll: Callable[..., WorkState | None],
        notification: bytes | None,
        *,
        max_polls: int,
        timeout_ns: int | None,
    ) -> WorkState | None:
        """Enter fused post/poll with a request-local notification override."""

        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return start_and_poll(
                        notification,
                        max_polls=max_polls,
                        timeout_ns=timeout_ns,
                    )
            return start_and_poll(
                notification,
                max_polls=max_polls,
                timeout_ns=timeout_ns,
            )
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed during fused persistent "
                "request notification start/poll"
            ) from error

    def _backend_request_slot_poll_bounded(
        self,
        poll_bounded: Callable[..., WorkState | None],
        *,
        max_polls: int,
        timeout_ns: int | None,
        timeout_check_interval: int,
    ) -> WorkState | None:
        """Enter the optional bounded persistent-request polling operation."""

        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return poll_bounded(
                        max_polls=max_polls,
                        timeout_ns=timeout_ns,
                        timeout_check_interval=timeout_check_interval,
                    )
            return poll_bounded(
                max_polls=max_polls,
                timeout_ns=timeout_ns,
                timeout_check_interval=timeout_check_interval,
            )
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed during bounded persistent "
                "request polling"
            ) from error

    def _backend_request_slot_state(
        self, poll_state: Callable[[], WorkState | None]
    ) -> WorkState | None:
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return poll_state()
            return poll_state()
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to query persistent request state"
            ) from error

    def _backend_request_slot_error(
        self, backend_slot: BackendRequestSlot
    ) -> BaseException | None:
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return backend_slot.error
            return backend_slot.error
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to query persistent request error"
            ) from error

    def _backend_request_slot_completion_event(
        self, backend_slot: BackendRequestSlot
    ) -> object | None:
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return backend_slot.completion_event
            return backend_slot.completion_event
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to query persistent request "
                "completion event"
            ) from error

    def _backend_request_slot_cancel(self, cancel: Callable[[], bool]) -> bool:
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return cancel()
            return cancel()
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to cancel a persistent request"
            ) from error

    def _backend_request_slot_recycle(self, recycle: Callable[[], None]) -> None:
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    result = recycle()
            else:
                result = recycle()
            if result is not None:
                raise BackendFailureError(
                    "backend request-slot recycle must return None"
                )
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to recycle a persistent request"
            ) from error

    def _backend_request_slot_close(self, close: Callable[[], None]) -> None:
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    result = close()
            else:
                result = close()
            if result is not None:
                raise BackendFailureError("backend request-slot close must return None")
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to release a persistent request"
            ) from error

    def _backend_work_state(self, backend_work: BackendWork) -> WorkState:
        """Read a provider Work state without a lambda or duplicate thread check."""

        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return backend_work.state
            return backend_work.state
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to query work state"
            ) from error

    def _backend_work_poll_bounded(
        self,
        backend_work: BackendWork,
        *,
        max_polls: int,
        timeout_ns: int | None,
        timeout_check_interval: int,
    ) -> WorkState:
        """Call a fused provider Work poll without a lambda."""

        poll = backend_work.poll_bounded
        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    return poll(
                        max_polls=max_polls,
                        timeout_ns=timeout_ns,
                        timeout_check_interval=timeout_check_interval,
                    )
            return poll(
                max_polls=max_polls,
                timeout_ns=timeout_ns,
                timeout_check_interval=timeout_check_interval,
            )
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to poll work"
            ) from error

    def _backend_work_close(self, backend_work: BackendWork) -> None:
        """Close provider Work without a bound-method/lambda dispatch wrapper."""

        try:
            if self._thread_mode is ThreadMode.SERIALIZED:
                with self._backend_lock:
                    result = backend_work.close()
            else:
                result = backend_work.close()
            if result is not None:
                raise BackendFailureError("backend work close must return None")
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                f"backend {self.backend!r} failed to release work"
            ) from error

    def _check_thread(self) -> None:
        if (
            self._thread_mode is ThreadMode.SINGLE
            and threading.get_ident() != self._owner_thread
        ):
            raise UnsupportedError(
                "single-thread endpoint used from a different thread"
            )

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _registered_memory(
    value: object,
    registration_id: str,
    name: str,
) -> RegisteredMemory:
    if isinstance(value, RawSpan):
        return RegisteredMemory(
            registration_id=registration_id,
            name=name,
            owner=_LifetimeAnchor(value.owner),
            address=value.address,
            nbytes=value.nbytes,
            device=value.device,
            memory_type=value.memory_type,
        )
    if isinstance(value, torch.Tensor):
        if value.layout is not torch.strided:
            raise InvalidRegionError("only strided tensor layouts can be registered")
        if not value.is_contiguous():
            raise InvalidRegionError(
                "tensor registration requires a contiguous tensor; "
                "no implicit copy is performed"
            )
        nbytes = value.numel() * value.element_size()
        address = value.data_ptr()
        device = str(value.device)
        memory_type = _tensor_memory_type(value)
        storage = value.untyped_storage()
        anchor = _LifetimeAnchor(
            owner=value,
            storage=storage,
            allocation_address=storage.data_ptr(),
            allocation_nbytes=storage.nbytes(),
        )
        return _checked_memory(
            registration_id,
            name,
            anchor,
            address,
            nbytes,
            device,
            memory_type,
        )

    untyped_storage_type = getattr(torch, "UntypedStorage", ())
    typed_storage_type = getattr(getattr(torch, "storage", object), "TypedStorage", ())
    if isinstance(value, (untyped_storage_type, typed_storage_type)):
        storage = getattr(value, "_untyped_storage", value)
        nbytes = storage.nbytes()
        address = storage.data_ptr()
        device = str(storage.device)
        if storage.device.type == "cuda":
            memory_type = "cuda"
        else:
            try:
                memory_type = "host-pinned" if _storage_is_pinned(storage) else "host"
            except RuntimeError:
                memory_type = "host"
        anchor = _LifetimeAnchor(
            owner=value,
            storage=storage,
            allocation_address=address,
            allocation_nbytes=nbytes,
        )
        return _checked_memory(
            registration_id,
            name,
            anchor,
            address,
            nbytes,
            device,
            memory_type,
        )

    try:
        view = memoryview(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise InvalidRegionError(
            "registration expects a contiguous Tensor, Storage, writable "
            "buffer, or RawSpan"
        ) from error
    if not view.c_contiguous:
        raise InvalidRegionError(
            "buffer registration requires C-contiguous memory; "
            "no implicit copy is performed"
        )
    if view.readonly:
        raise InvalidRegionError("buffer registration requires writable memory")
    try:
        address = ctypes.addressof(ctypes.c_char.from_buffer(view))
    except (TypeError, ValueError) as error:
        raise InvalidRegionError("could not obtain the buffer address") from error
    anchor = _LifetimeAnchor(
        owner=value,
        buffer=view,
        allocation_address=address,
        allocation_nbytes=view.nbytes,
    )
    return _checked_memory(
        registration_id,
        name,
        anchor,
        address,
        view.nbytes,
        "cpu",
        "host",
    )


def _checked_memory(
    registration_id: str,
    name: str,
    owner: object,
    address: int,
    nbytes: int,
    device: str,
    memory_type: str,
) -> RegisteredMemory:
    if not isinstance(address, int) or address <= 0 or nbytes <= 0:
        raise InvalidRegionError("cannot register empty or unallocated memory")
    return RegisteredMemory(
        registration_id=registration_id,
        name=name,
        owner=owner,
        address=address,
        nbytes=nbytes,
        device=device,
        memory_type=memory_type,
    )


def _released_memory(memory: RegisteredMemory) -> RegisteredMemory:
    return RegisteredMemory(
        registration_id=memory.registration_id,
        name=memory.name,
        owner=None,
        address=memory.address,
        nbytes=memory.nbytes,
        device=memory.device,
        memory_type=memory.memory_type,
    )


def _memory_ranges_overlap(first: RegisteredMemory, second: RegisteredMemory) -> bool:
    if first.device != second.device:
        return False
    return (
        first.address < second.address + second.nbytes
        and second.address < first.address + first.nbytes
    )


def _validate_memory_stable(memory: RegisteredMemory) -> None:
    anchor = memory.owner
    if not isinstance(anchor, _LifetimeAnchor):
        return
    if anchor.storage is not None:
        storage = cast(Any, anchor.storage)
        try:
            address = storage.data_ptr()
            nbytes = storage.nbytes()
        except Exception as error:
            raise InvalidRegionError("could not validate registered Storage") from error
        if address != anchor.allocation_address or nbytes != anchor.allocation_nbytes:
            raise InvalidRegionError(
                "registered Storage was resized; deregister and register it again"
            )
    if anchor.buffer is not None:
        try:
            address = ctypes.addressof(ctypes.c_char.from_buffer(anchor.buffer))
        except (TypeError, ValueError) as error:
            raise InvalidRegionError(
                "registered buffer no longer exposes its original address"
            ) from error
        if (
            address != anchor.allocation_address
            or anchor.buffer.nbytes != anchor.allocation_nbytes
        ):
            raise InvalidRegionError("registered buffer address or size changed")


def _tensor_memory_type(tensor: torch.Tensor) -> str:
    if tensor.device.type == "cuda":
        return "cuda"
    if tensor.device.type == "cpu":
        try:
            return "host-pinned" if tensor.is_pinned() else "host"
        except RuntimeError:
            return "host"
    return tensor.device.type


def _storage_is_pinned(storage: object) -> bool:
    view = torch.empty(
        0,
        dtype=torch.uint8,
        device=storage.device,  # type: ignore[attr-defined]
    )
    return view.set_(storage).is_pinned()  # type: ignore[arg-type]


def _validate_region_bounds(
    allocation_nbytes: int,
    offset: int,
    nbytes: int | None,
    stride: int | None,
    count: int | None,
) -> tuple[int, int, int | None, int | None]:
    _validate_nonnegative_int(offset, "offset")
    stride, count = _validate_stride(stride, count)
    if nbytes is None:
        if stride is not None:
            raise InvalidRegionError(
                "nbytes is required when stride and count are specified"
            )
        nbytes = allocation_nbytes - offset
    _validate_positive_int(nbytes, "nbytes")
    if stride is not None and stride < nbytes:
        raise InvalidRegionError("stride must be at least nbytes")
    extent = _region_extent(nbytes, stride, count)
    if offset + extent > allocation_nbytes:
        raise InvalidRegionError(
            f"region [{offset}, {offset + extent}) exceeds allocation size "
            f"{allocation_nbytes}"
        )
    return offset, nbytes, stride, count


def _validate_stride(
    stride: int | None,
    count: int | None,
) -> tuple[int | None, int | None]:
    if (stride is None) != (count is None):
        raise InvalidRegionError("stride and count must be specified together")
    if stride is not None:
        _validate_positive_int(stride, "stride")
        _validate_positive_int(count, "count")
    return stride, count


def _region_extent(
    nbytes: int,
    stride: int | None,
    count: int | None,
) -> int:
    if stride is None:
        return nbytes
    if count is None:
        raise InvalidRegionError("count is required with stride")
    return (count - 1) * stride + nbytes


def _validate_plan_indices(
    plan: TransferPlan,
    local_indices: Sequence[int] | None,
    remote_indices: Sequence[int] | None,
    op: TransferOp,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    if not plan.indexed:
        if local_indices is not None or remote_indices is not None:
            raise InvalidRegionError(
                "indices require a plan prepared with indexed=True"
            )
        local = remote = None
    else:
        if (local_indices is None) != (remote_indices is None):
            raise InvalidRegionError(
                "local_indices and remote_indices must be specified together"
            )
        if local_indices is None:
            local = remote = None
        else:
            remote_values = cast(Sequence[int], remote_indices)
            local = _validate_indices(
                local_indices, plan.local_block_count, "local_indices"
            )
            remote = _validate_indices(
                remote_values,
                plan.remote_block_count,
                "remote_indices",
            )
    if local is None:
        if plan._local_total_nbytes != plan._remote_total_nbytes:
            raise InvalidRegionError(
                f"selected local and remote byte totals differ: "
                f"{plan._local_total_nbytes} != {plan._remote_total_nbytes}"
            )
        destination_nonoverlap = (
            plan._remote_full_nonoverlap
            if op is TransferOp.WRITE
            else plan._local_full_nonoverlap
        )
        if not destination_nonoverlap:
            raise InvalidRegionError("selected destination regions must not overlap")
        return local, remote

    selected_local = (
        None
        if local is None
        else _select_region_blocks(plan._local, local, plan._local_starts)
    )
    selected_remote = (
        None
        if remote is None
        else _select_region_blocks(plan._remote, remote, plan._remote_starts)
    )
    local_size = _selected_region_nbytes(plan._local, selected_local)
    remote_size = _selected_region_nbytes(plan._remote, selected_remote)
    if local_size != remote_size:
        raise InvalidRegionError(
            f"selected local and remote byte totals differ: "
            f"{local_size} != {remote_size}"
        )
    if op is TransferOp.WRITE:
        _validate_nonoverlapping_destination(plan._remote, selected_remote)
    else:
        _validate_nonoverlapping_destination(plan._local, selected_local)
    return local, remote


def _snapshot_backend_index_selection(
    plan: TransferPlan,
    op: TransferOp,
    local_indices: Sequence[int],
    remote_indices: Sequence[int],
    *,
    collect_local_devices: bool = True,
) -> tuple[BackendIndexSelection, tuple[str, ...]]:
    """Create the sole private immutable selection from one input snapshot."""

    (
        packed,
        local_count,
        remote_count,
        selected_nbytes,
        uniform_pair_layout,
        destination_catalog_nonoverlapping,
        local_devices,
    ) = _pack_and_validate_index_selection(
        plan,
        op,
        local_indices,
        remote_indices,
        collect_local_devices=collect_local_devices,
    )
    # ``packed`` is owned immutable bytes, so the cast is already read-only.
    # Avoid allocating a second memoryview wrapper on every dynamic selection.
    packed_view = memoryview(packed).cast("i")
    local_view = packed_view[:local_count]
    remote_view = packed_view[local_count : local_count + remote_count]
    return (
        _make_backend_index_selection(
            op,
            plan._handle,
            local_view,
            remote_view,
            selected_nbytes=selected_nbytes,
            uniform_pair_layout=uniform_pair_layout,
            destination_catalog_nonoverlapping=destination_catalog_nonoverlapping,
            destination_indices_unique=True,
        ),
        local_devices,
    )


def _pack_and_validate_index_selection(
    plan: TransferPlan,
    op: TransferOp,
    local_indices: Sequence[int],
    remote_indices: Sequence[int],
    *,
    collect_local_devices: bool = True,
) -> tuple[bytes, int, int, int, bool, bool, tuple[str, ...]]:
    """Snapshot two vectors once and derive reusable Core proof facts."""

    local_input_view = _native_int32_buffer_view(local_indices)
    remote_input_view = _native_int32_buffer_view(remote_indices)
    native_destination_duplicate: bool | None = None
    if local_input_view is not None and remote_input_view is not None:
        local_count = local_input_view.nbytes // 4
        if local_count == 0:
            raise InvalidRegionError("local_indices must be non-empty")
        remote_count = remote_input_view.nbytes // 4
        if remote_count == 0:
            raise InvalidRegionError("remote_indices must be non-empty")

        if _compiled_index_analyzer is not None:
            # The helper owns the only copy: it creates one combined PyBytes
            # snapshot under the GIL, then derives every fact from that
            # immutable snapshot with the GIL released.
            (
                packed,
                analyzed_local_count,
                analyzed_remote_count,
                local_minimum,
                local_maximum,
                remote_minimum,
                remote_maximum,
                native_destination_duplicate,
            ) = _compiled_index_analyzer(
                local_input_view,
                remote_input_view,
                op is TransferOp.WRITE,
            )
            if (
                analyzed_local_count != local_count
                or analyzed_remote_count != remote_count
            ):
                raise RuntimeError(
                    "compiled int32 index analyzer returned inconsistent counts"
                )
            local_bounds = (local_minimum, local_maximum)
            remote_bounds = (remote_minimum, remote_maximum)
        else:
            # Compatibility path for source-tree Python loaded with an older
            # torch._C binary. It preserves the exact pre-helper behavior and
            # is deliberately selected only by a missing import, never by a
            # helper exception.
            packed = b"".join((local_input_view.cast("B"), remote_input_view.cast("B")))
            packed_indices = memoryview(packed).cast("i")
            local_view = packed_indices[:local_count]
            remote_view = packed_indices[local_count : local_count + remote_count]
            destination_view = remote_view if op is TransferOp.WRITE else local_view
            if max(local_count, remote_count) <= _INDEX_SELECTION_TORCH_THRESHOLD:
                local_bounds = (min(local_view), max(local_view))
                remote_bounds = (min(remote_view), max(remote_view))
                native_destination_duplicate = len(set(destination_view)) != len(
                    destination_view
                )
            else:
                validation_scratch = bytearray(packed)
                validation = torch.frombuffer(validation_scratch, dtype=torch.int32)
                local_tensor = validation[:local_count]
                remote_tensor = validation[local_count : local_count + remote_count]
                local_minimum, local_maximum = torch.aminmax(local_tensor)
                remote_minimum, remote_maximum = torch.aminmax(remote_tensor)
                destination_tensor = (
                    remote_tensor if op is TransferOp.WRITE else local_tensor
                )
                ordered_destination = torch.sort(destination_tensor).values
                local_bounds = (int(local_minimum), int(local_maximum))
                remote_bounds = (int(remote_minimum), int(remote_maximum))
                native_destination_duplicate = bool(
                    torch.any(ordered_destination[1:] == ordered_destination[:-1])
                )

        packed_indices = memoryview(packed).cast("i")
        local_view = packed_indices[:local_count]
        remote_view = packed_indices[local_count : local_count + remote_count]

        # Preserve the public error order: local bounds, remote bounds, byte
        # equality, then destination duplicate/physical-overlap diagnostics.
        _raise_packed_index_bounds(
            *local_bounds, plan.local_block_count, "local_indices"
        )
        _raise_packed_index_bounds(
            *remote_bounds, plan.remote_block_count, "remote_indices"
        )
    else:
        # Mixed buffers and general Sequences retain the established
        # operator.index normalization and validation behavior.
        packed = bytearray()
        local_count, local_needs_bounds = _append_index_snapshot(
            packed,
            local_indices,
            plan.local_block_count,
            "local_indices",
        )
        if local_needs_bounds:
            # Preserve the public left-to-right validation contract for mixed
            # inputs. A native local buffer is copied by _append_index_snapshot,
            # but its bounds must be diagnosed before the remote Sequence is
            # iterated (which may itself be invalid or have side effects).
            local_snapshot_view = memoryview(packed).cast("i")
            try:
                _validate_packed_index_bounds(
                    local_snapshot_view,
                    plan.local_block_count,
                    "local_indices",
                )
            finally:
                # bytearray cannot be resized while an exported view exists.
                local_snapshot_view.release()
        remote_count, remote_needs_bounds = _append_index_snapshot(
            packed,
            remote_indices,
            plan.remote_block_count,
            "remote_indices",
        )
        packed_indices = memoryview(packed).cast("i")
        local_view = packed_indices[:local_count]
        remote_view = packed_indices[local_count : local_count + remote_count]
        if remote_needs_bounds:
            _validate_packed_index_bounds(
                remote_view, plan.remote_block_count, "remote_indices"
            )

    selected_local: tuple[tuple[Region, int], ...] | None = None
    selected_remote: tuple[tuple[Region, int], ...] | None = None
    if plan._local_uniform_nbytes is None:
        selected_local = _select_region_blocks(
            plan._local, local_view, plan._local_starts
        )
        local_size = _selected_region_nbytes(plan._local, selected_local)
    else:
        local_size = local_count * plan._local_uniform_nbytes
    if plan._remote_uniform_nbytes is None:
        selected_remote = _select_region_blocks(
            plan._remote, remote_view, plan._remote_starts
        )
        remote_size = _selected_region_nbytes(plan._remote, selected_remote)
    else:
        remote_size = remote_count * plan._remote_uniform_nbytes
    if local_size != remote_size:
        raise InvalidRegionError(
            f"selected local and remote byte totals differ: "
            f"{local_size} != {remote_size}"
        )

    destination_full_nonoverlap = (
        plan._remote_full_nonoverlap
        if op is TransferOp.WRITE
        else plan._local_full_nonoverlap
    )
    if native_destination_duplicate:
        raise InvalidRegionError("selected destination regions must not overlap")
    if destination_full_nonoverlap:
        destination_view = remote_view if op is TransferOp.WRITE else local_view
        if native_destination_duplicate is None:
            _validate_unique_packed_indices(destination_view)
    elif op is TransferOp.WRITE:
        if selected_remote is None:
            selected_remote = _select_region_blocks(
                plan._remote, remote_view, plan._remote_starts
            )
        _validate_nonoverlapping_destination(plan._remote, selected_remote)
    else:
        if selected_local is None:
            selected_local = _select_region_blocks(
                plan._local, local_view, plan._local_starts
            )
        _validate_nonoverlapping_destination(plan._local, selected_local)

    if not collect_local_devices:
        local_devices = ()
    elif plan._local_uniform_device is None:
        if selected_local is None:
            selected_local = _select_region_blocks(
                plan._local, local_view, plan._local_starts
            )
        local_devices = tuple(
            dict.fromkeys(region.device for region, _ in selected_local)
        )
    else:
        local_devices = (plan._local_uniform_device,)

    uniform_pair_layout = (
        local_count == remote_count
        and plan._local_uniform_nbytes is not None
        and plan._local_uniform_nbytes == plan._remote_uniform_nbytes
    )
    return (
        bytes(packed),
        local_count,
        remote_count,
        local_size,
        uniform_pair_layout,
        destination_full_nonoverlap,
        local_devices,
    )


def _append_index_snapshot(
    packed: bytearray,
    values: Sequence[int],
    upper_bound: int,
    name: str,
) -> tuple[int, bool]:
    """Append one vector, combining general-sequence coercion and bounds."""

    if values is None:
        raise InvalidRegionError(f"{name} must be non-empty")
    view = _native_int32_buffer_view(values)
    if view is not None:
        count = view.nbytes // 4
        if count == 0:
            raise InvalidRegionError(f"{name} must be non-empty")
        packed.extend(view.cast("B"))
        return count, True

    count = 0
    try:
        iterator = iter(values)
    except TypeError as error:
        raise InvalidRegionError(f"{name} must contain integers") from error
    for raw_value in iterator:
        if isinstance(raw_value, bool):
            raise InvalidRegionError(f"{name} must contain integers")
        try:
            value = operator.index(raw_value)
        except TypeError as error:
            raise InvalidRegionError(f"{name} must contain integers") from error
        if value < 0 or value >= upper_bound:
            raise InvalidRegionError(
                f"{name} index {value} is outside [0, {upper_bound})"
            )
        if value > _INT32_MAX:
            raise InvalidRegionError(
                f"{name} index {value} exceeds the packed int32 range"
            )
        packed.extend(value.to_bytes(4, byteorder=sys.byteorder, signed=True))
        count += 1
    if count == 0:
        raise InvalidRegionError(f"{name} must be non-empty")
    return count, False


def _native_int32_buffer_view(values: object) -> memoryview | None:
    """Return the exact packed-input view, without copying or coercion."""

    try:
        view = memoryview(values)
    except TypeError:
        return None
    if (
        view.ndim == 1
        and view.c_contiguous
        and view.itemsize == 4
        and view.format in _NATIVE_INT32_FORMATS
    ):
        return view
    return None


def _raise_packed_index_bounds(
    low: int,
    high: int,
    upper_bound: int,
    name: str,
) -> None:
    if low < 0:
        raise InvalidRegionError(f"{name} index {low} is outside [0, {upper_bound})")
    if high >= upper_bound:
        raise InvalidRegionError(f"{name} index {high} is outside [0, {upper_bound})")


def _validate_packed_index_bounds(
    values: memoryview,
    upper_bound: int,
    name: str,
) -> None:
    tensor = torch.frombuffer(values, dtype=torch.int32)
    minimum, maximum = torch.aminmax(tensor)
    low = int(minimum)
    high = int(maximum)
    _raise_packed_index_bounds(low, high, upper_bound, name)


def _validate_unique_packed_indices(values: memoryview) -> None:
    if len(values) < 2:
        return
    tensor = torch.frombuffer(values, dtype=torch.int32)
    ordered = torch.sort(tensor).values
    if bool(torch.any(ordered[1:] == ordered[:-1])):
        raise InvalidRegionError("selected destination regions must not overlap")


def _validate_cuda_ordering_arguments(
    capabilities: Capabilities,
    backend: str,
    mode: CudaOrderingMode,
    stream: object | None,
    ready_event: object | None,
) -> None:
    if stream is not None and ready_event is not None:
        raise ValueError("stream and ready_event are mutually exclusive")
    ordering_token_count = int(stream is not None) + int(ready_event is not None)
    if mode is CudaOrderingMode.CALLER_READY:
        if ordering_token_count:
            raise ValueError(
                "caller_ready CUDA ordering does not accept a stream or ready_event"
            )
    elif ordering_token_count != 1:
        raise ValueError(
            "host_wait CUDA ordering requires exactly one stream or ready_event"
        )
    if mode not in capabilities.cuda_ordering_modes:
        raise UnsupportedError(
            f"backend {backend!r} does not support CUDA ordering "
            f"mode {mode.value!r}"
        )


def _validate_attached_notification(
    capabilities: Capabilities,
    backend: str,
    notification: bytes | bytearray | memoryview | None,
) -> bytes | None:
    if notification is None:
        return None
    if not capabilities.attached_notifications:
        raise UnsupportedError(f"backend {backend!r} does not support notifications")
    result = _as_bytes(
        notification,
        "notification",
        max_nbytes=_MAX_NOTIFICATION_BYTES,
    )
    if not result:
        raise ValueError("an attached notification payload must be non-empty")
    return result


def _validate_request_slot_notification(
    notification: bytes | bytearray | memoryview | None,
) -> bytes | None:
    """Validate an explicit per-generation slot notification or clear."""

    if notification is None:
        return None
    result = _as_bytes(
        notification,
        "notification",
        max_nbytes=_MAX_NOTIFICATION_BYTES,
    )
    if not result:
        raise ValueError("an attached notification payload must be non-empty")
    return result


def _validate_cuda_ordering_selection(
    plan: TransferPlan,
    local_indices: Sequence[int] | None,
    mode: CudaOrderingMode,
    stream: object | None,
    ready_event: object | None,
) -> torch.device | None:
    """Validate a CUDA submission shape without establishing readiness."""

    if mode is CudaOrderingMode.CALLER_READY:
        return None
    if mode is not CudaOrderingMode.HOST_WAIT:
        raise UnsupportedError(f"unsupported CUDA ordering mode {mode!r}")

    selected = (
        tuple((region, 0) for region in plan._local)
        if local_indices is None
        else _select_region_blocks(plan._local, local_indices, plan._local_starts)
    )
    return _validate_cuda_ordering_devices(
        tuple(dict.fromkeys(region.device for region, _ in selected)),
        mode,
        stream,
        ready_event,
    )


def _validate_cuda_ordering_devices(
    selected_devices: Sequence[str],
    mode: CudaOrderingMode,
    stream: object | None,
    ready_event: object | None,
) -> torch.device | None:
    """Validate ordering from device facts cached by an IndexSelection."""

    if mode is CudaOrderingMode.CALLER_READY:
        return None
    if mode is not CudaOrderingMode.HOST_WAIT:
        raise UnsupportedError(f"unsupported CUDA ordering mode {mode!r}")

    devices: set[torch.device] = set()
    for device_name in selected_devices:
        try:
            device = torch.device(device_name)
        except (RuntimeError, TypeError, ValueError) as error:
            raise InvalidRegionError(
                f"local region has invalid device {device_name!r}"
            ) from error
        if device.type != "cuda" or device.index is None:
            raise InvalidRegionError(
                "host_wait requires selected local regions on one explicitly "
                "indexed CUDA device"
            )
        devices.add(device)
    if len(devices) != 1:
        raise UnsupportedError(
            "host_wait requires selected local regions on exactly one CUDA device"
        )
    expected_device = next(iter(devices))

    if stream is not None:
        stream_types = tuple(
            stream_type
            for stream_type in (
                getattr(torch, "Stream", None),
                getattr(torch.cuda, "Stream", None),
            )
            if isinstance(stream_type, type)
        )
        if not stream_types or not isinstance(stream, stream_types):
            raise TypeError("stream must be a torch CUDA Stream")
        _validate_cuda_token_device(stream, expected_device, "stream")
    else:
        event_type = getattr(torch.cuda, "Event", None)
        if not isinstance(event_type, type) or not isinstance(ready_event, event_type):
            raise TypeError("ready_event must be a torch.cuda.Event")
        _validate_cuda_token_device(ready_event, expected_device, "ready_event")

    return expected_device


def _establish_cuda_ordering_prevalidated(
    mode: CudaOrderingMode,
    stream: object | None,
    ready_event: object | None,
    expected_device: torch.device | None,
) -> None:
    """Resolve Core-owned CUDA readiness before entering provider locks."""

    if mode is CudaOrderingMode.CALLER_READY:
        return
    if mode is not CudaOrderingMode.HOST_WAIT or expected_device is None:
        raise UnsupportedError(f"unsupported CUDA ordering mode {mode!r}")

    if stream is not None:
        try:
            if _cuda_token_stream_is_capturing(stream):
                raise UnsupportedError("host_wait is not CUDA Graph capture safe")
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                "failed to query CUDA Graph capture state"
            ) from error

    if stream is not None:
        try:
            event = stream.record_event()  # type: ignore[attr-defined]
        except Exception as error:
            raise BackendFailureError(
                "failed to record the CUDA readiness event"
            ) from error
    else:
        event = ready_event

    if stream is not None:
        # A stream implementation supplies the event on each submission; keep
        # this runtime check even though the stream itself was validated once.
        event_type = getattr(torch.cuda, "Event", None)
        if not isinstance(event_type, type) or not isinstance(event, event_type):
            raise TypeError("ready_event must be a torch.cuda.Event")
        _validate_cuda_token_device(event, expected_device, "ready_event")
    if event is None:
        raise BackendFailureError("CUDA readiness event is missing")
    try:
        event.synchronize()  # type: ignore[union-attr]
    except TransferError:
        raise
    except Exception as error:
        raise BackendFailureError("failed to wait for CUDA readiness") from error


def _cuda_token_stream_is_capturing(stream: object) -> bool:
    """Query the supplied stream before Core records a new readiness event."""

    direct_query = getattr(stream, "is_capturing", None)
    if callable(direct_query):
        return bool(direct_query())
    with torch.cuda.stream(stream):  # type: ignore[arg-type]
        return torch.cuda.is_current_stream_capturing()


def _make_reusable_cuda_stream_readiness(
    stream: object,
    expected_device: torch.device | None,
) -> Callable[[], None]:
    """Create one per-slot event and return its generation readiness gate."""

    if expected_device is None:
        raise UnsupportedError("host_wait stream requires one CUDA device")
    try:
        event = torch.cuda.Event()
        record = cast(Callable[[object], None], event.record)
        synchronize = cast(Callable[[], None], event.synchronize)
    except Exception as error:
        raise BackendFailureError(
            "failed to create the reusable CUDA readiness event"
        ) from error

    direct_capture_query = getattr(stream, "is_capturing", None)
    if callable(direct_capture_query):
        capture_query = direct_capture_query
    else:
        stream_context = torch.cuda.stream
        current_stream_is_capturing = torch.cuda.is_current_stream_capturing

        def capture_query() -> bool:
            with stream_context(stream):  # type: ignore[arg-type]
                return bool(current_stream_is_capturing())

    device_validated = False

    def synchronize_generation() -> None:
        nonlocal device_validated
        try:
            if capture_query():
                raise UnsupportedError("host_wait is not CUDA Graph capture safe")
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError(
                "failed to query CUDA Graph capture state"
            ) from error
        try:
            record(stream)
        except Exception as error:
            raise BackendFailureError(
                "failed to record the CUDA readiness event"
            ) from error
        if not device_validated:
            _validate_cuda_token_device(event, expected_device, "ready_event")
            device_validated = True
        try:
            synchronize()
        except TransferError:
            raise
        except Exception as error:
            raise BackendFailureError("failed to wait for CUDA readiness") from error

    return synchronize_generation


def _validate_cuda_token_device(
    token: object,
    expected: torch.device,
    name: str,
) -> None:
    device_value = getattr(token, "device", None)
    if device_value is None:
        qualifier = "recorded " if name == "ready_event" else ""
        raise InvalidRegionError(f"{name} must be a {qualifier}CUDA token")
    try:
        device = torch.device(device_value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise InvalidRegionError(f"{name} has an invalid device") from error
    if device != expected:
        raise InvalidRegionError(
            f"{name} device {device} does not match local region device {expected}"
        )


def _logical_block_count(regions: Sequence[Region]) -> int:
    return sum(region.count or 1 for region in regions)


def _uniform_region_nbytes(regions: Sequence[Region]) -> int | None:
    first = regions[0].nbytes
    return first if all(region.nbytes == first for region in regions[1:]) else None


def _uniform_region_device(regions: Sequence[Region]) -> str | None:
    first = regions[0].device
    return first if all(region.device == first for region in regions[1:]) else None


def _catalog_starts(regions: Sequence[Region]) -> tuple[int, ...]:
    starts: list[int] = []
    next_index = 0
    for region in regions:
        starts.append(next_index)
        next_index += region.count or 1
    return tuple(starts)


def _select_region_blocks(
    regions: Sequence[Region],
    indices: Sequence[int],
    starts: Sequence[int],
) -> tuple[tuple[Region, int], ...]:
    selected: list[tuple[Region, int]] = []
    for index in indices:
        region_index = bisect_right(starts, index) - 1
        if region_index < 0 or region_index >= len(regions):
            raise InvalidRegionError(
                "logical block index is outside the region catalog"
            )
        selected.append((regions[region_index], index - starts[region_index]))
    return tuple(selected)


def _selected_region_nbytes(
    regions: Sequence[Region],
    selected: Sequence[tuple[Region, int]] | None,
) -> int:
    if selected is None:
        return sum(region.total_nbytes for region in regions)
    return sum(region.nbytes for region, _ in selected)


def _validate_nonoverlapping_destination(
    regions: Sequence[Region],
    selected: Sequence[tuple[Region, int]] | None,
) -> None:
    intervals: dict[object, list[tuple[int, int]]] = {}
    blocks = (
        (
            (region, block_index)
            for region in regions
            for block_index in range(region.count or 1)
        )
        if selected is None
        else iter(selected)
    )
    for region, block_index in blocks:
        descriptor = region._backend_descriptor
        if isinstance(region._origin, Registration):
            key: object = ("local", region.device)
            base = region._origin._memory.address
        elif isinstance(descriptor, RegistrationMetadata):
            key = (
                "remote-registration",
                id(region._origin),
                descriptor.registration_id,
            )
            base = 0
        elif isinstance(descriptor, _RemoteRawDescriptor):
            key = (
                "remote-address",
                id(region._origin),
                descriptor.device,
                descriptor.memory_type,
            )
            base = descriptor.address
        else:
            raise InvalidRegionError("region has an invalid memory descriptor")
        stride = region.stride or region.nbytes
        start = base + region.offset + block_index * stride
        intervals.setdefault(key, []).append((start, start + region.nbytes))
    for values in intervals.values():
        values.sort()
        for previous, current in itertools.pairwise(values):
            if current[0] < previous[1]:
                raise InvalidRegionError(
                    "selected destination regions must not overlap"
                )


def _regions_are_nonoverlapping(regions: Sequence[Region]) -> bool:
    # Each compact Region is internally disjoint because stride >= nbytes.
    # Compare only its bounding envelope against other regions in the same
    # physical address space. An ambiguous interleaving is conservatively
    # rejected instead of expanding `count` logical blocks.
    envelopes: dict[object, list[tuple[int, int]]] = {}
    for region in regions:
        descriptor = region._backend_descriptor
        if isinstance(region._origin, Registration):
            key: object = ("local", region.device)
            base = region._origin._memory.address
        elif isinstance(descriptor, RegistrationMetadata):
            key = (
                "remote-registration",
                id(region._origin),
                descriptor.registration_id,
            )
            base = 0
        elif isinstance(descriptor, _RemoteRawDescriptor):
            key = (
                "remote-address",
                id(region._origin),
                descriptor.device,
                descriptor.memory_type,
            )
            base = descriptor.address
        else:
            return False
        stride = region.stride or region.nbytes
        start = base + region.offset
        end = start + ((region.count or 1) - 1) * stride + region.nbytes
        envelopes.setdefault(key, []).append((start, end))
    for values in envelopes.values():
        values.sort()
        for previous, current in itertools.pairwise(values):
            if current[0] < previous[1]:
                return False
    return True


def _validate_indices(
    values: Sequence[int],
    upper_bound: int,
    name: str,
) -> tuple[int, ...]:
    result: list[int] = []
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise InvalidRegionError(f"{name} must contain integers")
        try:
            value = operator.index(raw_value)
        except TypeError as error:
            raise InvalidRegionError(f"{name} must contain integers") from error
        if value < 0 or value >= upper_bound:
            raise InvalidRegionError(
                f"{name} index {value} is outside [0, {upper_bound})"
            )
        result.append(value)
    if not result:
        raise InvalidRegionError(f"{name} must be non-empty")
    return tuple(result)


def _adopt_resource_handle(resource: object, handle: object, kind: str) -> None:
    if handle is None or handle is _UNADOPTED_HANDLE:
        raise BackendFailureError(f"backend adopted an invalid {kind} handle")
    mutable_resource = cast(Any, resource)
    current = mutable_resource._handle
    if current is _UNADOPTED_HANDLE:
        mutable_resource._handle = handle
        return
    if current is handle:
        return
    raise BackendFailureError(f"backend attempted to adopt {kind} more than once")


def _creation_cleanup_failure(
    kind: str,
    original_error: BaseException,
    cleanup_error: BaseException,
) -> BackendFailureError:
    del cleanup_error
    failure = BackendFailureError(
        f"{kind} creation failed and cleanup is pending; recover it from the "
        f"Endpoint open-resource snapshots and retry close()"
    )
    add_note = getattr(failure, "add_note", None)
    if callable(add_note):
        add_note(f"original failure type: {type(original_error).__name__}")
    return failure


def _endpoint_creation_cleanup_failure(
    endpoint: Endpoint,
    original_error: BaseException,
    cleanup_error: BaseException,
) -> BackendFailureError:
    failure = BackendFailureError(
        "endpoint backend initialization failed and cleanup is pending; recover "
        "exception.recovery_endpoint and retry close()"
    )
    failure.recovery_endpoint = endpoint  # type: ignore[attr-defined]
    add_note = getattr(failure, "add_note", None)
    if callable(add_note):
        add_note(f"initialization failure type: {type(original_error).__name__}")
        add_note(f"cleanup failure type: {type(cleanup_error).__name__}")
    return failure


def _as_bytes(
    value: object,
    name: str,
    *,
    max_nbytes: int | None = None,
) -> bytes:
    if type(value) is bytes:
        if max_nbytes is not None and len(value) > max_nbytes:
            raise ValueError(f"{name} exceeds the {max_nbytes}-byte size limit")
        return value
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    if max_nbytes is not None:
        try:
            source_nbytes = (
                value.nbytes if isinstance(value, memoryview) else len(value)
            )
        except ValueError as error:
            raise ValueError(f"{name} buffer is unavailable") from error
        if source_nbytes > max_nbytes:
            raise ValueError(f"{name} exceeds the {max_nbytes}-byte size limit")
    result = bytes(value)
    if max_nbytes is not None and len(result) > max_nbytes:
        raise ValueError(f"{name} exceeds the {max_nbytes}-byte size limit")
    return result


def _validate_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidRegionError(f"{name} must be a positive integer")


def _transfer_slot_deadline_ns(timeout: float | None) -> int | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a non-negative number or None")
    if timeout < 0 or (isinstance(timeout, float) and not math.isfinite(timeout)):
        raise ValueError("timeout must be finite and non-negative")
    if isinstance(timeout, int):
        timeout_ns = timeout * 1_000_000_000
    else:
        numerator, denominator = timeout.as_integer_ratio()
        timeout_ns = (numerator * 1_000_000_000 + denominator - 1) // denominator
    return time.monotonic_ns() + timeout_ns


def _validate_bounded_poll_arguments(
    *,
    max_polls: object,
    timeout_ns: object,
    timeout_check_interval: object | None,
    allow_zero: bool,
) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(max_polls, bool) or not isinstance(max_polls, int):
        qualifier = "non-negative" if allow_zero else "positive"
        raise TypeError(f"max_polls must be a {qualifier} integer")
    if max_polls < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"max_polls must be a {qualifier} integer")
    if max_polls > _MAX_FUSED_REQUEST_SLOT_TIMEOUT_NS:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"max_polls must be a {qualifier} signed 64-bit integer")
    if timeout_ns is not None and (
        isinstance(timeout_ns, bool) or not isinstance(timeout_ns, int)
    ):
        raise TypeError(
            "timeout_ns must be a non-negative signed 64-bit integer or None"
        )
    if timeout_ns is not None and (
        timeout_ns < 0 or timeout_ns > _MAX_FUSED_REQUEST_SLOT_TIMEOUT_NS
    ):
        raise ValueError(
            "timeout_ns must be a non-negative signed 64-bit integer or None"
        )
    if timeout_check_interval is not None and (
        isinstance(timeout_check_interval, bool)
        or not isinstance(timeout_check_interval, int)
    ):
        raise TypeError("timeout_check_interval must be a positive integer")
    if timeout_check_interval is not None and timeout_check_interval <= 0:
        raise ValueError("timeout_check_interval must be a positive integer")
    if (
        timeout_check_interval is not None
        and timeout_check_interval > _MAX_FUSED_REQUEST_SLOT_TIMEOUT_NS
    ):
        raise ValueError(
            "timeout_check_interval must be a positive signed 64-bit integer"
        )


def _validate_nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidRegionError(f"{name} must be a non-negative integer")


def _validate_nonempty_str(
    value: object,
    name: str,
    error_type: type[Exception] = ValueError,
) -> None:
    if not isinstance(value, str) or not value:
        raise error_type(f"{name} must be a non-empty string")


def _coerce_enum(value: object, enum_type: type[Any], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"{name} must be one of {choices}") from error
