from __future__ import annotations

import ctypes
import itertools
import json
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any

from ._backend import (
    Backend,
    BackendWork,
    ImportedRawMemory,
    ImportedRegistration,
    RegisteredMemory,
    register_backend,
)
from ._errors import (
    BusyError,
    ClosedError,
    InvalidMetadataError,
    InvalidRegionError,
    UnsupportedError,
)
from ._types import (
    BackendNotification,
    BackendRegion,
    Capabilities,
    EndpointMetadata,
    NotificationCompletion,
    ProgressMode,
    ThreadMode,
    TransferOp,
    WorkState,
)


@dataclass(eq=False, slots=True)
class _ReferencePeer:
    backend: _ReferenceBackend
    endpoint_id: str
    incarnation: str
    allowed_registration_ids: frozenset[str]
    released: bool = False


@dataclass(eq=False, slots=True)
class _ReferencePlan:
    local: tuple[BackendRegion, ...]
    remote: tuple[BackendRegion, ...]
    indexed: bool
    released: bool = False


class _ReferenceWork(BackendWork):
    def __init__(
        self,
        state: WorkState,
        error: BaseException | None = None,
    ) -> None:
        self._state = state
        self._error = error
        self._closed = False

    @property
    def state(self) -> WorkState:
        if self._closed:
            raise ClosedError("reference work is closed")
        return self._state

    @property
    def error(self) -> BaseException | None:
        if self._closed:
            raise ClosedError("reference work is closed")
        return self._error

    def cancel(self) -> bool:
        if self._closed:
            raise ClosedError("reference work is closed")
        if self._state in {
            WorkState.COMPLETED,
            WorkState.FAILED,
            WorkState.CANCELLED,
        }:
            return self._state is WorkState.CANCELLED
        self._state = WorkState.CANCELLED
        return True

    def close(self) -> None:
        self._closed = True


class _ReferenceBackend(Backend):
    """Synchronous, in-process CPU backend used for contract testing."""

    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        raw_spans=False,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
        synchronous_notification_send=True,
        standalone_notification_completion=NotificationCompletion.REMOTE_DELIVERED,
    )

    def __init__(
        self,
        endpoint_id: str,
        name: str,
        incarnation: str,
        progress_mode: ProgressMode,
        thread_mode: ThreadMode,
        options: Mapping[str, Any],
    ) -> None:
        self._initialize_record(
            endpoint_id,
            name,
            incarnation,
            progress_mode,
            thread_mode,
            options,
        )
        try:
            self._activate()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(
                        "reference backend activation cleanup also raised: "
                        f"{cleanup_error}"
                    )
            raise

    @classmethod
    def _new_unactivated(
        cls,
        endpoint_id: str,
        name: str,
        incarnation: str,
        progress_mode: ProgressMode,
        thread_mode: ThreadMode,
        options: Mapping[str, Any],
    ) -> _ReferenceBackend:
        backend = cls.__new__(cls)
        backend._initialize_record(
            endpoint_id,
            name,
            incarnation,
            progress_mode,
            thread_mode,
            options,
        )
        return backend

    def _initialize_record(
        self,
        endpoint_id: str,
        name: str,
        incarnation: str,
        progress_mode: ProgressMode,
        thread_mode: ThreadMode,
        options: Mapping[str, Any],
    ) -> None:
        del progress_mode, thread_mode
        if options:
            unknown = ", ".join(sorted(options))
            raise ValueError(f"reference backend has no options: {unknown}")
        self._endpoint_id = endpoint_id
        self._name = name
        self._incarnation = incarnation
        self._registrations: dict[str, RegisteredMemory] = {}
        self._plans: set[_ReferencePlan] = set()
        self._peers: set[_ReferencePeer] = set()
        self._notifications: deque[BackendNotification] = deque()
        self._lock = threading.RLock()
        self._activation_started = False
        self._activated = False
        self._closed = False

    def _activate(self) -> None:
        if self._closed:
            raise ClosedError("cannot activate a closed reference backend")
        if self._activated:
            return
        if self._activation_started:
            raise BusyError("reference backend activation is already in progress")
        self._activation_started = True
        with _REFERENCE_ENDPOINTS_LOCK:
            if self._endpoint_id in _REFERENCE_ENDPOINTS:
                raise BusyError(f"duplicate reference endpoint {self._endpoint_id!r}")
            _REFERENCE_ENDPOINTS[self._endpoint_id] = self
        self._activated = True

    @property
    def capabilities(self) -> Capabilities:
        return self._CAPABILITIES

    @property
    def supported_thread_modes(self) -> frozenset[ThreadMode]:
        return frozenset(
            {
                ThreadMode.SINGLE,
                ThreadMode.CALLER_SERIALIZED,
                ThreadMode.SERIALIZED,
                ThreadMode.MULTIPLE,
            }
        )

    def endpoint_payload(self, registration_handles: Sequence[object]) -> bytes:
        self._check_open()
        registration_ids = [
            _require_registration_handle(handle) for handle in registration_handles
        ]
        with self._lock:
            if any(
                registration_id not in self._registrations
                for registration_id in registration_ids
            ):
                raise ClosedError("reference registration is not active")
        return json.dumps(
            {
                "endpoint_id": self._endpoint_id,
                "incarnation": self._incarnation,
                "registrations": sorted(registration_ids),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def register(
        self,
        memory: RegisteredMemory,
        *,
        adopt_handle: Callable[[object], None],
    ) -> None:
        self._check_open()
        if not memory.device.startswith("cpu"):
            raise UnsupportedError("reference backend supports CPU allocations only")
        with self._lock:
            if memory.registration_id in self._registrations:
                raise BusyError("duplicate reference registration")
            adopt_handle(memory.registration_id)
            self._registrations[memory.registration_id] = memory

    def deregister(self, registration_handle: object) -> None:
        self._check_open()
        registration_id = _require_registration_handle(registration_handle)
        with self._lock:
            self._registrations.pop(registration_id, None)

    def describe_registration(self, registration_handle: object) -> bytes:
        self._check_open()
        registration_id = _require_registration_handle(registration_handle)
        with self._lock:
            if registration_id not in self._registrations:
                raise ClosedError("reference registration is not active")
        return registration_id.encode("utf-8")

    def import_peer(
        self,
        metadata: EndpointMetadata,
        *,
        adopt_handle: Callable[[object], None],
    ) -> None:
        self._check_open()
        with _REFERENCE_ENDPOINTS_LOCK:
            try:
                remote = _REFERENCE_ENDPOINTS[metadata.endpoint_id]
            except KeyError as error:
                raise InvalidMetadataError(
                    "reference peer is not present in this process"
                ) from error
        if remote._incarnation != metadata.incarnation:
            raise InvalidMetadataError("reference peer incarnation is stale")
        try:
            payload = json.loads(metadata.backend_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidMetadataError("reference peer payload is malformed") from error
        if not isinstance(payload, dict) or set(payload) != {
            "endpoint_id",
            "incarnation",
            "registrations",
        }:
            raise InvalidMetadataError("reference peer payload has invalid fields")
        registration_ids = payload["registrations"]
        if (
            payload["endpoint_id"] != metadata.endpoint_id
            or payload["incarnation"] != metadata.incarnation
            or not isinstance(registration_ids, list)
            or not all(isinstance(value, str) for value in registration_ids)
            or len(registration_ids) != len(set(registration_ids))
        ):
            raise InvalidMetadataError("reference peer payload does not match")
        advertised_ids = {item.registration_id for item in metadata.registrations}
        if set(registration_ids) != advertised_ids:
            raise InvalidMetadataError(
                "reference payload registration grant does not match metadata"
            )
        peer = _ReferencePeer(
            backend=remote,
            endpoint_id=metadata.endpoint_id,
            incarnation=metadata.incarnation,
            allowed_registration_ids=frozenset(registration_ids),
        )
        adopt_handle(peer)
        self._peers.add(peer)

    def release_peer(self, peer_handle: object) -> None:
        self._check_open()
        peer = _require_peer(peer_handle)
        if peer.released:
            return
        peer.released = True
        self._peers.discard(peer)

    def prepare(
        self,
        local: Sequence[BackendRegion],
        remote: Sequence[BackendRegion],
        *,
        indexed: bool,
        adopt_handle: Callable[[object], None],
    ) -> None:
        self._check_open()
        for region in local:
            if region.remote:
                raise InvalidRegionError("local reference region is remote")
            self._resolve_local(region)
        for region in remote:
            if not region.remote:
                raise InvalidRegionError("remote reference region is local")
            self._resolve_remote(region)
        plan = _ReferencePlan(tuple(local), tuple(remote), indexed)
        adopt_handle(plan)
        self._plans.add(plan)

    def release_plan(self, plan_handle: object) -> None:
        self._check_open()
        plan = _require_plan(plan_handle)
        if plan.released:
            return
        plan.released = True
        self._plans.discard(plan)

    def submit(
        self,
        op: TransferOp,
        plan_handle: object,
        *,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        notification: bytes | None,
        reuse_token: object | None,
        adopt_work: Callable[[BackendWork], None],
    ) -> None:
        self._check_open()
        plan = _require_plan(plan_handle)
        if plan.released:
            raise ClosedError("reference plan is released")
        work = _ReferenceWork(WorkState.PENDING)
        try:
            adopt_work(work)
        except BaseException as error:
            # A callback interruption can occur immediately before or after
            # Core stores the reference. Never launch in either case; if Core
            # did adopt it, expose a terminal failure instead of re-raising.
            work._error = error
            work._state = WorkState.FAILED
            return
        try:
            local = _select(plan.local, local_indices, indexed=plan.indexed)
            remote = _select(plan.remote, remote_indices, indexed=plan.indexed)
            local_spans = [
                span for region in local for span in self._resolve_local(region)
            ]
            remote_memory = [self._resolve_remote(region) for region in remote]
            remote_spans = [span for spans, _ in remote_memory for span in spans]
            if op is TransferOp.WRITE:
                source, destination = local_spans, remote_spans
            elif op is TransferOp.READ:
                source, destination = remote_spans, local_spans
            else:
                raise UnsupportedError(f"unsupported transfer operation {op!r}")
            _copy_spans(
                source,
                destination,
                anchors=tuple(memory for _, memory in remote_memory),
            )
            if notification is not None:
                peer = self._plan_peer(plan)
                peer.backend._enqueue_notification(
                    BackendNotification(
                        payload=notification,
                        source_endpoint_id=self._endpoint_id,
                        source_incarnation=self._incarnation,
                    )
                )
            work._state = WorkState.COMPLETED
        except BaseException as error:
            work._error = error
            work._state = WorkState.FAILED

    def send_notification(
        self,
        peer_handle: object,
        payload: bytes,
        *,
        adopt_work: Callable[[BackendWork], None],
    ) -> None:
        self._check_open()
        work = _ReferenceWork(WorkState.PENDING)
        try:
            adopt_work(work)
        except BaseException as error:
            work._error = error
            work._state = WorkState.FAILED
            return
        try:
            peer = _require_peer(peer_handle)
            if peer.released:
                raise ClosedError("reference peer is released")
            peer.backend._enqueue_notification(
                BackendNotification(
                    payload=payload,
                    source_endpoint_id=self._endpoint_id,
                    source_incarnation=self._incarnation,
                )
            )
            work._state = WorkState.COMPLETED
        except BaseException as error:
            work._error = error
            work._state = WorkState.FAILED

    def send_notification_sync(self, peer_handle: object, payload: bytes) -> None:
        self._check_open()
        peer = _require_peer(peer_handle)
        if peer.released:
            raise ClosedError("reference peer is released")
        peer.backend._enqueue_notification(
            BackendNotification(
                payload=payload,
                source_endpoint_id=self._endpoint_id,
                source_incarnation=self._incarnation,
            )
        )

    def poll_notifications(
        self, max_items: int | None = None
    ) -> list[BackendNotification]:
        self._check_open()
        count = len(self._notifications) if max_items is None else max_items
        result: list[BackendNotification] = []
        with self._lock:
            for _ in range(min(count, len(self._notifications))):
                result.append(self._notifications.popleft())
        return result

    def progress(self) -> None:
        self._check_open()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._registrations or self._plans or self._peers:
                raise BusyError("reference backend still owns child resources")
            with _REFERENCE_ENDPOINTS_LOCK:
                if _REFERENCE_ENDPOINTS.get(self._endpoint_id) is self:
                    _REFERENCE_ENDPOINTS.pop(self._endpoint_id, None)
            self._notifications.clear()
            self._activated = False
            self._closed = True

    def _resolve_local(self, region: BackendRegion) -> list[tuple[int, int]]:
        registration_id = _require_registration_handle(region.backend_descriptor)
        with self._lock:
            try:
                memory = self._registrations[registration_id]
            except KeyError as error:
                raise ClosedError(
                    "local reference registration is not active"
                ) from error
        return _memory_spans(memory, region)

    def _resolve_remote(
        self, region: BackendRegion
    ) -> tuple[list[tuple[int, int]], RegisteredMemory]:
        descriptor = region.backend_descriptor
        if isinstance(descriptor, ImportedRawMemory):
            raise UnsupportedError("reference backend rejects remote raw spans")
        if not isinstance(descriptor, ImportedRegistration):
            raise InvalidRegionError("invalid remote registration descriptor")
        peer = _require_peer(descriptor.peer_handle)
        if peer.released:
            raise ClosedError("reference peer is released")
        if descriptor.descriptor != descriptor.registration_id.encode("utf-8"):
            raise InvalidMetadataError(
                "reference registration descriptor does not match its id"
            )
        if descriptor.registration_id not in peer.allowed_registration_ids:
            raise InvalidMetadataError(
                "remote registration is outside the imported grant"
            )
        with peer.backend._lock:
            try:
                memory = peer.backend._registrations[descriptor.registration_id]
            except KeyError as error:
                raise ClosedError(
                    "remote reference registration is not active"
                ) from error
        if (
            memory.name != descriptor.name
            or memory.nbytes != descriptor.nbytes
            or memory.device != descriptor.device
            or memory.memory_type != descriptor.memory_type
        ):
            raise InvalidMetadataError("remote registration attributes changed")
        return _memory_spans(memory, region), memory

    def _plan_peer(self, plan: _ReferencePlan) -> _ReferencePeer:
        descriptor = plan.remote[0].backend_descriptor
        if not isinstance(descriptor, ImportedRegistration):
            raise InvalidRegionError("reference plan has no named remote peer")
        return _require_peer(descriptor.peer_handle)

    def _enqueue_notification(self, notification: BackendNotification) -> None:
        with self._lock:
            self._check_open()
            self._notifications.append(notification)

    def _check_open(self) -> None:
        if self._closed or not self._activated:
            raise ClosedError(f"reference endpoint {self._name!r} is closed")


def _create_reference_backend(
    endpoint_id: str,
    name: str,
    incarnation: str,
    progress_mode: ProgressMode,
    thread_mode: ThreadMode,
    options: Mapping[str, Any],
    *,
    adopt_backend: Callable[[Backend], None],
) -> None:
    backend = _ReferenceBackend._new_unactivated(
        endpoint_id,
        name,
        incarnation,
        progress_mode,
        thread_mode,
        options,
    )
    adopt_backend(backend)
    backend._activate()


def _register_reference_backend() -> None:
    register_backend("reference", _create_reference_backend, factory_api_version=2)


def _require_registration_handle(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidRegionError("invalid reference registration handle")
    return value


def _require_peer(value: object) -> _ReferencePeer:
    if not isinstance(value, _ReferencePeer):
        raise InvalidRegionError("invalid reference peer handle")
    return value


def _require_plan(value: object) -> _ReferencePlan:
    if not isinstance(value, _ReferencePlan):
        raise InvalidRegionError("invalid reference plan handle")
    return value


def _select(
    regions: tuple[BackendRegion, ...],
    indices: Sequence[int] | None,
    *,
    indexed: bool,
) -> tuple[BackendRegion, ...]:
    if indices is None:
        return regions
    if not indexed:
        raise InvalidRegionError("indices require an indexed reference plan")
    return tuple(_logical_block(regions, index) for index in indices)


def _logical_block(
    regions: Sequence[BackendRegion], logical_index: int
) -> BackendRegion:
    if isinstance(logical_index, bool) or not isinstance(logical_index, int):
        raise InvalidRegionError("reference block indices must be integers")
    if logical_index < 0:
        raise InvalidRegionError("reference block index must be non-negative")
    first_index = 0
    for region in regions:
        count = region.count or 1
        if logical_index < first_index + count:
            block_index = logical_index - first_index
            stride = region.stride or region.nbytes
            return BackendRegion(
                owner=region.owner,
                offset=region.offset + block_index * stride,
                nbytes=region.nbytes,
                device=region.device,
                memory_type=region.memory_type,
                backend_descriptor=region.backend_descriptor,
                remote=region.remote,
            )
        first_index += count
    raise InvalidRegionError("reference block index is outside the region catalog")


def _memory_spans(
    memory: RegisteredMemory,
    region: BackendRegion,
) -> list[tuple[int, int]]:
    count = region.count or 1
    stride = region.stride or region.nbytes
    end = region.offset + (count - 1) * stride + region.nbytes
    if end > memory.nbytes:
        raise InvalidRegionError("reference region exceeds registered memory")
    return [
        (memory.address + region.offset + index * stride, region.nbytes)
        for index in range(count)
    ]


def _copy_spans(
    source: list[tuple[int, int]],
    destination: list[tuple[int, int]],
    *,
    anchors: tuple[RegisteredMemory, ...] = (),
) -> None:
    # This argument deliberately retains remote allocation owners across every
    # raw pointer access, even if the exporting endpoint deregisters meanwhile.
    _ = anchors
    source_size = sum(nbytes for _, nbytes in source)
    destination_size = sum(nbytes for _, nbytes in destination)
    if source_size != destination_size:
        raise InvalidRegionError(
            f"source and destination sizes differ: {source_size} != {destination_size}"
        )
    _reject_overlapping_spans(destination)
    for source_address, source_nbytes in source:
        for destination_address, destination_nbytes in destination:
            if _overlaps(
                source_address,
                source_nbytes,
                destination_address,
                destination_nbytes,
            ):
                raise InvalidRegionError(
                    "reference backend rejects overlapping source and "
                    "destination regions"
                )

    source_index = destination_index = 0
    source_offset = destination_offset = 0
    while source_index < len(source):
        source_address, source_nbytes = source[source_index]
        destination_address, destination_nbytes = destination[destination_index]
        chunk = min(
            source_nbytes - source_offset,
            destination_nbytes - destination_offset,
        )
        ctypes.memmove(
            destination_address + destination_offset,
            source_address + source_offset,
            chunk,
        )
        source_offset += chunk
        destination_offset += chunk
        if source_offset == source_nbytes:
            source_index += 1
            source_offset = 0
        if destination_offset == destination_nbytes:
            destination_index += 1
            destination_offset = 0


def _reject_overlapping_spans(spans: list[tuple[int, int]]) -> None:
    intervals = sorted((address, address + nbytes) for address, nbytes in spans)
    for previous, current in itertools.pairwise(intervals):
        if current[0] < previous[1]:
            raise InvalidRegionError(
                "reference backend rejects overlapping destination regions"
            )


def _overlaps(
    first_address: int,
    first_nbytes: int,
    second_address: int,
    second_nbytes: int,
) -> bool:
    return (
        first_address < second_address + second_nbytes
        and second_address < first_address + first_nbytes
    )


_REFERENCE_ENDPOINTS: WeakValueDictionary[str, _ReferenceBackend] = (
    WeakValueDictionary()
)
_REFERENCE_ENDPOINTS_LOCK = threading.Lock()
