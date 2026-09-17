from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ._errors import BackendNotFoundError, BusyError, UnsupportedError
from ._types import (
    BackendNotification,
    BackendNotificationBatch,
    BackendRegion,
    Capabilities,
    EndpointMetadata,
    ProgressMode,
    ThreadMode,
    TransferOp,
    WorkState,
)


__all__ = [
    "BACKEND_FACTORY_API_VERSION",
    "Backend",
    "BackendFactory",
    "BackendFactoryV2",
    "BackendIndexSelection",
    "BackendRequestSlot",
    "BackendWork",
    "ImportedRawMemory",
    "ImportedRegistration",
    "RegisteredMemory",
    "get_backend_factory",
    "register_backend",
    "unregister_backend",
]

_REQUEST_SLOT_TERMINAL_STATES = frozenset(
    {WorkState.COMPLETED, WorkState.FAILED, WorkState.CANCELLED}
)
_SIGNED_INT64_MAX = (1 << 63) - 1
_INDEX_SELECTION_TOKEN = object()


class BackendIndexSelection:
    """Core-authenticated, immutable indexed-selection lowering input.

    Instances are constructed only by Core after it has snapshotted and
    validated both index vectors against one exact plan and operation.  The
    read-only memoryviews share one private, Core-owned native-int32 buffer.
    Providers must validate the exact type and plan/operation binding before
    using the proof flags; :meth:`Backend.submit_prevalidated` does that for
    source-compatible providers.
    """

    __slots__ = (
        "_destination_catalog_nonoverlapping",
        "_destination_indices_unique",
        "_local_indices",
        "_op",
        "_plan_handle",
        "_remote_indices",
        "_sealed",
        "_selected_nbytes",
        "_uniform_pair_layout",
    )

    def __init__(
        self,
        op: TransferOp,
        plan_handle: object,
        local_indices: memoryview,
        remote_indices: memoryview,
        *,
        selected_nbytes: int,
        uniform_pair_layout: bool,
        destination_catalog_nonoverlapping: bool,
        destination_indices_unique: bool,
        _token: object,
    ) -> None:
        if _token is not _INDEX_SELECTION_TOKEN:
            raise TypeError("BackendIndexSelection objects are created by Core")
        # This object is created on every full-dynamic submission. Calling its
        # rejecting Python __setattr__ for each field would repeat getattr and
        # dispatch overhead nine times before the object is sealed.
        set_attribute = object.__setattr__
        set_attribute(self, "_op", op)
        set_attribute(self, "_plan_handle", plan_handle)
        set_attribute(self, "_local_indices", local_indices)
        set_attribute(self, "_remote_indices", remote_indices)
        set_attribute(self, "_selected_nbytes", selected_nbytes)
        set_attribute(self, "_uniform_pair_layout", uniform_pair_layout)
        set_attribute(
            self,
            "_destination_catalog_nonoverlapping",
            destination_catalog_nonoverlapping,
        )
        set_attribute(self, "_destination_indices_unique", destination_indices_unique)
        set_attribute(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("BackendIndexSelection is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("BackendIndexSelection is immutable")

    @property
    def local_indices(self) -> memoryview:
        return self._local_indices

    @property
    def remote_indices(self) -> memoryview:
        return self._remote_indices

    @property
    def selected_nbytes(self) -> int:
        return self._selected_nbytes

    @property
    def uniform_pair_layout(self) -> bool:
        """Whether Core proved equal, uniform per-pair block geometry."""

        return self._uniform_pair_layout

    @property
    def destination_catalog_nonoverlapping(self) -> bool:
        """Whether Core proved that the full destination catalog is disjoint."""

        return self._destination_catalog_nonoverlapping

    @property
    def destination_indices_unique(self) -> bool:
        """Whether Core proved the destination logical indices are unique."""

        return self._destination_indices_unique

    def _validate_binding(self, op: TransferOp, plan_handle: object) -> None:
        if type(self) is not BackendIndexSelection:
            raise TypeError("selection must be a Core BackendIndexSelection")
        if self._op is not op or self._plan_handle is not plan_handle:
            raise ValueError("prevalidated selection does not match plan and op")


def _make_backend_index_selection(
    op: TransferOp,
    plan_handle: object,
    local_indices: memoryview,
    remote_indices: memoryview,
    *,
    selected_nbytes: int,
    uniform_pair_layout: bool,
    destination_catalog_nonoverlapping: bool,
    destination_indices_unique: bool,
) -> BackendIndexSelection:
    return BackendIndexSelection(
        op,
        plan_handle,
        local_indices,
        remote_indices,
        selected_nbytes=selected_nbytes,
        uniform_pair_layout=uniform_pair_layout,
        destination_catalog_nonoverlapping=destination_catalog_nonoverlapping,
        destination_indices_unique=destination_indices_unique,
        _token=_INDEX_SELECTION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class RegisteredMemory:
    """A local allocation whose lifetime is retained by PyTorch Core."""

    registration_id: str
    name: str
    owner: object = field(repr=False)
    address: int
    nbytes: int
    device: str
    memory_type: str


@dataclass(frozen=True, slots=True)
class ImportedRegistration:
    """A named remote allocation imported from endpoint metadata."""

    peer_handle: object = field(repr=False)
    registration_id: str
    name: str
    nbytes: int
    device: str
    memory_type: str
    descriptor: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class ImportedRawMemory:
    """An expert-only remote span interpreted and validated by a backend."""

    peer_handle: object = field(repr=False)
    address: int
    nbytes: int
    device: str
    memory_type: str


class BackendWork(ABC):
    """Provider request used by the stable :class:`Work` wrapper.

    ``close`` must be idempotent. Core retains every submission argument until
    the public Work is closed. Core also retains any CUDA readiness token even
    though it resolves readiness before calling the provider. State may remain
    PENDING, advance to RUNNING, or advance directly to a terminal state;
    RUNNING may only remain RUNNING or advance to a terminal state.
    """

    # An empty base layout lets latency-sensitive provider subclasses opt into
    # real slots.  A subclass that omits ``__slots__`` still receives its normal
    # instance dictionary, preserving the source-compatible extension model.
    __slots__ = ()

    @property
    @abstractmethod
    def state(self) -> WorkState | None: ...

    @property
    @abstractmethod
    def error(self) -> BaseException | None: ...

    @property
    def completion_event(self) -> object | None:
        return None

    def poll_bounded(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
        timeout_check_interval: int,
    ) -> WorkState:
        """Perform at most ``max_polls`` state observations.

        Providers advertising ``capabilities.fused_work_poll`` should override
        this method so the bounded loop remains inside one provider boundary.
        The default preserves source compatibility with existing providers.
        A zero timeout still permits the first observation; the timeout is
        checked only between observations and cannot preempt a blocking poll.
        """

        if isinstance(max_polls, bool) or not isinstance(max_polls, int):
            raise TypeError("max_polls must be a positive integer")
        if max_polls <= 0 or max_polls > _SIGNED_INT64_MAX:
            raise ValueError("max_polls must be a positive signed 64-bit integer")
        if timeout_ns is not None:
            if isinstance(timeout_ns, bool) or not isinstance(timeout_ns, int):
                raise TypeError("timeout_ns must be a non-negative integer or None")
            if timeout_ns < 0 or timeout_ns > _SIGNED_INT64_MAX:
                raise ValueError(
                    "timeout_ns must be a non-negative signed 64-bit integer or None"
                )
        if isinstance(timeout_check_interval, bool) or not isinstance(
            timeout_check_interval, int
        ):
            raise TypeError("timeout_check_interval must be a positive integer")
        if timeout_check_interval <= 0 or timeout_check_interval > _SIGNED_INT64_MAX:
            raise ValueError(
                "timeout_check_interval must be a positive signed 64-bit integer"
            )
        deadline_ns = None if timeout_ns is None else time.monotonic_ns() + timeout_ns
        state = WorkState.PENDING
        for poll_index in range(max_polls):
            state = self.state
            if (
                not isinstance(state, WorkState)
                or state in _REQUEST_SLOT_TERMINAL_STATES
            ):
                break
            if (
                deadline_ns is not None
                and (poll_index == 0 or (poll_index + 1) % timeout_check_interval == 0)
                and time.monotonic_ns() >= deadline_ns
            ):
                break
        return state

    @abstractmethod
    def cancel(self) -> bool:
        """Request best-effort cancellation.

        Returns ``True`` only when the request is known to be cancelled.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the provider request after it reaches a terminal state."""


class BackendRequestSlot(ABC):
    """Provider-owned persistent request behind a public ``TransferSlot``.

    Core queries ``state``, ``error``, and ``completion_event`` only while one
    unrecycled generation is active. ``start`` rearms the immutable transfer
    described at construction. It must return ``None`` and, once native work
    may have launched, must convert every launch failure (including a caught
    ``BaseException``) into observable terminal slot state rather than raise.

    ``recycle`` is called only after terminal state and returns the request to
    provider-idle state without changing its immutable transfer description.
    It must return ``None`` and be idempotent after successful recycle so Core
    can retry if an asynchronous interruption lands after the provider state
    transition. Ordinarily, ``close`` is called only while provider-idle and
    must likewise be idempotent after successful release.

    A provider advertising
    ``capabilities.deferred_successful_request_slot_recycle`` additionally
    retains an observable, clean ``COMPLETED`` receipt when Core omits the
    successful recycle in ``TransferSlot.execute``. Its next ``start``,
    ``start_and_poll``, ``start_with_notification``,
    ``start_and_poll_with_notification``, or ``close`` must consume that
    receipt inside the same provider boundary. Explicit ``recycle`` remains
    supported and idempotent.
    Failure, cancellation, ambiguous launch, and nonterminal state must never
    be consumed implicitly, and ``poll_state`` must continue to report
    ``COMPLETED`` until the receipt is consumed.

    Core serializes calls to one slot under ``SERIALIZED`` and ``MULTIPLE``;
    ``SINGLE`` supplies the same guarantee through its one-owner-thread
    contract. Providers therefore need synchronization only for state shared
    between different slots or other backend resources. Independent slots may
    make progress concurrently under ``MULTIPLE``. Asynchronous interruption
    is recoverable at every provider boundary, but request-slot methods are not
    recursively reentrant from a Python signal handler while the interrupted
    method is still unwinding.
    """

    # See BackendWork: provider slots can be dictionary-free without requiring
    # existing unslotted third-party implementations to change.
    __slots__ = ()

    @property
    @abstractmethod
    def state(self) -> WorkState | None: ...

    def poll_state(self) -> WorkState | None:
        """Return current state through a cacheable method dispatch.

        Providers with a latency-sensitive request path should override this
        method directly.  The default preserves source compatibility with
        providers that implement only the ``state`` property. ``None`` means
        the provider is still idle because an interrupted ``start`` never
        entered or acknowledged a generation. Once a generation is
        acknowledged, the provider must report a ``WorkState`` until recycle.
        """

        return self.state

    @property
    @abstractmethod
    def error(self) -> BaseException | None: ...

    @property
    def completion_event(self) -> object | None:
        return None

    @abstractmethod
    def start(self) -> None:
        """Start one generation of the immutable request."""

    def start_with_notification(self, notification: bytes | None) -> None:
        """Start with an explicit request-local notification override.

        ``None`` explicitly clears the notification for this generation;
        non-``None`` payloads are non-empty immutable bytes already validated
        by Core. Before returning, the provider must synchronously copy or
        retain the payload for as long as native work can access it; Core need
        not retain the argument after this boundary. Providers advertise this
        optional operation through
        ``request_slot_notification_overrides``. The default is deliberately
        unsupported so an older provider can never silently retain a stale
        notification from a prior repost.
        """

        del notification
        raise UnsupportedError(
            "backend request slot does not support notification overrides"
        )

    def start_and_poll(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
    ) -> WorkState | None:
        """Start one generation and perform a bounded initial state poll.

        This source-compatible default composes :meth:`start` and
        :meth:`poll_state`. Providers advertising
        ``capabilities.fused_request_slot_start_poll`` should override it so
        native post and status polling remain inside one provider boundary.

        At most ``max_polls`` state observations may be made. Zero is a
        post-only budget: fused providers should return the state implied by
        the native post result without issuing a status query. The default
        source-compatible composition returns the cached ``state`` after
        ``start``; only an overriding fused provider can guarantee that reading
        it performs no native probe. ``timeout_ns`` bounds elapsed time from
        entry, checked between observations; it cannot preempt a blocking
        provider operation. With a positive budget, one observation is made
        after start even for a zero timeout. A successful implementation must
        return a :class:`WorkState`. ``None`` retains its idle/unacknowledged
        meaning so Core can recover an interrupted start, but is not a valid
        successful acknowledgement of a generation.
        """

        if isinstance(max_polls, bool) or not isinstance(max_polls, int):
            raise TypeError("max_polls must be a non-negative integer")
        if max_polls < 0 or max_polls > _SIGNED_INT64_MAX:
            raise ValueError("max_polls must be a non-negative signed 64-bit integer")
        if timeout_ns is not None:
            if isinstance(timeout_ns, bool) or not isinstance(timeout_ns, int):
                raise TypeError("timeout_ns must be a non-negative integer or None")
            if timeout_ns < 0 or timeout_ns > _SIGNED_INT64_MAX:
                raise ValueError(
                    "timeout_ns must be a non-negative signed 64-bit integer or None"
                )
        start_result = self.start()
        if start_result is not None:
            raise TypeError("backend request-slot start must return None")
        if max_polls == 0:
            return self.state
        deadline_ns = None if timeout_ns is None else time.monotonic_ns() + timeout_ns
        state: WorkState | None = None
        for _ in range(max_polls):
            state = self.poll_state()
            if (
                state is None
                or not isinstance(state, WorkState)
                or state in _REQUEST_SLOT_TERMINAL_STATES
                or (deadline_ns is not None and time.monotonic_ns() >= deadline_ns)
            ):
                break
        return state

    def start_and_poll_with_notification(
        self,
        notification: bytes | None,
        *,
        max_polls: int,
        timeout_ns: int | None,
    ) -> WorkState | None:
        """Start with a notification override and perform bounded polling.

        This source-compatible default composes
        :meth:`start_with_notification` and :meth:`poll_state`. Providers that
        also advertise ``fused_request_slot_start_poll`` should override it so
        notification selection, native post, and the initial status probes
        remain in one provider boundary.

        The same copy-or-retain requirement as :meth:`start_with_notification`
        applies before this method returns. A zero poll budget has the same
        post-only semantics and default limitation as :meth:`start_and_poll`.
        """

        if isinstance(max_polls, bool) or not isinstance(max_polls, int):
            raise TypeError("max_polls must be a non-negative integer")
        if max_polls < 0 or max_polls > _SIGNED_INT64_MAX:
            raise ValueError("max_polls must be a non-negative signed 64-bit integer")
        if timeout_ns is not None:
            if isinstance(timeout_ns, bool) or not isinstance(timeout_ns, int):
                raise TypeError("timeout_ns must be a non-negative integer or None")
            if timeout_ns < 0 or timeout_ns > _SIGNED_INT64_MAX:
                raise ValueError(
                    "timeout_ns must be a non-negative signed 64-bit integer or None"
                )
        result = self.start_with_notification(notification)
        if result is not None:
            raise TypeError(
                "backend request-slot start_with_notification must return None"
            )
        if max_polls == 0:
            return self.state
        deadline_ns = None if timeout_ns is None else time.monotonic_ns() + timeout_ns
        state: WorkState | None = None
        for _ in range(max_polls):
            state = self.poll_state()
            if (
                state is None
                or not isinstance(state, WorkState)
                or state in _REQUEST_SLOT_TERMINAL_STATES
                or (deadline_ns is not None and time.monotonic_ns() >= deadline_ns)
            ):
                break
        return state

    def poll_bounded(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
        timeout_check_interval: int,
    ) -> WorkState | None:
        """Perform bounded state observations for an active generation.

        Providers advertising ``capabilities.fused_request_slot_poll`` should
        override this source-compatible default so multiple native status
        observations remain inside one provider boundary. The capability does
        not require start/poll fusion.

        At most ``max_polls`` observations are made. ``timeout_ns`` measures a
        soft monotonic elapsed-time bound from entry; it is checked after the
        first observation and then every ``timeout_check_interval``
        observations. It cannot preempt a blocking provider call, and sampled
        checking may overshoot by up to one sampling interval of provider
        calls. A zero timeout still makes exactly one observation. ``None``
        disables clock checks. A returned ``None`` retains its provider-idle
        meaning and is invalid after Core has observed an acknowledged active
        generation.
        """

        if isinstance(max_polls, bool) or not isinstance(max_polls, int):
            raise TypeError("max_polls must be a positive integer")
        if max_polls <= 0 or max_polls > _SIGNED_INT64_MAX:
            raise ValueError("max_polls must be a positive signed 64-bit integer")
        if timeout_ns is not None:
            if isinstance(timeout_ns, bool) or not isinstance(timeout_ns, int):
                raise TypeError("timeout_ns must be a non-negative integer or None")
            if timeout_ns < 0 or timeout_ns > _SIGNED_INT64_MAX:
                raise ValueError(
                    "timeout_ns must be a non-negative signed 64-bit integer or None"
                )
        if isinstance(timeout_check_interval, bool) or not isinstance(
            timeout_check_interval, int
        ):
            raise TypeError("timeout_check_interval must be a positive integer")
        if timeout_check_interval <= 0 or timeout_check_interval > _SIGNED_INT64_MAX:
            raise ValueError(
                "timeout_check_interval must be a positive signed 64-bit integer"
            )
        deadline_ns = None if timeout_ns is None else time.monotonic_ns() + timeout_ns
        state: WorkState | None = None
        for poll in range(max_polls):
            state = self.poll_state()
            if (
                state is None
                or not isinstance(state, WorkState)
                or state in _REQUEST_SLOT_TERMINAL_STATES
            ):
                break
            observations = poll + 1
            if (
                deadline_ns is not None
                and (observations == 1 or observations % timeout_check_interval == 0)
                and time.monotonic_ns() >= deadline_ns
            ):
                break
        return state

    @abstractmethod
    def cancel(self) -> bool:
        """Request best-effort cancellation of the active generation."""

    @abstractmethod
    def recycle(self) -> None:
        """Return a terminal generation to retry-safe provider-idle state."""

    @abstractmethod
    def close(self) -> None:
        """Release this provider request while it is idle."""


class Backend(ABC):
    """Experimental provider SPI for one-sided endpoint transfer.

    Background progress is provider-owned. Release methods must be idempotent
    after a successful release and must make partial cleanup retry-safe.
    Handle-producing methods construct a provider transaction record, pass it
    to ``adopt_handle`` before their first native mutation, and return ``None``.
    Core can therefore roll back an adopted partial mutation or retain it for a
    public close retry. Before adoption, a provider must fully roll back any
    internal state before raising. If native cleanup fails, it may additionally
    retain the orphan internally, reject conflicting operations, and retry
    cleanup on later calls and close.
    """

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @property
    def supported_thread_modes(self) -> frozenset[ThreadMode]:
        return frozenset({ThreadMode.SERIALIZED})

    @abstractmethod
    def endpoint_payload(self, registration_handles: Sequence[object]) -> bytes:
        """Export connection metadata limited to the selected registrations.

        A provider must not silently grant access to omitted registrations. It
        may reject selective export if its transport cannot enforce it.
        """

        ...

    @abstractmethod
    def register(
        self,
        memory: RegisteredMemory,
        *,
        adopt_handle: Callable[[object], None],
    ) -> None:
        """Adopt a retry-safe provider record before native registration."""

    @abstractmethod
    def deregister(self, registration_handle: object) -> None: ...

    @abstractmethod
    def describe_registration(self, registration_handle: object) -> bytes: ...

    @abstractmethod
    def import_peer(
        self,
        metadata: EndpointMetadata,
        *,
        adopt_handle: Callable[[object], None],
    ) -> None:
        """Adopt a retry-safe provider record before native peer mutation."""

    @abstractmethod
    def release_peer(self, peer_handle: object) -> None: ...

    @abstractmethod
    def prepare(
        self,
        local: Sequence[BackendRegion],
        remote: Sequence[BackendRegion],
        *,
        indexed: bool,
        adopt_handle: Callable[[object], None],
    ) -> None:
        """Adopt a retry-safe provider record before native preparation."""

    @abstractmethod
    def release_plan(self, plan_handle: object) -> None: ...

    @abstractmethod
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
        """Submit work after Core has established local CUDA readiness.

        Core resolves the selected CUDA ordering mode before entering the
        serialized provider call, so this method must not add an implicit device
        synchronization. Terminal completion means local memory may be safely
        reused. ``completion_event``, when present, must express that completion
        to the device. An attached notification may become visible only after the
        remote writes are visible.

        Providers must also reject overlapping physical destination spans,
        including aliases advertised as distinct remote registrations.

        For an indexed plan, each prepared ``BackendRegion`` contributes
        ``count`` logical blocks, or one block when ``count`` is absent. Indices
        address the flattened region-major, block-minor catalogs. A logical
        block has ``nbytes`` bytes at ``offset + within_region_index * stride``
        (with an effective stride of ``nbytes`` for an unstrided region);
        providers may keep this representation compact rather than expanding it
        at prepare. ``None`` indices select every logical block in that catalog.

        ``reuse_token`` is an opaque identity hint. Core passes the same
        non-``None`` object for every submission from one ``BoundTransfer`` and
        for repeated full-selection, notification-free, caller-ready convenience
        submissions of one operation on one plan. A provider may key immutable
        cached lowering state by object identity, but must not inspect the token,
        derive transfer semantics from it, or require it to be non-``None``.
        Core tokens support weak references; caches should be weak-keyed or
        scoped to and cleared with the provider plan.

        ``submit_prevalidated`` has a narrower token contract: its token
        identifies the exact plan/operation/index selection, while notification
        is request-local and may change. Only an override that can update the
        native notification at post time may reuse that token. The default
        therefore clears it when forwarding to this legacy method.

        Submission is transactional with respect to ownership. The provider
        must construct its BackendWork and synchronously call ``adopt_work``
        before launching transport work. Before adoption it may raise only if
        no transport access can remain in flight. After successful adoption it
        must not raise: launch failures, including asynchronous interruptions,
        must instead become observable through the adopted BackendWork. This
        handshake ensures Core owns all memory and plan lifetimes before native
        code can access them. If the adoption callback itself is interrupted,
        the provider must not launch; it should make its Work terminally failed
        and return normally, allowing Core to determine whether adoption took
        effect before the interruption.
        """

        ...

    def submit_prevalidated(
        self,
        op: TransferOp,
        plan_handle: object,
        *,
        selection: BackendIndexSelection,
        notification: bytes | None,
        reuse_token: object | None,
        adopt_work: Callable[[BackendWork], None],
    ) -> None:
        """Submit a Core-authenticated immutable indexed selection.

        The default preserves provider source compatibility by forwarding the
        two read-only native-int32 views to :meth:`submit`. A provider may
        override this method to consume the packed views without rebuilding
        them and to combine Core's proof flags with its own plan-resolution
        proofs. It must not trust a duck-typed object.
        """

        if type(selection) is not BackendIndexSelection:
            raise TypeError("selection must be a Core BackendIndexSelection")
        selection._validate_binding(op, plan_handle)
        del reuse_token
        return self.submit(
            op,
            plan_handle,
            local_indices=selection.local_indices,
            remote_indices=selection.remote_indices,
            notification=notification,
            reuse_token=None,
            adopt_work=adopt_work,
        )

    def submit_prevalidated_and_poll(
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
        """Submit one authenticated selection and perform a bounded probe.

        This is capability-gated rather than composed through a callback
        wrapper: only a provider that can keep preparation, post, and polling
        in its intended native boundary should advertise
        ``fused_prevalidated_submit_poll``. Existing providers remain source
        compatible and use :meth:`submit_prevalidated` plus public Work polls.
        """

        del (
            op,
            plan_handle,
            selection,
            notification,
            max_polls,
            timeout_ns,
            adopt_work,
        )
        raise UnsupportedError(
            "backend does not support fused prevalidated submit and poll"
        )

    def create_request_slot(
        self,
        op: TransferOp,
        plan_handle: object,
        *,
        local_indices: Sequence[int] | None,
        remote_indices: Sequence[int] | None,
        notification: bytes | None,
        adopt_slot: Callable[..., None],
    ) -> None:
        """Create an opt-in persistent request for one immutable transfer.

        The provider must construct a retry-safe :class:`BackendRequestSlot`
        record and synchronously call ``adopt_slot`` before its first native
        mutation. A provider may pass an optional keyword-only
        ``execution_dispatch`` object to that callback. Core then validates and
        caches the dispatch object's execution methods while retaining the
        :class:`BackendRequestSlot` wrapper for diagnostics, completion events,
        and close. Omitting the keyword keeps the wrapper as the dispatch
        object and is source compatible with existing providers.

        Before adoption the provider may raise only after fully rolling back
        provider state. If the adoption callback is interrupted, it must not
        create or launch native work and should return normally when Core may
        already have retained the slot.

        A provider advertises this optional SPI with
        ``capabilities.persistent_request_slots``. The default rejects it so
        existing providers remain source compatible.
        """

        del op, plan_handle, local_indices, remote_indices, notification, adopt_slot
        raise UnsupportedError("backend does not support persistent request slots")

    @abstractmethod
    def send_notification(
        self,
        peer_handle: object,
        payload: bytes,
        *,
        adopt_work: Callable[[BackendWork], None],
    ) -> None:
        """Send a standalone notification.

        Terminal Work completion has the meaning declared by
        ``capabilities.standalone_notification_completion``.
        The provider must use the same adopt-before-launch transaction described
        by :meth:`submit`.
        """

        ...

    def send_notification_sync(self, peer_handle: object, payload: bytes) -> None:
        """Send without creating a provider or Core Work.

        Return means the completion point declared by
        ``capabilities.standalone_notification_completion``; it does not imply
        remote delivery unless that declaration is ``REMOTE_DELIVERED``.
        Providers opt in with ``capabilities.synchronous_notification_send``.
        Under ``MULTIPLE``, Core pins the peer lifetime but does not serialize
        independent native calls; providers must preserve their advertised
        concurrent-entry contract.
        """

        del peer_handle, payload
        raise UnsupportedError(
            "backend does not support synchronous standalone notification send"
        )

    @abstractmethod
    def poll_notifications(
        self, max_items: int | None = None
    ) -> list[BackendNotification]:
        """Destructively receive up to ``max_items`` notifications.

        This v0 operation is at-most-once: the SPI has no batch token or
        acknowledge/rollback phase.  Providers must validate native ingress
        before exposing it, while applications that require reliable delivery
        must sequence, acknowledge, deduplicate, and retransmit messages.
        """

        ...

    def poll_notification_batches(
        self, max_items: int | None = None
    ) -> list[BackendNotificationBatch]:
        """Destructively receive notifications while preserving source groups.

        ``max_items`` counts payloads, not batches. Providers with a native
        grouped receive should override this method and advertise
        ``capabilities.grouped_notification_receive``. This compatibility
        default groups only adjacent notifications with a complete matching
        identity. Anonymous notifications remain singleton batches because
        scalar input cannot prove that they share a native source.
        """

        notifications = self.poll_notifications(max_items)
        batches: list[BackendNotificationBatch] = []
        source: tuple[str, str] | None = None
        payloads: list[bytes] = []

        def flush() -> None:
            nonlocal source, payloads
            if not payloads or source is None:
                return
            batches.append(
                BackendNotificationBatch(
                    payloads=tuple(payloads),
                    source_endpoint_id=source[0],
                    source_incarnation=source[1],
                )
            )
            source = None
            payloads = []

        for notification in notifications:
            if not isinstance(notification, BackendNotification):
                raise TypeError(
                    "backend notifications must contain BackendNotification values"
                )
            if notification.source_endpoint_id is None:
                flush()
                batches.append(BackendNotificationBatch((notification.payload,)))
                continue
            assert notification.source_incarnation is not None
            next_source = (
                notification.source_endpoint_id,
                notification.source_incarnation,
            )
            if source != next_source:
                flush()
                source = next_source
            payloads.append(notification.payload)
        flush()
        return batches

    @abstractmethod
    def progress(self) -> None: ...

    @abstractmethod
    def close(self) -> None:
        """Close the provider; this must be idempotent after native success."""

        ...


BackendFactory = Callable[
    [
        str,
        str,
        str,
        ProgressMode,
        ThreadMode,
        Mapping[str, Any],
    ],
    Backend,
]


class BackendFactoryV2(Protocol):
    """Factory that transfers backend ownership before native activation.

    Version-2 providers must construct a pure, closeable Python backend record,
    pass it exactly once to ``adopt_backend``, and only then activate or publish
    native state.  A successful call returns ``None``.  Core deliberately does
    not inspect callable signatures or retry a factory under another contract.
    """

    def __call__(
        self,
        endpoint_id: str,
        name: str,
        incarnation: str,
        progress_mode: ProgressMode,
        thread_mode: ThreadMode,
        options: Mapping[str, Any],
        *,
        adopt_backend: Callable[[Backend], None],
    ) -> None: ...


# This is the maximum backward-compatible factory contract understood by this
# Core.  A provider selecting version N may run when this value is >= N.
BACKEND_FACTORY_API_VERSION = 2


@dataclass(frozen=True, slots=True)
class _BackendRegistration:
    factory: BackendFactory | BackendFactoryV2
    factory_api_version: int


_BACKENDS: dict[str, _BackendRegistration] = {}
_BACKENDS_LOCK = threading.Lock()


def register_backend(
    name: str,
    factory: BackendFactory | BackendFactoryV2,
    *,
    replace: bool = False,
    factory_api_version: int = 1,
) -> None:
    """Register an endpoint-transfer backend factory.

    Providers should call this explicitly from an integration module. Importing
    :mod:`torch` never imports an external provider.  Version 1 is retained for
    source compatibility and returns a :class:`Backend`; its return boundary
    cannot close the final CALL-to-adoption interruption window.  Version 2
    receives an ``adopt_backend`` keyword callback, must call it exactly once
    before native activation, and returns ``None``.  Repeating a registration
    with the same factory object and exact API version is idempotent.  This
    closes the caller's post-commit/lost-return window without weakening name
    collision protection: a different factory object or version remains busy.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("backend name must be a non-empty string")
    if not callable(factory):
        raise TypeError("backend factory must be callable")
    if type(factory_api_version) is not int:
        raise TypeError("factory_api_version must be an integer")
    if factory_api_version not in (1, 2):
        raise ValueError("factory_api_version must be 1 or 2")
    with _BACKENDS_LOCK:
        existing = _BACKENDS.get(name)
        if existing is not None and not replace:
            if (
                existing.factory is factory
                and existing.factory_api_version == factory_api_version
            ):
                return
            raise BusyError(f"transfer backend {name!r} is already registered")
        _BACKENDS[name] = _BackendRegistration(factory, factory_api_version)


def unregister_backend(name: str) -> None:
    """Remove a provider factory.

    This only affects future endpoints and is primarily intended for tests.
    Existing endpoints continue to own their backend instances.
    """

    with _BACKENDS_LOCK:
        if name not in _BACKENDS:
            raise BackendNotFoundError(f"transfer backend {name!r} is not registered")
        del _BACKENDS[name]


def _get_backend_registration(name: str) -> _BackendRegistration:
    with _BACKENDS_LOCK:
        try:
            return _BACKENDS[name]
        except KeyError as error:
            raise BackendNotFoundError(
                f"transfer backend {name!r} is not registered"
            ) from error


def get_backend_factory(name: str) -> BackendFactory | BackendFactoryV2:
    """Return the registered callable without changing the legacy query API."""

    return _get_backend_registration(name).factory
