import ast
import gc
from array import array
from pathlib import Path
import sys
import threading
import weakref
from unittest import mock

import torch
from torch.distributed._transfer import (
    BACKEND_FACTORY_API_VERSION,
    BackendFailureError,
    BackendIndexSelection,
    BackendNotificationBatch,
    BackendRequestSlot,
    BackendRegion,
    BackendWork,
    BoundTransfer,
    BusyError,
    Capabilities,
    ClosedError,
    CudaOrderingMode,
    Endpoint,
    EndpointIdentity,
    EndpointMetadata,
    ImportedRawMemory,
    ImportedRegistration,
    IndexSelection,
    InvalidMetadataError,
    InvalidRegionError,
    NotificationCompletion,
    NotificationBatch,
    RawSpan,
    RequestLocalNotificationSlot,
    RegisteredMemory,
    RegistrationMetadata,
    TransferCancelledError,
    TransferOp,
    TransferSlot,
    TransferTimeoutError,
    ThreadMode,
    UnsupportedError,
    Work,
    WorkState,
    register_backend,
)
from torch.distributed._transfer import _api, _backend, _reference, _types
from torch.distributed._transfer._reference import _ReferenceBackend
from torch.testing._internal.common_utils import run_tests, TestCase


def _emulated_compiled_index_analyzer(
    local_indices: object,
    remote_indices: object,
    destination_is_remote: bool,
) -> tuple[bytes, int, int, int, int, int, int, bool]:
    local_bytes = bytes(memoryview(local_indices).cast("B"))
    remote_bytes = bytes(memoryview(remote_indices).cast("B"))
    packed = local_bytes + remote_bytes
    values = memoryview(packed).cast("i")
    local_count = len(local_bytes) // 4
    remote_count = len(remote_bytes) // 4
    local = values[:local_count]
    remote = values[local_count:]
    destination = remote if destination_is_remote else local
    return (
        packed,
        local_count,
        remote_count,
        min(local),
        max(local),
        min(remote),
        max(remote),
        len(set(destination)) != len(destination),
    )


class _PendingWork(BackendWork):
    def __init__(self) -> None:
        self._state = WorkState.PENDING

    @property
    def state(self) -> WorkState:
        return self._state

    @property
    def error(self) -> BaseException | None:
        return None

    def cancel(self) -> bool:
        self._state = WorkState.CANCELLED
        return True

    def close(self) -> None:
        pass


class _BadErrorWork(_PendingWork):
    @property
    def state(self) -> WorkState:
        return WorkState.FAILED

    @property
    def error(self):
        return "not an exception"


class _RegressingWork(_PendingWork):
    def __init__(self) -> None:
        self._queries = 0
        self._cancelled = False

    @property
    def state(self) -> WorkState:
        if self._cancelled:
            return WorkState.CANCELLED
        self._queries += 1
        return WorkState.PENDING if self._queries == 2 else WorkState.RUNNING

    def cancel(self) -> bool:
        self._cancelled = True
        return True


class _TransientStateErrorWork(_PendingWork):
    def __init__(self) -> None:
        self._queries = 0
        self._cancelled = False

    @property
    def state(self) -> WorkState:
        self._queries += 1
        if self._queries == 1:
            raise RuntimeError("transient query failure")
        if self._cancelled:
            return WorkState.CANCELLED
        return WorkState.RUNNING

    def cancel(self) -> bool:
        self._cancelled = True
        return True


class _CountingPendingWork(_PendingWork):
    def __init__(self) -> None:
        super().__init__()
        self.state_queries = 0

    @property
    def state(self) -> WorkState:
        self.state_queries += 1
        return self._state


class _InterruptingFailureWork(_PendingWork):
    def __init__(self) -> None:
        super().__init__()
        self.failure = RuntimeError("provider failure")
        self.error_queries = 0

    @property
    def state(self) -> WorkState:
        return WorkState.FAILED

    @property
    def error(self) -> BaseException:
        self.error_queries += 1
        if self.error_queries == 1:
            raise KeyboardInterrupt
        return self.failure


class _PendingReferenceBackend(_ReferenceBackend):
    def submit(self, *args, **kwargs) -> None:
        kwargs["adopt_work"](_PendingWork())


class _CountingPendingReferenceBackend(_ReferenceBackend):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_work: _CountingPendingWork | None = None

    def submit(self, *args, **kwargs) -> None:
        self.last_work = _CountingPendingWork()
        kwargs["adopt_work"](self.last_work)


class _InterruptingFailureReferenceBackend(_ReferenceBackend):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_work: _InterruptingFailureWork | None = None

    def submit(self, *args, **kwargs) -> None:
        self.last_work = _InterruptingFailureWork()
        kwargs["adopt_work"](self.last_work)


class _FusedSelectionWork(_PendingWork):
    def __init__(self, state: WorkState) -> None:
        super().__init__()
        self._state = state
        self.state_queries = 0
        self.bounded_calls: list[tuple[int, int | None, int]] = []
        self.next_bounded_state = WorkState.COMPLETED
        self._error: BaseException | None = None

    @property
    def state(self) -> WorkState:
        self.state_queries += 1
        return self._state

    @property
    def error(self) -> BaseException | None:
        return self._error

    def poll_bounded(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
        timeout_check_interval: int,
    ) -> WorkState:
        self.bounded_calls.append((max_polls, timeout_ns, timeout_check_interval))
        self._state = self.next_bounded_state
        return self._state


class _FusedSelectionReferenceBackend(_ReferenceBackend):
    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
        fused_prevalidated_submit_poll=True,
        fused_work_poll=True,
        standalone_notification_completion=NotificationCompletion.REMOTE_DELIVERED,
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.initial_state = WorkState.RUNNING
        self.fused_calls: list[tuple[int, int | None, bytes | None]] = []
        self.last_work: _FusedSelectionWork | None = None

    def submit_prevalidated_and_poll(
        self,
        op,
        plan_handle,
        *,
        selection,
        notification,
        max_polls,
        timeout_ns,
        adopt_work,
    ) -> WorkState:
        selection._validate_binding(op, plan_handle)
        work = _FusedSelectionWork(self.initial_state)
        self.last_work = work
        adopt_work(work)
        self.fused_calls.append((max_polls, timeout_ns, notification))
        return self.initial_state


class _BackgroundOnlyReferenceBackend(_ReferenceBackend):
    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        strided_regions=True,
        background_progress=True,
        standalone_notification_completion=NotificationCompletion.REMOTE_DELIVERED,
    )

    def progress(self) -> None:
        raise AssertionError("background-only backend must not be progressed")


class _AttachedOnlyReferenceBackend(_ReferenceBackend):
    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
    )


class _BadErrorReferenceBackend(_ReferenceBackend):
    def submit(self, *args, **kwargs) -> None:
        kwargs["adopt_work"](_BadErrorWork())


class _InterruptReferenceBackend(_ReferenceBackend):
    def submit(self, *args, **kwargs) -> None:
        raise KeyboardInterrupt


class _AdoptThenInterruptReferenceBackend(_ReferenceBackend):
    def submit(self, *args, **kwargs) -> None:
        kwargs["adopt_work"](_PendingWork())
        raise KeyboardInterrupt


class _RegressingReferenceBackend(_ReferenceBackend):
    def submit(self, *args, **kwargs) -> None:
        kwargs["adopt_work"](_RegressingWork())


class _TransientStateErrorReferenceBackend(_ReferenceBackend):
    def submit(self, *args, **kwargs) -> None:
        kwargs["adopt_work"](_TransientStateErrorWork())


class _InterruptAfterCloseReferenceBackend(_ReferenceBackend):
    def close(self) -> None:
        was_closed = self._closed
        super().close()
        if not was_closed:
            raise KeyboardInterrupt


class _TestRequestSlot(BackendRequestSlot):
    def __init__(self) -> None:
        self._state: WorkState | None = None
        self._error: BaseException | None = None
        self.next_state = WorkState.COMPLETED
        self.next_error: BaseException | None = None
        self.start_count = 0
        self.recycle_count = 0
        self.close_count = 0
        self.closed = False
        self.fail_close_once = False
        self.raise_error_once: BaseException | None = None
        self.start_entered: threading.Event | None = None
        self.start_release: threading.Event | None = None
        self.completion = object()

    @property
    def state(self) -> WorkState | None:
        if self.closed:
            raise ClosedError("test request slot is closed")
        return self._state

    @property
    def error(self) -> BaseException | None:
        if self.closed:
            raise ClosedError("test request slot is closed")
        if self.raise_error_once is not None:
            error = self.raise_error_once
            self.raise_error_once = None
            raise error
        return self._error

    @property
    def completion_event(self) -> object:
        return self.completion

    def start(self) -> None:
        if self.closed:
            raise ClosedError("test request slot is closed")
        self.start_count += 1
        if self.start_entered is not None:
            self.start_entered.set()
        if self.start_release is not None and not self.start_release.wait(2):
            raise RuntimeError("test request slot start timed out")
        self._state = self.next_state
        self._error = self.next_error

    def cancel(self) -> bool:
        if self.closed:
            raise ClosedError("test request slot is closed")
        if self._state in {
            WorkState.COMPLETED,
            WorkState.FAILED,
            WorkState.CANCELLED,
        }:
            return self._state is WorkState.CANCELLED
        self._state = WorkState.CANCELLED
        self._error = None
        return True

    def recycle(self) -> None:
        if self.closed:
            raise ClosedError("test request slot is closed")
        self.recycle_count += 1
        self._state = None
        self._error = None

    def close(self) -> None:
        self.close_count += 1
        if self.fail_close_once:
            self.fail_close_once = False
            raise RuntimeError("injected request-slot cleanup failure")
        self.closed = True


class _UnacknowledgedStartRequestSlot(_TestRequestSlot):
    @property
    def state(self) -> WorkState | None:
        return None

    def start(self) -> None:
        raise KeyboardInterrupt


class _NormallyUnacknowledgedRequestSlot(_UnacknowledgedStartRequestSlot):
    def start(self) -> None:
        self.start_count += 1


class _BlockingObservationRequestSlot(_TestRequestSlot):
    def __init__(self) -> None:
        super().__init__()
        self.block_next_observation = False
        self.observation_entered = threading.Event()
        self.observation_release = threading.Event()

    @property
    def state(self) -> WorkState | None:
        if self.block_next_observation:
            self.block_next_observation = False
            self.observation_entered.set()
            if not self.observation_release.wait(2):
                raise RuntimeError("test request-slot observation timed out")
        return super().state


_RETURN_CURRENT_SLOT_STATE = object()


class _FusedTestRequestSlot(_TestRequestSlot):
    def __init__(self) -> None:
        super().__init__()
        self.fused_calls: list[tuple[int, int | None]] = []
        self.fused_result: object = _RETURN_CURRENT_SLOT_STATE
        self.fused_error: BaseException | None = None
        self.poll_states: list[WorkState | None] = []
        self.poll_count = 0
        self.bounded_calls: list[tuple[int, int | None, int]] = []
        self.bounded_results: list[object] = []
        self.bounded_error: BaseException | None = None
        self.bounded_entered: threading.Event | None = None
        self.bounded_release: threading.Event | None = None

    def poll_state(self) -> WorkState | None:
        self.poll_count += 1
        if self.poll_states:
            state = self.poll_states.pop(0)
            if isinstance(state, WorkState):
                self._state = state
            return state
        return self.state

    def start_and_poll(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
    ):
        self.fused_calls.append((max_polls, timeout_ns))
        self.start()
        if self.fused_error is not None:
            raise self.fused_error
        if self.fused_result is _RETURN_CURRENT_SLOT_STATE:
            return self._state
        return self.fused_result

    def poll_bounded(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
        timeout_check_interval: int,
    ):
        self.bounded_calls.append((max_polls, timeout_ns, timeout_check_interval))
        if self.bounded_entered is not None:
            self.bounded_entered.set()
        if self.bounded_release is not None and not self.bounded_release.wait(2):
            raise RuntimeError("test bounded request-slot poll timed out")
        if self.bounded_error is not None:
            raise self.bounded_error
        if self.bounded_results:
            state = self.bounded_results.pop(0)
            if isinstance(state, WorkState):
                self._state = state
            return state
        return super().poll_bounded(
            max_polls=max_polls,
            timeout_ns=timeout_ns,
            timeout_check_interval=timeout_check_interval,
        )


class _NotificationFusedTestRequestSlot(_FusedTestRequestSlot):
    def __init__(self) -> None:
        super().__init__()
        self.notification_starts: list[bytes | None] = []
        self.notification_fused_calls: list[tuple[bytes | None, int, int | None]] = []

    def start_with_notification(self, notification: bytes | None) -> None:
        self.notification_starts.append(notification)
        self.start()

    def start_and_poll_with_notification(
        self,
        notification: bytes | None,
        *,
        max_polls: int,
        timeout_ns: int | None,
    ) -> WorkState | None:
        self.notification_fused_calls.append((notification, max_polls, timeout_ns))
        self.start()
        if self.fused_error is not None:
            raise self.fused_error
        if self.fused_result is _RETURN_CURRENT_SLOT_STATE:
            return self._state
        return self.fused_result


class _TestExecutionDispatch:
    _OPERATION_NAMES = frozenset(
        {
            "start",
            "start_and_poll",
            "poll_bounded",
            "poll_state",
            "cancel",
            "recycle",
        }
    )

    def __init__(self, slot: _FusedTestRequestSlot) -> None:
        self.slot = slot
        self.attribute_reads = {name: 0 for name in self._OPERATION_NAMES}

    def __getattribute__(self, name):
        if name in object.__getattribute__(self, "_OPERATION_NAMES"):
            reads = object.__getattribute__(self, "attribute_reads")
            reads[name] += 1
        return object.__getattribute__(self, name)

    def start(self) -> None:
        self.slot.start()

    def start_and_poll(self, *, max_polls: int, timeout_ns: int | None):
        return self.slot.start_and_poll(
            max_polls=max_polls,
            timeout_ns=timeout_ns,
        )

    def poll_bounded(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
        timeout_check_interval: int,
    ):
        return self.slot.poll_bounded(
            max_polls=max_polls,
            timeout_ns=timeout_ns,
            timeout_check_interval=timeout_check_interval,
        )

    def poll_state(self) -> WorkState | None:
        return self.slot.poll_state()

    def cancel(self) -> bool:
        return self.slot.cancel()

    def recycle(self) -> None:
        self.slot.recycle()

    @property
    def error(self):
        raise AssertionError("execution dispatch must not own diagnostics")

    @property
    def completion_event(self):
        raise AssertionError("execution dispatch must not own completion events")

    def close(self) -> None:
        raise AssertionError("execution dispatch must not own close")


class _CapabilityGatedExecutionDispatch(_TestExecutionDispatch):
    def __getattribute__(self, name):
        if name in {"start_and_poll", "poll_bounded"}:
            raise AssertionError("disabled fused dispatch operation was resolved")
        return super().__getattribute__(name)


class _NonCallableExecutionDispatch(_TestExecutionDispatch):
    start = 1


class _NoneStartExecutionDispatch(_TestExecutionDispatch):
    start = None


class _NoneFusedExecutionDispatch(_TestExecutionDispatch):
    start_and_poll = None


class _UnretainedExecutionDispatch:
    def __init__(self, slot: _FusedTestRequestSlot) -> None:
        # None of these callable objects retains this dispatch container.
        self.start = slot.start
        self.start_and_poll = slot.start_and_poll
        self.poll_bounded = slot.poll_bounded
        self.poll_state = slot.poll_state
        self.cancel = slot.cancel
        self.recycle = slot.recycle


class _UnacknowledgedFusedRequestSlot(_FusedTestRequestSlot):
    @property
    def state(self) -> WorkState | None:
        return None

    def poll_state(self) -> WorkState | None:
        self.poll_count += 1
        return None

    def start_and_poll(
        self,
        *,
        max_polls: int,
        timeout_ns: int | None,
    ) -> WorkState | None:
        self.fused_calls.append((max_polls, timeout_ns))
        self.start_count += 1
        raise KeyboardInterrupt


class _SlotReferenceBackend(_ReferenceBackend):
    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
        persistent_request_slots=True,
        cuda_completion_event=True,
        cuda_ordering_modes=frozenset(
            {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
        ),
        standalone_notification_completion=NotificationCompletion.REMOTE_DELIVERED,
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.request_slots: list[_TestRequestSlot] = []

    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _TestRequestSlot()
        try:
            kwargs["adopt_slot"](slot)
        except BaseException:
            return
        self.request_slots.append(slot)


class _FusedSlotReferenceBackend(_SlotReferenceBackend):
    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
        persistent_request_slots=True,
        fused_request_slot_start_poll=True,
        fused_request_slot_poll=True,
        cuda_completion_event=True,
        cuda_ordering_modes=frozenset(
            {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
        ),
        standalone_notification_completion=NotificationCompletion.REMOTE_DELIVERED,
    )

    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _FusedTestRequestSlot()
        try:
            kwargs["adopt_slot"](slot)
        except BaseException:
            return
        self.request_slots.append(slot)


class _BlockingObservationSlotReferenceBackend(_SlotReferenceBackend):
    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _BlockingObservationRequestSlot()
        try:
            kwargs["adopt_slot"](slot)
        except BaseException:
            return
        self.request_slots.append(slot)


class _ExecutionDispatchSlotBackend(_FusedSlotReferenceBackend):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.execution_dispatches: list[_TestExecutionDispatch] = []

    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _FusedTestRequestSlot()
        dispatch = _TestExecutionDispatch(slot)
        try:
            kwargs["adopt_slot"](slot, execution_dispatch=dispatch)
        except BaseException:
            return
        self.request_slots.append(slot)
        self.execution_dispatches.append(dispatch)


class _CapabilityGatedDispatchSlotBackend(_SlotReferenceBackend):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.execution_dispatches: list[_CapabilityGatedExecutionDispatch] = []

    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _FusedTestRequestSlot()
        dispatch = _CapabilityGatedExecutionDispatch(slot)
        try:
            kwargs["adopt_slot"](slot, execution_dispatch=dispatch)
        except BaseException:
            return
        self.request_slots.append(slot)
        self.execution_dispatches.append(dispatch)


class _InvalidExecutionDispatchSlotBackend(_FusedSlotReferenceBackend):
    _DISPATCH_TYPE = _NonCallableExecutionDispatch

    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _FusedTestRequestSlot()
        dispatch = self._DISPATCH_TYPE(slot)
        try:
            kwargs["adopt_slot"](slot, execution_dispatch=dispatch)
        except BaseException:
            return
        raise AssertionError("non-callable execution dispatch was adopted")


class _NoneStartExecutionDispatchSlotBackend(_InvalidExecutionDispatchSlotBackend):
    _DISPATCH_TYPE = _NoneStartExecutionDispatch


class _NoneFusedExecutionDispatchSlotBackend(_InvalidExecutionDispatchSlotBackend):
    _DISPATCH_TYPE = _NoneFusedExecutionDispatch


class _UnretainedExecutionDispatchSlotBackend(_FusedSlotReferenceBackend):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.execution_dispatch_ref = None

    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _FusedTestRequestSlot()
        dispatch = _UnretainedExecutionDispatch(slot)
        self.execution_dispatch_ref = weakref.ref(dispatch)
        try:
            kwargs["adopt_slot"](slot, execution_dispatch=dispatch)
        except BaseException:
            return
        self.request_slots.append(slot)


class _StartFusedOnlySlotReferenceBackend(_FusedSlotReferenceBackend):
    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
        persistent_request_slots=True,
        fused_request_slot_start_poll=True,
        cuda_completion_event=True,
        cuda_ordering_modes=frozenset(
            {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
        ),
        standalone_notification_completion=NotificationCompletion.REMOTE_DELIVERED,
    )


class _PollOnlySlotReferenceBackend(_FusedSlotReferenceBackend):
    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
        persistent_request_slots=True,
        fused_request_slot_poll=True,
        cuda_completion_event=True,
        cuda_ordering_modes=frozenset(
            {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
        ),
        standalone_notification_completion=NotificationCompletion.REMOTE_DELIVERED,
    )


class _DeferredRecycleSlotReferenceBackend(_SlotReferenceBackend):
    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
        persistent_request_slots=True,
        cuda_completion_event=True,
        cuda_ordering_modes=frozenset(
            {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
        ),
        standalone_notification_completion=NotificationCompletion.REMOTE_DELIVERED,
        deferred_successful_request_slot_recycle=True,
    )


class _DeferredRecycleFusedSlotReferenceBackend(_FusedSlotReferenceBackend):
    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
        persistent_request_slots=True,
        fused_request_slot_start_poll=True,
        fused_request_slot_poll=True,
        cuda_completion_event=True,
        cuda_ordering_modes=frozenset(
            {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
        ),
        standalone_notification_completion=NotificationCompletion.REMOTE_DELIVERED,
        deferred_successful_request_slot_recycle=True,
    )


class _NotificationSlotReferenceBackend(_FusedSlotReferenceBackend):
    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
        persistent_request_slots=True,
        fused_request_slot_start_poll=True,
        fused_request_slot_poll=True,
        cuda_completion_event=True,
        cuda_ordering_modes=frozenset(
            {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
        ),
        standalone_notification_completion=NotificationCompletion.REMOTE_DELIVERED,
        deferred_successful_request_slot_recycle=True,
        request_slot_notification_overrides=True,
    )

    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _NotificationFusedTestRequestSlot()
        try:
            kwargs["adopt_slot"](slot)
        except BaseException:
            return
        self.request_slots.append(slot)


class _MissingNotificationSlotBackend(_NotificationSlotReferenceBackend):
    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _FusedTestRequestSlot()
        try:
            kwargs["adopt_slot"](slot)
        except BaseException:
            return
        raise AssertionError("default notification dispatch was adopted")


class _UnacknowledgedFusedSlotBackend(_FusedSlotReferenceBackend):
    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _UnacknowledgedFusedRequestSlot()
        kwargs["adopt_slot"](slot)
        self.request_slots.append(slot)


class _UnacknowledgedStartSlotBackend(_SlotReferenceBackend):
    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _UnacknowledgedStartRequestSlot()
        kwargs["adopt_slot"](slot)
        self.request_slots.append(slot)


class _NormallyUnacknowledgedSlotBackend(_SlotReferenceBackend):
    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _NormallyUnacknowledgedRequestSlot()
        kwargs["adopt_slot"](slot)
        self.request_slots.append(slot)


class _InterruptSlotCreationBackend(_SlotReferenceBackend):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.created_slot: _TestRequestSlot | None = None

    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _TestRequestSlot()
        try:
            kwargs["adopt_slot"](slot)
        except BaseException:
            return
        self.created_slot = slot
        raise KeyboardInterrupt


class _InterruptSlotCleanupBackend(_InterruptSlotCreationBackend):
    def create_request_slot(self, *args, **kwargs) -> None:
        del args
        slot = _TestRequestSlot()
        slot.fail_close_once = True
        try:
            kwargs["adopt_slot"](slot)
        except BaseException:
            return
        self.created_slot = slot
        raise KeyboardInterrupt


class _InterruptBeforeSlotAdoptionBackend(_SlotReferenceBackend):
    def create_request_slot(self, *args, **kwargs) -> None:
        del args, kwargs
        raise KeyboardInterrupt


class _NoCompletionEventSlotBackend(_SlotReferenceBackend):
    _CAPABILITIES = Capabilities(
        read=True,
        write=True,
        attached_notifications=True,
        standalone_notifications=True,
        strided_regions=True,
        manual_progress=True,
        background_progress=True,
        persistent_request_slots=True,
        cuda_ordering_modes=frozenset(
            {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
        ),
        standalone_notification_completion=NotificationCompletion.REMOTE_DELIVERED,
    )


register_backend("test-pending", _PendingReferenceBackend, replace=True)
register_backend(
    "test-counting-pending", _CountingPendingReferenceBackend, replace=True
)
register_backend(
    "test-interrupting-failure",
    _InterruptingFailureReferenceBackend,
    replace=True,
)
register_backend("test-fused-selection", _FusedSelectionReferenceBackend, replace=True)
register_backend("test-background-only", _BackgroundOnlyReferenceBackend, replace=True)
register_backend("test-attached-only", _AttachedOnlyReferenceBackend, replace=True)
register_backend("test-bad-error", _BadErrorReferenceBackend, replace=True)
register_backend("test-interrupt", _InterruptReferenceBackend, replace=True)
register_backend(
    "test-adopt-then-interrupt", _AdoptThenInterruptReferenceBackend, replace=True
)
register_backend(
    "test-request-slot-no-completion-event",
    _NoCompletionEventSlotBackend,
    replace=True,
)
register_backend("test-regressing", _RegressingReferenceBackend, replace=True)
register_backend(
    "test-transient-state-error",
    _TransientStateErrorReferenceBackend,
    replace=True,
)
register_backend(
    "test-interrupt-after-close", _InterruptAfterCloseReferenceBackend, replace=True
)
register_backend("test-request-slot", _SlotReferenceBackend, replace=True)
register_backend(
    "test-request-slot-blocking-observation",
    _BlockingObservationSlotReferenceBackend,
    replace=True,
)
register_backend("test-request-slot-fused", _FusedSlotReferenceBackend, replace=True)
register_backend(
    "test-request-slot-execution-dispatch",
    _ExecutionDispatchSlotBackend,
    replace=True,
)
register_backend(
    "test-request-slot-execution-dispatch-gated",
    _CapabilityGatedDispatchSlotBackend,
    replace=True,
)
register_backend(
    "test-request-slot-execution-dispatch-invalid",
    _InvalidExecutionDispatchSlotBackend,
    replace=True,
)
register_backend(
    "test-request-slot-execution-dispatch-none-start",
    _NoneStartExecutionDispatchSlotBackend,
    replace=True,
)
register_backend(
    "test-request-slot-execution-dispatch-none-fused",
    _NoneFusedExecutionDispatchSlotBackend,
    replace=True,
)
register_backend(
    "test-request-slot-execution-dispatch-lifetime",
    _UnretainedExecutionDispatchSlotBackend,
    replace=True,
)
register_backend(
    "test-request-slot-start-fused-only",
    _StartFusedOnlySlotReferenceBackend,
    replace=True,
)
register_backend(
    "test-request-slot-poll-only",
    _PollOnlySlotReferenceBackend,
    replace=True,
)
register_backend(
    "test-request-slot-deferred-recycle",
    _DeferredRecycleSlotReferenceBackend,
    replace=True,
)
register_backend(
    "test-request-slot-fused-deferred-recycle",
    _DeferredRecycleFusedSlotReferenceBackend,
    replace=True,
)
register_backend(
    "test-request-slot-notification",
    _NotificationSlotReferenceBackend,
    replace=True,
)
register_backend(
    "test-request-slot-notification-missing",
    _MissingNotificationSlotBackend,
    replace=True,
)
register_backend(
    "test-request-slot-fused-unacknowledged",
    _UnacknowledgedFusedSlotBackend,
    replace=True,
)
register_backend(
    "test-request-slot-unacknowledged-start",
    _UnacknowledgedStartSlotBackend,
    replace=True,
)
register_backend(
    "test-request-slot-no-ack",
    _NormallyUnacknowledgedSlotBackend,
    replace=True,
)
register_backend(
    "test-request-slot-create-interrupt",
    _InterruptSlotCreationBackend,
    replace=True,
)
register_backend(
    "test-request-slot-cleanup-interrupt",
    _InterruptSlotCleanupBackend,
    replace=True,
)
register_backend(
    "test-request-slot-before-adoption-interrupt",
    _InterruptBeforeSlotAdoptionBackend,
    replace=True,
)


class EndpointTransferTest(TestCase):
    def test_backend_registration_lost_return_retry_is_exactly_idempotent(
        self,
    ) -> None:
        class Factory:
            def __call__(self, *args, adopt_backend, **kwargs) -> None:
                del args, adopt_backend, kwargs

        backend_name = "test-registration-lost-return"
        factory = Factory()
        register_backend(backend_name, factory, replace=True, factory_api_version=2)

        # Model a retry after Core committed the registry write but the caller
        # was interrupted before observing the return.
        register_backend(backend_name, factory, factory_api_version=2)
        registration = _backend._get_backend_registration(backend_name)
        self.assertIs(registration.factory, factory)
        self.assertEqual(registration.factory_api_version, 2)

        # Idempotence is deliberately exact; neither a new callable nor a
        # different contract can adopt an already-published name implicitly.
        with self.assertRaises(BusyError):
            register_backend(backend_name, factory, factory_api_version=1)
        with self.assertRaises(BusyError):
            register_backend(backend_name, Factory(), factory_api_version=2)

    def test_backend_factory_versions_are_explicit_and_never_introspected(self) -> None:
        self.assertEqual(BACKEND_FACTORY_API_VERSION, 2)
        for value, error_type in (
            (True, TypeError),
            (2.0, TypeError),
            (0, ValueError),
            (3, ValueError),
        ):
            with self.subTest(value=value), self.assertRaises(error_type):
                register_backend(
                    f"test-invalid-factory-version-{value!r}",
                    lambda *args: None,
                    factory_api_version=value,
                )

        events = []

        class Factory:
            @property
            def __signature__(self):
                raise AssertionError("factory signatures must not be inspected")

            def __call__(self, *args, adopt_backend, **kwargs) -> None:
                del kwargs
                backend = _ReferenceBackend._new_unactivated(*args)
                events.append("pure")
                self.assert_not_activated(backend)
                adopt_backend(backend)
                events.append("adopted")
                backend._activate()
                events.append("activated")

            @staticmethod
            def assert_not_activated(backend) -> None:
                if backend._activated:
                    raise AssertionError("native/global activation preceded adoption")

        backend_name = "test-factory-v2-no-introspection"
        register_backend(backend_name, Factory(), replace=True, factory_api_version=2)
        endpoint = Endpoint(
            "factory-v2", backend=backend_name, endpoint_id="factory-v2-order"
        )
        self.assertEqual(events, ["pure", "adopted", "activated"])
        self.assertTrue(endpoint._backend_receipt.published)
        endpoint.close()

        # The legacy callable contract remains the default.
        legacy_name = "test-factory-v1-compatible"
        register_backend(legacy_name, _ReferenceBackend, replace=True)
        legacy = Endpoint("factory-v1", backend=legacy_name)
        self.assertEqual(
            _backend._get_backend_registration(legacy_name).factory_api_version, 1
        )
        legacy.close()

    def test_backend_factory_v2_rejects_invalid_adoption_contracts(self) -> None:
        retained_callbacks = []

        for behavior in ("missing", "duplicate", "wrong", "non-none"):
            instances = []

            def factory(*args, adopt_backend, _behavior=behavior, **kwargs):
                del kwargs
                if _behavior == "missing":
                    retained_callbacks.append(adopt_backend)
                    return None
                if _behavior == "wrong":
                    adopt_backend(object())
                    return None
                backend = _ReferenceBackend._new_unactivated(*args)
                instances.append(backend)
                adopt_backend(backend)
                if _behavior == "duplicate":
                    adopt_backend(backend)
                backend._activate()
                return object() if _behavior == "non-none" else None

            backend_name = f"test-factory-v2-invalid-{behavior}"
            register_backend(backend_name, factory, replace=True, factory_api_version=2)
            with (
                self.subTest(behavior=behavior),
                self.assertRaises((BackendFailureError, TypeError)),
            ):
                Endpoint("invalid-v2", backend=backend_name)
            for instance in instances:
                self.assertTrue(instance._closed)

        late_backend = _ReferenceBackend(
            "late-adoption",
            "late-adoption",
            "late-incarnation",
            _types.ProgressMode.MANUAL,
            ThreadMode.SERIALIZED,
            {},
        )
        with self.assertRaisesRegex(BackendFailureError, "after its call returned"):
            retained_callbacks[0](late_backend)
        late_backend.close()

    def test_backend_factory_v1_wrong_type_is_foreign_and_never_touched(self) -> None:
        calls = []

        class ForeignResult:
            def close(self) -> None:
                calls.append("close")

        candidate = ForeignResult()
        backend_name = "test-factory-v1-foreign-result"
        register_backend(
            backend_name,
            lambda *args, **kwargs: candidate,
            replace=True,
            factory_api_version=1,
        )

        with self.assertRaisesRegex(TypeError, "must return a Backend instance"):
            Endpoint("foreign-result", backend=backend_name)
        self.assertEqual(calls, [])

    def test_backend_factory_v2_post_adoption_failure_exposes_recovery(self) -> None:
        instances = []

        class PendingCloseBackend(_ReferenceBackend):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def close(self) -> None:
                if not hasattr(self, "allow_close"):
                    self.allow_close = False
                if not self.allow_close:
                    raise BusyError("injected activation cleanup hold")
                super().close()

        def factory(*args, adopt_backend, **kwargs) -> None:
            del kwargs
            backend = PendingCloseBackend._new_unactivated(*args)
            instances.append(backend)
            adopt_backend(backend)
            backend._activate()
            raise KeyboardInterrupt("post-adoption activation cut")

        backend_name = "test-factory-v2-post-adoption-recovery"
        register_backend(backend_name, factory, replace=True, factory_api_version=2)
        with self.assertRaisesRegex(
            BackendFailureError, "recovery_endpoint"
        ) as context:
            Endpoint("interrupted-v2", backend=backend_name)
        recovery = context.exception.recovery_endpoint
        self.assertIs(recovery._backend, instances[0])
        self.assertFalse(recovery._backend_receipt.published)
        self.assertTrue(recovery._creation_failed)
        instances[0].allow_close = True
        recovery.close()
        self.assertTrue(recovery.closed)

    def test_backend_finalizer_orphans_are_isolated_by_exact_identity(self) -> None:
        blocked_ids = {"published-orphan"}
        instances = []

        class OrphanBackend(_ReferenceBackend):
            def close(self) -> None:
                if self._endpoint_id in blocked_ids:
                    raise BusyError("injected live backend child")
                super().close()

        def factory(*args, adopt_backend, **kwargs) -> None:
            del kwargs
            backend = OrphanBackend._new_unactivated(*args)
            instances.append(backend)
            adopt_backend(backend)
            backend._activate()

        backend_name = "test-factory-v2-published-orphan"
        register_backend(backend_name, factory, replace=True, factory_api_version=2)
        abandoned = Endpoint(
            "abandoned", backend=backend_name, endpoint_id="published-orphan"
        )
        abandoned_ref = weakref.ref(abandoned)
        del abandoned
        gc.collect()
        self.assertIsNone(abandoned_ref())
        self.assertIn(
            (backend_name, "published-orphan"),
            _api._ORPHANED_BACKEND_RECEIPTS,
        )

        # Auditing the failed orphan does not gate an unrelated endpoint id.
        unrelated = Endpoint(
            "unrelated", backend=backend_name, endpoint_id="fresh-identity"
        )
        unrelated.close()
        call_count = len(instances)
        with self.assertRaisesRegex(
            BackendFailureError, "abandoned published endpoint"
        ):
            Endpoint("collision", backend=backend_name, endpoint_id="published-orphan")
        self.assertEqual(len(instances), call_count)

        blocked_ids.clear()
        replacement = Endpoint(
            "replacement", backend=backend_name, endpoint_id="published-orphan"
        )
        replacement.close()
        self.assertNotIn(
            (backend_name, "published-orphan"),
            _api._ORPHANED_BACKEND_RECEIPTS,
        )

    def test_unpublished_factory_orphan_has_distinct_exact_id_diagnostic(self) -> None:
        block_cleanup = True
        fail_factory = True

        class OrphanBackend(_ReferenceBackend):
            def close(self) -> None:
                if block_cleanup:
                    raise BusyError("injected construction cleanup hold")
                super().close()

        def factory(*args, adopt_backend, **kwargs) -> None:
            del kwargs
            backend = OrphanBackend._new_unactivated(*args)
            adopt_backend(backend)
            backend._activate()
            if fail_factory:
                raise RuntimeError("injected post-activation failure")

        backend_name = "test-factory-v2-unpublished-orphan"
        register_backend(backend_name, factory, replace=True, factory_api_version=2)
        with self.assertRaises(BackendFailureError) as context:
            Endpoint(
                "unpublished", backend=backend_name, endpoint_id="unpublished-orphan"
            )
        recovery = context.exception.recovery_endpoint
        recovery_ref = weakref.ref(recovery)
        # The test harness retains the raised exception until the test returns;
        # sever its documented recovery edge to model an application dropping
        # both objects and allow the Endpoint weak finalizer to run now.
        context.exception.recovery_endpoint = None
        context.exception.__traceback__ = None
        context.exception.__cause__ = None
        del recovery
        del context
        gc.collect()
        self.assertIsNone(recovery_ref())

        fail_factory = False
        with self.assertRaisesRegex(
            BackendFailureError, "unpublished endpoint construction"
        ):
            Endpoint(
                "collision", backend=backend_name, endpoint_id="unpublished-orphan"
            )
        block_cleanup = False
        replacement = Endpoint(
            "replacement", backend=backend_name, endpoint_id="unpublished-orphan"
        )
        replacement.close()

    def test_endpoint_close_cut_after_receipt_commit_is_retryable(self) -> None:
        endpoint = Endpoint("receipt-close-cut")
        original_detach = Endpoint._detach_backend_finalizer
        interrupt = True

        def detach_with_cut(owner) -> None:
            nonlocal interrupt
            if interrupt:
                interrupt = False
                raise KeyboardInterrupt("after receipt close commit")
            original_detach(owner)

        with mock.patch.object(Endpoint, "_detach_backend_finalizer", detach_with_cut):
            with self.assertRaisesRegex(
                KeyboardInterrupt, "after receipt close commit"
            ):
                endpoint.close()
            self.assertTrue(endpoint._backend_receipt.closed)
            self.assertFalse(endpoint.closed)
            self.assertTrue(endpoint._backend_finalizer.alive)
            endpoint.close()
        self.assertTrue(endpoint.closed)
        self.assertFalse(endpoint._backend_finalizer.alive)

    def test_endpoint_close_does_not_implicitly_progress(self) -> None:
        class NoCloseProgressBackend(_ReferenceBackend):
            def progress(self) -> None:
                raise AssertionError("close must not inject provider progress")

        backend_name = "test-close-without-progress"
        register_backend(backend_name, NoCloseProgressBackend, replace=True)
        endpoint = Endpoint("no-close-progress", backend=backend_name)
        endpoint.close()
        self.assertTrue(endpoint.closed)

    def test_backend_receipt_empty_slots_preserve_unslotted_subclasses(self) -> None:
        self.assertEqual(BackendWork.__slots__, ())
        self.assertEqual(BackendRequestSlot.__slots__, ())

        work = _PendingWork()
        slot = _TestRequestSlot()
        work.extension_value = "work"
        slot.extension_value = "slot"

        self.assertEqual(work.__dict__["extension_value"], "work")
        self.assertEqual(slot.__dict__["extension_value"], "slot")

    def test_endpoint_close_retries_without_progress_after_interruption(self) -> None:
        endpoint = Endpoint("endpoint", backend="test-interrupt-after-close")

        with self.assertRaises(KeyboardInterrupt):
            endpoint.close()
        self.assertFalse(endpoint.closed)
        self.assertTrue(endpoint._closing)

        endpoint.close()
        self.assertTrue(endpoint.closed)

    def test_registration_factory_interruption_rolls_back_adopted_handle(self) -> None:
        endpoint = Endpoint("endpoint")
        original_register = endpoint._backend.register

        def register_then_interrupt(*args, **kwargs) -> None:
            original_register(*args, **kwargs)
            raise KeyboardInterrupt

        endpoint._backend.register = register_then_interrupt
        with self.assertRaises(KeyboardInterrupt):
            endpoint.register(bytearray(4), name="buffer")

        self.assertEqual(endpoint.open_registrations, ())
        self.assertFalse(endpoint._backend._registrations)
        endpoint.close()

    def test_failed_registration_rollback_is_publicly_recoverable(self) -> None:
        endpoint = Endpoint("endpoint")
        original_register = endpoint._backend.register
        original_deregister = endpoint._backend.deregister
        fail_cleanup = True

        def register_then_interrupt(*args, **kwargs) -> None:
            original_register(*args, **kwargs)
            raise KeyboardInterrupt

        def deregister_once(handle) -> None:
            nonlocal fail_cleanup
            if fail_cleanup:
                fail_cleanup = False
                raise RuntimeError("injected cleanup failure")
            original_deregister(handle)

        endpoint._backend.register = register_then_interrupt
        endpoint._backend.deregister = deregister_once
        with self.assertRaisesRegex(BackendFailureError, "cleanup is pending"):
            endpoint.register(bytearray(4), name="buffer")

        (registration,) = endpoint.open_registrations
        with self.assertRaisesRegex(BackendFailureError, "cleanup only"):
            registration.region()
        registration.close()
        self.assertEqual(endpoint.open_registrations, ())
        endpoint.close()

    def test_peer_and_plan_factory_interruptions_roll_back(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(4), name="local")
        remote = destination.register(bytearray(4), name="remote")
        metadata = bytes(destination.export_metadata())
        original_import = source._backend.import_peer

        def import_then_interrupt(*args, **kwargs) -> None:
            original_import(*args, **kwargs)
            raise KeyboardInterrupt

        source._backend.import_peer = import_then_interrupt
        with self.assertRaises(KeyboardInterrupt):
            source.import_peer(metadata)
        self.assertEqual(source.open_peers, ())
        self.assertFalse(source._backend._peers)

        source._backend.import_peer = original_import
        peer = source.import_peer(metadata)
        original_prepare = source._backend.prepare

        def prepare_then_interrupt(*args, **kwargs) -> None:
            original_prepare(*args, **kwargs)
            raise KeyboardInterrupt

        source._backend.prepare = prepare_then_interrupt
        with self.assertRaises(KeyboardInterrupt):
            source.prepare(local=[local.region()], remote=[peer.region("remote")])
        self.assertEqual(source.open_plans, ())
        self.assertFalse(source._backend._plans)

        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_reference_write_and_metadata_round_trip(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        source_buffer = bytearray(b"abcdefgh")
        destination_buffer = bytearray(8)
        source_registration = source.register(source_buffer, name="blocks")
        destination_registration = destination.register(
            destination_buffer, name="blocks"
        )

        encoded = bytes(destination.export_metadata())
        self.assertEqual(bytes(EndpointMetadata.from_bytes(encoded)), encoded)
        peer = source.import_peer(encoded)
        plan = source.prepare(
            local=[source_registration.region()],
            remote=[peer.region("blocks")],
        )
        work = source.write(plan)
        self.assertTrue(work.test())
        self.assertEqual(work.state, WorkState.COMPLETED)
        self.assertEqual(destination_buffer, source_buffer)

        work.close()
        plan.close()
        peer.close()
        source_registration.close()
        destination_registration.close()
        source.close()
        destination.close()

    def test_reference_read(self) -> None:
        sender = Endpoint("sender")
        receiver = Endpoint("receiver")
        source_buffer = bytearray(b"remote")
        destination_buffer = bytearray(6)
        source_registration = sender.register(source_buffer, name="weights")
        destination_registration = receiver.register(destination_buffer, name="weights")
        peer = receiver.import_peer(bytes(sender.export_metadata()))
        plan = receiver.prepare(
            local=[destination_registration.region()],
            remote=[peer.region("weights")],
        )

        work = receiver.read(plan)
        work.wait()
        self.assertEqual(destination_buffer, source_buffer)

        work.close()
        plan.close()
        peer.close()
        destination_registration.close()
        source_registration.close()
        receiver.close()
        sender.close()

    def test_bound_transfer_full_reuse_and_provider_tokens(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        source_buffer = bytearray(b"data")
        destination_buffer = bytearray(4)
        local = source.register(source_buffer, name="buffer")
        remote = destination.register(destination_buffer, name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        tokens: list[object | None] = []
        original_submit = source._backend.submit

        def capture_submit(*args, **kwargs) -> None:
            tokens.append(kwargs["reuse_token"])
            original_submit(*args, **kwargs)

        source._backend.submit = capture_submit
        first = plan.bind("write")
        self.assertIsInstance(first, BoundTransfer)
        self.assertEqual(first.op, TransferOp.WRITE)
        self.assertEqual(source.open_bound_transfers, (first,))

        for value in (b"one!", b"two!"):
            source_buffer[:] = value
            work = first.submit()
            work.wait()
            self.assertEqual(destination_buffer, value)
            work.close()

        second = plan.bind(TransferOp.WRITE)
        work = second.submit()
        work.close()
        reader = plan.bind(TransferOp.READ)
        destination_buffer[:] = b"back"
        work = reader.submit()
        work.wait()
        self.assertEqual(source_buffer, b"back")
        work.close()

        convenience_start = len(tokens)
        for _ in range(2):
            work = source.write(plan)
            work.close()

        self.assertIs(tokens[0], tokens[1])
        self.assertIsNot(tokens[0], tokens[2])
        self.assertIsNot(tokens[0], tokens[3])
        self.assertIsNot(tokens[2], tokens[3])
        self.assertIs(tokens[convenience_start], tokens[convenience_start + 1])
        self.assertIsNot(tokens[0], tokens[convenience_start])

        reader.close()
        second.close()
        first.close()
        self.assertEqual(source.open_bound_transfers, ())
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_bound_transfer_canonicalizes_indexed_selection_and_notification(
        self,
    ) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        source_buffer = bytearray(b"a0b0c0d0")
        destination_buffer = bytearray(b"--------")
        local = source.register(source_buffer, name="source")
        remote = destination.register(destination_buffer, name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=2, count=4)],
            remote=[peer.region("destination", 0, 1, stride=2, count=4)],
            indexed=True,
        )
        local_indices = [3, 1]
        remote_indices = [0, 2]
        notification = bytearray(b"ready")
        bound = plan.bind(
            TransferOp.WRITE,
            local_indices,
            remote_indices,
            notification,
        )
        local_indices[:] = [0]
        remote_indices[:] = [1]
        notification[:] = b"later"

        work = bound.submit()
        work.wait()
        self.assertEqual(destination_buffer, b"d---b---")
        self.assertEqual(
            [item.payload for item in destination.poll_notifications()],
            [b"ready"],
        )

        work.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_bound_transfer_enforces_work_plan_endpoint_close_order(self) -> None:
        source = Endpoint("source", backend="test-pending")
        destination = Endpoint("destination", backend="test-pending")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)

        with self.assertRaisesRegex(BusyError, "bound transfers"):
            plan.close()
        with self.assertRaises(BusyError):
            source.close()
        work = bound.submit()
        with self.assertRaisesRegex(BusyError, "open work"):
            bound.close()
        with self.assertRaisesRegex(BusyError, "open work"):
            plan.close()
        self.assertIn(work, source.open_works)
        self.assertIn(bound, source.open_bound_transfers)

        self.assertTrue(work.cancel())
        work.close()
        bound.close()
        self.assertTrue(bound.closed)
        with self.assertRaises(ClosedError):
            bound.submit()
        self.assertEqual(source.open_bound_transfers, ())

        plan.close()
        with self.assertRaises(ClosedError):
            plan.bind(TransferOp.WRITE)
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_bound_transfer_preserves_adoption_interruption_recovery(self) -> None:
        source = Endpoint("source", backend="test-adopt-then-interrupt")
        destination = Endpoint("destination", backend="test-adopt-then-interrupt")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)

        work = bound.submit()
        self.assertEqual(source.open_works, (work,))
        work.cancel()
        self.assertEqual(work.state, WorkState.FAILED)
        self.assertIsInstance(work.error, KeyboardInterrupt)
        work.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_bound_transfer_rolls_back_interruption_before_adoption(self) -> None:
        source = Endpoint("source", backend="test-interrupt")
        destination = Endpoint("destination", backend="test-interrupt")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)

        with self.assertRaises(KeyboardInterrupt):
            bound.submit()
        self.assertEqual(source.open_works, ())
        self.assertFalse(bound._works)
        self.assertFalse(plan._submitting)
        self.assertFalse(plan._works)

        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_bound_transfer_close_retry_repairs_post_commit_interruption(self) -> None:
        class InterruptingBoundTransfer(BoundTransfer):
            __slots__ = ("interrupt_close_commit",)

            def __setattr__(self, name: str, value: object) -> None:
                object.__setattr__(self, name, value)
                if (
                    name == "_closed"
                    and value is True
                    and getattr(self, "interrupt_close_commit", False)
                ):
                    object.__setattr__(self, "interrupt_close_commit", False)
                    raise KeyboardInterrupt

        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        with mock.patch.object(_api, "BoundTransfer", InterruptingBoundTransfer):
            bound = plan.bind(TransferOp.WRITE)
        bound.interrupt_close_commit = True

        with self.assertRaises(KeyboardInterrupt):
            bound.close()
        self.assertTrue(bound.closed)
        # The parent blockers were removed before the interrupted commit, but
        # the retained parent reference lets the same-operation retry finish.
        self.assertNotIn(bound, plan._bindings)
        self.assertNotIn(bound, source.open_bound_transfers)
        self.assertIs(bound._plan, plan)

        bound.close()
        self.assertIsNone(bound._plan)
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_resource_close_retries_finish_post_commit_epilogues(self) -> None:
        class InterruptOnCloseCommit:
            __slots__ = ()

            def __setattr__(self, name: str, value: object) -> None:
                object.__setattr__(self, name, value)
                if (
                    name == "_closed"
                    and value is True
                    and getattr(self, "interrupt_close_commit", False)
                ):
                    object.__setattr__(self, "interrupt_close_commit", False)
                    raise KeyboardInterrupt

        class InterruptingRegistration(InterruptOnCloseCommit, _api.Registration):
            __slots__ = ("interrupt_close_commit",)

        class InterruptingPeer(InterruptOnCloseCommit, _api.Peer):
            __slots__ = ("interrupt_close_commit",)

        class InterruptingPlan(InterruptOnCloseCommit, _api.TransferPlan):
            __slots__ = ("interrupt_close_commit",)

        class InterruptingWork(InterruptOnCloseCommit, _api.Work):
            __slots__ = ("interrupt_close_commit",)

        source = Endpoint("source")
        destination = Endpoint("destination")
        with mock.patch.object(_api, "Registration", InterruptingRegistration):
            local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        with mock.patch.object(_api, "Peer", InterruptingPeer):
            peer = source.import_peer(bytes(destination.export_metadata()))
        with mock.patch.object(_api, "TransferPlan", InterruptingPlan):
            plan = source.prepare(
                local=[local.region()], remote=[peer.region("buffer")]
            )
        with mock.patch.object(_api, "Work", InterruptingWork):
            work = source.write(plan)
        work.wait()

        for resource in (work, plan, peer, local):
            resource.interrupt_close_commit = True
            with self.assertRaises(KeyboardInterrupt):
                resource.close()
            self.assertTrue(resource.closed)

            if resource is work:
                self.assertIn(work, source.open_works)
            elif resource is plan:
                self.assertIn(plan, source.open_plans)
                self.assertTrue(plan._origins)
            elif resource is peer:
                self.assertIn(peer, source.open_peers)
                self.assertIsNotNone(peer._handle)
            else:
                self.assertIn(local, source.open_registrations)
                self.assertIsNotNone(local._memory.owner)

            resource.close()

        self.assertEqual(source.open_works, ())
        self.assertEqual(source.open_plans, ())
        self.assertEqual(source.open_peers, ())
        self.assertEqual(source.open_registrations, ())
        remote.close()
        source.close()
        destination.close()

    def test_bound_host_wait_establishes_ordering_on_every_submit(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        local_region = local.region()
        local_region._device = "cuda:0"
        plan = source.prepare(local=[local_region], remote=[peer.region("buffer")])
        source._capabilities = Capabilities(
            read=True,
            write=True,
            manual_progress=True,
            background_progress=True,
            cuda_ordering_modes=frozenset(
                {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
            ),
        )
        actions: list[str] = []

        class FakeEvent:
            device = torch.device("cuda:0")

            def synchronize(self) -> None:
                actions.append("wait")

        with (
            mock.patch.object(torch.cuda, "Event", FakeEvent),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ),
        ):
            bound = plan.bind(
                TransferOp.WRITE,
                cuda_ordering=CudaOrderingMode.HOST_WAIT,
                ready_event=FakeEvent(),
            )
            self.assertEqual(actions, [])
            for _ in range(2):
                work = bound.submit()
                work.close()
        self.assertEqual(actions, ["wait", "wait"])

        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_backend_request_slot_default_start_and_poll_is_bounded(self) -> None:
        class CountingRequestSlot(_TestRequestSlot):
            def __init__(self) -> None:
                super().__init__()
                self.next_state = WorkState.RUNNING
                self.poll_count = 0

            def poll_state(self) -> WorkState:
                self.poll_count += 1
                return self.state

        backend_slot = CountingRequestSlot()
        self.assertEqual(
            backend_slot.start_and_poll(max_polls=3, timeout_ns=None),
            WorkState.RUNNING,
        )
        self.assertEqual(backend_slot.start_count, 1)
        self.assertEqual(backend_slot.poll_count, 3)

        backend_slot = CountingRequestSlot()
        self.assertEqual(
            backend_slot.start_and_poll(max_polls=64, timeout_ns=0),
            WorkState.RUNNING,
        )
        self.assertEqual(backend_slot.poll_count, 1)
        backend_slot = CountingRequestSlot()
        self.assertEqual(
            backend_slot.start_and_poll(max_polls=0, timeout_ns=None),
            WorkState.RUNNING,
        )
        self.assertEqual(backend_slot.start_count, 1)
        self.assertEqual(backend_slot.poll_count, 0)
        with self.assertRaisesRegex(ValueError, "max_polls"):
            backend_slot.start_and_poll(max_polls=-1, timeout_ns=None)
        with self.assertRaisesRegex(ValueError, "max_polls"):
            backend_slot.start_and_poll(max_polls=1 << 63, timeout_ns=None)
        with self.assertRaisesRegex(TypeError, "max_polls"):
            backend_slot.start_and_poll(max_polls=True, timeout_ns=None)
        with self.assertRaisesRegex(ValueError, "timeout_ns"):
            backend_slot.start_and_poll(max_polls=1, timeout_ns=-1)
        with self.assertRaisesRegex(ValueError, "timeout_ns"):
            backend_slot.start_and_poll(max_polls=1, timeout_ns=1 << 63)
        with self.assertRaisesRegex(TypeError, "timeout_ns"):
            backend_slot.start_and_poll(max_polls=1, timeout_ns=True)

        backend_slot = CountingRequestSlot()
        backend_slot.start = mock.Mock(return_value=object())
        with self.assertRaisesRegex(TypeError, "start must return None"):
            backend_slot.start_and_poll(max_polls=1, timeout_ns=None)

    def test_backend_request_slot_default_poll_bounded_contract(self) -> None:
        class CountingRequestSlot(_TestRequestSlot):
            def __init__(self) -> None:
                super().__init__()
                self._state = WorkState.RUNNING
                self.poll_count = 0

            def poll_state(self) -> WorkState:
                self.poll_count += 1
                assert isinstance(self._state, WorkState)
                return self._state

        backend_slot = CountingRequestSlot()
        self.assertEqual(
            backend_slot.poll_bounded(
                max_polls=3, timeout_ns=None, timeout_check_interval=32
            ),
            WorkState.RUNNING,
        )
        self.assertEqual(backend_slot.poll_count, 3)

        backend_slot = CountingRequestSlot()
        with mock.patch(
            "torch.distributed._transfer._backend.time.monotonic_ns",
            side_effect=[10, 10],
        ):
            self.assertEqual(
                backend_slot.poll_bounded(
                    max_polls=64, timeout_ns=0, timeout_check_interval=32
                ),
                WorkState.RUNNING,
            )
        self.assertEqual(backend_slot.poll_count, 1)

        backend_slot = CountingRequestSlot()
        with mock.patch(
            "torch.distributed._transfer._backend.time.monotonic_ns",
            side_effect=[10, 10, 12],
        ):
            self.assertEqual(
                backend_slot.poll_bounded(
                    max_polls=64, timeout_ns=1, timeout_check_interval=1
                ),
                WorkState.RUNNING,
            )
        self.assertEqual(backend_slot.poll_count, 2)

        backend_slot = CountingRequestSlot()
        with mock.patch(
            "torch.distributed._transfer._backend.time.monotonic_ns",
            side_effect=[10, 10, 12],
        ):
            self.assertEqual(
                backend_slot.poll_bounded(
                    max_polls=64, timeout_ns=1, timeout_check_interval=32
                ),
                WorkState.RUNNING,
            )
        self.assertEqual(backend_slot.poll_count, 32)

        for max_polls, timeout_ns, timeout_check_interval, error_type in (
            (0, None, 32, ValueError),
            (True, None, 32, TypeError),
            (1, -1, 32, ValueError),
            (1 << 63, None, 32, ValueError),
            (1, 1 << 63, 32, ValueError),
            (1, True, 32, TypeError),
            (1, None, 0, ValueError),
            (1, None, 1 << 63, ValueError),
            (1, None, True, TypeError),
        ):
            with self.assertRaises(error_type):
                backend_slot.poll_bounded(
                    max_polls=max_polls,
                    timeout_ns=timeout_ns,
                    timeout_check_interval=timeout_check_interval,
                )

    def test_transfer_slot_capability_is_validated_and_opt_in(self) -> None:
        with self.assertRaisesRegex(TypeError, "persistent_request_slots"):
            Capabilities(persistent_request_slots=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "fused_request_slot_start_poll"):
            Capabilities(
                persistent_request_slots=True,
                fused_request_slot_start_poll=1,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "requires persistent_request_slots"):
            Capabilities(fused_request_slot_start_poll=True)
        with self.assertRaisesRegex(TypeError, "fused_request_slot_poll"):
            Capabilities(
                persistent_request_slots=True,
                fused_request_slot_poll=1,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "requires persistent_request_slots"):
            Capabilities(fused_request_slot_poll=True)
        with self.assertRaisesRegex(
            TypeError, "deferred_successful_request_slot_recycle"
        ):
            Capabilities(
                persistent_request_slots=True,
                deferred_successful_request_slot_recycle=1,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "requires persistent_request_slots"):
            Capabilities(deferred_successful_request_slot_recycle=True)
        with self.assertRaisesRegex(TypeError, "request_slot_notification_overrides"):
            Capabilities(
                persistent_request_slots=True,
                attached_notifications=True,
                request_slot_notification_overrides=1,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(TypeError, "fused_prevalidated_submit_poll"):
            Capabilities(fused_prevalidated_submit_poll=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "fused_work_poll"):
            Capabilities(fused_work_poll=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "grouped_notification_receive"):
            Capabilities(grouped_notification_receive=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "requires notifications"):
            Capabilities(grouped_notification_receive=True)
        with self.assertRaisesRegex(TypeError, "synchronous_notification_send"):
            Capabilities(synchronous_notification_send=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "requires standalone_notifications"):
            Capabilities(synchronous_notification_send=True)
        synchronous_notifications = Capabilities(
            standalone_notifications=True,
            standalone_notification_completion=NotificationCompletion.LOCAL_ACCEPTED,
            synchronous_notification_send=True,
        )
        self.assertTrue(synchronous_notifications.synchronous_notification_send)
        self.assertTrue(
            Capabilities(
                attached_notifications=True,
                grouped_notification_receive=True,
            ).grouped_notification_receive
        )
        with self.assertRaisesRegex(ValueError, "requires persistent_request_slots"):
            Capabilities(
                attached_notifications=True,
                request_slot_notification_overrides=True,
            )
        with self.assertRaisesRegex(ValueError, "requires attached_notifications"):
            Capabilities(
                persistent_request_slots=True,
                request_slot_notification_overrides=True,
            )
        poll_only = Capabilities(
            persistent_request_slots=True,
            fused_request_slot_poll=True,
        )
        self.assertTrue(poll_only.fused_request_slot_poll)
        self.assertFalse(poll_only.fused_request_slot_start_poll)
        self.assertFalse(poll_only.deferred_successful_request_slot_recycle)

        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)

        self.assertFalse(source.capabilities.persistent_request_slots)
        self.assertFalse(source.capabilities.fused_request_slot_start_poll)
        self.assertFalse(source.capabilities.fused_request_slot_poll)
        self.assertFalse(source.capabilities.deferred_successful_request_slot_recycle)
        self.assertFalse(source.capabilities.request_slot_notification_overrides)
        self.assertFalse(source.capabilities.fused_prevalidated_submit_poll)
        self.assertFalse(source.capabilities.fused_work_poll)
        with self.assertRaisesRegex(UnsupportedError, "persistent request slots"):
            bound.create_slot()
        self.assertEqual(source.open_transfer_slots, ())

        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_request_local_notification_slot_is_explicit_reusable_and_work_free(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-notification")
        destination = Endpoint("destination", backend="test-request-slot-notification")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)

        with self.assertRaisesRegex(TypeError, "must be a bool"):
            bound.create_slot(request_local_notifications=1)  # type: ignore[arg-type]

        fixed_slot = bound.create_slot()
        self.assertIs(type(fixed_slot), TransferSlot)
        fixed_slot.close()

        slot = bound.create_slot(request_local_notifications=True)
        self.assertIs(type(slot), RequestLocalNotificationSlot)
        self.assertIsInstance(slot, TransferSlot)
        self.assertEqual(source.open_works, ())
        backend_slot = source._backend.request_slots[-1]

        with self.assertRaisesRegex(TypeError, "notification"):
            slot.start()  # type: ignore[call-arg]
        self.assertEqual(slot.generation, 0)
        self.assertEqual(backend_slot.start_count, 0)

        released_view = memoryview(bytearray(b"released"))
        released_view.release()
        for invalid, error_type, message in (
            (b"", ValueError, "must be non-empty"),
            (b"x" * (_api._MAX_NOTIFICATION_BYTES + 1), ValueError, "size limit"),
            ("wrong", TypeError, "bytes-like"),
            (released_view, ValueError, "unavailable"),
        ):
            with self.assertRaisesRegex(error_type, message):
                slot.start(notification=invalid)  # type: ignore[arg-type]
        self.assertEqual(slot.generation, 0)
        self.assertEqual(backend_slot.start_count, 0)

        mutable = bytearray(b"alpha")
        slot.start(notification=mutable)
        mutable[:] = b"other"
        self.assertEqual(backend_slot.notification_starts, [b"alpha"])
        self.assertEqual(slot.state, WorkState.COMPLETED)
        slot.recycle()

        view_owner = bytearray(b"view")
        slot.start(notification=memoryview(view_owner))
        view_owner[:] = b"xxxx"
        self.assertEqual(backend_slot.notification_starts[-1], b"view")
        slot.recycle()

        generation = slot.execute(notification=b"b\x00\xff")
        self.assertEqual(generation, 3)
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(
            backend_slot.notification_fused_calls,
            [(b"b\x00\xff", 64, None)],
        )

        self.assertEqual(slot.execute(notification=None), 4)
        self.assertEqual(
            backend_slot.notification_fused_calls[-1],
            (None, 64, None),
        )
        self.assertEqual(source.open_works, ())

        prior_calls = len(backend_slot.notification_fused_calls)
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            slot.execute(notification=b"")
        self.assertEqual(slot.generation, 4)
        self.assertEqual(len(backend_slot.notification_fused_calls), prior_calls)

        tagged = plan.bind(TransferOp.WRITE, notification=b"fixed")
        with self.assertRaisesRegex(ValueError, "notification=None"):
            tagged.create_slot(request_local_notifications=True)
        tagged.close()

        second = bound.create_slot(request_local_notifications=True)
        slot.start(notification=b"one")
        second.start(notification=b"two")
        self.assertEqual(slot.state, WorkState.COMPLETED)
        self.assertEqual(second.state, WorkState.COMPLETED)
        slot.recycle()
        second.recycle()
        second.close()
        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_request_local_notification_slots_cover_indexed_write_and_read(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-notification")
        destination = Endpoint("destination", backend="test-request-slot-notification")
        local = source.register(bytearray(b"abcdefgh"), name="buffer")
        remote = destination.register(bytearray(8), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 4, stride=4, count=2)],
            remote=[peer.region("buffer", 0, 4, stride=4, count=2)],
            indexed=True,
        )

        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=[1],
            remote_indices=[0],
        )
        write_bound = selection.bind()
        write_slot = write_bound.create_slot(request_local_notifications=True)
        self.assertEqual(write_slot.execute(notification=b"indexed"), 1)

        read_bound = plan.bind(
            TransferOp.READ,
            local_indices=[0],
            remote_indices=[1],
        )
        read_slot = read_bound.create_slot(request_local_notifications=True)
        self.assertEqual(read_slot.execute(notification=b"read"), 1)

        read_slot.close()
        read_bound.close()
        write_slot.close()
        write_bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_request_local_notification_slot_is_capability_and_spi_gated(
        self,
    ) -> None:
        for backend, message in (
            ("test-request-slot", "does not support request-local"),
            ("test-request-slot-notification-missing", "without adoption"),
        ):
            source = Endpoint("source", backend=backend)
            destination = Endpoint("destination", backend=backend)
            local = source.register(bytearray(b"data"), name="buffer")
            remote = destination.register(bytearray(4), name="buffer")
            peer = source.import_peer(bytes(destination.export_metadata()))
            plan = source.prepare(
                local=[local.region()], remote=[peer.region("buffer")]
            )
            bound = plan.bind(TransferOp.WRITE)
            with self.assertRaisesRegex(
                (UnsupportedError, BackendFailureError), message
            ):
                bound.create_slot(request_local_notifications=True)
            self.assertEqual(source.open_transfer_slots, ())
            bound.close()
            plan.close()
            peer.close()
            local.close()
            remote.close()
            source.close()
            destination.close()

    def test_transfer_slot_reuses_one_public_resource_and_counts_generations(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot")
        destination = Endpoint("destination", backend="test-request-slot")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()

        self.assertIsInstance(slot, TransferSlot)
        self.assertEqual(source.open_transfer_slots, (slot,))
        self.assertEqual(source.open_works, ())
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(slot.generation, 0)

        self.assertIsNone(slot.start())
        self.assertTrue(slot.active)
        self.assertEqual(slot.generation, 1)
        self.assertEqual(slot.state, WorkState.COMPLETED)
        self.assertIs(slot.wait(), slot)
        backend_slot = source._backend.request_slots[0]
        self.assertIs(slot.completion_event, backend_slot.completion)
        with self.assertRaisesRegex(BusyError, "unrecycled generation"):
            slot.start()
        with self.assertRaisesRegex(BusyError, "unrecycled generation"):
            slot.close()

        slot.recycle()
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(slot.generation, 1)
        self.assertIsNone(slot.start())
        self.assertEqual(slot.generation, 2)
        self.assertEqual(slot.state, WorkState.COMPLETED)
        slot.recycle()
        self.assertEqual(backend_slot.start_count, 2)
        self.assertEqual(backend_slot.recycle_count, 2)
        self.assertEqual(source.open_works, ())

        slot.close()
        self.assertTrue(slot.closed)
        self.assertEqual(source.open_transfer_slots, ())
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_execution_dispatch_is_cached_and_wrapper_owned(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-execution-dispatch")
        destination = Endpoint(
            "destination", backend="test-request-slot-execution-dispatch"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        dispatch = source._backend.execution_dispatches[0]
        self.assertIsNone(slot._fast_execute)

        self.assertEqual(
            dispatch.attribute_reads,
            {name: 1 for name in _TestExecutionDispatch._OPERATION_NAMES},
        )
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_results = [WorkState.COMPLETED]
        self.assertIs(slot.start_and_wait(), slot)
        self.assertIs(slot.completion_event, backend_slot.completion)
        slot.recycle()

        backend_slot.next_state = WorkState.RUNNING
        slot.start()
        self.assertEqual(slot.state, WorkState.RUNNING)
        self.assertTrue(slot.cancel())
        self.assertEqual(slot.state, WorkState.CANCELLED)
        slot.recycle()

        failure = RuntimeError("wrapper-owned diagnostic")
        backend_slot.next_state = WorkState.FAILED
        backend_slot.next_error = failure
        slot.start()
        self.assertIs(slot.error, failure)
        slot.recycle()
        self.assertEqual(
            dispatch.attribute_reads,
            {name: 1 for name in _TestExecutionDispatch._OPERATION_NAMES},
        )

        slot.close()
        self.assertEqual(backend_slot.close_count, 1)
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_execution_dispatch_respects_capability_gating(
        self,
    ) -> None:
        source = Endpoint(
            "source", backend="test-request-slot-execution-dispatch-gated"
        )
        destination = Endpoint(
            "destination", backend="test-request-slot-execution-dispatch-gated"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        dispatch = source._backend.execution_dispatches[0]

        self.assertEqual(dispatch.attribute_reads["start"], 1)
        self.assertEqual(dispatch.attribute_reads["poll_state"], 1)
        self.assertEqual(dispatch.attribute_reads["cancel"], 1)
        self.assertEqual(dispatch.attribute_reads["recycle"], 1)
        self.assertEqual(dispatch.attribute_reads["start_and_poll"], 0)
        self.assertEqual(dispatch.attribute_reads["poll_bounded"], 0)
        slot.start()
        slot.recycle()
        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_strongly_retains_execution_dispatch_until_close(
        self,
    ) -> None:
        source = Endpoint(
            "source", backend="test-request-slot-execution-dispatch-lifetime"
        )
        destination = Endpoint(
            "destination", backend="test-request-slot-execution-dispatch-lifetime"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        dispatch_ref = source._backend.execution_dispatch_ref

        gc.collect()
        self.assertIsNotNone(dispatch_ref())
        self.assertIs(slot._backend_execution_dispatch, dispatch_ref())
        slot.close()
        gc.collect()
        self.assertIsNone(dispatch_ref())

        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_execution_dispatch_rejects_noncallable_operation(
        self,
    ) -> None:
        for backend in (
            "test-request-slot-execution-dispatch-invalid",
            "test-request-slot-execution-dispatch-none-start",
            "test-request-slot-execution-dispatch-none-fused",
        ):
            with self.subTest(backend=backend):
                source = Endpoint("source", backend=backend)
                destination = Endpoint("destination", backend=backend)
                local = source.register(bytearray(b"data"), name="buffer")
                remote = destination.register(bytearray(4), name="buffer")
                peer = source.import_peer(bytes(destination.export_metadata()))
                plan = source.prepare(
                    local=[local.region()], remote=[peer.region("buffer")]
                )
                bound = plan.bind(TransferOp.WRITE)

                with self.assertRaisesRegex(BackendFailureError, "without adoption"):
                    bound.create_slot()
                self.assertEqual(source.open_transfer_slots, ())
                self.assertEqual(bound._slots, set())

                bound.close()
                plan.close()
                peer.close()
                local.close()
                remote.close()
                source.close()
                destination.close()

    def test_transfer_slot_pending_cancel_failure_and_recycle_contract(self) -> None:
        source = Endpoint("source", backend="test-request-slot")
        destination = Endpoint("destination", backend="test-request-slot")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        with self.assertRaisesRegex(BusyError, "no generation"):
            slot.recycle()
        with self.assertRaisesRegex(BusyError, "no active generation"):
            slot.wait()
        backend_slot.next_state = WorkState.PENDING
        slot.start()
        self.assertFalse(slot.test())
        with self.assertRaises(TransferTimeoutError):
            slot.wait(0)
        with self.assertRaisesRegex(BusyError, "non-terminal"):
            slot.recycle()
        self.assertTrue(slot.cancel())
        self.assertEqual(slot.state, WorkState.CANCELLED)
        self.assertIsInstance(slot.error, TransferCancelledError)
        with self.assertRaises(TransferCancelledError):
            slot.wait()
        slot.recycle()

        failure = RuntimeError("injected slot failure")
        backend_slot.next_state = WorkState.FAILED
        backend_slot.next_error = failure
        slot.start()
        self.assertEqual(slot.state, WorkState.FAILED)
        self.assertIs(slot.error, failure)
        with self.assertRaises(BackendFailureError) as context:
            slot.wait()
        self.assertIs(context.exception.__cause__, failure)
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_start_and_wait_fuses_then_falls_back_or_times_out(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-fused")
        destination = Endpoint("destination", backend="test-request-slot-fused")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        self.assertTrue(source.capabilities.fused_request_slot_start_poll)
        self.assertIs(slot.start_and_wait(), slot)
        self.assertEqual(backend_slot.fused_calls, [(64, None)])
        self.assertEqual(backend_slot.poll_count, 0)
        self.assertEqual(slot.state, WorkState.COMPLETED)
        slot.recycle()

        self.assertIs(slot.start_and_wait(1 << 80), slot)
        self.assertEqual(backend_slot.fused_calls[-1], (64, (1 << 63) - 1))
        slot.recycle()

        backend_slot.next_state = WorkState.RUNNING
        backend_slot.poll_states = [WorkState.COMPLETED]
        self.assertIs(slot.start_and_wait(1), slot)
        max_polls, timeout_ns = backend_slot.fused_calls[-1]
        self.assertEqual(max_polls, 64)
        self.assertIsInstance(timeout_ns, int)
        self.assertGreaterEqual(timeout_ns, 0)
        self.assertLessEqual(timeout_ns, 1_000_000_000)
        self.assertEqual(backend_slot.poll_count, 1)
        slot.recycle()

        backend_slot.next_state = WorkState.RUNNING
        with self.assertRaises(TransferTimeoutError):
            slot.start_and_wait(0)
        self.assertEqual(backend_slot.fused_calls[-1], (64, 0))
        self.assertTrue(slot.active)
        self.assertEqual(backend_slot.poll_count, 1)
        self.assertEqual(slot.state, WorkState.RUNNING)
        self.assertTrue(slot.cancel())
        slot.recycle()

        generation = slot.generation
        for timeout, error_type in (
            (True, TypeError),
            ("one", TypeError),
            (-1, ValueError),
            (float("inf"), ValueError),
        ):
            with self.assertRaises(error_type):
                slot.start_and_wait(timeout)  # type: ignore[arg-type]
        self.assertEqual(slot.generation, generation)
        self.assertFalse(slot.active)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_start_and_wait_uses_bounded_poll_chunks(self) -> None:
        source = Endpoint(
            "source", backend="test-request-slot-fused", thread_mode="single"
        )
        destination = Endpoint(
            "destination", backend="test-request-slot-fused", thread_mode="single"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_results = [
            WorkState.RUNNING,
            WorkState.RUNNING,
            WorkState.COMPLETED,
        ]

        with mock.patch.object(
            source,
            "_backend_request_slot_poll_bounded",
            side_effect=AssertionError(
                "SINGLE bounded polling must use direct cached dispatch"
            ),
        ):
            self.assertIs(slot.start_and_wait(), slot)
        self.assertEqual(
            backend_slot.bounded_calls,
            [
                (
                    _api._FUSED_REQUEST_SLOT_POLL_MAX_POLLS,
                    _api._FUSED_REQUEST_SLOT_POLL_TIMEOUT_NS,
                    _api._FUSED_REQUEST_SLOT_POLL_TIMEOUT_CHECK_INTERVAL,
                )
            ]
            * 3,
        )
        self.assertEqual(backend_slot.poll_count, 0)
        self.assertEqual(slot.state, WorkState.COMPLETED)
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_poll_only_continuation_does_not_require_start_fusion(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-poll-only")
        destination = Endpoint("destination", backend="test-request-slot-poll-only")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_results = [WorkState.COMPLETED]

        self.assertFalse(source.capabilities.fused_request_slot_start_poll)
        self.assertTrue(source.capabilities.fused_request_slot_poll)
        self.assertIs(slot.start_and_wait(), slot)
        self.assertEqual(backend_slot.start_count, 1)
        self.assertEqual(backend_slot.fused_calls, [])
        self.assertEqual(
            backend_slot.bounded_calls,
            [
                (
                    _api._FUSED_REQUEST_SLOT_POLL_MAX_POLLS,
                    _api._FUSED_REQUEST_SLOT_POLL_TIMEOUT_NS,
                    _api._FUSED_REQUEST_SLOT_POLL_TIMEOUT_CHECK_INTERVAL,
                )
            ],
        )
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_bounded_poll_timeout_leaves_generation_active(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-fused")
        destination = Endpoint("destination", backend="test-request-slot-fused")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_results = [WorkState.RUNNING]

        with mock.patch(
            "torch.distributed._transfer._api.time.monotonic_ns",
            side_effect=[0, 0, 0, 999_950_000, 1_000_000_000],
        ):
            with self.assertRaises(TransferTimeoutError):
                slot.start_and_wait(1)
        self.assertTrue(slot.active)
        self.assertEqual(slot.state, WorkState.RUNNING)
        self.assertEqual(
            backend_slot.bounded_calls,
            [
                (
                    _api._FUSED_REQUEST_SLOT_POLL_MAX_POLLS,
                    50_000,
                    _api._FUSED_REQUEST_SLOT_POLL_TIMEOUT_CHECK_INTERVAL,
                )
            ],
        )
        self.assertTrue(slot.cancel())
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_bounded_poll_validates_provider_results(self) -> None:
        source = Endpoint("source", backend="test-request-slot-fused")
        destination = Endpoint("destination", backend="test-request-slot-fused")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        for result, message in (
            (None, "invalid transfer slot state"),
            ("running", "invalid transfer slot state"),
            (WorkState.PENDING, "regressed"),
        ):
            backend_slot.next_state = WorkState.RUNNING
            backend_slot.bounded_results = [result]
            with self.assertRaisesRegex(BackendFailureError, message):
                slot.start_and_wait()
            self.assertTrue(slot.active)
            self.assertEqual(slot._state, WorkState.RUNNING)
            backend_slot._state = WorkState.RUNNING
            self.assertTrue(slot.cancel())
            slot.recycle()

        provider_error = RuntimeError("injected bounded-poll failure")
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_error = provider_error
        with self.assertRaises(BackendFailureError) as context:
            slot.start_and_wait()
        self.assertIs(context.exception.__cause__, provider_error)
        self.assertTrue(slot.active)
        self.assertEqual(slot._state, WorkState.RUNNING)
        backend_slot.bounded_error = None
        backend_slot._state = WorkState.RUNNING
        self.assertTrue(slot.cancel())
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_cancel_can_acquire_between_bounded_poll_chunks(
        self,
    ) -> None:
        source = Endpoint(
            "source", backend="test-request-slot-fused", thread_mode="multiple"
        )
        destination = Endpoint(
            "destination", backend="test-request-slot-fused", thread_mode="multiple"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_results = [WorkState.RUNNING]
        between_chunks = threading.Event()
        resume_waiter = threading.Event()
        waiter_errors: list[BaseException] = []
        original_progress = Endpoint.progress

        def pause_between_chunks(self):
            if self is source and not between_chunks.is_set():
                between_chunks.set()
                if not resume_waiter.wait(2):
                    raise RuntimeError("bounded-poll cancellation test timed out")
            return original_progress(self)

        def start_and_wait() -> None:
            try:
                slot.start_and_wait()
            except BaseException as error:
                waiter_errors.append(error)

        with mock.patch.object(Endpoint, "progress", pause_between_chunks):
            waiter = threading.Thread(target=start_and_wait)
            waiter.start()
            self.assertTrue(between_chunks.wait(1))
            self.assertTrue(slot.cancel())
            resume_waiter.set()
            waiter.join(2)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(waiter_errors), 1)
        self.assertIsInstance(waiter_errors[0], TransferCancelledError)
        self.assertEqual(len(backend_slot.bounded_calls), 1)
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_bounded_wait_is_pinned_to_started_generation(self) -> None:
        source = Endpoint(
            "source", backend="test-request-slot-fused", thread_mode="multiple"
        )
        destination = Endpoint(
            "destination", backend="test-request-slot-fused", thread_mode="multiple"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.next_state = WorkState.RUNNING
        entered = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []
        original_wait_bounded = TransferSlot._wait_bounded_until

        def pause_before_bounded_wait(self, *args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("bounded-poll generation test timed out")
            return original_wait_bounded(self, *args, **kwargs)

        def run_waiter() -> None:
            try:
                slot.start_and_wait()
            except BaseException as error:
                errors.append(error)

        with mock.patch.object(
            TransferSlot, "_wait_bounded_until", pause_before_bounded_wait
        ):
            waiter = threading.Thread(target=run_waiter)
            waiter.start()
            self.assertTrue(entered.wait(1))
            backend_slot._state = WorkState.COMPLETED
            self.assertEqual(slot.state, WorkState.COMPLETED)
            slot.recycle()
            backend_slot.next_state = WorkState.COMPLETED
            slot.start()
            replacement_generation = slot.generation
            release.set()
            waiter.join(2)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], BusyError)
        self.assertIn("changed while waiting", str(errors[0]))
        self.assertEqual(slot.generation, replacement_generation)
        self.assertEqual(backend_slot.bounded_calls, [])
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_start_and_wait_validates_fused_provider_results(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-fused")
        destination = Endpoint("destination", backend="test-request-slot-fused")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        failure = RuntimeError("fused provider failure")
        backend_slot.next_state = WorkState.FAILED
        backend_slot.next_error = failure
        with self.assertRaises(BackendFailureError) as context:
            slot.start_and_wait()
        self.assertIs(context.exception.__cause__, failure)
        self.assertIs(slot.error, failure)
        slot.recycle()

        backend_slot.next_state = WorkState.CANCELLED
        backend_slot.next_error = None
        with self.assertRaises(TransferCancelledError):
            slot.start_and_wait()
        self.assertIsInstance(slot.error, TransferCancelledError)
        slot.recycle()

        for result, message in (
            (None, "without acknowledging"),
            ("completed", "invalid fused request-slot state"),
        ):
            backend_slot.next_state = WorkState.COMPLETED
            backend_slot.fused_result = result
            with self.assertRaisesRegex(BackendFailureError, message):
                slot.start_and_wait()
            self.assertEqual(slot.state, WorkState.FAILED)
            slot.recycle()

        backend_slot.fused_result = _RETURN_CURRENT_SLOT_STATE
        backend_slot.fused_error = KeyboardInterrupt()
        backend_slot.next_state = WorkState.COMPLETED
        with self.assertRaises(BackendFailureError) as context:
            slot.start_and_wait()
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)
        self.assertEqual(slot.state, WorkState.FAILED)
        slot.recycle()

        backend_slot.fused_error = None
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.poll_states = [WorkState.PENDING]
        with self.assertRaisesRegex(BackendFailureError, "regressed"):
            slot.start_and_wait(1)
        self.assertTrue(slot.active)
        self.assertEqual(slot._state, WorkState.RUNNING)
        backend_slot._state = WorkState.RUNNING
        self.assertTrue(slot.cancel())
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_start_and_wait_fallback_ignores_fused_spi(self) -> None:
        source = Endpoint("source", backend="test-request-slot")
        destination = Endpoint("destination", backend="test-request-slot")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.start_and_poll = mock.Mock(
            side_effect=AssertionError("fused SPI must not be entered")
        )
        backend_slot.poll_bounded = mock.Mock(
            side_effect=AssertionError("bounded-poll SPI must not be entered")
        )

        self.assertFalse(source.capabilities.fused_request_slot_start_poll)
        self.assertFalse(source.capabilities.fused_request_slot_poll)
        self.assertIs(slot.start_and_wait(), slot)
        backend_slot.start_and_poll.assert_not_called()
        backend_slot.poll_bounded.assert_not_called()
        self.assertEqual(backend_slot.start_count, 1)
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_execute_source_compatible_fallback_recycles(self) -> None:
        source = Endpoint("source", backend="test-request-slot")
        destination = Endpoint("destination", backend="test-request-slot")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        self.assertFalse(source.capabilities.deferred_successful_request_slot_recycle)
        self.assertEqual(slot.execute(), 1)
        self.assertEqual(slot.generation, 1)
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(backend_slot.start_count, 1)
        self.assertEqual(backend_slot.recycle_count, 1)
        with self.assertRaisesRegex(BusyError, "no active generation"):
            _ = slot.completion_event
        with self.assertRaisesRegex(BusyError, "no generation"):
            slot.recycle()

        generation = slot.generation
        for timeout, error_type in (
            (True, TypeError),
            ("one", TypeError),
            (-1, ValueError),
            (float("inf"), ValueError),
        ):
            with self.assertRaises(error_type):
                slot.execute(timeout)  # type: ignore[arg-type]
        self.assertEqual(slot.generation, generation)
        self.assertFalse(slot.active)

        self.assertEqual(slot.execute(), 2)
        self.assertEqual(backend_slot.recycle_count, 2)
        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_execute_deferred_recycle_preserves_explicit_api(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-deferred-recycle")
        destination = Endpoint(
            "destination", backend="test-request-slot-deferred-recycle"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        self.assertTrue(source.capabilities.deferred_successful_request_slot_recycle)
        self.assertEqual(slot.execute(), 1)
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertFalse(slot.test())
        self.assertIsNone(slot.error)
        self.assertEqual(backend_slot.recycle_count, 0)
        # The test provider models the retained, observable completion receipt.
        self.assertEqual(backend_slot.state, WorkState.COMPLETED)
        with self.assertRaisesRegex(BusyError, "no active generation"):
            _ = slot.completion_event

        self.assertIs(slot.start_and_wait(), slot)
        self.assertEqual(slot.generation, 2)
        self.assertTrue(slot.active)
        self.assertEqual(slot.state, WorkState.COMPLETED)
        with self.assertRaisesRegex(BusyError, "unrecycled generation"):
            slot.start()
        with self.assertRaisesRegex(BusyError, "recycle it before close"):
            slot.close()
        slot.recycle()
        self.assertEqual(backend_slot.recycle_count, 1)

        # Close after execute is legal even though the provider still owns its
        # deferred successful receipt.
        self.assertEqual(slot.execute(), 3)
        self.assertEqual(backend_slot.recycle_count, 1)
        slot.close()
        self.assertTrue(slot.closed)
        self.assertEqual(backend_slot.close_count, 1)
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_execute_deferred_fused_immediate_success(self) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        with (
            mock.patch.object(
                source,
                "_backend_request_slot_start_and_poll",
                side_effect=AssertionError("SINGLE execute must use cached dispatch"),
            ),
            mock.patch.object(
                _api,
                "_transfer_slot_deadline_ns",
                side_effect=AssertionError("timeout=None must bypass validation"),
            ),
            mock.patch.object(
                TransferSlot,
                "_check_access",
                side_effect=AssertionError("fast execute must inline access checks"),
            ),
            mock.patch.object(
                TransferSlot,
                "_begin_generation_unlocked",
                side_effect=AssertionError("fast execute must inline generation begin"),
            ),
            mock.patch.object(
                TransferSlot,
                "_start_and_poll_unlocked",
                side_effect=AssertionError("fast execute must inline fused start"),
            ),
            mock.patch.object(
                TransferSlot,
                "_observe_backend_state_unlocked",
                side_effect=AssertionError("fast execute must inline completion"),
            ),
        ):
            self.assertEqual(slot.execute(), 1)
            self.assertEqual(slot.execute(), 2)
        self.assertEqual(backend_slot.fused_calls, [(64, None), (64, None)])
        self.assertEqual(backend_slot.poll_count, 0)
        self.assertEqual(backend_slot.recycle_count, 0)
        self.assertEqual(slot.generation, 2)
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_execute_exact_running_uses_direct_continuation(
        self,
    ) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        observe = TransferSlot._observe_backend_state_unlocked

        def reject_initial_running(self, state):
            if state is WorkState.RUNNING:
                raise AssertionError("initial RUNNING must use direct continuation")
            return observe(self, state)

        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_results = [WorkState.COMPLETED]
        with mock.patch.object(
            TransferSlot,
            "_observe_backend_state_unlocked",
            reject_initial_running,
        ):
            self.assertEqual(slot.execute(), 1)
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(len(backend_slot.bounded_calls), 1)

        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_results = [WorkState.PENDING]
        with (
            mock.patch.object(
                TransferSlot,
                "_observe_backend_state_unlocked",
                reject_initial_running,
            ),
            self.assertRaisesRegex(BackendFailureError, "regressed"),
        ):
            slot.execute()
        self.assertTrue(slot.active)
        self.assertTrue(slot._backend_generation_active)
        self.assertIs(slot._state, WorkState.RUNNING)
        backend_slot._state = WorkState.RUNNING
        self.assertTrue(slot.cancel())
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_start_test_recycles_only_success(self) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        with mock.patch.object(
            TransferSlot,
            "_start_and_poll_unlocked",
            side_effect=AssertionError("exact start/test must use cached dispatch"),
        ):
            self.assertTrue(slot.start_and_test())
        self.assertFalse(slot.active)
        self.assertEqual(slot.generation, 1)
        self.assertEqual(backend_slot.fused_calls, [(0, None)])
        self.assertEqual(backend_slot.recycle_count, 0)

        backend_slot.next_state = WorkState.RUNNING
        backend_slot.poll_states = [WorkState.RUNNING, WorkState.COMPLETED]
        with mock.patch.object(
            TransferSlot,
            "_observe_backend_state_unlocked",
            side_effect=AssertionError(
                "exact RUNNING start/test must not reclassify provider state"
            ),
        ):
            self.assertFalse(slot.start_and_test())
        self.assertTrue(slot.active)
        with (
            mock.patch.object(
                TransferSlot,
                "_check_access",
                side_effect=AssertionError("exact continuation must inline access"),
            ),
            mock.patch.object(
                TransferSlot,
                "_refresh_unlocked",
                side_effect=AssertionError("exact continuation must poll directly"),
            ),
            mock.patch.object(
                TransferSlot,
                "_finish_execute_success_unlocked",
                side_effect=AssertionError("exact continuation must finish inline"),
            ),
            mock.patch.object(
                TransferSlot,
                "_observe_backend_state_unlocked",
                side_effect=AssertionError(
                    "exact continuation must inline common provider states"
                ),
            ),
        ):
            self.assertFalse(slot.test_and_recycle())
            self.assertTrue(slot.test_and_recycle())
        self.assertFalse(slot.active)
        self.assertEqual(backend_slot.recycle_count, 0)
        with self.assertRaisesRegex(BusyError, "no active generation"):
            slot.test_and_recycle()

        backend_slot.next_state = WorkState.FAILED
        backend_slot.next_error = RuntimeError("injected failure")
        with self.assertRaises(BackendFailureError):
            slot.start_and_test()
        self.assertTrue(slot.active)
        slot.recycle()
        self.assertEqual(backend_slot.recycle_count, 1)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_cached_success_recycles_without_another_poll(self) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        backend_slot.next_state = WorkState.RUNNING
        backend_slot.poll_states = [WorkState.COMPLETED]
        self.assertFalse(slot.start_and_test())
        self.assertTrue(slot.test())
        self.assertTrue(slot.active)
        self.assertEqual(slot.state, WorkState.COMPLETED)
        polls_before_recycle = backend_slot.poll_count
        with mock.patch.object(
            slot,
            "_backend_poll_state",
            side_effect=AssertionError("cached completion must not poll again"),
        ):
            self.assertTrue(slot.test_and_recycle())
        self.assertEqual(backend_slot.poll_count, polls_before_recycle)
        self.assertEqual(backend_slot.recycle_count, 0)
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_serialized_fallback_start_test_is_one_transaction(
        self,
    ) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-blocking-observation",
            thread_mode="serialized",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-blocking-observation",
            thread_mode="serialized",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.block_next_observation = True
        results: dict[str, bool] = {}
        errors: list[BaseException] = []
        second_entered = threading.Event()
        second_done = threading.Event()

        def start_and_test(label: str) -> None:
            if label == "second":
                second_entered.set()
            try:
                results[label] = slot.start_and_test()
            except BaseException as error:
                errors.append(error)
            finally:
                if label == "second":
                    second_done.set()

        first = threading.Thread(target=start_and_test, args=("first",))
        second = threading.Thread(target=start_and_test, args=("second",))
        first.start()
        self.assertTrue(backend_slot.observation_entered.wait(1))
        second.start()
        self.assertTrue(second_entered.wait(1))
        try:
            self.assertFalse(second_done.wait(0.05))
            self.assertEqual(backend_slot.start_count, 1)
            self.assertEqual(slot.generation, 1)
            self.assertTrue(slot.active)
        finally:
            backend_slot.observation_release.set()
        first.join(2)
        second.join(2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, {"first": True, "second": True})
        self.assertEqual(backend_slot.start_count, 2)
        self.assertEqual(backend_slot.recycle_count, 2)
        self.assertEqual(slot.generation, 2)
        self.assertFalse(slot.active)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_start_test_retains_cancel_and_failure(self) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        backend_slot.next_state = WorkState.RUNNING
        self.assertFalse(slot.start_and_test())
        self.assertTrue(slot.cancel())
        with self.assertRaises(TransferCancelledError):
            slot.test_and_recycle()
        self.assertTrue(slot.active)
        with self.assertRaisesRegex(BusyError, "unrecycled generation"):
            slot.start_and_test()
        with self.assertRaisesRegex(BusyError, "recycle it before close"):
            slot.close()
        slot.recycle()
        self.assertEqual(backend_slot.recycle_count, 1)

        failure = RuntimeError("retained start/test failure")
        backend_slot.next_state = WorkState.FAILED
        backend_slot.next_error = failure
        with self.assertRaises(BackendFailureError) as context:
            slot.start_and_test()
        self.assertIs(context.exception.__cause__, failure)
        self.assertTrue(slot.active)
        with self.assertRaises(BackendFailureError) as context:
            slot.test_and_recycle()
        self.assertIs(context.exception.__cause__, failure)
        self.assertTrue(slot.active)
        slot.recycle()
        self.assertEqual(backend_slot.recycle_count, 2)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_start_test_deferred_success_closes_without_recycle(
        self,
    ) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        self.assertTrue(slot.start_and_test())
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(backend_slot.recycle_count, 0)
        slot.close()
        self.assertTrue(slot.closed)
        self.assertEqual(backend_slot.recycle_count, 0)
        self.assertEqual(backend_slot.close_count, 1)

        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_start_test_interruption_boundaries_are_recoverable(
        self,
    ) -> None:
        class InterruptingIdleCommitTransferSlot(TransferSlot):
            __slots__ = ("interrupt_idle_commit",)

            def __init__(self, *args, **kwargs) -> None:
                self.interrupt_idle_commit = False
                super().__init__(*args, **kwargs)

            def __setattr__(self, name, value) -> None:
                object.__setattr__(self, name, value)
                if (
                    name == "_active"
                    and value is False
                    and getattr(self, "interrupt_idle_commit", False)
                ):
                    object.__setattr__(self, "interrupt_idle_commit", False)
                    raise KeyboardInterrupt

        source = Endpoint(
            "source",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        with mock.patch.object(
            _api, "TransferSlot", InterruptingIdleCommitTransferSlot
        ):
            slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        backend_slot.fused_error = KeyboardInterrupt()
        with self.assertRaises(BackendFailureError) as context:
            slot.start_and_test()
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)
        self.assertTrue(slot.active)
        self.assertEqual(slot.state, WorkState.FAILED)
        slot.recycle()
        self.assertEqual(backend_slot.recycle_count, 1)

        backend_slot.fused_error = None
        slot.interrupt_idle_commit = True
        with self.assertRaises(KeyboardInterrupt):
            slot.start_and_test()
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(slot.generation, 2)
        self.assertEqual(backend_slot.recycle_count, 1)
        self.assertTrue(slot.start_and_test())
        self.assertEqual(slot.generation, 3)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_request_local_slot_start_test_snapshots_notification(self) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-notification",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-notification",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot(request_local_notifications=True)
        backend_slot = source._backend.request_slots[0]

        payload = bytearray(b"A\x00\xff")
        with mock.patch.object(
            RequestLocalNotificationSlot,
            "_start_and_poll_with_notification_unlocked",
            side_effect=AssertionError(
                "exact notification start/test must use cached dispatch"
            ),
        ):
            self.assertTrue(slot.start_and_test(notification=payload))
        payload[:] = b"BAD"
        self.assertEqual(
            backend_slot.notification_fused_calls,
            [(b"A\x00\xff", 0, None)],
        )
        self.assertTrue(slot.start_and_test(notification=None))
        self.assertEqual(
            backend_slot.notification_fused_calls[-1],
            (None, 0, None),
        )
        self.assertFalse(slot.active)
        self.assertEqual(backend_slot.recycle_count, 0)

        backend_slot.next_state = WorkState.RUNNING
        backend_slot.poll_states = [WorkState.COMPLETED]
        with mock.patch.object(
            RequestLocalNotificationSlot,
            "_observe_backend_state_unlocked",
            side_effect=AssertionError(
                "exact notification RUNNING start/test must not reclassify state"
            ),
        ):
            self.assertFalse(slot.start_and_test(notification=b"pending"))
        self.assertTrue(slot.active)
        self.assertTrue(slot.test_and_recycle())
        self.assertFalse(slot.active)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_request_local_slot_execute_exact_running_uses_direct_continuation(
        self,
    ) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-notification",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-notification",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot(request_local_notifications=True)
        backend_slot = source._backend.request_slots[0]
        observe = RequestLocalNotificationSlot._observe_backend_state_unlocked

        def reject_initial_running(self, state):
            if state is WorkState.RUNNING:
                raise AssertionError("initial RUNNING must use direct continuation")
            return observe(self, state)

        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_results = [WorkState.COMPLETED]
        with mock.patch.object(
            RequestLocalNotificationSlot,
            "_observe_backend_state_unlocked",
            reject_initial_running,
        ):
            self.assertEqual(slot.execute(notification=b"ready"), 1)
        self.assertEqual(
            backend_slot.notification_fused_calls,
            [(b"ready", _api._FUSED_REQUEST_SLOT_MAX_POLLS, None)],
        )
        self.assertEqual(len(backend_slot.bounded_calls), 1)
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertIs(
            slot._notification_fast_execute,
            slot._backend_start_and_poll_with_notification,
        )

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_request_local_slot_interrupted_completion_cannot_reuse_stale_state(
        self,
    ) -> None:
        class InterruptingNotificationSlot(RequestLocalNotificationSlot):
            __slots__ = ("interrupt_idle_commit",)

            def __init__(self, *args, **kwargs) -> None:
                self.interrupt_idle_commit = False
                super().__init__(*args, **kwargs)

            def __setattr__(self, name, value) -> None:
                object.__setattr__(self, name, value)
                if (
                    name == "_active"
                    and value is False
                    and getattr(self, "interrupt_idle_commit", False)
                ):
                    object.__setattr__(self, "interrupt_idle_commit", False)
                    raise KeyboardInterrupt

        source = Endpoint(
            "source", backend="test-request-slot-notification", thread_mode="single"
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-notification",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        with mock.patch.object(
            _api, "RequestLocalNotificationSlot", InterruptingNotificationSlot
        ):
            slot = bound.create_slot(request_local_notifications=True)
        backend_slot = source._backend.request_slots[0]
        notification_fast_execute = slot._notification_fast_execute
        self.assertIsNotNone(notification_fast_execute)

        backend_slot.next_state = WorkState.RUNNING
        self.assertFalse(slot.start_and_test(notification=b"first"))
        backend_slot.poll_states = [WorkState.COMPLETED]
        slot.interrupt_idle_commit = True
        with self.assertRaises(KeyboardInterrupt):
            slot.test_and_recycle()
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertIsNone(slot._state)
        self.assertIsNone(slot._start_error)
        self.assertIs(
            slot._notification_fast_execute,
            notification_fast_execute,
        )

        prior_poll_count = backend_slot.poll_count
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.fused_error = KeyboardInterrupt()
        backend_slot.poll_states = [WorkState.COMPLETED]
        with self.assertRaises(BackendFailureError) as context:
            slot.start_and_test(notification=b"second")
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)
        self.assertEqual(backend_slot.poll_count, prior_poll_count + 1)
        self.assertTrue(slot.active)
        self.assertEqual(slot.state, WorkState.FAILED)

        slot.recycle()
        self.assertFalse(slot.active)
        self.assertIs(
            slot._notification_fast_execute,
            slot._backend_start_and_poll_with_notification,
        )
        backend_slot.fused_error = None
        backend_slot.next_state = WorkState.COMPLETED
        self.assertTrue(slot.start_and_test(notification=None))

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_request_local_slot_interrupted_bounded_execute_uses_safe_fallback(
        self,
    ) -> None:
        class InterruptingNotificationSlot(RequestLocalNotificationSlot):
            __slots__ = ("interrupt_idle_commit",)

            def __init__(self, *args, **kwargs) -> None:
                self.interrupt_idle_commit = False
                super().__init__(*args, **kwargs)

            def __setattr__(self, name, value) -> None:
                object.__setattr__(self, name, value)
                if (
                    name == "_active"
                    and value is False
                    and getattr(self, "interrupt_idle_commit", False)
                ):
                    object.__setattr__(self, "interrupt_idle_commit", False)
                    raise KeyboardInterrupt

        source = Endpoint(
            "source", backend="test-request-slot-notification", thread_mode="single"
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-notification",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        with mock.patch.object(
            _api, "RequestLocalNotificationSlot", InterruptingNotificationSlot
        ):
            slot = bound.create_slot(request_local_notifications=True)
        backend_slot = source._backend.request_slots[0]

        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_results = [WorkState.COMPLETED]
        slot.interrupt_idle_commit = True
        with self.assertRaises(KeyboardInterrupt):
            slot.execute(notification=b"bounded")
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(slot._state, WorkState.COMPLETED)
        self.assertIsNone(slot._notification_fast_execute)

        backend_slot.next_state = WorkState.COMPLETED
        self.assertEqual(slot.execute(notification=b"fallback"), 2)
        self.assertFalse(slot.active)
        self.assertIsNone(slot._state)
        self.assertIs(
            slot._notification_fast_execute,
            slot._backend_start_and_poll_with_notification,
        )

        backend_slot.next_state = WorkState.FAILED
        backend_slot.next_error = RuntimeError("notification failure")
        with self.assertRaises(BackendFailureError):
            slot.start_and_test(notification=b"failure")
        self.assertIsNotNone(slot._notification_fast_execute)
        slot.interrupt_idle_commit = True
        with self.assertRaises(KeyboardInterrupt):
            slot.recycle()
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(slot._state, WorkState.FAILED)
        self.assertIsNone(slot._notification_fast_execute)

        backend_slot.next_state = WorkState.COMPLETED
        backend_slot.next_error = None
        self.assertTrue(slot.start_and_test(notification=None))
        self.assertIs(
            slot._notification_fast_execute,
            slot._backend_start_and_poll_with_notification,
        )

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_execute_single_fused_fast_path_preserves_guards(
        self,
    ) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        wrong_thread_errors: list[BaseException] = []

        def execute_from_wrong_thread() -> None:
            try:
                slot.execute()
            except BaseException as error:
                wrong_thread_errors.append(error)

        wrong_thread = threading.Thread(target=execute_from_wrong_thread)
        wrong_thread.start()
        wrong_thread.join(1)
        self.assertFalse(wrong_thread.is_alive())
        self.assertEqual(len(wrong_thread_errors), 1)
        self.assertIsInstance(wrong_thread_errors[0], UnsupportedError)
        self.assertEqual(slot.generation, 0)
        self.assertFalse(slot.active)
        self.assertEqual(backend_slot.fused_calls, [])

        backend_slot.next_state = WorkState.COMPLETED
        backend_slot.fused_error = KeyboardInterrupt()
        with self.assertRaises(BackendFailureError) as context:
            slot.execute()
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)
        self.assertEqual(slot.generation, 1)
        self.assertTrue(slot.active)
        self.assertEqual(slot.state, WorkState.FAILED)
        slot.recycle()

        backend_slot.fused_error = None
        slot.start()
        fused_call_count = len(backend_slot.fused_calls)
        with self.assertRaisesRegex(BusyError, "unrecycled generation"):
            slot.execute()
        self.assertEqual(len(backend_slot.fused_calls), fused_call_count)
        self.assertEqual(slot.generation, 2)
        slot.recycle()

        self.assertEqual(slot.execute(), 3)
        slot.close()
        with self.assertRaises(ClosedError):
            slot.execute()
        self.assertEqual(slot.generation, 3)

        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_execute_bounded_success_is_atomic_in_multiple(
        self,
    ) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="multiple",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="multiple",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_results = [WorkState.COMPLETED]
        finish_entered = threading.Event()
        finish_release = threading.Event()
        observer_done = threading.Event()
        execute_results: list[int] = []
        errors: list[BaseException] = []
        observed_states: list[WorkState | None] = []
        original_finish = TransferSlot._finish_execute_success_unlocked

        def pause_in_finish(self, expected_generation):
            finish_entered.set()
            if not finish_release.wait(2):
                raise RuntimeError("execute atomicity test timed out")
            return original_finish(self, expected_generation)

        def run_execute() -> None:
            try:
                execute_results.append(slot.execute())
            except BaseException as error:
                errors.append(error)

        def observe_state() -> None:
            try:
                observed_states.append(slot.state)
            except BaseException as error:
                errors.append(error)
            finally:
                observer_done.set()

        with mock.patch.object(
            TransferSlot, "_finish_execute_success_unlocked", pause_in_finish
        ):
            executor = threading.Thread(target=run_execute)
            executor.start()
            self.assertTrue(finish_entered.wait(1))
            observer = threading.Thread(target=observe_state)
            observer.start()
            self.assertFalse(observer_done.wait(0.05))
            finish_release.set()
            executor.join(2)
            observer.join(2)

        self.assertFalse(executor.is_alive())
        self.assertFalse(observer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(execute_results, [1])
        self.assertEqual(observed_states, [None])
        self.assertEqual(len(backend_slot.bounded_calls), 1)
        self.assertEqual(backend_slot.recycle_count, 0)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_execute_retains_unsuccessful_generations(self) -> None:
        source = Endpoint("source", backend="test-request-slot-fused-deferred-recycle")
        destination = Endpoint(
            "destination", backend="test-request-slot-fused-deferred-recycle"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        failure = RuntimeError("execute provider failure")
        backend_slot.next_state = WorkState.FAILED
        backend_slot.next_error = failure
        with self.assertRaises(BackendFailureError) as context:
            slot.execute()
        self.assertIs(context.exception.__cause__, failure)
        self.assertTrue(slot.active)
        self.assertEqual(slot.generation, 1)
        slot.recycle()

        backend_slot.next_state = WorkState.CANCELLED
        backend_slot.next_error = None
        with self.assertRaises(TransferCancelledError):
            slot.execute()
        self.assertTrue(slot.active)
        self.assertEqual(slot.generation, 2)
        slot.recycle()

        backend_slot.next_state = WorkState.RUNNING
        with self.assertRaises(TransferTimeoutError):
            slot.execute(0)
        self.assertTrue(slot.active)
        self.assertEqual(slot.generation, 3)
        self.assertEqual(slot.state, WorkState.RUNNING)
        self.assertTrue(slot.cancel())
        slot.recycle()
        self.assertEqual(backend_slot.recycle_count, 3)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_execute_interruption_preserves_a_safe_boundary(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-fused-deferred-recycle")
        destination = Endpoint(
            "destination", backend="test-request-slot-fused-deferred-recycle"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        original_finish = TransferSlot._finish_execute_success_unlocked

        def interrupt_before_finish(self, expected_generation):
            raise KeyboardInterrupt

        with mock.patch.object(
            TransferSlot,
            "_finish_execute_success_unlocked",
            interrupt_before_finish,
        ):
            with self.assertRaises(KeyboardInterrupt):
                slot.execute()
        self.assertTrue(slot.active)
        self.assertEqual(slot.state, WorkState.COMPLETED)
        slot.recycle()
        self.assertEqual(backend_slot.recycle_count, 1)

        def interrupt_after_finish(self, expected_generation):
            original_finish(self, expected_generation)
            raise KeyboardInterrupt

        with mock.patch.object(
            TransferSlot,
            "_finish_execute_success_unlocked",
            interrupt_after_finish,
        ):
            with self.assertRaises(KeyboardInterrupt):
                slot.execute()
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(slot.generation, 2)
        self.assertEqual(backend_slot.recycle_count, 1)

        self.assertEqual(slot.execute(), 3)
        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_deferred_execute_idle_boundaries_are_recoverable(
        self,
    ) -> None:
        class InterruptingIdleCommitTransferSlot(TransferSlot):
            __slots__ = ("interrupt_fast_restore", "interrupt_idle_commit")

            def __init__(self, *args, **kwargs) -> None:
                self.interrupt_fast_restore = False
                self.interrupt_idle_commit = False
                super().__init__(*args, **kwargs)

            def __setattr__(self, name, value) -> None:
                if (
                    name == "_fast_execute"
                    and value is not None
                    and getattr(self, "interrupt_fast_restore", False)
                ):
                    object.__setattr__(self, "interrupt_fast_restore", False)
                    raise KeyboardInterrupt
                object.__setattr__(self, name, value)
                if (
                    name == "_active"
                    and value is False
                    and getattr(self, "interrupt_idle_commit", False)
                ):
                    object.__setattr__(self, "interrupt_idle_commit", False)
                    raise KeyboardInterrupt

        source = Endpoint(
            "source",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        with mock.patch.object(
            _api, "TransferSlot", InterruptingIdleCommitTransferSlot
        ):
            slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        fast_execute = slot._fast_execute
        self.assertIs(fast_execute, slot._backend_start_and_poll)
        self.assertTrue(slot._caller_ready)

        def interrupt_before_provider_entry(**_kwargs) -> WorkState:
            raise KeyboardInterrupt

        slot._fast_execute = interrupt_before_provider_entry
        with self.assertRaises(BackendFailureError) as context:
            slot.execute()
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)
        self.assertTrue(slot.active)
        self.assertEqual(slot.state, WorkState.FAILED)
        self.assertEqual(backend_slot.fused_calls, [])
        slot.recycle()
        self.assertEqual(backend_slot.recycle_count, 0)
        self.assertIs(slot._fast_execute, slot._backend_start_and_poll)

        slot.interrupt_idle_commit = True
        with self.assertRaises(KeyboardInterrupt):
            slot.execute()
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertIsNone(slot.error)
        self.assertIsNone(slot._state)
        self.assertEqual(slot.generation, 2)
        self.assertEqual(backend_slot.state, WorkState.COMPLETED)
        self.assertIs(slot._fast_execute, fast_execute)
        self.assertEqual(slot.execute(), 3)
        self.assertEqual(slot.execute(), 4)
        self.assertIsNone(slot._state)
        self.assertEqual(backend_slot.recycle_count, 0)

        # A timed call uses the generic success finalizer. It disables exact
        # execution before public idle, so an interrupt at that commit leaves a
        # dirty but fast-ineligible slot that the next generic call repairs.
        slot.interrupt_idle_commit = True
        with self.assertRaises(KeyboardInterrupt):
            slot.execute(1.0)
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(slot._state, WorkState.COMPLETED)
        self.assertIsNone(slot._fast_execute)

        # Publishing exact eligibility is the last private-idle store. An
        # interrupt just before it leaves a completely clean generic fallback;
        # one successful fallback restores the exact path for its successor.
        slot.interrupt_fast_restore = True
        with self.assertRaises(KeyboardInterrupt):
            slot.execute(1.0)
        self.assertFalse(slot.active)
        self.assertIsNone(slot._state)
        self.assertIsNone(slot._error)
        self.assertIsNone(slot._start_error)
        self.assertFalse(slot._backend_generation_active)
        self.assertIsNone(slot._fast_execute)
        self.assertEqual(slot.execute(), 7)
        self.assertIs(slot._fast_execute, slot._backend_start_and_poll)
        self.assertEqual(slot.execute(), 8)

        slot.close()
        self.assertIsNone(slot._fast_execute)
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_dirty_idle_repairs_ambiguous_failure_before_begin(
        self,
    ) -> None:
        class InterruptingIdleCommitTransferSlot(TransferSlot):
            __slots__ = ("interrupt_fast_restore", "interrupt_idle_commit")

            def __init__(self, *args, **kwargs) -> None:
                self.interrupt_fast_restore = False
                self.interrupt_idle_commit = False
                super().__init__(*args, **kwargs)

            def __setattr__(self, name, value) -> None:
                if (
                    name == "_fast_execute"
                    and value is not None
                    and getattr(self, "interrupt_fast_restore", False)
                ):
                    object.__setattr__(self, "interrupt_fast_restore", False)
                    raise KeyboardInterrupt
                object.__setattr__(self, name, value)
                if (
                    name == "_active"
                    and value is False
                    and getattr(self, "interrupt_idle_commit", False)
                ):
                    object.__setattr__(self, "interrupt_idle_commit", False)
                    raise KeyboardInterrupt

        source = Endpoint(
            "source",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-fused-deferred-recycle",
            thread_mode="single",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        with mock.patch.object(
            _api, "TransferSlot", InterruptingIdleCommitTransferSlot
        ):
            slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        backend_slot.next_state = WorkState.COMPLETED
        backend_slot.fused_error = KeyboardInterrupt()
        with self.assertRaises(BackendFailureError):
            slot.execute()
        self.assertEqual(slot.state, WorkState.FAILED)
        self.assertIs(slot._fast_execute, slot._backend_start_and_poll)

        slot.interrupt_idle_commit = True
        with self.assertRaises(KeyboardInterrupt):
            slot.recycle()
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertIsNone(slot.error)
        self.assertEqual(slot._state, WorkState.FAILED)
        self.assertIsNotNone(slot._start_error)
        self.assertIsNone(slot._fast_execute)

        backend_slot.fused_error = None
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.bounded_results = [WorkState.COMPLETED]
        self.assertEqual(slot.execute(), 2)
        self.assertFalse(slot.active)
        self.assertIsNone(slot.error)
        self.assertIs(slot._fast_execute, slot._backend_start_and_poll)

        # CANCELLED is never folded. Its explicit recycle follows the same
        # disable-clean-restore protocol; interruption before the last restore
        # leaves a clean generic fallback and cannot expose stale diagnostics.
        backend_slot.next_state = WorkState.CANCELLED
        with self.assertRaises(TransferCancelledError):
            slot.execute()
        self.assertIs(slot._fast_execute, slot._backend_start_and_poll)
        slot.interrupt_fast_restore = True
        with self.assertRaises(KeyboardInterrupt):
            slot.recycle()
        self.assertFalse(slot.active)
        self.assertIsNone(slot._state)
        self.assertIsNone(slot._error)
        self.assertIsNone(slot._start_error)
        self.assertFalse(slot._backend_generation_active)
        self.assertIsNone(slot._fast_execute)

        backend_slot.next_state = WorkState.COMPLETED
        self.assertEqual(slot.execute(), 4)
        self.assertIs(slot._fast_execute, slot._backend_start_and_poll)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_fused_fallback_releases_slot_lock(self) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-start-fused-only",
            thread_mode="multiple",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-start-fused-only",
            thread_mode="multiple",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.start_entered = threading.Event()
        errors: list[BaseException] = []

        def start_and_wait() -> None:
            try:
                slot.start_and_wait(2)
            except BaseException as error:
                errors.append(error)

        waiter = threading.Thread(target=start_and_wait)
        waiter.start()
        self.assertTrue(backend_slot.start_entered.wait(1))
        self.assertTrue(slot.cancel())
        waiter.join(2)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], TransferCancelledError)
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_waits_are_pinned_to_the_started_generation(self) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-start-fused-only",
            thread_mode="multiple",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-start-fused-only",
            thread_mode="multiple",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        original_wait_until = TransferSlot._wait_until

        def replace_generation_while_waiter_is_paused(start_and_wait: bool) -> None:
            entered = threading.Event()
            release = threading.Event()
            errors: list[BaseException] = []

            def pause_before_first_observation(self, *args, **kwargs):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("generation-pinning test timed out")
                return original_wait_until(self, *args, **kwargs)

            if start_and_wait:
                backend_slot.next_state = WorkState.RUNNING
                operation = slot.start_and_wait
            else:
                backend_slot.next_state = WorkState.COMPLETED
                slot.start()
                operation = slot.wait

            def run_waiter() -> None:
                try:
                    operation()
                except BaseException as error:
                    errors.append(error)

            with mock.patch.object(
                TransferSlot, "_wait_until", pause_before_first_observation
            ):
                waiter = threading.Thread(target=run_waiter)
                waiter.start()
                self.assertTrue(entered.wait(1))
                backend_slot._state = WorkState.COMPLETED
                self.assertEqual(slot.state, WorkState.COMPLETED)
                slot.recycle()
                backend_slot.next_state = WorkState.COMPLETED
                slot.start()
                replacement_generation = slot.generation
                release.set()
                waiter.join(2)

            self.assertFalse(waiter.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], BusyError)
            self.assertIn("changed while waiting", str(errors[0]))
            self.assertEqual(slot.generation, replacement_generation)
            slot.recycle()

        replace_generation_while_waiter_is_paused(start_and_wait=False)
        replace_generation_while_waiter_is_paused(start_and_wait=True)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_start_and_wait_fallback_remains_spin_oriented(self) -> None:
        source = Endpoint("source", backend="test-request-slot-start-fused-only")
        destination = Endpoint(
            "destination", backend="test-request-slot-start-fused-only"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.next_state = WorkState.RUNNING
        backend_slot.poll_states = [WorkState.RUNNING] * 130 + [WorkState.COMPLETED]

        with mock.patch("torch.distributed._transfer._api.time.sleep") as sleep:
            self.assertIs(slot.start_and_wait(), slot)
        sleep.assert_not_called()
        self.assertEqual(backend_slot.poll_count, 131)
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_wait_retains_cooperative_backoff(self) -> None:
        source = Endpoint("source", backend="test-request-slot")
        destination = Endpoint("destination", backend="test-request-slot")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.next_state = WorkState.RUNNING
        slot.start()

        def complete_on_yield(delay: float) -> None:
            self.assertEqual(delay, 0)
            backend_slot._state = WorkState.COMPLETED

        with mock.patch(
            "torch.distributed._transfer._api.time.sleep",
            side_effect=complete_on_yield,
        ) as sleep:
            self.assertIs(slot.wait(), slot)
        sleep.assert_called_once_with(0)
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_rejects_concurrent_start(self) -> None:
        source = Endpoint("source", backend="test-request-slot", thread_mode="multiple")
        destination = Endpoint(
            "destination", backend="test-request-slot", thread_mode="multiple"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.start_entered = threading.Event()
        backend_slot.start_release = threading.Event()
        errors: list[BaseException] = []

        def start_slot() -> None:
            try:
                slot.start()
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=start_slot)
        second = threading.Thread(target=start_slot)
        first.start()
        self.assertTrue(backend_slot.start_entered.wait(1))
        second.start()
        backend_slot.start_release.set()
        first.join(2)
        second.join(2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], BusyError)
        self.assertEqual(backend_slot.start_count, 1)
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_close_is_linearized_with_fused_start(self) -> None:
        source = Endpoint(
            "source", backend="test-request-slot-fused", thread_mode="multiple"
        )
        destination = Endpoint(
            "destination", backend="test-request-slot-fused", thread_mode="multiple"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        backend_slot.start_entered = threading.Event()
        backend_slot.start_release = threading.Event()
        start_errors: list[BaseException] = []
        close_errors: list[BaseException] = []

        def start_slot() -> None:
            try:
                slot.start_and_wait()
            except BaseException as error:
                start_errors.append(error)

        def close_slot() -> None:
            try:
                slot.close()
            except BaseException as error:
                close_errors.append(error)

        starter = threading.Thread(target=start_slot)
        closer = threading.Thread(target=close_slot)
        starter.start()
        self.assertTrue(backend_slot.start_entered.wait(1))
        closer.start()
        backend_slot.start_release.set()
        starter.join(2)
        closer.join(2)

        self.assertFalse(starter.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual(start_errors, [])
        self.assertEqual(len(close_errors), 1)
        self.assertIsInstance(close_errors[0], BusyError)
        self.assertTrue(slot.active)
        self.assertFalse(slot.closed)

        slot.recycle()
        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_interrupted_before_provider_ack_is_recyclable(self) -> None:
        source = Endpoint("source", backend="test-request-slot-unacknowledged-start")
        destination = Endpoint(
            "destination", backend="test-request-slot-unacknowledged-start"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()

        self.assertIsNone(slot.start())
        self.assertEqual(slot.state, WorkState.FAILED)
        self.assertIsInstance(slot.error, KeyboardInterrupt)
        with self.assertRaises(BackendFailureError) as context:
            slot.wait()
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)
        slot.recycle()
        self.assertFalse(slot.active)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_normal_return_without_provider_ack_is_recyclable(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-no-ack")
        destination = Endpoint("destination", backend="test-request-slot-no-ack")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()

        self.assertIsNone(slot.start())
        self.assertEqual(slot.state, WorkState.FAILED)
        with self.assertRaisesRegex(BackendFailureError, "without acknowledging"):
            slot.wait()
        slot.recycle()
        self.assertFalse(slot.active)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_fused_interruption_before_ack_is_recyclable(self) -> None:
        source = Endpoint("source", backend="test-request-slot-fused-unacknowledged")
        destination = Endpoint(
            "destination", backend="test-request-slot-fused-unacknowledged"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()

        with self.assertRaises(BackendFailureError) as context:
            slot.start_and_wait()
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)
        self.assertEqual(slot.state, WorkState.FAILED)
        self.assertTrue(slot.active)
        slot.recycle()
        self.assertFalse(slot.active)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_single_mode_rejects_every_wrong_thread_operation(
        self,
    ) -> None:
        source = Endpoint(
            "source", backend="test-request-slot-fused", thread_mode="single"
        )
        destination = Endpoint(
            "destination", backend="test-request-slot-fused", thread_mode="single"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()

        def wrong_thread_error(function) -> BaseException:
            errors: list[BaseException] = []

            def invoke() -> None:
                try:
                    function()
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=invoke)
            thread.start()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            return errors[0]

        for function in (
            slot.start,
            slot.start_and_wait,
            lambda: slot.state,
            slot.test,
            slot.recycle,
            slot.close,
        ):
            self.assertIsInstance(wrong_thread_error(function), UnsupportedError)

        slot.start()
        slot.recycle()
        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_lifetime_blockers_and_recovery_snapshot(self) -> None:
        source = Endpoint("source", backend="test-request-slot")
        destination = Endpoint("destination", backend="test-request-slot")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()

        self.assertEqual(source.open_transfer_slots, (slot,))
        with self.assertRaisesRegex(BusyError, "request slots"):
            bound.close()
        with self.assertRaisesRegex(BusyError, "transfer slot"):
            source.close()
        slot.start()
        with self.assertRaisesRegex(BusyError, "unrecycled generation"):
            slot.close()
        slot.recycle()
        slot.close()
        self.assertEqual(source.open_transfer_slots, ())

        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_host_wait_runs_on_every_fused_start(self) -> None:
        source = Endpoint("source", backend="test-request-slot-fused")
        destination = Endpoint("destination", backend="test-request-slot-fused")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        local_region = local.region()
        local_region._device = "cuda:0"
        plan = source.prepare(local=[local_region], remote=[peer.region("buffer")])
        actions: list[str] = []
        synchronize_bind_count = 0

        class SynchronizeDescriptor:
            def __get__(self, instance: object, owner: object) -> object:
                nonlocal synchronize_bind_count
                synchronize_bind_count += 1
                return instance._synchronize

        class FakeEvent:
            device = torch.device("cuda:0")
            synchronize = SynchronizeDescriptor()

            def _synchronize(self) -> None:
                actions.append("wait")

        with (
            mock.patch.object(torch.cuda, "Event", FakeEvent),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ) as capture_query,
        ):
            bound = plan.bind(
                TransferOp.WRITE,
                cuda_ordering=CudaOrderingMode.HOST_WAIT,
                ready_event=FakeEvent(),
            )
            slot = bound.create_slot()
            self.assertEqual(synchronize_bind_count, 1)
            for _ in range(2):
                slot.start_and_wait()
                slot.recycle()

        self.assertEqual(actions, ["wait", "wait"])
        self.assertEqual(synchronize_bind_count, 1)
        capture_query.assert_not_called()
        slot.close()
        self.assertIsNone(slot._ready_event_synchronize)
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_cached_ready_event_failure_precedes_generation(self) -> None:
        source = Endpoint("source", backend="test-request-slot-fused")
        destination = Endpoint("destination", backend="test-request-slot-fused")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        local_region = local.region()
        local_region._device = "cuda:0"
        plan = source.prepare(local=[local_region], remote=[peer.region("buffer")])

        class FailingEvent:
            device = torch.device("cuda:0")

            def synchronize(self) -> None:
                raise RuntimeError("injected CUDA wait failure")

        with mock.patch.object(torch.cuda, "Event", FailingEvent):
            bound = plan.bind(
                TransferOp.WRITE,
                cuda_ordering=CudaOrderingMode.HOST_WAIT,
                ready_event=FailingEvent(),
            )
            slot = bound.create_slot()
            with self.assertRaisesRegex(BackendFailureError, "CUDA readiness"):
                slot.start_and_wait()

        self.assertEqual(slot.generation, 0)
        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertEqual(source._backend.request_slots[0].start_count, 0)
        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_host_wait_stream_reuses_one_event(self) -> None:
        source = Endpoint("source", backend="test-request-slot-fused")
        destination = Endpoint("destination", backend="test-request-slot-fused")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        local_region = local.region()
        local_region._device = "cuda:0"
        plan = source.prepare(local=[local_region], remote=[peer.region("buffer")])
        actions: list[tuple[str, int]] = []
        event_count = 0

        class FakeEvent:
            device = torch.device("cuda:0")

            def __init__(self) -> None:
                nonlocal event_count
                event_count += 1
                self.event_id = event_count

            def record(self, stream: object) -> None:
                self.assert_stream(stream)
                actions.append(("record", self.event_id))

            def assert_stream(self, stream: object) -> None:
                if not isinstance(stream, FakeStream):
                    raise AssertionError("readiness event recorded on wrong stream")

            def synchronize(self) -> None:
                actions.append(("wait", self.event_id))

        class FakeStream:
            device = torch.device("cuda:0")

            def is_capturing(self) -> bool:
                actions.append(("capture", event_count))
                return False

        with (
            mock.patch.object(torch, "Stream", FakeStream),
            mock.patch.object(torch.cuda, "Stream", FakeStream),
            mock.patch.object(torch.cuda, "Event", FakeEvent),
        ):
            bound = plan.bind(
                TransferOp.WRITE,
                cuda_ordering=CudaOrderingMode.HOST_WAIT,
                stream=FakeStream(),
            )
            slot = bound.create_slot()
            self.assertEqual(event_count, 1)
            self.assertEqual(slot.execute(), 1)
            self.assertEqual(slot.execute(), 2)

        self.assertEqual(
            actions,
            [
                ("capture", 1),
                ("record", 1),
                ("wait", 1),
                ("capture", 1),
                ("record", 1),
                ("wait", 1),
            ],
        )
        self.assertEqual(source._backend.request_slots[0].start_count, 2)
        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_reusable_stream_event_failure_precedes_generation(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-fused")
        destination = Endpoint("destination", backend="test-request-slot-fused")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        local_region = local.region()
        local_region._device = "cuda:0"
        plan = source.prepare(local=[local_region], remote=[peer.region("buffer")])
        event_count = 0
        wait_count = 0

        class FakeEvent:
            device = torch.device("cuda:0")

            def __init__(self) -> None:
                nonlocal event_count
                event_count += 1

            def record(self, _stream: object) -> None:
                return None

            def synchronize(self) -> None:
                nonlocal wait_count
                wait_count += 1
                if wait_count == 1:
                    raise RuntimeError("injected reusable-event wait failure")

        class FakeStream:
            device = torch.device("cuda:0")

            def is_capturing(self) -> bool:
                return False

        with (
            mock.patch.object(torch, "Stream", FakeStream),
            mock.patch.object(torch.cuda, "Stream", FakeStream),
            mock.patch.object(torch.cuda, "Event", FakeEvent),
        ):
            bound = plan.bind(
                TransferOp.WRITE,
                cuda_ordering=CudaOrderingMode.HOST_WAIT,
                stream=FakeStream(),
            )
            slot = bound.create_slot()
            with self.assertRaisesRegex(BackendFailureError, "CUDA readiness"):
                slot.execute()
            self.assertEqual(slot.generation, 0)
            self.assertFalse(slot.active)
            self.assertEqual(source._backend.request_slots[0].start_count, 0)
            self.assertEqual(slot.execute(), 1)

        self.assertEqual(event_count, 1)
        self.assertEqual(wait_count, 2)
        self.assertEqual(source._backend.request_slots[0].start_count, 1)
        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_creation_interruption_rolls_back_or_is_recoverable(
        self,
    ) -> None:
        for backend_name in (
            "test-request-slot-before-adoption-interrupt",
            "test-request-slot-create-interrupt",
        ):
            source = Endpoint("source", backend=backend_name)
            destination = Endpoint("destination", backend=backend_name)
            local = source.register(bytearray(b"data"), name="buffer")
            remote = destination.register(bytearray(4), name="buffer")
            peer = source.import_peer(bytes(destination.export_metadata()))
            plan = source.prepare(
                local=[local.region()], remote=[peer.region("buffer")]
            )
            bound = plan.bind(TransferOp.WRITE)

            with self.assertRaises(KeyboardInterrupt):
                bound.create_slot()
            self.assertEqual(source.open_transfer_slots, ())
            self.assertFalse(bound._slots)
            if backend_name == "test-request-slot-create-interrupt":
                self.assertEqual(source._backend.created_slot.close_count, 1)

            bound.close()
            plan.close()
            peer.close()
            local.close()
            remote.close()
            source.close()
            destination.close()

        source = Endpoint("source", backend="test-request-slot-cleanup-interrupt")
        destination = Endpoint(
            "destination", backend="test-request-slot-cleanup-interrupt"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)

        with self.assertRaisesRegex(BackendFailureError, "cleanup is pending"):
            bound.create_slot()
        (slot,) = source.open_transfer_slots
        with self.assertRaisesRegex(BackendFailureError, "cleanup only"):
            slot.start()
        slot.close()
        self.assertEqual(source.open_transfer_slots, ())

        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_creation_publication_interrupt_has_no_hidden_blocker(
        self,
    ) -> None:
        class InterruptAfterAdd(set):
            def add(self, value) -> None:
                super().add(value)
                raise KeyboardInterrupt

        source = Endpoint("source", backend="test-request-slot")
        destination = Endpoint("destination", backend="test-request-slot")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        bound._slots = InterruptAfterAdd(bound._slots)

        with self.assertRaises(KeyboardInterrupt):
            bound.create_slot()
        self.assertEqual(source.open_transfer_slots, ())
        self.assertFalse(bound._slots)
        self.assertEqual(source._backend.request_slots, [])

        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_close_commit_interruptions_are_retryable(self) -> None:
        class InterruptingSet(set):
            def __init__(self, values, *, before: bool) -> None:
                super().__init__(values)
                self.before = before
                self.armed = True

            def discard(self, value) -> None:
                if self.armed:
                    self.armed = False
                    if self.before:
                        raise KeyboardInterrupt
                    super().discard(value)
                    raise KeyboardInterrupt
                super().discard(value)

        for boundary in ("endpoint-after", "bound-before", "bound-after"):
            with self.subTest(boundary=boundary):
                source = Endpoint("source", backend="test-request-slot")
                destination = Endpoint("destination", backend="test-request-slot")
                local = source.register(bytearray(b"data"), name="buffer")
                remote = destination.register(bytearray(4), name="buffer")
                peer = source.import_peer(bytes(destination.export_metadata()))
                plan = source.prepare(
                    local=[local.region()], remote=[peer.region("buffer")]
                )
                bound = plan.bind(TransferOp.WRITE)
                slot = bound.create_slot()
                backend_slot = source._backend.request_slots[0]

                if boundary == "endpoint-after":
                    source._transfer_slots = InterruptingSet(
                        source._transfer_slots, before=False
                    )
                else:
                    bound._slots = InterruptingSet(
                        bound._slots, before=boundary == "bound-before"
                    )

                with self.assertRaises(KeyboardInterrupt):
                    slot.close()
                if boundary == "endpoint-after":
                    self.assertTrue(slot.closed)
                    self.assertNotIn(slot, bound._slots)
                else:
                    self.assertTrue(slot.closed)
                slot.close()
                self.assertTrue(slot.closed)
                self.assertIsNone(slot._bound)
                self.assertNotIn(slot, source._transfer_slots)
                self.assertNotIn(slot, bound._slots)
                self.assertGreaterEqual(backend_slot.close_count, 1)

                bound.close()
                plan.close()
                peer.close()
                local.close()
                remote.close()
                source.close()
                destination.close()

    def test_transfer_slot_pre_provider_interrupt_recycles_core_only(self) -> None:
        source = Endpoint("source", backend="test-request-slot")
        destination = Endpoint("destination", backend="test-request-slot")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]
        original_begin = TransferSlot._begin_generation_unlocked

        def interrupt_after_core_publication(self) -> None:
            original_begin(self)
            raise KeyboardInterrupt

        with mock.patch.object(
            TransferSlot,
            "_begin_generation_unlocked",
            interrupt_after_core_publication,
        ):
            self.assertIsNone(slot.start())

        self.assertTrue(slot.active)
        self.assertEqual(slot.generation, 1)
        self.assertEqual(backend_slot.start_count, 0)
        self.assertEqual(slot.state, WorkState.FAILED)
        slot.recycle()
        self.assertEqual(backend_slot.recycle_count, 0)

        backend_slot.next_state = WorkState.COMPLETED
        slot.start()
        self.assertEqual(slot.generation, 2)
        slot.recycle()
        self.assertEqual(backend_slot.recycle_count, 1)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_recycle_interrupt_publishes_clean_idle_last(self) -> None:
        class InterruptingIdleCommitTransferSlot(TransferSlot):
            __slots__ = ("interrupt_idle_commit",)

            def __init__(self, *args, **kwargs) -> None:
                self.interrupt_idle_commit = False
                super().__init__(*args, **kwargs)

            def __setattr__(self, name, value) -> None:
                object.__setattr__(self, name, value)
                if (
                    name == "_active"
                    and value is False
                    and getattr(self, "interrupt_idle_commit", False)
                ):
                    object.__setattr__(self, "interrupt_idle_commit", False)
                    raise KeyboardInterrupt

        source = Endpoint("source", backend="test-request-slot")
        destination = Endpoint("destination", backend="test-request-slot")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        with mock.patch.object(
            _api, "TransferSlot", InterruptingIdleCommitTransferSlot
        ):
            slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        prior_error = RuntimeError("prior generation failed")
        backend_slot.next_state = WorkState.FAILED
        backend_slot.next_error = prior_error
        slot.start()
        self.assertTrue(slot.test())
        self.assertIs(slot.error, prior_error)

        slot.interrupt_idle_commit = True
        with self.assertRaises(KeyboardInterrupt):
            slot.recycle()

        self.assertFalse(slot.active)
        self.assertIsNone(slot.state)
        self.assertIsNone(slot.error)
        # The public idle commit precedes private cleanup. The retained
        # terminal state is an explicit dirty-idle marker that the next begin
        # must repair before publishing another generation.
        self.assertEqual(slot._state, WorkState.FAILED)
        self.assertTrue(slot._backend_generation_active)
        self.assertIs(slot._error, prior_error)

        backend_slot.next_state = WorkState.COMPLETED
        backend_slot.next_error = None
        self.assertEqual(slot.execute(), 2)
        self.assertIsNone(slot.error)

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_terminal_error_publication_retries_after_interrupt(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot")
        destination = Endpoint("destination", backend="test-request-slot")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()
        backend_slot = source._backend.request_slots[0]

        failure = RuntimeError("retryable provider failure")
        backend_slot.next_state = WorkState.FAILED
        backend_slot.next_error = failure
        backend_slot.raise_error_once = KeyboardInterrupt()
        slot.start()
        with self.assertRaises(KeyboardInterrupt):
            _ = slot.state
        self.assertEqual(slot._state, WorkState.PENDING)
        self.assertEqual(slot.state, WorkState.FAILED)
        self.assertIs(slot.error, failure)
        slot.recycle()

        backend_slot.next_state = WorkState.CANCELLED
        slot.start()
        with (
            mock.patch(
                "torch.distributed._transfer._api.TransferCancelledError",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            _ = slot.state
        self.assertEqual(slot._state, WorkState.PENDING)
        self.assertEqual(slot.state, WorkState.CANCELLED)
        self.assertIsInstance(slot.error, TransferCancelledError)
        slot.recycle()

        backend_slot.next_state = None
        slot.start()
        with (
            mock.patch(
                "torch.distributed._transfer._api.BackendFailureError",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            _ = slot.state
        self.assertEqual(slot._state, WorkState.PENDING)
        self.assertEqual(slot.state, WorkState.FAILED)
        self.assertIsInstance(slot.error, BackendFailureError)
        slot.recycle()

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_transfer_slot_completion_event_is_capability_gated_and_generation_local(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-request-slot-no-completion-event")
        destination = Endpoint(
            "destination", backend="test-request-slot-no-completion-event"
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        slot = bound.create_slot()

        with self.assertRaisesRegex(BusyError, "no active generation"):
            _ = slot.completion_event
        slot.start()
        self.assertIsNone(slot.completion_event)
        slot.recycle()
        with self.assertRaisesRegex(BusyError, "no active generation"):
            _ = slot.completion_event

        slot.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_reference_retains_remote_owner_during_copy(self) -> None:
        class WeakBytearray(bytearray):
            pass

        source = Endpoint("source", thread_mode="multiple")
        destination = Endpoint("destination", thread_mode="multiple")
        source_registration = source.register(bytearray(b"data"), name="buffer")
        destination_buffer = WeakBytearray(4)
        destination_owner = weakref.ref(destination_buffer)
        destination_registration = destination.register(
            destination_buffer, name="buffer"
        )
        del destination_buffer
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[source_registration.region()],
            remote=[peer.region("buffer")],
        )

        entered_copy = threading.Event()
        continue_copy = threading.Event()
        original_copy = _reference._copy_spans

        def blocking_copy(source_spans, destination_spans, *, anchors=()):
            entered_copy.set()
            self.assertTrue(continue_copy.wait(1))
            self.assertIsNotNone(destination_owner())
            return original_copy(source_spans, destination_spans, anchors=anchors)

        _reference._copy_spans = blocking_copy
        result = []
        writer = threading.Thread(target=lambda: result.append(source.write(plan)))
        try:
            writer.start()
            self.assertTrue(entered_copy.wait(1))
            destination_registration.close()
            gc.collect()
            self.assertIsNotNone(destination_owner())
            continue_copy.set()
            writer.join(1)
            self.assertFalse(writer.is_alive())
            result[0].wait()
        finally:
            continue_copy.set()
            writer.join(1)
            _reference._copy_spans = original_copy

        result[0].close()
        plan.close()
        peer.close()
        source_registration.close()
        source.close()
        destination.close()

    def test_reference_tensor_and_storage_registration(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        source_tensor = torch.arange(8, dtype=torch.uint8)
        destination_tensor = torch.zeros(8, dtype=torch.uint8)
        source_registration = source.register(
            source_tensor.untyped_storage(), name="storage"
        )
        destination_registration = destination.register(
            destination_tensor, name="tensor"
        )
        peer = destination.import_peer(bytes(source.export_metadata()))
        plan = destination.prepare(
            local=[destination_registration.region()],
            remote=[peer.region("storage")],
        )

        work = destination.read(plan)
        work.wait()
        self.assertTrue(torch.equal(destination_tensor, source_tensor))

        work.close()
        plan.close()
        peer.close()
        destination_registration.close()
        source_registration.close()
        destination.close()
        source.close()

    def test_tensor_rebinding_keeps_registered_storage_alive(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        source_tensor = torch.arange(8, dtype=torch.uint8)
        expected = source_tensor.clone()
        destination_tensor = torch.zeros_like(source_tensor)
        source_registration = source.register(source_tensor, name="source")
        destination_registration = destination.register(
            destination_tensor, name="destination"
        )
        source_tensor.set_(torch.full_like(source_tensor, 99))
        peer = destination.import_peer(bytes(source.export_metadata()))
        plan = destination.prepare(
            local=[destination_registration.region()],
            remote=[peer.region("source")],
        )

        work = destination.read(plan)
        work.wait()
        self.assertTrue(torch.equal(destination_tensor, expected))

        work.close()
        plan.close()
        peer.close()
        destination_registration.close()
        source_registration.close()
        destination.close()
        source.close()

    def test_indexed_plan_reuse(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        source_buffer = bytearray(b"00112233")
        destination_buffer = bytearray(b"----------")
        source_registration = source.register(source_buffer, name="blocks")
        destination_registration = destination.register(
            destination_buffer, name="blocks"
        )
        peer = source.import_peer(bytes(destination.export_metadata()))
        local = [source_registration.region(index * 2, 2) for index in range(4)]
        remote = [peer.region("blocks", index * 2, 2) for index in range(5)]
        plan = source.prepare(local=local, remote=remote, indexed=True)

        first = source.write(plan, local_indices=[3, 1], remote_indices=[0, 2])
        self.assertTrue(first.test())
        self.assertEqual(destination_buffer, bytearray(b"33--11----"))
        first.close()
        second = source.write(plan, local_indices=[0], remote_indices=[4])
        second.wait()
        self.assertEqual(destination_buffer, bytearray(b"33--11--00"))

        second.close()
        plan.close()
        peer.close()
        source_registration.close()
        destination_registration.close()
        source.close()
        destination.close()

    def test_index_selection_snapshots_packed_int32_and_reuses_dynamic_tag(
        self,
    ) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        source_buffer = bytearray(b"a0b0c0d0")
        destination_buffer = bytearray(b"--------")
        local = source.register(source_buffer, name="source")
        remote = destination.register(destination_buffer, name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=2, count=4)],
            remote=[peer.region("destination", 0, 1, stride=2, count=4)],
            indexed=True,
        )
        local_indices = array("i", [3, 1])
        remote_indices = array("i", [0, 2])
        self.assertEqual(local_indices.itemsize, 4)
        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=local_indices,
            remote_indices=remote_indices,
        )
        self.assertIsInstance(selection, IndexSelection)
        self.assertEqual(selection.local_count, 2)
        self.assertEqual(selection.remote_count, 2)
        self.assertEqual(selection.selected_nbytes, 2)
        backend_selection = selection._backend_selection
        self.assertIsInstance(backend_selection, BackendIndexSelection)
        self.assertTrue(backend_selection.local_indices.readonly)
        self.assertTrue(backend_selection.remote_indices.readonly)
        self.assertIsInstance(backend_selection.local_indices.obj, bytes)
        self.assertIs(
            backend_selection.local_indices.obj,
            backend_selection.remote_indices.obj,
        )
        with self.assertRaisesRegex(AttributeError, "immutable"):
            selection._op = TransferOp.READ
        with self.assertRaisesRegex(AttributeError, "immutable"):
            backend_selection._destination_indices_unique = False

        # The caller-owned mutable arrays no longer affect the snapshot.
        local_indices[:] = array("i", [0, 0])
        remote_indices[:] = array("i", [1, 1])
        work = selection.submit(notification=b"first")
        self.assertIn(selection, work._dependencies)
        work.wait()
        self.assertEqual(destination_buffer, bytearray(b"d---b---"))
        self.assertEqual(
            [item.payload for item in destination.poll_notifications()],
            [b"first"],
        )
        work.close()

        destination_buffer[:] = b"--------"
        bound = selection.bind()
        self.assertIs(bound._selection, selection)
        work = bound.submit(notification=b"second")
        work.wait()
        self.assertEqual(destination_buffer, bytearray(b"d---b---"))
        self.assertEqual(
            [item.payload for item in destination.poll_notifications()],
            [b"second"],
        )
        work.close()
        bound.close()

        immutable_bound = plan.bind(
            TransferOp.WRITE,
            local_indices=[3, 1],
            remote_indices=[0, 2],
        )
        with self.assertRaisesRegex(TypeError, "IndexSelection binding"):
            immutable_bound.submit(notification=b"override")
        immutable_bound.close()
        plan.close()
        with self.assertRaises(ClosedError):
            selection.submit()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_index_selection_fused_submit_and_bounded_work_poll(self) -> None:
        source = Endpoint("source", backend="test-fused-selection")
        destination = Endpoint("destination", backend="test-fused-selection")
        local = source.register(bytearray(b"abcd"), name="source")
        remote = destination.register(bytearray(4), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("destination", 0, 1, stride=1, count=4)],
            indexed=True,
        )
        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=array("i", [3, 1]),
            remote_indices=array("i", [0, 2]),
        )
        backend = source._backend
        self.assertIsInstance(backend, _FusedSelectionReferenceBackend)

        work, terminal = selection.submit_and_test(
            notification=b"pending",
            max_polls=17,
            timeout_ns=23,
        )
        self.assertFalse(terminal)
        backend_work = backend.last_work
        self.assertIsNotNone(backend_work)
        assert backend_work is not None
        self.assertEqual(backend_work.state_queries, 0)
        self.assertEqual(
            work.poll_bounded(
                max_polls=31,
                timeout_ns=37,
                timeout_check_interval=7,
            ),
            WorkState.COMPLETED,
        )
        self.assertEqual(backend_work.bounded_calls, [(31, 37, 7)])
        self.assertEqual(backend_work.state_queries, 0)
        self.assertEqual(work.state, WorkState.COMPLETED)
        work.close()

        backend.initial_state = WorkState.COMPLETED
        work, state = selection.submit_and_poll(max_polls=0)
        self.assertIs(state, WorkState.COMPLETED)
        immediate_work = backend.last_work
        assert immediate_work is not None
        self.assertTrue(work.test())
        self.assertEqual(immediate_work.state_queries, 0)
        self.assertEqual(
            backend.fused_calls,
            [(17, 23, b"pending"), (0, None, None)],
        )

        work.close()
        work, terminal = selection.submit_and_test(max_polls=0)
        self.assertTrue(terminal)
        self.assertEqual(backend.last_work.state_queries, 0)

        for arguments, error_type in (
            ({"max_polls": True}, TypeError),
            ({"max_polls": -1}, ValueError),
            ({"timeout_ns": True}, TypeError),
            ({"timeout_ns": -1}, ValueError),
        ):
            with self.assertRaises(error_type):
                selection.submit_and_test(**arguments)
        with self.assertRaises(TypeError):
            work.test_bounded(max_polls=True)
        with self.assertRaises(ValueError):
            work.test_bounded(max_polls=0)
        work.close()

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_plan_one_shot_indices_use_private_snapshot_and_fallback(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        source_buffer = bytearray(b"a0b0c0d0")
        destination_buffer = bytearray(b"--------")
        local = source.register(source_buffer, name="source")
        remote = destination.register(destination_buffer, name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=2, count=4)],
            remote=[peer.region("destination", 0, 1, stride=2, count=4)],
            indexed=True,
        )
        local_indices = array("i", [3, 1])
        remote_indices = array("i", [0, 2])
        backend = source._backend
        original_submit = backend.submit_prevalidated
        captured: list[BackendIndexSelection] = []

        def capture_submit(*args, **kwargs) -> None:
            captured.append(kwargs["selection"])
            local_indices[:] = array("i", [0, 0])
            remote_indices[:] = array("i", [1, 1])
            original_submit(*args, **kwargs)

        backend.submit_prevalidated = capture_submit
        work = plan.submit_indices(
            TransferOp.WRITE,
            local_indices=local_indices,
            remote_indices=remote_indices,
            notification=b"one-shot",
        )
        self.assertEqual(destination_buffer, bytearray(b"d---b---"))
        self.assertEqual(len(captured), 1)
        backend_selection = captured[0]
        self.assertIn(backend_selection, work._dependencies)
        self.assertFalse(
            any(isinstance(value, IndexSelection) for value in work._dependencies)
        )
        self.assertTrue(backend_selection.local_indices.readonly)
        self.assertTrue(backend_selection.remote_indices.readonly)
        self.assertEqual(list(backend_selection.local_indices), [3, 1])
        self.assertEqual(list(backend_selection.remote_indices), [0, 2])
        self.assertEqual(
            [item.payload for item in destination.poll_notifications()],
            [b"one-shot"],
        )
        work.close()

        destination_buffer[:] = b"--------"
        work, terminal = plan.submit_indices_and_test(
            TransferOp.WRITE,
            local_indices=array("i", [2]),
            remote_indices=array("i", [3]),
            max_polls=0,
        )
        # The source-compatible provider cannot expose its post-derived state
        # without a poll, so zero remains conservatively nonterminal.
        self.assertFalse(terminal)
        self.assertTrue(work.test())
        self.assertEqual(destination_buffer, bytearray(b"------c-"))
        work.close()

        destination_buffer[:] = b"--------"
        work, state = plan.submit_indices_and_poll(
            TransferOp.WRITE,
            local_indices=array("i", [1]),
            remote_indices=array("i", [1]),
            max_polls=1,
            timeout_ns=None,
        )
        self.assertIs(state, WorkState.COMPLETED)
        self.assertEqual(destination_buffer, bytearray(b"--b-----"))
        work.close()

        # caller_ready never consumes local device identities. In particular,
        # a nonuniform-device catalog must not trigger an O(k) region walk just
        # to build facts that this ordering mode discards.
        plan._local_uniform_device = None
        with mock.patch.object(
            _api,
            "_select_region_blocks",
            side_effect=AssertionError("caller_ready walked local devices"),
        ):
            work = plan.submit_indices(
                TransferOp.WRITE,
                local_indices=[0],
                remote_indices=[0],
            )
        work.close()

        with self.assertRaisesRegex(ValueError, "non-empty"):
            plan.submit_indices(
                TransferOp.WRITE,
                local_indices=[0],
                remote_indices=[0],
                notification=b"",
            )
        with self.assertRaisesRegex(ValueError, "does not accept"):
            plan.submit_indices(
                TransferOp.WRITE,
                local_indices=[0],
                remote_indices=[0],
                stream=object(),
            )
        for method_name in (
            "submit_indices",
            "submit_indices_and_poll",
            "submit_indices_and_test",
        ):
            method = getattr(plan, method_name)
            with self.subTest(method=method_name, argument="op"):
                with self.assertRaisesRegex(ValueError, "op must be one of"):
                    method(
                        None,
                        local_indices=[0],
                        remote_indices=[0],
                    )
            with self.subTest(method=method_name, argument="local_indices"):
                with self.assertRaisesRegex(
                    InvalidRegionError, "local_indices must be non-empty"
                ):
                    method(
                        TransferOp.WRITE,
                        local_indices=None,
                        remote_indices=[0],
                    )
            with self.subTest(method=method_name, argument="remote_indices"):
                with self.assertRaisesRegex(
                    InvalidRegionError, "remote_indices must be non-empty"
                ):
                    method(
                        TransferOp.WRITE,
                        local_indices=[0],
                        remote_indices=None,
                    )

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_plan_one_shot_indices_enter_fused_provider_boundary(self) -> None:
        source = Endpoint("source", backend="test-fused-selection")
        destination = Endpoint("destination", backend="test-fused-selection")
        local = source.register(bytearray(b"abcd"), name="source")
        remote = destination.register(bytearray(4), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("destination", 0, 1, stride=1, count=4)],
            indexed=True,
        )
        backend = source._backend
        self.assertIsInstance(backend, _FusedSelectionReferenceBackend)
        backend.initial_state = WorkState.COMPLETED

        work, state = plan.submit_indices_and_poll(
            TransferOp.WRITE,
            local_indices=array("i", [3, 1]),
            remote_indices=array("i", [0, 2]),
            notification=b"fused-one-shot",
            max_polls=0,
            timeout_ns=None,
        )
        self.assertIs(state, WorkState.COMPLETED)
        self.assertEqual(backend.fused_calls, [(0, None, b"fused-one-shot")])
        self.assertTrue(
            any(
                isinstance(value, BackendIndexSelection) for value in work._dependencies
            )
        )
        self.assertFalse(
            any(isinstance(value, IndexSelection) for value in work._dependencies)
        )
        work.close()

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_fused_selection_immediate_state_retains_epilogue_interrupt(self) -> None:
        class InterruptOnceOnDiscard(set):
            armed = True

            def discard(self, value) -> None:
                super().discard(value)
                if self.armed:
                    self.armed = False
                    raise KeyboardInterrupt

        source = Endpoint("source", backend="test-fused-selection")
        destination = Endpoint("destination", backend="test-fused-selection")
        local = source.register(bytearray(b"abcd"), name="source")
        remote = destination.register(bytearray(4), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("destination", 0, 1, stride=1, count=4)],
            indexed=True,
        )
        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=array("i", [3, 1]),
            remote_indices=array("i", [0, 2]),
        )
        backend = source._backend
        self.assertIsInstance(backend, _FusedSelectionReferenceBackend)
        backend.initial_state = WorkState.COMPLETED
        plan._submitting = InterruptOnceOnDiscard()

        work, terminal = selection.submit_and_test(max_polls=0)
        self.assertTrue(terminal)
        self.assertFalse(plan._submitting)
        self.assertEqual(source.open_works, (work,))
        self.assertEqual(work.state, WorkState.FAILED)
        self.assertIsInstance(work.error, KeyboardInterrupt)
        with self.assertRaises(BackendFailureError) as context:
            work.wait()
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)

        work.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_fused_selection_pending_state_reconciles_epilogue_interrupt(self) -> None:
        class InterruptOnceOnDiscard(set):
            armed = True

            def discard(self, value) -> None:
                super().discard(value)
                if self.armed:
                    self.armed = False
                    raise KeyboardInterrupt

        source = Endpoint("source", backend="test-fused-selection")
        destination = Endpoint("destination", backend="test-fused-selection")
        local = source.register(bytearray(b"abcd"), name="source")
        remote = destination.register(bytearray(4), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("destination", 0, 1, stride=1, count=4)],
            indexed=True,
        )
        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=array("i", [3, 1]),
            remote_indices=array("i", [0, 2]),
        )
        backend = source._backend
        self.assertIsInstance(backend, _FusedSelectionReferenceBackend)
        backend.initial_state = WorkState.PENDING
        plan._submitting = InterruptOnceOnDiscard()

        work, terminal = selection.submit_and_test(max_polls=0)
        self.assertFalse(terminal)
        self.assertFalse(plan._submitting)
        self.assertEqual(source.open_works, (work,))
        self.assertTrue(work.test_bounded(max_polls=1, timeout_ns=None))
        self.assertEqual(work.state, WorkState.FAILED)
        self.assertIsInstance(work.error, KeyboardInterrupt)
        with self.assertRaises(BackendFailureError) as context:
            work.wait()
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)

        work.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_index_selection_accepts_index_scalars_and_rejects_unsafe_values(
        self,
    ) -> None:
        class IndexScalar:
            def __init__(self, value: int) -> None:
                self.value = value

            def __index__(self) -> int:
                return self.value

        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"abcd"), name="source")
        remote = destination.register(bytearray(4), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("destination", 0, 1, stride=1, count=4)],
            indexed=True,
        )

        selection = plan.select_indices(
            "write",
            local_indices=[IndexScalar(3), IndexScalar(1)],
            remote_indices=[IndexScalar(0), IndexScalar(2)],
        )
        work = selection.submit()
        work.close()
        for local_indices, remote_indices, message in (
            ([True], [0], "integers"),
            ([0], [False], "integers"),
            ([4], [0], "outside"),
            ([0, 1], [2, 2], "overlap"),
            ([], [], "non-empty"),
        ):
            with self.assertRaisesRegex(InvalidRegionError, message):
                plan.select_indices(
                    TransferOp.WRITE,
                    local_indices=local_indices,
                    remote_indices=remote_indices,
                )

        # The raw convenience fallback now follows the same operator.index
        # contract, without requiring callers to construct a selection.
        work = source.write(
            plan,
            local_indices=[IndexScalar(0)],
            remote_indices=[IndexScalar(1)],
        )
        work.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_index_selection_native_int32_analyzer_matches_fallback(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        block_count = _api._INDEX_SELECTION_TORCH_THRESHOLD + 3
        local = source.register(bytearray(block_count), name="source")
        remote = destination.register(bytearray(block_count), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=block_count)],
            remote=[peer.region("destination", 0, 1, stride=1, count=block_count)],
            indexed=True,
        )
        cutoff = _api._INDEX_SELECTION_TORCH_THRESHOLD
        signatures = []
        for analyzer in (None, _emulated_compiled_index_analyzer):
            with mock.patch.object(_api, "_compiled_index_analyzer", analyzer):
                selection = plan.select_indices(
                    TransferOp.WRITE,
                    local_indices=array("i", range(cutoff + 1)),
                    remote_indices=array("i", reversed(range(cutoff + 1))),
                )
                backend_selection = selection._backend_selection
                signatures.append(
                    (
                        bytes(backend_selection.local_indices),
                        bytes(backend_selection.remote_indices),
                        selection.local_count,
                        selection.remote_count,
                        selection.selected_nbytes,
                    )
                )
        self.assertEqual(signatures[0], signatures[1])

        caller_local = array("i", range(cutoff + 1))
        caller_remote = array("i", reversed(range(cutoff + 1)))

        def mutate_after_snapshot(*args, **kwargs):
            result = _emulated_compiled_index_analyzer(*args, **kwargs)
            caller_local[:] = array("i", [0]) * len(caller_local)
            caller_remote[:] = array("i", [0]) * len(caller_remote)
            return result

        analyzer = mock.Mock(side_effect=mutate_after_snapshot)
        with mock.patch.object(_api, "_compiled_index_analyzer", analyzer):
            selection = plan.select_indices(
                TransferOp.WRITE,
                local_indices=caller_local,
                remote_indices=caller_remote,
            )
        analyzer.assert_called_once()
        self.assertEqual(
            list(selection._backend_selection.local_indices),
            list(range(cutoff + 1)),
        )
        self.assertEqual(
            list(selection._backend_selection.remote_indices),
            list(reversed(range(cutoff + 1))),
        )

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_index_selection_compiled_analyzer_preserves_error_precedence(
        self,
    ) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(8), name="source")
        remote = destination.register(bytearray(8), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=8)],
            remote=[peer.region("destination", 0, 1, stride=1, count=8)],
            indexed=True,
        )
        cases = (
            (array("i", [-1]), array("i", [8]), "local_indices.*outside"),
            (array("i", [0]), array("i", [8]), "remote_indices.*outside"),
            (
                array("i", [0, 1, 2]),
                array("i", [0, 0]),
                "byte totals differ",
            ),
            (array("i", [0, 1]), array("i", [0, 0]), "overlap"),
        )
        for analyzer in (None, _emulated_compiled_index_analyzer):
            for local_indices, remote_indices, message in cases:
                with (
                    self.subTest(analyzer=analyzer, message=message),
                    mock.patch.object(_api, "_compiled_index_analyzer", analyzer),
                    self.assertRaisesRegex(InvalidRegionError, message),
                ):
                    plan.select_indices(
                        TransferOp.WRITE,
                        local_indices=local_indices,
                        remote_indices=remote_indices,
                    )

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_index_selection_mixed_inputs_preserve_error_order_and_snapshot(
        self,
    ) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(4), name="source")
        remote = destination.register(bytearray(4), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("destination", 0, 1, stride=1, count=4)],
            indexed=True,
        )

        cases = (
            (array("i", [-1]), [True], r"local_indices index -1.*outside"),
            (array("i", [4]), [False], r"local_indices index 4.*outside"),
            (array("i", [-1]), [4], r"local_indices index -1.*outside"),
            (array("i", [-1]), [], r"local_indices index -1.*outside"),
            ([True], array("i", [4]), r"local_indices must contain integers"),
            ([4], array("i", [-1]), r"local_indices index 4.*outside"),
            ([], array("i", [-1]), r"local_indices must be non-empty"),
        )
        for local_indices, remote_indices, message in cases:
            with (
                self.subTest(
                    local_type=type(local_indices).__name__,
                    remote_type=type(remote_indices).__name__,
                ),
                self.assertRaisesRegex(InvalidRegionError, message),
            ):
                plan.select_indices(
                    TransferOp.WRITE,
                    local_indices=local_indices,
                    remote_indices=remote_indices,
                )

        caller_local = array("i", [3, 1])

        class MutatingRemote:
            def __iter__(self):
                caller_local[:] = array("i", [0, 0])
                return iter((0, 2))

        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=caller_local,
            remote_indices=MutatingRemote(),
        )
        self.assertEqual(list(selection._backend_selection.local_indices), [3, 1])
        self.assertEqual(list(selection._backend_selection.remote_indices), [0, 2])

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_index_selection_compiled_analyzer_exception_is_not_retried(
        self,
    ) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(4), name="source")
        remote = destination.register(bytearray(4), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("destination", 0, 1, stride=1, count=4)],
            indexed=True,
        )
        analyzer = mock.Mock(side_effect=RuntimeError("compiled helper failed"))
        with (
            mock.patch.object(_api, "_compiled_index_analyzer", analyzer),
            self.assertRaisesRegex(RuntimeError, "compiled helper failed"),
        ):
            plan.select_indices(
                TransferOp.WRITE,
                local_indices=array("i", [0, 1]),
                remote_indices=array("i", [2, 3]),
            )
        analyzer.assert_called_once()

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_index_selection_missing_helper_and_third_party_sequences(self) -> None:
        class IndexScalar:
            def __init__(self, value: int) -> None:
                self.value = value

            def __index__(self) -> int:
                return self.value

        class ThirdPartyIndices:
            def __init__(self, *values: int) -> None:
                self.values = values

            def __iter__(self):
                return iter(IndexScalar(value) for value in self.values)

        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"abcd"), name="source")
        remote = destination.register(bytearray(4), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("destination", 0, 1, stride=1, count=4)],
            indexed=True,
        )

        with mock.patch.object(_api, "_compiled_index_analyzer", None):
            fallback = plan.select_indices(
                TransferOp.WRITE,
                local_indices=array("i", [3, 1]),
                remote_indices=array("i", [0, 2]),
            )
        self.assertEqual(list(fallback._backend_selection.local_indices), [3, 1])

        analyzer = mock.Mock(side_effect=AssertionError("must not be called"))
        with mock.patch.object(_api, "_compiled_index_analyzer", analyzer):
            mixed = plan.select_indices(
                TransferOp.WRITE,
                local_indices=array("i", [2, 0]),
                remote_indices=ThirdPartyIndices(1, 3),
            )
        analyzer.assert_not_called()
        self.assertEqual(list(mixed._backend_selection.remote_indices), [1, 3])

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_compiled_index_analyzer_defensive_buffer_contract(self) -> None:
        analyzer = _api._compiled_index_analyzer
        if analyzer is None:
            self.skipTest("requires torch._C rebuilt with the transfer helper")

        local = array("i", [3, 1, 2])
        remote = array("i", [0, 0, 2])
        result = analyzer(local, remote, True)
        self.assertEqual(result[0], local.tobytes() + remote.tobytes())
        self.assertEqual(result[1:7], (3, 3, 1, 3, 0, 2))
        self.assertTrue(result[7])

        targeted_destinations = (
            array("i", [5]),
            array("i", [5, 5]),
            array("i", range(4096)),
            array("i", reversed(range(4096))),
            array("i", [2, 0, 3, 1]),
            array("i", [7, 7, -5, 100]),
            array("i", ((index * 2053) % 4096 for index in range(4096))),
            array(
                "i",
                (((index * 37) % 64) * 104729 for index in range(64)),
            ),
            array("i", [0, 104729, -65537, 209458]),
            array("i", [7, 8, 7]),
            array("i", [0, 1, 2, 3, 4, 5, 6, 7, -1, 9, 10]),
            array("i", [*range(4095), -1]),
            array("i", [*reversed(range(1, 4096)), 4096]),
            array("i", [-(2**31), 2**31 - 1, 0]),
        )
        for destination in targeted_destinations:
            expected_duplicate = len(set(destination)) != len(destination)
            for destination_is_remote in (False, True):
                with self.subTest(
                    destination_is_remote=destination_is_remote,
                    destination=list(destination[:8]),
                ):
                    if destination_is_remote:
                        candidate = analyzer(local, destination, True)
                    else:
                        candidate = analyzer(destination, remote, False)
                    self.assertEqual(candidate[7], expected_duplicate)
                    destination_bounds = (
                        candidate[5:7] if destination_is_remote else candidate[3:5]
                    )
                    self.assertEqual(
                        destination_bounds, (min(destination), max(destination))
                    )

        invalid_buffers = (
            array("i"),
            array("I", [0]),
            array("q", [0]),
            memoryview(array("i", [0, 1, 2, 3]))[::2],
        )
        for invalid in invalid_buffers:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                analyzer(invalid, array("i", [0]), True)

    def test_index_selection_reuse_token_is_lazy_and_not_redundant(self) -> None:
        source = Endpoint("source", backend="test-request-slot")
        destination = Endpoint("destination", backend="test-request-slot")
        local = source.register(bytearray(b"abcd"), name="source")
        remote = destination.register(bytearray(4), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("destination", 0, 1, stride=1, count=4)],
            indexed=True,
        )
        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=array("i", [3, 1]),
            remote_indices=array("i", [0, 2]),
        )
        backend = source._backend
        original_submit = backend.submit_prevalidated
        observed_tokens: list[object | None] = []

        def capture_submit(*args, **kwargs) -> None:
            observed_tokens.append(kwargs["reuse_token"])
            original_submit(*args, **kwargs)

        backend.submit_prevalidated = capture_submit
        self.assertIsNone(selection._reuse_token)
        direct = selection.submit()
        direct.close()
        self.assertEqual(observed_tokens, [None])
        self.assertIsNone(selection._reuse_token)

        first_bound = selection.bind()
        self.assertIsNone(first_bound._reuse_token)
        slot = first_bound.create_slot()
        self.assertIsNone(selection._reuse_token)
        slot.close()

        first = first_bound.submit()
        token = selection._reuse_token
        self.assertIsNotNone(token)
        first.close()
        second = first_bound.submit()
        second.close()
        second_bound = selection.bind()
        self.assertIsNone(second_bound._reuse_token)
        third = second_bound.submit()
        third.close()
        self.assertEqual(len(observed_tokens), 4)
        self.assertIsNone(observed_tokens[0])
        self.assertIs(observed_tokens[1], token)
        self.assertIs(observed_tokens[2], token)
        self.assertIs(observed_tokens[3], token)

        second_bound.close()
        first_bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_index_selection_irregular_layout_keeps_conservative_fallback(
        self,
    ) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"abc"), name="source")
        remote = destination.register(bytearray(3), name="destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1), local.region(1, 2)],
            remote=[peer.region("destination", 0, 3)],
            indexed=True,
        )
        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=[0, 1],
            remote_indices=[0],
        )
        self.assertFalse(selection._backend_selection.uniform_pair_layout)
        self.assertEqual(selection.selected_nbytes, 3)
        work = selection.submit()
        work.wait()
        work.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_backend_index_selection_cannot_be_forged(self) -> None:
        with self.assertRaisesRegex(TypeError, "created by Core"):
            BackendIndexSelection(
                TransferOp.WRITE,
                object(),
                memoryview(array("i", [0])),
                memoryview(array("i", [0])),
                selected_nbytes=1,
                uniform_pair_layout=True,
                destination_catalog_nonoverlapping=True,
                destination_indices_unique=True,
                _token=object(),
            )

    def test_indexed_compact_strided_block_catalog(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        source_buffer = bytearray(b"a0b0c0d0")
        destination_buffer = bytearray(b"--------")
        source_registration = source.register(source_buffer, name="source")
        destination_registration = destination.register(
            destination_buffer, name="destination"
        )
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[source_registration.region(0, 1, stride=2, count=4)],
            remote=[peer.region("destination", 0, 1, stride=2, count=4)],
            indexed=True,
        )

        self.assertEqual(plan.local_block_count, 4)
        self.assertEqual(plan.remote_block_count, 4)
        work = source.write(
            plan,
            local_indices=[3, 1],
            remote_indices=[0, 2],
        )
        work.wait()
        self.assertEqual(destination_buffer, bytearray(b"d---b---"))
        work.close()
        with self.assertRaises(InvalidRegionError):
            source.write(plan, local_indices=[4], remote_indices=[0])

        plan.close()
        peer.close()
        source_registration.close()
        destination_registration.close()
        source.close()
        destination.close()

    def test_indexed_logical_block_sizes_must_match(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        destination_buffer = bytearray(4)
        source_registration = source.register(bytearray(b"a0b0"), name="source")
        destination_registration = destination.register(
            destination_buffer, name="destination"
        )
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[source_registration.region(0, 1, stride=2, count=2)],
            remote=[peer.region("destination", 0, 2, stride=2, count=2)],
            indexed=True,
        )

        with self.assertRaisesRegex(InvalidRegionError, "byte totals differ"):
            source.write(plan, local_indices=[0], remote_indices=[0])
        work = source.write(
            plan,
            local_indices=[0, 1],
            remote_indices=[1],
        )
        work.wait()
        self.assertEqual(destination_buffer, bytearray(b"\x00\x00ab"))
        work.close()

        plan.close()
        peer.close()
        source_registration.close()
        destination_registration.close()
        source.close()
        destination.close()

    def test_strided_regions(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        source_buffer = bytearray(b"a0b0c0d0")
        destination_buffer = bytearray(4)
        source_registration = source.register(source_buffer, name="source")
        destination_registration = destination.register(
            destination_buffer, name="destination"
        )
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[source_registration.region(0, 1, stride=2, count=4)],
            remote=[peer.region("destination")],
        )

        work = source.write(plan)
        work.wait()
        self.assertEqual(destination_buffer, bytearray(b"abcd"))

        work.close()
        plan.close()
        peer.close()
        source_registration.close()
        destination_registration.close()
        source.close()
        destination.close()

    def test_notifications_identify_source(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])

        with self.assertRaisesRegex(ValueError, "non-empty"):
            source.write(plan, notification=b"")
        transfer = source.write(plan, notification=b"visible")
        transfer.wait()
        notices = destination.poll_notifications()
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0].payload, b"visible")
        self.assertEqual(notices[0].source_endpoint_id, source.endpoint_id)
        self.assertEqual(notices[0].source_incarnation, source.incarnation)
        self.assertEqual(notices[0].source, source.identity)

        class OversizedPayload(bytearray):
            converted = False

            def __bytes__(self) -> bytes:
                self.converted = True
                return super().__bytes__()

        oversized = OversizedPayload(b"1234")
        with (
            mock.patch("torch.distributed._transfer._api._MAX_NOTIFICATION_BYTES", 3),
            self.assertRaisesRegex(ValueError, "size limit"),
        ):
            source.send_notification(peer, oversized)
        self.assertFalse(oversized.converted)

        standalone = source.send_notification(peer, b"standalone")
        standalone.wait()
        self.assertEqual(
            [item.payload for item in destination.poll_notifications()],
            [b"standalone"],
        )

        self.assertTrue(source.capabilities.synchronous_notification_send)
        self.assertEqual(set(source.open_works), {standalone, transfer})
        sync_payload = b"sync-exact-payload"
        self.assertIsNone(source.send_notification_sync(peer, sync_payload))
        self.assertEqual(set(source.open_works), {standalone, transfer})
        sync_notifications = destination.poll_notifications()
        self.assertEqual([item.payload for item in sync_notifications], [sync_payload])
        self.assertIs(sync_notifications[0].payload, sync_payload)

        provider_failure = RuntimeError("provider failed after entry")

        def fail_after_entry(*args, **kwargs) -> None:
            del args, kwargs
            raise provider_failure

        source._send_notification_sync_dispatch = fail_after_entry
        with self.assertRaisesRegex(
            BackendFailureError, "delivery outcome may be ambiguous"
        ) as context:
            source.send_notification_sync(peer, b"ambiguous")
        self.assertIs(context.exception.__cause__, provider_failure)

        transfer_error = ClosedError("provider is closed")

        def fail_with_transfer_error(*args, **kwargs) -> None:
            del args, kwargs
            raise transfer_error

        source._send_notification_sync_dispatch = fail_with_transfer_error
        with self.assertRaises(ClosedError) as context:
            source.send_notification_sync(peer, b"rejected")
        self.assertIs(context.exception, transfer_error)

        source._send_notification_sync_dispatch = lambda *args, **kwargs: object()
        with self.assertRaisesRegex(
            BackendFailureError, "must return None.*outcome may be ambiguous"
        ):
            source.send_notification_sync(peer, b"invalid-result")
        self.assertEqual(set(source.open_works), {standalone, transfer})

        standalone.close()
        transfer.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_attached_only_backend_can_poll_notifications(self) -> None:
        endpoint = Endpoint("endpoint", backend="test-attached-only")
        destination = Endpoint("destination", backend="test-attached-only")
        peer = endpoint.import_peer(bytes(destination.export_metadata()))
        self.assertTrue(endpoint.capabilities.attached_notifications)
        self.assertFalse(endpoint.capabilities.standalone_notifications)
        self.assertFalse(endpoint.capabilities.synchronous_notification_send)
        with self.assertRaisesRegex(UnsupportedError, "synchronous notification"):
            endpoint.send_notification_sync(peer, b"unsupported")
        with self.assertRaisesRegex(UnsupportedError, "synchronous standalone"):
            _api.Backend.send_notification_sync(
                endpoint._backend, object(), b"unsupported"
            )
        self.assertEqual(endpoint.poll_notifications(), [])
        peer.close()
        endpoint.close()
        destination.close()

    def test_multiple_synchronous_notification_sends_overlap_and_pin_peer(
        self,
    ) -> None:
        source = Endpoint("source", thread_mode="multiple")
        destination = Endpoint("destination", thread_mode="multiple")
        peer = source.import_peer(bytes(destination.export_metadata()))
        original_send = source._send_notification_sync_dispatch
        self.assertIsNotNone(original_send)
        entered_first = threading.Event()
        entered_both = threading.Event()
        release = threading.Event()
        count_lock = threading.Lock()
        entered = 0
        errors: list[BaseException] = []

        def blocking_send(peer_handle, payload) -> None:
            nonlocal entered
            with count_lock:
                entered += 1
                entered_first.set()
                if entered == 2:
                    entered_both.set()
            if not release.wait(2):
                raise RuntimeError("timed out waiting to release concurrent sends")
            assert original_send is not None
            original_send(peer_handle, payload)

        source._send_notification_sync_dispatch = blocking_send

        def send(payload: bytes) -> None:
            try:
                source.send_notification_sync(peer, payload)
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=send, args=(payload,))
            for payload in (b"first", b"second")
        ]
        try:
            for thread in threads:
                thread.start()
            self.assertTrue(entered_first.wait(1))
            self.assertTrue(
                entered_both.wait(1),
                "MULTIPLE synchronous sends were serialized before native entry",
            )
            with self.assertRaisesRegex(BusyError, "synchronous notification"):
                peer.close()
        finally:
            release.set()
            for thread in threads:
                thread.join(2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(
            sorted(item.payload for item in destination.poll_notifications()),
            [b"first", b"second"],
        )
        peer.close()
        source.close()
        destination.close()

    def test_multiple_synchronous_notification_interrupted_cleanup_cannot_wedge_peer(
        self,
    ) -> None:
        class InterruptSecondEnter:
            def __init__(self, lock) -> None:
                self.lock = lock
                self.enters = 0

            def __enter__(self):
                self.enters += 1
                if self.enters == 2:
                    raise KeyboardInterrupt
                return self.lock.__enter__()

            def __exit__(self, *args):
                return self.lock.__exit__(*args)

        source = Endpoint("source", thread_mode="multiple")
        destination = Endpoint("destination", thread_mode="multiple")
        peer = source.import_peer(bytes(destination.export_metadata()))
        original_lock = source._state_lock
        source._state_lock = InterruptSecondEnter(original_lock)
        retained_error = None
        try:
            try:
                source.send_notification_sync(peer, b"accepted-before-interrupt")
            except KeyboardInterrupt as error:
                # Retain the traceback, and therefore the completed send frame,
                # through peer.close(). Close must use the explicit completion
                # bit rather than mistake weakref liveness for native activity.
                retained_error = error
            else:
                self.fail("cleanup-lock interruption did not propagate")
        finally:
            source._state_lock = original_lock

        gc.collect()
        self.assertIsNotNone(retained_error)
        self.assertTrue(peer._active_sync_notification_sends)
        self.assertTrue(
            all(
                reference() is None or not reference().active
                for reference in peer._active_sync_notification_sends
            )
        )
        # close() prunes dead reservations and cannot mistake interrupted
        # cleanup for a still-running native call.
        peer.close()
        self.assertFalse(peer._active_sync_notification_sends)
        retained_error = None
        source.close()
        destination.close()

    def test_multiple_async_notification_send_overlaps_manual_progress(self) -> None:
        # This is a backend-entry concurrency test, not a timing benchmark.
        source = Endpoint("source", thread_mode=ThreadMode.MULTIPLE)
        destination = Endpoint("destination", thread_mode=ThreadMode.MULTIPLE)
        peer = source.import_peer(bytes(destination.export_metadata()))
        backend_work = _PendingWork()
        send_entered = threading.Event()
        progress_entered = threading.Event()
        release_send = threading.Event()
        errors: list[BaseException] = []
        returned: list[Work] = []

        def blocked_send(*args, **kwargs) -> None:
            kwargs["adopt_work"](backend_work)
            send_entered.set()
            if not release_send.wait(2):
                raise RuntimeError("test did not release notification send")

        def observe_progress() -> None:
            progress_entered.set()

        source._backend.send_notification = blocked_send
        source._backend.progress = observe_progress

        def send() -> None:
            try:
                returned.append(source.send_notification(peer, b"overlap"))
            except BaseException as error:
                errors.append(error)

        send_thread = threading.Thread(target=send)
        send_thread.start()
        try:
            self.assertTrue(send_entered.wait(1))
            progress_thread = threading.Thread(target=source.progress)
            progress_thread.start()
            progress_thread.join(1)
            self.assertFalse(progress_thread.is_alive())
            self.assertTrue(progress_entered.is_set())
            with self.assertRaises(BusyError):
                peer.close()
        finally:
            release_send.set()
            send_thread.join(2)

        self.assertFalse(send_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(returned), 1)
        backend_work._state = WorkState.COMPLETED
        returned[0].close()
        peer.close()
        source.close()
        destination.close()

    def test_multiple_notification_consumer_admission_serializes_and_blocks_close(
        self,
    ) -> None:
        endpoint = Endpoint("consumer", thread_mode=ThreadMode.MULTIPLE)
        first_entered = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        count_lock = threading.Lock()
        entries = 0
        errors: list[BaseException] = []

        def blocked_poll(max_items=None):
            del max_items
            nonlocal entries
            with count_lock:
                entries += 1
                if entries == 1:
                    first_entered.set()
                else:
                    second_entered.set()
            if entries == 1 and not release_first.wait(2):
                raise RuntimeError("test did not release first notification poll")
            return []

        endpoint._backend.poll_notifications = blocked_poll

        def poll() -> None:
            try:
                endpoint.poll_notifications()
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=poll)
        second = threading.Thread(target=poll)
        first.start()
        self.assertTrue(first_entered.wait(1))
        second.start()
        self.assertFalse(second_entered.wait(0.1))

        close_errors: list[BaseException] = []

        def close() -> None:
            try:
                endpoint.close()
            except BaseException as error:
                close_errors.append(error)

        close_thread = threading.Thread(target=close)
        close_thread.start()
        close_thread.join(1)
        self.assertFalse(close_thread.is_alive())
        self.assertEqual(len(close_errors), 1)
        self.assertIsInstance(close_errors[0], BusyError)
        self.assertFalse(endpoint.closed)

        release_first.set()
        first.join(2)
        second.join(2)
        self.assertTrue(second_entered.is_set())
        self.assertTrue(all(not thread.is_alive() for thread in (first, second)))
        self.assertEqual(errors, [])
        (warmed_reservation,) = endpoint._multiple_backend_call_reservation_pool
        endpoint.poll_notifications()
        self.assertEqual(len(endpoint._multiple_backend_call_reservation_pool), 1)
        self.assertIs(
            endpoint._multiple_backend_call_reservation_pool[0], warmed_reservation
        )
        endpoint.close()

    def test_notification_batches_are_immutable_bounded_and_source_compatible(
        self,
    ) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        first = source.send_notification(peer, b"first")
        second_payload = bytearray(b"second")
        second = source.send_notification(peer, second_payload)
        first.wait()
        second.wait()
        second_payload[:] = b"mutate"

        with mock.patch.object(
            destination._backend,
            "poll_notification_batches",
            side_effect=AssertionError("zero max_items must not enter the backend"),
        ):
            self.assertEqual(destination.poll_notification_batches(max_items=0), [])
        for invalid, error_type in (
            (True, TypeError),
            (-1, ValueError),
            (1.0, TypeError),
        ):
            with self.assertRaises(error_type):
                destination.poll_notification_batches(max_items=invalid)  # type: ignore[arg-type]

        batches = destination.poll_notification_batches()
        self.assertEqual(len(batches), 1)
        self.assertIsInstance(batches[0], NotificationBatch)
        self.assertEqual(batches[0].payloads, (b"first", b"second"))
        self.assertEqual(batches[0].source, source.identity)

        exact_payloads = (b"retained-one", b"retained-two")
        retained = BackendNotificationBatch(
            exact_payloads,
            source_endpoint_id=source.endpoint_id,
            source_incarnation=source.incarnation,
        )
        self.assertIs(retained.payloads, exact_payloads)
        self.assertEqual(retained.total_nbytes, sum(map(len, exact_payloads)))
        with (
            mock.patch.object(
                destination._backend,
                "poll_notification_batches",
                return_value=[retained],
            ),
            mock.patch.object(
                sys.modules[Endpoint.__module__],
                "NotificationBatch",
                side_effect=lambda *args, **kwargs: (
                    self.assertFalse(destination._state_lock._is_owned())
                    or NotificationBatch(*args, **kwargs)
                ),
            ),
        ):
            public = destination.poll_notification_batches()
        self.assertIs(public[0].payloads, exact_payloads)

        mutable_payload = bytearray(b"snapshotted")
        snapshotted = BackendNotificationBatch([mutable_payload])
        mutable_payload[:] = b"overwritten!"
        self.assertEqual(snapshotted.payloads, (b"snapshotted",))
        self.assertEqual(snapshotted.total_nbytes, len(b"snapshotted"))

        class OversizedBytearray(bytearray):
            def __bytes__(self) -> bytes:
                raise AssertionError("oversized mutable input must not be copied")

        oversized_mutable = OversizedBytearray(
            b"x" * (BackendNotificationBatch.MAX_PAYLOAD_BYTES + 1)
        )
        with self.assertRaisesRegex(ValueError, "wire size limit"):
            BackendNotificationBatch((oversized_mutable,))

        overrun = BackendNotificationBatch(
            (b"one", b"two"),
            source_endpoint_id=source.endpoint_id,
            source_incarnation=source.incarnation,
        )
        with (
            mock.patch.object(
                destination._backend,
                "poll_notification_batches",
                return_value=[overrun],
            ),
            self.assertRaisesRegex(BackendFailureError, "more notification payloads"),
        ):
            destination.poll_notification_batches(max_items=1)

        with self.assertRaisesRegex(ValueError, "at least one payload"):
            BackendNotificationBatch(())
        with self.assertRaisesRegex(ValueError, "specified together"):
            NotificationBatch((b"payload",), source_endpoint_id="incomplete")

        first.close()
        second.close()
        peer.close()
        source.close()
        destination.close()

    def test_notification_send_uses_work_adoption_and_direct_dispatch(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        backend_send = source._backend.send_notification
        observed_adoption = None

        def capture_send(*args, **kwargs) -> None:
            nonlocal observed_adoption
            observed_adoption = kwargs["adopt_work"]
            backend_send(*args, **kwargs)

        source._backend.send_notification = capture_send
        with mock.patch.object(
            source,
            "_backend_call",
            side_effect=AssertionError(
                "standalone notification must use its direct backend helper"
            ),
        ):
            work = source.send_notification(peer, b"direct")

        self.assertIs(observed_adoption.__self__, work)
        self.assertIs(observed_adoption.__func__, Work._adopt_backend_work)
        work.wait()
        work.close()
        peer.close()
        source.close()
        destination.close()

    def test_notification_interruption_after_adoption_retains_peer(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        peer = source.import_peer(bytes(destination.export_metadata()))
        backend_work = _PendingWork()

        def adopt_then_interrupt(*args, **kwargs) -> None:
            kwargs["adopt_work"](backend_work)
            raise KeyboardInterrupt

        source._backend.send_notification = adopt_then_interrupt
        recovery_token = object()
        work = source.send_notification(
            peer,
            b"message",
            recovery_token=recovery_token,
        )

        self.assertEqual(source.open_works, (work,))
        self.assertIs(work.recovery_token, recovery_token)
        with self.assertRaises(BusyError):
            peer.close()
        backend_work._state = WorkState.COMPLETED
        with self.assertRaises(BackendFailureError) as context:
            work.wait()
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)

        work.close()
        self.assertIsNone(work.recovery_token)
        peer.close()
        source.close()
        destination.close()

    def test_lifetime_order_is_enforced(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        work = source.write(plan)

        with self.assertRaises(BusyError):
            local.close()
        with self.assertRaises(BusyError):
            peer.close()
        with self.assertRaises(BusyError):
            plan.close()
        with self.assertRaises(BusyError):
            source.close()

        work.close()
        self.assertIsNone(work._plan)
        work.close()
        plan.close()
        self.assertFalse(plan._local)
        self.assertFalse(plan._remote)
        plan.close()
        peer.close()
        self.assertIsNone(peer._handle)
        peer.close()
        local.close()
        self.assertIsNone(local._memory.owner)
        local.close()
        remote.close()
        source.close()
        source.close()
        destination.close()

    def test_plan_cannot_close_during_submit_bookkeeping(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        submit_entered = threading.Event()
        allow_submit = threading.Event()
        close_done = threading.Event()
        close_errors: list[BaseException] = []
        original_submit = source._backend.submit

        def blocked_submit(*args, **kwargs):
            submit_entered.set()
            allow_submit.wait()
            return original_submit(*args, **kwargs)

        source._backend.submit = blocked_submit
        work_items: list[Work] = []
        submit_thread = threading.Thread(
            target=lambda: work_items.append(source.write(plan))
        )

        def close_plan() -> None:
            try:
                plan.close()
            except BaseException as error:
                close_errors.append(error)
            finally:
                close_done.set()

        submit_thread.start()
        self.assertTrue(submit_entered.wait(1))
        close_thread = threading.Thread(target=close_plan)
        close_thread.start()
        self.assertTrue(close_done.wait(1))
        self.assertEqual(len(close_errors), 1)
        self.assertIsInstance(close_errors[0], BusyError)
        self.assertFalse(plan.closed)
        allow_submit.set()
        submit_thread.join()
        close_thread.join()
        self.assertEqual(len(work_items), 1)
        self.assertFalse(plan.closed)

        work_items[0].close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_overlapping_transfer_is_request_local_failure(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        shared = bytearray(b"data")
        local = source.register(shared, name="local")
        remote = destination.register(shared, name="remote")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("remote")])

        work = source.write(plan)
        self.assertEqual(work.state, WorkState.FAILED)
        self.assertIsInstance(work.error, InvalidRegionError)
        with self.assertRaises(InvalidRegionError):
            work.wait()

        work.close()
        self.assertIsNone(work._error)
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_indexed_destination_must_not_overlap(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        destination_buffer = bytearray(b"---")
        local = source.register(bytearray(b"ab"), name="buffer")
        remote = destination.register(destination_buffer, name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=2)],
            remote=[
                peer.region("buffer", 0, 1, stride=2, count=2),
                peer.region("buffer", 1, 1, stride=1, count=2),
            ],
            indexed=True,
        )

        work = source.write(
            plan,
            local_indices=[0, 1],
            remote_indices=[0, 2],
        )
        work.wait()
        self.assertEqual(destination_buffer, bytearray(b"ab-"))
        work.close()

        with self.assertRaises(InvalidRegionError):
            source.write(
                plan,
                local_indices=[0, 1],
                remote_indices=[1, 3],
            )

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_overlapping_registrations_are_rejected(self) -> None:
        endpoint = Endpoint("endpoint")
        storage = bytearray(b"abcd")
        registration = endpoint.register(storage, name="first")
        with self.assertRaisesRegex(
            InvalidRegionError, "overlaps existing registration"
        ):
            endpoint.register(storage, name="alias")
        registration.close()
        endpoint.close()

    def test_reference_deregister_is_idempotent_after_success(self) -> None:
        endpoint = Endpoint("endpoint")
        registration = endpoint.register(bytearray(4), name="buffer")
        handle = registration._handle

        endpoint._backend.deregister(handle)
        endpoint._backend.deregister(handle)
        registration.close()
        endpoint.close()

    def test_one_shot_plan_is_released_with_work(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))

        work = source.write(local=[local.region()], remote=[peer.region("buffer")])
        work.wait()
        work.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_submit_reservation_is_released_on_base_exception(self) -> None:
        source = Endpoint("source", backend="test-interrupt")
        destination = Endpoint("destination", backend="test-interrupt")
        local = source.register(bytearray(4), name="local")
        remote = destination.register(bytearray(4), name="remote")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("remote")])

        with self.assertRaises(KeyboardInterrupt):
            source.write(plan)

        self.assertFalse(plan._submitting)
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_submit_interruption_after_adoption_retains_lifetimes(self) -> None:
        source = Endpoint("source", backend="test-adopt-then-interrupt")
        destination = Endpoint("destination", backend="test-adopt-then-interrupt")
        local = source.register(bytearray(4), name="local")
        remote = destination.register(bytearray(4), name="remote")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("remote")])

        recovery_token = object()
        work = source.write(plan, recovery_token=recovery_token)

        self.assertFalse(plan._submitting)
        self.assertIs(work.recovery_token, recovery_token)
        self.assertIn(work, plan._works)
        self.assertIn(work, source._works)
        with self.assertRaises(BusyError):
            plan.close()
        work._backend_work._state = WorkState.COMPLETED
        with self.assertRaises(BackendFailureError) as context:
            work.wait()
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)

        work.close()
        self.assertIsNone(work.recovery_token)
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_submit_without_work_adoption_is_rolled_back(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(4), name="local")
        remote = destination.register(bytearray(4), name="remote")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("remote")])
        source._backend.submit = lambda *args, **kwargs: None

        with self.assertRaisesRegex(BackendFailureError, "without adopting"):
            source.write(plan)

        self.assertFalse(plan._submitting)
        self.assertFalse(plan._works)
        self.assertFalse(source._works)
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_submit_epilogue_interruption_is_recoverable(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(4), name="local")
        remote = destination.register(bytearray(4), name="remote")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("remote")])

        class InterruptOnceOnDiscard(set):
            armed = True

            def discard(self, value) -> None:
                super().discard(value)
                if self.armed:
                    self.armed = False
                    raise KeyboardInterrupt

        plan._submitting = InterruptOnceOnDiscard()
        work = source.write(plan)

        self.assertFalse(plan._submitting)
        self.assertEqual(source.open_works, (work,))
        with self.assertRaises(BackendFailureError) as context:
            work.wait()
        self.assertIsInstance(context.exception.__cause__, KeyboardInterrupt)
        work.close()
        self.assertEqual(source.open_works, ())

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_recovery_token_covers_every_work_producing_surface(self) -> None:
        class IdentityOnlyToken:
            def __hash__(self):
                raise AssertionError("recovery token must not be hashed")

            def __eq__(self, other):
                raise AssertionError("recovery token must not be compared")

            def __bool__(self):
                raise AssertionError("recovery token must not be coerced")

        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"abcd"), name="local")
        remote = destination.register(bytearray(4), name="remote")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("remote", 0, 1, stride=1, count=4)],
            indexed=True,
        )
        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=array("i", [0]),
            remote_indices=array("i", [0]),
        )
        bound = selection.bind()
        calls = (
            lambda token: plan.submit_indices(
                TransferOp.WRITE,
                local_indices=[0],
                remote_indices=[0],
                recovery_token=token,
            ),
            lambda token: plan.submit_indices_and_test(
                TransferOp.WRITE,
                local_indices=[0],
                remote_indices=[0],
                recovery_token=token,
                max_polls=0,
            ),
            lambda token: plan.submit_indices_and_poll(
                TransferOp.WRITE,
                local_indices=[0],
                remote_indices=[0],
                recovery_token=token,
                max_polls=0,
            ),
            lambda token: selection.submit(recovery_token=token),
            lambda token: selection.submit_and_test(recovery_token=token, max_polls=0),
            lambda token: selection.submit_and_poll(recovery_token=token, max_polls=0),
            lambda token: bound.submit(recovery_token=token),
            lambda token: source.read(
                plan,
                local_indices=[0],
                remote_indices=[0],
                recovery_token=token,
            ),
            lambda token: source.write(
                plan,
                local_indices=[0],
                remote_indices=[0],
                recovery_token=token,
            ),
            lambda token: source.send_notification(
                peer, b"recovery", recovery_token=token
            ),
        )

        self.assertEqual(len(calls), 10)
        for call in calls:
            token = IdentityOnlyToken()
            result = call(token)
            work = result[0] if isinstance(result, tuple) else result
            self.assertIs(work.recovery_token, token)
            matches = [
                candidate
                for candidate in source.open_works
                if candidate.recovery_token is token
            ]
            self.assertEqual(matches, [work])
            work.wait()
            work.close()
            self.assertIsNone(work.recovery_token)

        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_recovery_token_is_strong_and_close_commit_is_retryable(self) -> None:
        class Token:
            pass

        class InterruptOnceOnDiscard(set):
            armed = True

            def discard(self, value) -> None:
                super().discard(value)
                if self.armed:
                    self.armed = False
                    raise KeyboardInterrupt("unlink interrupted")

        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="local")
        remote = destination.register(bytearray(4), name="remote")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("remote")])
        token = Token()
        token_ref = weakref.ref(token)
        work = source.write(plan, recovery_token=token)
        del token
        gc.collect()
        self.assertIs(token_ref(), work.recovery_token)
        work.wait()

        source._works = InterruptOnceOnDiscard(source._works)
        with self.assertRaisesRegex(KeyboardInterrupt, "unlink interrupted"):
            work.close()
        self.assertTrue(work.closed)
        self.assertIs(token_ref(), work.recovery_token)

        work.close()
        self.assertIsNone(work.recovery_token)
        gc.collect()
        self.assertIsNone(token_ref())
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_recovery_token_clears_on_every_preadoption_rollback(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"abcd"), name="local")
        remote = destination.register(bytearray(4), name="remote")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("remote", 0, 1, stride=1, count=4)],
            indexed=True,
        )
        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=[0],
            remote_indices=[0],
        )
        bound = plan.bind(
            TransferOp.WRITE,
            local_indices=[0],
            remote_indices=[0],
        )
        captured = []

        def reject_adoption(*args, **kwargs) -> None:
            captured.append(kwargs["adopt_work"].__self__)

        cases = (
            (
                "submit",
                lambda token: source.write(
                    plan,
                    local_indices=[0],
                    remote_indices=[0],
                    recovery_token=token,
                ),
            ),
            ("submit", lambda token: bound.submit(recovery_token=token)),
            (
                "submit_prevalidated",
                lambda token: selection.submit(recovery_token=token),
            ),
            (
                "send_notification",
                lambda token: source.send_notification(
                    peer, b"rollback", recovery_token=token
                ),
            ),
        )
        for backend_method, submit in cases:
            token = object()
            with (
                mock.patch.object(
                    source._backend,
                    backend_method,
                    side_effect=reject_adoption,
                ),
                self.assertRaisesRegex(BackendFailureError, "without adopting"),
            ):
                submit(token)
            self.assertIsNone(captured[-1].recovery_token)
            self.assertEqual(source.open_works, ())

        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_one_shot_validation_failure_does_not_leak_plan(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))

        with self.assertRaises(InvalidRegionError):
            source.write(
                local=[local.region()],
                remote=[peer.region("buffer")],
                local_indices=[0],
                remote_indices=[0],
            )
        self.assertFalse(source._plans)

        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_one_shot_plan_release_can_be_retried(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        work = source.write(local=[local.region()], remote=[peer.region("buffer")])
        original_release = source._backend.release_plan
        fail_once = True

        def release_plan(plan) -> None:
            nonlocal fail_once
            if fail_once:
                fail_once = False
                raise RuntimeError("injected release failure")
            original_release(plan)

        source._backend.release_plan = release_plan
        with self.assertRaises(BackendFailureError):
            work.close()
        self.assertFalse(work.closed)
        self.assertTrue(source._plans)
        work.close()
        self.assertFalse(source._plans)

        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_failed_one_shot_cleanup_is_retried_by_child_close(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        original_release = source._backend.release_plan
        fail_once = True

        def release_plan(plan) -> None:
            nonlocal fail_once
            if fail_once:
                fail_once = False
                raise RuntimeError("injected cleanup failure")
            original_release(plan)

        source._backend.release_plan = release_plan
        with self.assertRaises(InvalidRegionError):
            source.write(
                local=[local.region()],
                remote=[peer.region("buffer")],
                local_indices=[0],
                remote_indices=[0],
            )
        self.assertTrue(source._orphaned_plans)
        peer.close()
        self.assertFalse(source._orphaned_plans)

        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_rejects_invalid_regions_and_implicit_copies(self) -> None:
        endpoint = Endpoint("endpoint")
        registration = endpoint.register(bytearray(8), name="buffer")
        with self.assertRaises(InvalidRegionError):
            registration.region(7, 2)
        with self.assertRaises(InvalidRegionError):
            registration.region(0, 1, stride=2)
        with self.assertRaises(UnsupportedError):
            endpoint.register(RawSpan(1, 8, "cpu", "host", bytearray(8)), name="raw")

        tensor = torch.empty((2, 2)).t()
        with self.assertRaises(InvalidRegionError):
            endpoint.register(tensor, name="noncontiguous")
        with self.assertRaises(InvalidRegionError):
            endpoint.register(torch.empty(0), name="empty")

        registration.close()
        endpoint.close()

    def test_resized_storage_is_rejected_before_use(self) -> None:
        endpoint = Endpoint("endpoint")
        tensor = torch.arange(8, dtype=torch.uint8)
        storage = tensor.untyped_storage()
        registration = endpoint.register(tensor, name="tensor")
        storage.resize_(16)
        with self.assertRaises(InvalidRegionError):
            registration.region()
        with self.assertRaises(InvalidRegionError):
            endpoint.export_metadata()
        registration.close()
        endpoint.close()

    def test_cuda_ordering_modes_are_explicit(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])

        self.assertEqual(
            source.capabilities.cuda_ordering_modes,
            frozenset({CudaOrderingMode.CALLER_READY}),
        )
        self.assertTrue(source.capabilities.attached_notifications)
        self.assertTrue(source.capabilities.standalone_notifications)
        self.assertEqual(
            source.capabilities.standalone_notification_completion,
            NotificationCompletion.REMOTE_DELIVERED,
        )
        with self.assertRaises(ValueError):
            source.write(plan, stream=object())
        with self.assertRaises(ValueError):
            source.write(plan, cuda_ordering="host_wait")
        with self.assertRaises(UnsupportedError):
            source.write(
                plan,
                cuda_ordering=CudaOrderingMode.HOST_WAIT,
                ready_event=object(),
            )

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_host_wait_is_core_owned_and_runs_outside_backend_lock(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        source_buffer = bytearray(b"old!")
        destination_buffer = bytearray(4)
        local_registration = source.register(source_buffer, name="buffer")
        remote_registration = destination.register(destination_buffer, name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        local_region = local_registration.region()
        plan = source.prepare(local=[local_region], remote=[peer.region("buffer")])
        local_region._device = "cuda:0"
        source._capabilities = Capabilities(
            read=True,
            write=True,
            attached_notifications=True,
            standalone_notifications=True,
            strided_regions=True,
            manual_progress=True,
            background_progress=True,
            cuda_ordering_modes=frozenset(
                {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
            ),
            standalone_notification_completion=(
                NotificationCompletion.REMOTE_DELIVERED
            ),
        )

        progressed = threading.Event()
        test_case = self

        class FakeEvent:
            device = torch.device("cuda:0")

            def synchronize(self) -> None:
                progress_thread = threading.Thread(
                    target=lambda: (source.progress(), progressed.set())
                )
                progress_thread.start()
                if not progressed.wait(1):
                    test_case.fail("CUDA wait ran while the provider lock was held")
                progress_thread.join(1)
                source_buffer[:] = b"new!"

        with (
            mock.patch.object(torch.cuda, "Event", FakeEvent),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ),
        ):
            work = source.write(
                plan,
                cuda_ordering=CudaOrderingMode.HOST_WAIT,
                ready_event=FakeEvent(),
            )
        self.assertEqual(destination_buffer, b"new!")

        work.close()
        plan.close()
        peer.close()
        local_registration.close()
        remote_registration.close()
        source.close()
        destination.close()

    def test_host_wait_stream_records_once_and_wait_failure_rolls_back(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local_registration = source.register(bytearray(b"data"), name="buffer")
        remote_registration = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        local_region = local_registration.region()
        plan = source.prepare(local=[local_region], remote=[peer.region("buffer")])
        local_region._device = "cuda:1"
        source._capabilities = Capabilities(
            read=True,
            write=True,
            manual_progress=True,
            background_progress=True,
            cuda_ordering_modes=frozenset(
                {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
            ),
        )
        actions: list[str] = []

        class FakeEvent:
            device = torch.device("cuda:1")

            def synchronize(self) -> None:
                actions.append("wait")

        class FakeStream:
            device = torch.device("cuda:1")

            def record_event(self) -> FakeEvent:
                actions.append("record")
                return FakeEvent()

        class FakeStreamContext:
            def __enter__(self) -> None:
                actions.append("capture-check-enter")

            def __exit__(self, *args: object) -> None:
                actions.append("capture-check-exit")

        with (
            mock.patch.object(torch, "Stream", FakeStream),
            mock.patch.object(torch.cuda, "Stream", FakeStream),
            mock.patch.object(torch.cuda, "Event", FakeEvent),
            mock.patch.object(torch.cuda, "stream", return_value=FakeStreamContext()),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ),
        ):
            work = source.write(
                plan,
                cuda_ordering=CudaOrderingMode.HOST_WAIT,
                stream=FakeStream(),
            )
        self.assertEqual(
            actions,
            ["capture-check-enter", "capture-check-exit", "record", "wait"],
        )
        work.close()

        actions.clear()
        with (
            mock.patch.object(torch, "Stream", FakeStream),
            mock.patch.object(torch.cuda, "Stream", FakeStream),
            mock.patch.object(torch.cuda, "Event", FakeEvent),
            mock.patch.object(torch.cuda, "stream", return_value=FakeStreamContext()),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ),
            self.assertRaisesRegex(UnsupportedError, "Graph capture"),
        ):
            source.write(
                plan,
                cuda_ordering=CudaOrderingMode.HOST_WAIT,
                stream=FakeStream(),
            )
        self.assertEqual(actions, ["capture-check-enter", "capture-check-exit"])
        self.assertFalse(plan._submitting)

        class FailingEvent(FakeEvent):
            def synchronize(self) -> None:
                raise RuntimeError("wait failed")

        with (
            mock.patch.object(torch.cuda, "Event", FailingEvent),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ),
            self.assertRaisesRegex(BackendFailureError, "CUDA readiness"),
        ):
            source.write(
                plan,
                cuda_ordering=CudaOrderingMode.HOST_WAIT,
                ready_event=FailingEvent(),
            )
        self.assertFalse(plan._submitting)

        plan.close()
        peer.close()
        local_registration.close()
        remote_registration.close()
        source.close()
        destination.close()

    def test_host_wait_rejects_wrong_memory_token_and_skips_event_capture_query(
        self,
    ) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local_registration = source.register(bytearray(b"data"), name="buffer")
        remote_registration = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        local_region = local_registration.region()
        plan = source.prepare(local=[local_region], remote=[peer.region("buffer")])
        source._capabilities = Capabilities(
            read=True,
            write=True,
            manual_progress=True,
            background_progress=True,
            cuda_ordering_modes=frozenset(
                {CudaOrderingMode.CALLER_READY, CudaOrderingMode.HOST_WAIT}
            ),
        )

        class FakeEvent:
            def __init__(self, device: object) -> None:
                self.device = device

            def synchronize(self) -> None:
                pass

        with mock.patch.object(torch.cuda, "Event", FakeEvent):
            with self.assertRaisesRegex(InvalidRegionError, "selected local"):
                source.write(
                    plan,
                    cuda_ordering=CudaOrderingMode.HOST_WAIT,
                    ready_event=FakeEvent(torch.device("cuda:0")),
                )
            local_region._device = "cuda:0"
            with self.assertRaisesRegex(InvalidRegionError, "recorded CUDA"):
                source.write(
                    plan,
                    cuda_ordering=CudaOrderingMode.HOST_WAIT,
                    ready_event=FakeEvent(None),
                )
            with self.assertRaisesRegex(InvalidRegionError, "does not match"):
                source.write(
                    plan,
                    cuda_ordering=CudaOrderingMode.HOST_WAIT,
                    ready_event=FakeEvent(torch.device("cuda:1")),
                )
            with mock.patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                side_effect=AssertionError("ready-event path queried current stream"),
            ) as capture_query:
                work = source.write(
                    plan,
                    cuda_ordering=CudaOrderingMode.HOST_WAIT,
                    ready_event=FakeEvent(torch.device("cuda:0")),
                )
            capture_query.assert_not_called()
            work.close()
        self.assertFalse(plan._submitting)

        plan.close()
        peer.close()
        local_registration.close()
        remote_registration.close()
        source.close()
        destination.close()

    def test_expected_peer_identity_is_checked_before_import(self) -> None:
        local = Endpoint("local")
        remote = Endpoint("remote", endpoint_id="remote-id")
        metadata = bytes(remote.export_metadata())

        with self.assertRaisesRegex(InvalidMetadataError, "name"):
            local.import_peer(metadata, expected_name="other")
        self.assertEqual(local._peers, {})

        peer = local.import_peer(
            metadata,
            expected_identity=EndpointIdentity("remote-id", remote.incarnation),
            expected_name="remote",
        )
        peer.close()
        local.close()
        remote.close()

    def test_work_timeout_and_best_effort_cancel(self) -> None:
        source = Endpoint("source", backend="test-pending")
        destination = Endpoint("destination", backend="test-pending")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        work = source.write(plan)
        with self.assertRaises(TransferTimeoutError):
            work.wait(0)
        with self.assertRaises(ValueError):
            work.wait(float("nan"))
        self.assertTrue(work.cancel())
        self.assertEqual(work.state, WorkState.CANCELLED)
        with self.assertRaises(TransferCancelledError):
            work.wait()
        work.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_zero_timeout_bounded_work_fallback_observes_once(self) -> None:
        backend_work = _CountingPendingWork()
        self.assertEqual(
            backend_work.poll_bounded(
                max_polls=64,
                timeout_ns=0,
                timeout_check_interval=32,
            ),
            WorkState.PENDING,
        )
        self.assertEqual(backend_work.state_queries, 1)

        for max_polls, timeout_ns, timeout_check_interval in (
            (1 << 63, None, 32),
            (1, 1 << 63, 32),
            (1, None, 1 << 63),
        ):
            with self.assertRaises(ValueError):
                backend_work.poll_bounded(
                    max_polls=max_polls,
                    timeout_ns=timeout_ns,
                    timeout_check_interval=timeout_check_interval,
                )

        source = Endpoint("source", backend="test-counting-pending")
        destination = Endpoint("destination", backend="test-counting-pending")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        work = source.write(plan)
        self.assertFalse(
            work.test_bounded(
                max_polls=64,
                timeout_ns=0,
                timeout_check_interval=32,
            )
        )
        provider_work = source._backend.last_work
        self.assertIsNotNone(provider_work)
        assert provider_work is not None
        self.assertEqual(provider_work.state_queries, 1)
        for arguments in (
            {"max_polls": 1 << 63},
            {"timeout_ns": 1 << 63},
            {"timeout_check_interval": 1 << 63},
        ):
            with self.assertRaises(ValueError):
                work.test_bounded(**arguments)

        self.assertTrue(work.cancel())
        work.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_work_failure_diagnostic_publication_retries_after_interrupt(
        self,
    ) -> None:
        source = Endpoint("source", backend="test-interrupting-failure")
        destination = Endpoint("destination", backend="test-interrupting-failure")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        work = source.write(plan)
        provider_work = source._backend.last_work
        self.assertIsNotNone(provider_work)
        assert provider_work is not None

        with self.assertRaises(KeyboardInterrupt):
            _ = work.state
        self.assertEqual(work._state, WorkState.PENDING)
        self.assertIsNone(work._error)
        self.assertEqual(provider_work.error_queries, 1)

        self.assertEqual(work.state, WorkState.FAILED)
        self.assertIs(work.error, provider_work.failure)
        with self.assertRaises(BackendFailureError) as context:
            work.wait()
        self.assertIs(context.exception.__cause__, provider_work.failure)

        work.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_lifecycle_constructors_cannot_be_forged(self) -> None:
        endpoint = Endpoint("endpoint")
        with self.assertRaises(TypeError):
            Work(endpoint, _PendingWork())
        endpoint.close()

    def test_background_only_backend_closes_without_manual_progress(self) -> None:
        endpoint = Endpoint(
            "endpoint",
            backend="test-background-only",
            progress_mode="background",
        )
        with self.assertRaises(UnsupportedError):
            endpoint.progress()
        endpoint.close()

    def test_backend_error_must_be_an_exception(self) -> None:
        source = Endpoint("source", backend="test-bad-error")
        destination = Endpoint("destination", backend="test-bad-error")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        work = source.write(plan)

        self.assertEqual(work.state, WorkState.FAILED)
        self.assertIsInstance(work.error, BackendFailureError)

        work.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_backend_work_state_must_not_regress(self) -> None:
        source = Endpoint("source", backend="test-regressing")
        destination = Endpoint("destination", backend="test-regressing")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        work = source.write(plan)

        self.assertEqual(work.state, WorkState.RUNNING)
        with self.assertRaisesRegex(BackendFailureError, "regressed"):
            _ = work.state
        self.assertEqual(work._state, WorkState.RUNNING)
        self.assertTrue(work.cancel())

        work.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_state_query_failure_does_not_fake_terminal_completion(self) -> None:
        source = Endpoint("source", backend="test-transient-state-error")
        destination = Endpoint("destination", backend="test-transient-state-error")
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        work = source.write(plan)

        with self.assertRaises(BackendFailureError):
            work.test()
        self.assertEqual(work._state, WorkState.PENDING)
        with self.assertRaises(BusyError):
            work.close()
        self.assertFalse(plan.closed)
        self.assertFalse(local.closed)
        self.assertTrue(work.cancel())

        work.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_endpoint_id_detects_reincarnation(self) -> None:
        local = Endpoint("local")
        first = Endpoint("remote", endpoint_id="stable-remote")
        old_peer = local.import_peer(bytes(first.export_metadata()))
        first.close()
        second = Endpoint("remote", endpoint_id="stable-remote")

        with self.assertRaises(BusyError):
            local.import_peer(bytes(second.export_metadata()))
        old_peer.close()
        new_peer = local.import_peer(bytes(second.export_metadata()))
        self.assertNotEqual(old_peer.incarnation, new_peer.incarnation)

        new_peer.close()
        local.close()
        second.close()

    def test_reference_selective_metadata_is_a_registration_grant(self) -> None:
        remote = Endpoint("remote")
        first = remote.register(bytearray(4), name="first")
        second = remote.register(bytearray(4), name="second")
        full = remote.export_metadata()
        subset = remote.export_metadata([first])
        local = Endpoint("local")
        peer = local.import_peer(bytes(subset))
        with self.assertRaises(InvalidRegionError):
            peer.region("second")

        second_metadata = next(
            item for item in full.registrations if item.name == "second"
        )
        spliced = EndpointMetadata(
            wire_major=subset.wire_major,
            wire_minor=subset.wire_minor,
            required_features=subset.required_features,
            backend=subset.backend,
            endpoint_id=subset.endpoint_id,
            name=subset.name,
            incarnation=subset.incarnation,
            backend_payload=subset.backend_payload,
            registrations=(second_metadata,),
        )
        other_local = Endpoint("other-local")
        with self.assertRaises(InvalidMetadataError):
            other_local.import_peer(bytes(spliced))

        peer.close()
        local.close()
        other_local.close()
        first.close()
        second.close()
        remote.close()

    def test_reference_rejects_renamed_registration_metadata(self) -> None:
        remote = Endpoint("remote")
        remote_registration = remote.register(bytearray(4), name="original")
        metadata = remote.export_metadata()
        original = metadata.registrations[0]
        renamed = EndpointMetadata(
            wire_major=metadata.wire_major,
            wire_minor=metadata.wire_minor,
            required_features=metadata.required_features,
            backend=metadata.backend,
            endpoint_id=metadata.endpoint_id,
            name=metadata.name,
            incarnation=metadata.incarnation,
            backend_payload=metadata.backend_payload,
            registrations=(
                RegistrationMetadata(
                    registration_id=original.registration_id,
                    name="forged",
                    nbytes=original.nbytes,
                    device=original.device,
                    memory_type=original.memory_type,
                    backend_descriptor=original.backend_descriptor,
                ),
            ),
        )
        local = Endpoint("local")
        local_registration = local.register(bytearray(4), name="local")
        peer = local.import_peer(renamed)

        with self.assertRaises(InvalidMetadataError):
            local.prepare(
                local=[local_registration.region()],
                remote=[peer.region("forged")],
            )

        peer.close()
        local_registration.close()
        local.close()
        remote_registration.close()
        remote.close()

    def test_single_thread_mode_rejects_other_thread(self) -> None:
        endpoint = Endpoint("endpoint", thread_mode="single")
        errors: list[BaseException] = []

        def use_endpoint() -> None:
            try:
                endpoint.progress()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=use_endpoint)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], UnsupportedError)
        endpoint.close()

    def test_caller_serialized_mode_allows_sequential_thread_migration(self) -> None:
        source = Endpoint("source", thread_mode="caller_serialized")
        destination = Endpoint("destination", thread_mode=ThreadMode.CALLER_SERIALIZED)
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        work = source.write(plan)

        self.assertIs(source._state_lock, _api._NOOP_LOCK)
        self.assertIsNone(work._lock)
        self.assertIsNone(work._owner_thread)
        failures: list[BaseException] = []
        external_lock = threading.Lock()

        def observe_and_close() -> None:
            try:
                with external_lock:
                    self.assertEqual(work.state, WorkState.COMPLETED)
                    work.close()
                    source.progress()
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=observe_and_close)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(work.closed)

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_single_indexed_success_never_enters_core_noop_lock(self) -> None:
        source = Endpoint("source", thread_mode=ThreadMode.SINGLE)
        destination = Endpoint("destination")
        local = source.register(bytearray(b"abcdefgh"), name="buffer")
        remote = destination.register(bytearray(8), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 4, stride=4, count=2)],
            remote=[peer.region("buffer", 0, 4, stride=4, count=2)],
            indexed=True,
        )
        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=[1],
            remote_indices=[0],
        )

        with mock.patch.object(
            _api._NoopLock,
            "__enter__",
            side_effect=AssertionError("successful indexed path entered _NoopLock"),
        ):
            work = selection.submit()
            self.assertEqual(work.state, WorkState.COMPLETED)
            work.close()

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_caller_serialized_exact_slots_migrate_without_core_locks(self) -> None:
        source = Endpoint(
            "source",
            backend="test-request-slot-notification",
            thread_mode="caller_serialized",
        )
        destination = Endpoint(
            "destination",
            backend="test-request-slot-notification",
            thread_mode="caller_serialized",
        )
        local = source.register(bytearray(b"data"), name="buffer")
        remote = destination.register(bytearray(4), name="buffer")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("buffer")])
        bound = plan.bind(TransferOp.WRITE)
        fixed = bound.create_slot()
        notification = bound.create_slot(request_local_notifications=True)

        self.assertIsNone(fixed._lock)
        self.assertIsNone(fixed._owner_thread)
        self.assertIs(fixed._fast_execute, fixed._backend_start_and_poll)
        self.assertIsNone(notification._lock)
        self.assertIsNone(notification._owner_thread)
        self.assertIs(
            notification._notification_fast_execute,
            notification._backend_start_and_poll_with_notification,
        )

        external_lock = threading.Lock()

        def on_new_thread(function):
            results = []
            failures: list[BaseException] = []

            def invoke() -> None:
                try:
                    with external_lock:
                        results.append(function())
                except BaseException as error:
                    failures.append(error)

            thread = threading.Thread(target=invoke)
            thread.start()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(len(results), 1)
            return results[0]

        with mock.patch.object(
            _api,
            "_GET_IDENT",
            side_effect=AssertionError(
                "caller-serialized slot queried an unowned thread identity"
            ),
        ):
            self.assertEqual(on_new_thread(fixed.execute), 1)
            fixed_backend = source._backend.request_slots[0]
            fixed_backend.next_state = WorkState.RUNNING
            self.assertFalse(on_new_thread(fixed.start_and_test))
            fixed_backend.poll_states = [WorkState.COMPLETED]
            self.assertTrue(on_new_thread(fixed.test_and_recycle))

            self.assertEqual(
                on_new_thread(lambda: notification.execute(notification=b"first")),
                1,
            )
            notification_backend = source._backend.request_slots[1]
            notification_backend.next_state = WorkState.RUNNING
            self.assertFalse(
                on_new_thread(lambda: notification.start_and_test(notification=None))
            )
            notification_backend.poll_states = [WorkState.COMPLETED]
            self.assertTrue(on_new_thread(notification.test_and_recycle))

        self.assertIs(fixed._fast_execute, fixed._backend_start_and_poll)
        self.assertIs(
            notification._notification_fast_execute,
            notification._backend_start_and_poll_with_notification,
        )
        notification.close()
        fixed.close()
        bound.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_metadata_rejects_unknown_fields(self) -> None:
        endpoint = Endpoint("endpoint")
        encoded = bytes(endpoint.export_metadata())
        corrupted = encoded[:-1] + b',"unknown":1}'
        with self.assertRaises(InvalidMetadataError):
            EndpointMetadata.from_bytes(corrupted)
        overlong_integer = encoded.replace(b'"major":1', b'"major":' + b"9" * 4301, 1)
        with self.assertRaises(InvalidMetadataError):
            EndpointMetadata.from_bytes(overlong_integer)
        with self.assertRaises(InvalidMetadataError):
            EndpointMetadata.from_bytes(b"[" * 2000 + b"0" + b"]" * 2000)
        endpoint.close()

    def test_metadata_wire_minor_extensions_and_bounds(self) -> None:
        endpoint = Endpoint("endpoint")
        current = endpoint.export_metadata()
        future_optional = EndpointMetadata(
            wire_major=current.wire_major,
            wire_minor=current.wire_minor + 1,
            required_features=(),
            backend=current.backend,
            endpoint_id=current.endpoint_id,
            name=current.name,
            incarnation=current.incarnation,
            backend_payload=current.backend_payload,
            registrations=current.registrations,
            extensions={"future.optional": b"opaque"},
        )
        decoded = EndpointMetadata.from_bytes(bytes(future_optional))
        self.assertEqual(decoded.wire_minor, current.wire_minor + 1)
        self.assertEqual(decoded.extensions, (("future.optional", b"opaque"),))

        with self.assertRaisesRegex(InvalidMetadataError, "required"):
            EndpointMetadata(
                wire_major=current.wire_major,
                wire_minor=current.wire_minor + 1,
                required_features=("future.required",),
                backend=current.backend,
                endpoint_id=current.endpoint_id,
                name=current.name,
                incarnation=current.incarnation,
                backend_payload=current.backend_payload,
                registrations=current.registrations,
            )
        with self.assertRaisesRegex(InvalidMetadataError, "size limit"):
            EndpointMetadata.from_bytes(b" " * (EndpointMetadata.MAX_ENCODED_BYTES + 1))

        multidimensional = memoryview(bytearray(b"    ")).cast("B", (1, 4))
        self.assertEqual(len(multidimensional), 1)
        self.assertEqual(multidimensional.nbytes, 4)
        with (
            mock.patch.object(EndpointMetadata, "MAX_ENCODED_BYTES", 3),
            self.assertRaisesRegex(InvalidMetadataError, "size limit"),
        ):
            EndpointMetadata.from_bytes(multidimensional)

        registration = RegistrationMetadata(
            registration_id="r",
            name="n",
            nbytes=1,
            device="cuda",
            memory_type="m",
            backend_descriptor=b"",
        )
        with (
            mock.patch.object(EndpointMetadata, "MAX_STRING_BYTES", 3),
            self.assertRaisesRegex(InvalidMetadataError, "registration device"),
        ):
            EndpointMetadata(
                wire_major=current.wire_major,
                wire_minor=current.wire_minor,
                required_features=(),
                backend="ref",
                endpoint_id="id",
                name="n",
                incarnation="i",
                backend_payload=b"",
                registrations=(registration,),
            )

        with (
            mock.patch.object(EndpointMetadata, "MAX_EXTENSION_BYTES", 1),
            self.assertRaisesRegex(InvalidMetadataError, "extension.*size limit"),
        ):
            EndpointMetadata(
                wire_major=current.wire_major,
                wire_minor=current.wire_minor,
                required_features=(),
                backend=current.backend,
                endpoint_id=current.endpoint_id,
                name=current.name,
                incarnation=current.incarnation,
                backend_payload=current.backend_payload,
                registrations=current.registrations,
                extensions={"optional": b"xx"},
            )

        with self.assertRaisesRegex(InvalidMetadataError, "valid Unicode"):
            EndpointMetadata(
                wire_major=current.wire_major,
                wire_minor=current.wire_minor,
                required_features=(),
                backend=current.backend,
                endpoint_id=current.endpoint_id,
                name="invalid-\ud800",
                incarnation=current.incarnation,
                backend_payload=current.backend_payload,
                registrations=current.registrations,
            )
        endpoint.close()

    def test_core_has_no_exception_unsafe_manual_lock_acquisition(self) -> None:
        tree = ast.parse(Path(_api.__file__).read_text(encoding="utf-8"))
        acquisitions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "acquire"
        ]
        self.assertEqual(acquisitions, [])

    def test_first_publication_interruptions_roll_back_every_resource_family(
        self,
    ) -> None:
        class InterruptAfterAdd(set):
            def __init__(self, values=()) -> None:
                super().__init__(values)
                self.armed = True
                self.last = None

            def add(self, value) -> None:
                super().add(value)
                self.last = value
                if self.armed:
                    self.armed = False
                    raise KeyboardInterrupt("post-add publication cut")

        class InterruptAfterSet(dict):
            def __init__(self, values=()) -> None:
                super().__init__(values)
                self.armed = True
                self.last = None

            def __setitem__(self, key, value) -> None:
                super().__setitem__(key, value)
                self.last = value
                if self.armed:
                    self.armed = False
                    raise KeyboardInterrupt("post-set publication cut")

        endpoint = Endpoint("registration-publication")
        for attribute, name in (
            ("_registrations", "primary"),
            ("_registration_names", "name-index"),
        ):
            collection = InterruptAfterSet(getattr(endpoint, attribute))
            setattr(endpoint, attribute, collection)
            with self.assertRaisesRegex(KeyboardInterrupt, "publication cut"):
                endpoint.register(bytearray(4), name=name)
            self.assertTrue(collection.last.closed)
            self.assertEqual(endpoint.open_registrations, ())
            self.assertFalse(endpoint._registration_names)
        endpoint.close()

        source = Endpoint("source-publication")
        destination = Endpoint("destination-publication")
        local = source.register(bytearray(b"abcd"), name="local")
        remote = destination.register(bytearray(4), name="remote")
        metadata = bytes(destination.export_metadata())

        peer_publication = InterruptAfterSet(source._peers)
        source._peers = peer_publication
        with self.assertRaisesRegex(KeyboardInterrupt, "publication cut"):
            source.import_peer(metadata)
        self.assertTrue(peer_publication.last.closed)
        self.assertEqual(source.open_peers, ())
        peer = source.import_peer(metadata)

        plan_publication = InterruptAfterAdd(source._plans)
        source._plans = plan_publication
        with self.assertRaisesRegex(KeyboardInterrupt, "publication cut"):
            source.prepare(local=[local.region()], remote=[peer.region("remote")])
        self.assertTrue(plan_publication.last.closed)
        self.assertEqual(source.open_plans, ())

        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("remote", 0, 1, stride=1, count=4)],
            indexed=True,
        )
        selection = plan.select_indices(
            TransferOp.WRITE,
            local_indices=[0],
            remote_indices=[0],
        )
        for public_first in (True, False):
            if public_first:
                publication = InterruptAfterAdd(source._bound_transfers)
                source._bound_transfers = publication
            else:
                publication = InterruptAfterAdd(plan._bindings)
                plan._bindings = publication
            with self.assertRaisesRegex(KeyboardInterrupt, "publication cut"):
                selection.bind()
            self.assertTrue(publication.last.closed)
            self.assertEqual(source.open_bound_transfers, ())
            self.assertFalse(plan._bindings)

        work_publication = InterruptAfterAdd(source._works)
        source._works = work_publication
        token = object()
        with self.assertRaisesRegex(KeyboardInterrupt, "publication cut"):
            source.send_notification(peer, b"not-sent", recovery_token=token)
        self.assertTrue(work_publication.last.closed)
        self.assertFalse(work_publication.last._submitting)
        self.assertIsNone(work_publication.last.recovery_token)
        self.assertEqual(source.open_works, ())

        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_published_work_is_observation_gated_until_submit_reconciles(self) -> None:
        for indexed in (False, True):
            with self.subTest(indexed=indexed):
                source = Endpoint("source", thread_mode=ThreadMode.MULTIPLE)
                destination = Endpoint("destination", thread_mode=ThreadMode.MULTIPLE)
                local = source.register(bytearray(b"abcd"), name="local")
                remote = destination.register(bytearray(4), name="remote")
                peer = source.import_peer(bytes(destination.export_metadata()))
                plan = source.prepare(
                    local=[local.region(0, 1, stride=1, count=4)],
                    remote=[peer.region("remote", 0, 1, stride=1, count=4)],
                    indexed=indexed,
                )
                backend_work = _CountingPendingWork()
                entered = threading.Event()
                release = threading.Event()

                def blocked_submit(*args, **kwargs) -> None:
                    kwargs["adopt_work"](backend_work)
                    entered.set()
                    release.wait(2)

                if indexed:
                    source._backend.submit_prevalidated = blocked_submit
                    selection = plan.select_indices(
                        TransferOp.WRITE,
                        local_indices=[0],
                        remote_indices=[0],
                    )
                    submit = selection.submit
                else:
                    source._backend.submit = blocked_submit
                    submit = lambda *, recovery_token: source.write(
                        plan, recovery_token=recovery_token
                    )
                token = object()
                returned: list[Work] = []
                submit_thread = threading.Thread(
                    target=lambda: returned.append(submit(recovery_token=token))
                )
                submit_thread.start()
                self.assertTrue(entered.wait(1))
                try:
                    (work,) = source.open_works
                    self.assertIs(work.recovery_token, token)
                    observations = (
                        lambda: work.state,
                        lambda: work.error,
                        work.test,
                        lambda: work.poll_bounded(max_polls=1, timeout_ns=None),
                        work.cancel,
                        lambda: work.completion_event,
                        work.close,
                    )
                    for observation in observations:
                        with self.assertRaisesRegex(
                            BusyError, "submission is in progress"
                        ):
                            observation()
                    self.assertEqual(backend_work.state_queries, 0)
                    with self.assertRaisesRegex(BusyError, "submissions in progress"):
                        plan.close()
                finally:
                    release.set()
                    submit_thread.join(2)
                self.assertFalse(submit_thread.is_alive())
                self.assertEqual(returned, [work])
                self.assertFalse(work._submitting)
                backend_work._state = WorkState.COMPLETED
                self.assertEqual(work.state, WorkState.COMPLETED)
                work.close()
                plan.close()
                peer.close()
                local.close()
                remote.close()
                source.close()
                destination.close()

    def test_indexed_publication_is_owned_before_lock_exit_and_every_graph_edge(
        self,
    ) -> None:
        class InterruptAfterAdd(set):
            def __init__(self, values=()) -> None:
                super().__init__(values)
                self.armed = True
                self.last = None

            def add(self, value) -> None:
                super().add(value)
                self.last = value
                if self.armed:
                    self.armed = False
                    raise KeyboardInterrupt("indexed graph publication cut")

        class InterruptAfterExit:
            def __init__(self, lock) -> None:
                self.lock = lock
                self.armed = True

            def __enter__(self):
                self.lock.__enter__()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                result = self.lock.__exit__(exc_type, exc_value, traceback)
                if self.armed:
                    self.armed = False
                    raise KeyboardInterrupt("indexed state-lock exit cut")
                return result

        source = Endpoint("source", thread_mode=ThreadMode.MULTIPLE)
        destination = Endpoint("destination", thread_mode=ThreadMode.MULTIPLE)
        local = source.register(bytearray(b"abcd"), name="local")
        remote = destination.register(bytearray(4), name="remote")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(
            local=[local.region(0, 1, stride=1, count=4)],
            remote=[peer.region("remote", 0, 1, stride=1, count=4)],
            indexed=True,
        )
        selection = plan.select_indices(
            TransferOp.WRITE, local_indices=[0], remote_indices=[0]
        )
        backend_entries = 0

        def unexpected_backend_entry(*args, **kwargs) -> None:
            nonlocal backend_entries
            backend_entries += 1
            raise AssertionError("publication interruption reached provider")

        source._backend.submit_prevalidated = unexpected_backend_entry
        original_state_lock = source._state_lock
        source._state_lock = InterruptAfterExit(original_state_lock)
        try:
            with self.assertRaisesRegex(KeyboardInterrupt, "state-lock exit cut"):
                selection.submit(recovery_token=object())
        finally:
            source._state_lock = original_state_lock
        self.assertEqual(source.open_works, ())
        self.assertFalse(plan._submitting)
        self.assertFalse(plan._works)

        for owner, attribute in (
            (source, "_works"),
            (plan, "_submitting"),
            (plan, "_works"),
        ):
            with self.subTest(edge=f"{type(owner).__name__}.{attribute}"):
                original = getattr(owner, attribute)
                publication = InterruptAfterAdd(original)
                setattr(owner, attribute, publication)
                try:
                    with self.assertRaisesRegex(
                        KeyboardInterrupt, "graph publication cut"
                    ):
                        selection.submit(recovery_token=object())
                    self.assertIsNotNone(publication.last)
                    self.assertTrue(publication.last.closed)
                    self.assertFalse(publication.last._submitting)
                    self.assertIsNone(publication.last.recovery_token)
                finally:
                    setattr(owner, attribute, set(publication))
                self.assertEqual(source.open_works, ())
                self.assertFalse(plan._submitting)
                self.assertFalse(plan._works)

        self.assertEqual(backend_entries, 0)
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_endpoint_post_adoption_initialization_cleanup_is_complete(self) -> None:
        for cut in ("capabilities", "dispatch", "endpoint-info"):
            with self.subTest(cut=cut):
                instances = []

                class InitializationBackend(_ReferenceBackend):
                    def __init__(self, *args, **kwargs) -> None:
                        super().__init__(*args, **kwargs)
                        self.close_calls = 0

                    @property
                    def capabilities(self):
                        if cut == "capabilities":
                            raise KeyboardInterrupt("capabilities cut")
                        return super().capabilities

                    def __getattribute__(self, name):
                        if name == "send_notification_sync":
                            current_cut = object.__getattribute__(self, "_test_cut")
                            if current_cut == "dispatch":
                                raise KeyboardInterrupt("dispatch cut")
                        return super().__getattribute__(name)

                    def close(self) -> None:
                        self.close_calls += 1
                        return super().close()

                def factory(*args, **kwargs):
                    implementation = InitializationBackend(*args, **kwargs)
                    implementation._test_cut = cut
                    instances.append(implementation)
                    return implementation

                backend_name = f"test-endpoint-init-cleanup-{cut}"
                register_backend(backend_name, factory, replace=True)
                if cut == "endpoint-info":
                    with (
                        mock.patch.object(
                            _api,
                            "EndpointInfo",
                            side_effect=KeyboardInterrupt("EndpointInfo cut"),
                        ),
                        self.assertRaisesRegex(KeyboardInterrupt, "EndpointInfo cut"),
                    ):
                        Endpoint("interrupted", backend=backend_name)
                else:
                    with self.assertRaisesRegex(KeyboardInterrupt, f"{cut} cut"):
                        Endpoint("interrupted", backend=backend_name)
                self.assertEqual(len(instances), 1)
                self.assertEqual(instances[0].close_calls, 1)
                self.assertTrue(instances[0]._closed)

    def test_endpoint_failed_initialization_exposes_retryable_cleanup_owner(
        self,
    ) -> None:
        marker = object()
        instances = []

        class HeldInitializationBackend(_ReferenceBackend):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                super().close()
                return marker if self.close_calls == 1 else None

        def factory(*args, **kwargs):
            implementation = HeldInitializationBackend(*args, **kwargs)
            instances.append(implementation)
            return implementation

        backend_name = "test-endpoint-init-held-cleanup"
        register_backend(backend_name, factory, replace=True)
        with (
            mock.patch.object(
                _api,
                "EndpointInfo",
                side_effect=KeyboardInterrupt("EndpointInfo cut"),
            ),
            self.assertRaisesRegex(BackendFailureError, "recovery_endpoint") as context,
        ):
            Endpoint("interrupted", backend=backend_name)

        recovery_endpoint = context.exception.recovery_endpoint
        self.assertIs(recovery_endpoint._backend, instances[0])
        self.assertFalse(recovery_endpoint.closed)
        self.assertTrue(recovery_endpoint._creation_failed)
        self.assertEqual(instances[0].close_calls, 1)
        with self.assertRaisesRegex(BackendFailureError, "creation cleanup only"):
            recovery_endpoint.register(bytearray(1))
        recovery_endpoint.close()
        self.assertTrue(recovery_endpoint.closed)
        self.assertFalse(recovery_endpoint._creation_failed)
        self.assertEqual(instances[0].close_calls, 2)

    def test_submit_success_epilogues_keep_noop_lock_fast_branches(self) -> None:
        source = Path(_api.__file__).read_text(encoding="utf-8")
        generic = source[source.index("    def _submit(\n") :]
        generic = generic[: generic.index("    def _validate_regions(\n")]
        bound = source[source.index("    def _submit_bound(\n") :]
        bound = bound[: bound.index("    def send_notification(\n")]
        self.assertIn("if state_lock is _NOOP_LOCK:", generic)
        self.assertIn("if state_lock is _NOOP_LOCK:", bound)
        poll = source[source.index("    def poll_notification_batches(\n") :]
        poll = poll[: poll.index("    def progress(")]
        conditional = poll.index("if max_items is not None:")
        payload_count = poll.index(
            "payload_count = sum(len(batch.payloads) for batch in batches)"
        )
        self.assertLess(conditional, payload_count)

    def test_teardown_requires_exact_none_before_core_commit(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="local")
        remote = destination.register(bytearray(4), name="remote")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("remote")])
        work = source.write(plan)
        self.assertEqual(work.state, WorkState.COMPLETED)

        marker = object()

        def expect_retryable(resource, owner, attribute, original, *args) -> None:
            def close_then_return_marker(*call_args, **call_kwargs):
                original(*call_args, **call_kwargs)
                return marker

            setattr(owner, attribute, close_then_return_marker)
            with self.assertRaisesRegex(BackendFailureError, "must return None"):
                resource.close()
            self.assertFalse(resource.closed)
            setattr(owner, attribute, original)
            resource.close()

        backend_work = work._backend_work
        expect_retryable(work, backend_work, "close", backend_work.close)
        expect_retryable(
            plan,
            source._backend,
            "release_plan",
            source._backend.release_plan,
            plan._handle,
        )
        expect_retryable(
            peer,
            source._backend,
            "release_peer",
            source._backend.release_peer,
            peer._handle,
        )
        expect_retryable(
            local,
            source._backend,
            "deregister",
            source._backend.deregister,
            local._handle,
        )
        remote.close()
        original_close = source._backend.close
        expect_retryable(source, source._backend, "close", original_close)
        destination.close()

    def test_work_completion_event_is_capability_gated(self) -> None:
        source = Endpoint("source")
        destination = Endpoint("destination")
        local = source.register(bytearray(b"data"), name="local")
        remote = destination.register(bytearray(4), name="remote")
        peer = source.import_peer(bytes(destination.export_metadata()))
        plan = source.prepare(local=[local.region()], remote=[peer.region("remote")])
        work = source.write(plan)
        with mock.patch.object(
            type(work._backend_work),
            "completion_event",
            new_callable=mock.PropertyMock,
            side_effect=AssertionError("disabled capability entered provider"),
        ):
            self.assertIsNone(work.completion_event)
        work.close()
        plan.close()
        peer.close()
        local.close()
        remote.close()
        source.close()
        destination.close()

    def test_notification_batch_exact_type_fast_path_skips_payload_rescan(self) -> None:
        endpoint = Endpoint("receiver")
        payloads = (b"one", b"two")
        batch = BackendNotificationBatch(payloads)
        endpoint._backend.poll_notification_batches = lambda max_items: [batch]
        with mock.patch.object(
            _types,
            "_immutable_notification_payloads",
            side_effect=AssertionError("validated backend tuple was rescanned"),
        ):
            (public,) = endpoint.poll_notification_batches()
        self.assertIs(public.payloads, payloads)

        class BackendBatchSubclass(BackendNotificationBatch):
            pass

        endpoint._backend.poll_notification_batches = lambda max_items: [
            BackendBatchSubclass((b"subclass",))
        ]
        with self.assertRaisesRegex(BackendFailureError, "must be a list"):
            endpoint.poll_notification_batches()
        endpoint.close()

    def test_provider_objects_redact_owners_and_capability_descriptors(self) -> None:
        secret = b"provider-capability-secret"
        owner = {"secret": secret}
        registered = RegisteredMemory("id", "name", owner, 1, 1, "cpu", "host")
        imported = ImportedRegistration(
            owner,
            "id",
            "name",
            1,
            "cpu",
            "host",
            secret,
        )
        raw = ImportedRawMemory(owner, 1, 1, "cpu", "host")
        region = BackendRegion(
            owner=owner,
            offset=0,
            nbytes=1,
            device="cpu",
            memory_type="host",
            backend_descriptor=secret,
        )

        for value in (registered, imported, raw, region):
            self.assertNotIn("provider-capability-secret", repr(value))


if __name__ == "__main__":
    run_tests()
