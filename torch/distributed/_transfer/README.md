# Experimental endpoint transfer

`torch.distributed._transfer` is a private, backend-neutral data-plane API for
moving registered tensor or buffer bytes between independently managed
endpoints. It is intended for inference KV movement and live weight streaming,
where collective process-group semantics are the wrong abstraction.

NIXL is the first external provider. PyTorch does not import, link, or otherwise
depend on NIXL. The in-process CPU `reference` backend exists for contract tests,
not performance.

## Scope

The layer owns:

- allocation lifetime anchors and byte-range validation;
- versioned endpoint and named-registration metadata;
- peer, plan, work, binding, and persistent-request-slot resource ordering;
- READ, WRITE, indexed transfer, progress, cancellation, and notifications;
- typed errors and explicit backend capabilities;
- an experimental Python provider SPI.

Frameworks continue to own discovery, routing, leases, cache/model versioning,
layout policy, scheduling, and failure recovery. This package is not a new
control plane and does not provide durable checkpointing.

## Basic flow

```python
from torch.distributed import _transfer

source = _transfer.Endpoint("source", backend="reference")
destination = _transfer.Endpoint("destination", backend="reference")

source_registration = source.register(source_tensor, name="kv")
destination_registration = destination.register(destination_tensor, name="kv")

# The application transports these opaque bytes over its control plane.
destination_metadata = bytes(destination.export_metadata())
destination_peer = source.import_peer(destination_metadata)

plan = source.prepare(
    local=[source_registration.region()],
    remote=[destination_peer.region("kv")],
)
work = source.write(plan, notification=b"kv-visible")
work.wait()

work.close()
plan.close()
destination_peer.close()
source_registration.close()
destination_registration.close()
source.close()
destination.close()
```

External NIXL registration is explicit and lazy:

```python
from nixl.torch_transfer import register_torch_backend

register_torch_backend()
endpoint = _transfer.Endpoint("worker", backend="nixl")
```

Importing `torch` or `nixl` alone does not activate the integration.

## Provider construction ownership

Existing providers remain source-compatible with the version-1 factory, which
returns a `Backend`. New native providers should register FactoryV2 explicitly:

```python
def factory(endpoint_id, name, incarnation, progress_mode, thread_mode, options,
            *, adopt_backend):
    backend = ProviderBackend.new_unactivated(...)
    adopt_backend(backend)       # Core owns cleanup from this point
    backend.activate_native()    # may publish native identity or start threads
    return None

register_backend("provider", factory, factory_api_version=2)
```

A version-1 factory return is adopted only after Core verifies that it is a
`Backend`. A wrong-type return is foreign and unowned: Core raises `TypeError`
without invoking `close()` or any other method on that object. Version 1 cannot
close the interruption window between the factory return and this adoption;
new providers should use FactoryV2.

Core advertises `BACKEND_FACTORY_API_VERSION` as the maximum backward-compatible
factory contract it understands. It never inspects a callable signature or
retries under another version. A version-2 provider must adopt exactly once
before activation and return `None`. Core retains an interruption-durable close
receipt and weak finalizer; a failed orphan is quarantined by exact
`(backend_name, endpoint_id)`, so an unresolved identity collision fails closed
without blocking unrelated endpoint IDs.

Backend registration itself is retry-safe across a lost return. Re-registering
the same name with the identical callable object and exact factory API version
is an idempotent no-op under the registry lock. A different callable identity
or version still raises `BusyError`; providers cannot use retry recovery to
replace another owner's registration.

`Endpoint.close()` does not call `progress()` implicitly. Applications using
manual progress own their drain policy; Core still rejects active Work and open
child resources before it starts retry-only backend teardown.

## Memory model

`Endpoint.register` accepts a contiguous dense Tensor, Storage, writable
C-contiguous buffer, or capability-gated `RawSpan`. It registers the exact
logical byte interval and never creates a contiguous staging copy. A Tensor
registration anchors its original Storage, so rebinding the Tensor with `set_`
does not invalidate the registered address.

The caller must keep the underlying allocation address and extent unchanged
until the Registration is closed. In particular, Storage must not be resized
while a plan, Work, or TransferSlot can reference it. Core checks the anchored
address and size
before metadata export, plan creation, and submission, but that check is only a
diagnostic: Python cannot close the race with a concurrent resize after work is
submitted. Overlapping local registrations are rejected, including aliases of
the same Tensor, Storage, or writable buffer.

Named `Peer.region` is the safe remote API. `Peer.unsafe_region` exists only for
legacy protocols that exchange addresses; the provider must prove that the raw
span is covered by imported remote metadata. Backends that cannot prove this
must reject it.

A region can represent one interval or a compact strided layout:

```python
region = registration.region(
    offset=0,
    nbytes=block_bytes,
    stride=block_stride,
    count=num_blocks,
)
```

The allocation extent is bounds checked; `nbytes * count` is the transfer size.

## Indexed plans

`prepare(..., indexed=True)` creates reusable local and remote descriptor
catalogs. Each Region contributes `count` logical blocks, or one block when
`count` is absent. Indices address the flattened region-major, block-minor
catalog, so a compact strided Region can expose many independently selectable
blocks without expanding the plan input. `TransferPlan.local_block_count` and
`remote_block_count` report the resulting bounds.

Local and remote block counts and aggregate sizes may differ. Each submission
selects independent local and remote block-index sequences, whose selected byte
totals must match. Each selected block contributes its Region's `nbytes` value.
Only selected destination blocks participate in alias validation; duplicate or
physically overlapping destination selections are rejected. Omitting both index
sequences selects every logical block in each catalog.

```python
plan = endpoint.prepare(
    local=[local_registration.region(0, block_bytes, stride=64, count=8)],
    remote=[remote_peer.region("blocks", 0, block_bytes, stride=64, count=32)],
    indexed=True,
)
work = endpoint.write(
    plan,
    local_indices=[7, 2],
    remote_indices=[19, 4],
)
```

Individual providers may impose a stricter mapping. The current NIXL provider
requires selected local and remote sequences to have equal lengths and matching
per-block sizes.

### One-shot and reusable dynamic index selections

Frameworks that produce a new packed block table per request should use the
one-shot plan API. Core snapshots and authenticates the indices, but does not
allocate a transient public `IndexSelection` or enter the endpoint state lock a
second time:

```python
import numpy as np

work, state = plan.submit_indices_and_poll(
    _transfer.TransferOp.WRITE,
    local_indices=np.asarray(local_block_ids, dtype=np.int32),
    remote_indices=np.asarray(remote_block_ids, dtype=np.int32),
    notification=request_tag,
    max_polls=0,
)
```

`plan.submit_indices(...)` is the post-only form. Providers without fused
submission support use the same provider-neutral fallback; framework adapters
do not need a capability branch.

When the same validated selection will be bound, reused, or turned into a
persistent slot, retain it explicitly:

```python
selection = plan.select_indices(
    _transfer.TransferOp.WRITE,
    local_indices=np.asarray(local_block_ids, dtype=np.int32),
    remote_indices=np.asarray(remote_block_ids, dtype=np.int32),
)
work, state = selection.submit_and_poll(
    notification=request_tag,
    max_polls=64,
    timeout_ns=None,
)
while state not in {
    _transfer.WorkState.COMPLETED,
    _transfer.WorkState.FAILED,
    _transfer.WorkState.CANCELLED,
}:
    state = work.poll_bounded(
        max_polls=4096,
        timeout_ns=100_000,
        timeout_check_interval=32,
    )
work.wait()  # raises for FAILED/CANCELLED, otherwise returns immediately
work.close()
```

Both `submit()` forms remain strictly post-only and are the correct QD>1
primitive. A
provider may advertise `fused_prevalidated_submit_poll` and `fused_work_poll`
independently. The NIXL provider uses them to keep each bounded loop behind one
GIL release; providers without either capability use the source-compatible
composition. A zero initial poll budget posts only. On a non-fused provider its
returned Boolean is conservatively false because Core cannot recover a
post-derived native state without issuing a status observation. Boolean
`submit_indices_and_test`, `submit_and_test`, and `test_bounded` wrappers
remain available when a caller needs only terminal versus nonterminal.

All bounded-poll counts, nanosecond budgets, and timeout-check intervals must
fit in a non-negative (or positive, where zero is not meaningful) signed
64-bit integer, matching the native provider ABI. A zero timeout still permits
exactly the first state observation even when the sampling interval is larger
than one; later clock checks occur at interval multiples.

`IndexSelection` is immutable and belongs to the exact plan and operation used
to create it. Core copies both mutable, one-dimensional, C-contiguous native
`int32` inputs into one private packed buffer; NIXL reads that snapshot through
`numpy.frombuffer` without another copy. A general `Sequence` is still accepted:
Core applies `operator.index`, rejects booleans, validates bounds, and packs it
in one pass. Because this provider-neutral fast representation is signed
32-bit, an indexed catalog used through `select_indices` is limited to
2,147,483,648 logical entries. The format is deliberately native-endian so it
can be consumed directly by Torch and native providers; non-native-endian or
non-contiguous buffers take the safe sequence path.

Uniform block size and full-catalog destination disjointness are plan facts.
Core can therefore validate byte totals without expanding regions and validate
destination uniqueness with one packed sort. Irregular or conservatively
ambiguous catalogs retain exact region expansion and physical-overlap checks.
The provider may omit its duplicate per-selection work only when Core's
authenticated token proves uniform geometry, full-catalog disjointness, and
unique destination indices *and* the provider independently proves the matching
full-catalog facts.

Bind a selection when the same mapping is submitted repeatedly with a
request-local notification:

```python
bound = selection.bind()
for request_tag in generation_tags:
    work = bound.submit(notification=request_tag)
    work.wait()
    work.close()
bound.close()
```

The current NIXL raw binding updates or explicitly clears the notification at
every post. The Core reuse token identifies only the exact plan, operation, and
index snapshot; it never encodes provider-specific notification behavior.
Older NIXL bindings remain safe because the provider retires a cached request
before crossing between notification-present and notification-absent states.
Changing one non-empty tag to another still reuses the same request. Closing the
plan invalidates an otherwise live selection; each open Work or
selection-backed binding retains its packed snapshot until native access is
finished.

## Bound submissions

Latency-sensitive paths can bind one fully specified submission shape once and
reuse it without repeating plan ownership, storage stability, index, overlap,
notification, or CUDA-token validation:

```python
bound = plan.bind(
    _transfer.TransferOp.WRITE,
    local_indices=[7, 2],
    remote_indices=[19, 4],
    cuda_ordering=_transfer.CudaOrderingMode.CALLER_READY,
)

for _ in range(iterations):
    work = bound.submit()
    work.wait()
    work.close()

bound.close()
```

Every `submit()` creates a normal, independently owned `Work`; concurrent open
Work objects are permitted. Core still performs constant-time endpoint,
binding, and plan lifecycle checks and publishes all lifetime reservations
before calling a provider. For `host_wait`, static device and token validation
happens at bind time, while capture detection, event recording, and the actual
host wait happen on every submission.

When a provider reports `FAILED`, Core derives and publishes the failure
diagnostic before committing the terminal Work state. If an asynchronous
`BaseException` interrupts that diagnostic lookup, Work remains nonterminal and
the next observation retries it; a terminal state can therefore never make a
missing diagnostic permanently unobservable.

An ordinary `plan.bind(...)` binding remains fully immutable: operation,
indices, notification, CUDA ordering mode, stream, and ready event cannot be
overridden by `submit()`. The narrower `IndexSelection.bind()` form above owns
an immutable operation and index snapshot but intentionally accepts a
per-submission notification. Its provider path distinguishes a missing payload
from a replacement payload and cannot retain a stale tag.

Providers receive an opaque `reuse_token`. Its identity is stable across all
submissions from one binding, allowing a provider to cache immutable native
lowering. An `IndexSelection` allocates its shared geometry token lazily on the
first selection-backed `BoundTransfer.submit()`: direct one-shot submissions
carry no token, and a binding used only to create a persistent request slot
allocates none. Selection-backed bindings also avoid a redundant private
binding token. The full-selection, notification-free, `caller_ready`
convenience forms `endpoint.read(plan)` and `endpoint.write(plan)` use stable
tokens per plan and operation. Tokens are hints only and never define transfer
semantics.

### Persistent request slots

A provider can advertise `Capabilities.persistent_request_slots` to expose the
lowest-allocation single-flight form of a bound transfer. A slot is created once
and owns one stable provider request; starting a generation does not allocate a
public `Work` or modify Core's resource sets:

```python
bound = plan.bind(
    _transfer.TransferOp.WRITE,
    local_indices=[7, 2],
    remote_indices=[19, 4],
)
slot = bound.create_slot()

for _ in range(iterations):
    slot.start_and_wait()
    slot.recycle()

slot.close()
bound.close()
```

For a blocking full-cycle loop, `execute()` combines start, completion, and
successful logical recycle under the slot's serialization boundary:

```python
for _ in range(iterations):
    completed_generation = slot.execute()
```

For queue-depth loops, `start_and_test()` exposes the initial post result with
zero additional status observations when a fused provider supports that exact
contract, while preserving Core ownership. It returns `True` with the slot logically
idle on immediate success, or `False` with an active generation. Poll pending
members with `test_and_recycle()`; it likewise returns `True` only after clean
successful completion has made the slot repostable. Failure and cancellation
are raised and remain owned for explicit recovery. Request-local notification
slots require `notification=` on `start_and_test`, exactly as on `start` and
`execute`.

It returns the completed generation number. On success, `active` is false,
`state` is `None`, and the next generation or `close()` may proceed immediately.
Failure, cancellation, and timeout still leave the generation active, exactly
like `start_and_wait()`, so the caller can inspect or cancel it and must recycle
it explicitly after terminal state. The successful idle publication is atomic
with terminal observation under `SERIALIZED` and `MULTIPLE`; `SINGLE` provides
the same guarantee through its owner-thread contract, and
`CALLER_SERIALIZED` through the caller's whole-resource-graph exclusion.

Providers may advertise
`Capabilities.deferred_successful_request_slot_recycle`. Core then omits the
provider `recycle()` call after clean `COMPLETED` in `execute()`,
`start_and_test()`, or `test_and_recycle()`. The provider retains that
observable completion receipt and consumes it inside the next `start()`,
`start_and_poll()`, or `close()` boundary. Explicit
`start()`/`wait()`/`start_and_wait()` behavior is unchanged and still requires
explicit `recycle()`. Providers without the capability remain source-compatible:
Core performs provider recycle before publishing execute success.

Explicit `recycle()` completes provider recycle before making `active` false,
the sole public idle commit. Private cleanup follows, with `state = None` last
as a clean-idle marker. Public state and error ignore private terminal
diagnostics while idle; if asynchronous interruption stops cleanup, the next
begin observes the marker and idempotently repairs it before publishing another
generation.

The exact `timeout=None` + lock-free + fused-start + deferred-success case is a
straight-line Core specialization. It applies to `SINGLE` and correctly used
`CALLER_SERIALIZED` endpoints and performs the same closed/creation,
conditional owner-thread, active-generation, CUDA-ordering,
generation-publication, and ambiguous-launch checks without crossing the
generic helper frames. Immediate clean completion publishes idle directly.
Because `_active` is its sole Core publication commit and the provider owns
durable state after dispatch entry, this exact synchronous path needs no
transient Core PENDING or success-side state-clear store; pre-entry interruption
becomes a retained Core-only failure. Nonterminal and exceptional results rejoin
the ordinary bounded/observed recovery paths. This is an internal latency
optimization only and does not change the public or provider contract.

`TransferSlot` is intentionally mutable and single-flight. Its `state` is
`None` while idle. `start()` creates one owned generation, increments
`generation`, and changes `active` to true. On the explicit lifecycle, a second
start is rejected until the first generation reaches `COMPLETED`, `FAILED`, or
`CANCELLED` and the caller invokes `recycle()`. `active` remains true in terminal
state until that recycle boundary. Every `wait()` and `start_and_wait()` pins
the generation it began observing; if another thread recycles and restarts the
mutable slot before the waiter makes its next observation, that waiter raises
`BusyError` instead of reporting the replacement generation. A slot can close
only while idle.

The ordinary binding and `TransferSlot` remain fully immutable: operation,
indices, attached notification, stream/event, and CUDA ordering cannot change
between generations. With a bound stream in `host_wait`, each slot owns one
reusable, lazily materialized CUDA event. Core caches its record/synchronize
callables and the stream capture query; every `start()` still checks capture,
records that same event after prior producer work, and host-waits before entering
the provider. An explicitly supplied ready event keeps its cached synchronize
callable. No readiness event is created in the steady generation path.

When a framework reuses one exact operation/descriptor/index selection but
changes its completion tag, it may instead call
`binding.create_slot(request_local_notifications=True)`. The returned
`RequestLocalNotificationSlot` requires `notification=` on every `start`,
`start_and_wait`, or `execute`; `None` explicitly clears the native tag. This
mode is separately capability-gated so the fixed `TransferSlot.execute()` path
does not gain a request-local branch. The operation, selected indices, CUDA
ordering, and all memory ownership remain immutable.

`start_and_wait(timeout)` is the latency-oriented equivalent of `start()` plus
`wait(timeout)`. A provider may additionally advertise
`Capabilities.fused_request_slot_start_poll` and implement the bounded
`BackendRequestSlot.start_and_poll(max_polls=..., timeout_ns=...)` SPI. Core
then keeps native post and up to 64 initial status observations inside one
provider boundary. The same SPI accepts `max_polls=0` for post-only execution:
an overriding fused provider returns the post-derived state, while the default
source composition can only return its cached `state`. A provider may independently advertise
`Capabilities.fused_request_slot_poll` and implement
`BackendRequestSlot.poll_bounded(max_polls=..., timeout_ns=...,
timeout_check_interval=...)`. If the request
is still pending or running, Core continues in chunks of at most 4,096 native
observations and a 100-microsecond soft residency target. The default interval
checks the timeout after observation 1 and then every 32 observations; one
provider call is non-preemptible and the sampled target can overshoot. Every
chunk reacquires the slot lock, verifies the pinned generation, invokes the
provider, publishes monotonic state, and releases the lock. This removes
per-observation Python crossings and most clock-read tax while retaining
count-bounded cancellation and signal checkpoints between chunks. Providers
without poll-only fusion retain the latency-oriented single-poll spin path;
providers without start fusion may still accelerate polling after ordinary
`start()`.

At slot adoption, a provider may pass
`adopt_slot(wrapper, execution_dispatch=dispatch)`. Core strongly retains the
dispatch object and validates and caches its `start`, `poll_state`, `cancel`,
and `recycle` callables, plus `start_and_poll` and `poll_bounded` when their
capabilities are advertised. The wrapper remains authoritative for `error`,
`completion_event`, and `close`. Omitting `execution_dispatch` caches the
wrapper methods exactly as before, so existing providers remain source
compatible. The hook lets a native object own latency-sensitive execution
state without weakening Core's generation, timeout, CUDA-ordering, or lifetime
authority.

For the deferred-success, fused, `SINGLE` `execute()` specialization, adoption
caches the eligible fused callable itself and whether CUDA readiness is already
caller-owned. The steady path branches on that callable before loading the
generic lock or dispatch table; serialized, timed, non-fused, and non-deferred
slots continue through the general path.

Request-slot operations are interruption-safe but not signal-handler-reentrant:
after a Python signal interrupts an operation, let that frame unwind before
calling the same slot again. A subsequent ordinary call recovers the durable
state. This avoids charging every hot dispatch for a recursive-entry guard that
cannot make arbitrary signal-handler use safe.

The standalone `wait()` operation retains cooperative yield/backoff for callers
that prefer lower host-CPU pressure; both operations have identical state,
timeout, cancellation, and lifetime semantics.

Core passes the provider timeout as integer nanoseconds and saturates extremely
large finite public timeouts at signed 64-bit maximum for that bounded call;
the subsequent Core deadline remains an arbitrary-precision Python integer.
The private, introspectable policy constants are
`_FUSED_REQUEST_SLOT_POLL_MAX_POLLS` and
`_FUSED_REQUEST_SLOT_POLL_TIMEOUT_NS`, and
`_FUSED_REQUEST_SLOT_POLL_TIMEOUT_CHECK_INTERVAL`; benchmarks should read them rather than
duplicate the current 4,096/100-microsecond policy.

The timeout is a total host-side budget from entry to `start_and_wait`. It
cannot preempt one blocking CUDA-readiness or provider call. A timeout never
recycles implicitly: the generation remains active and cancellable, and the
caller must observe a terminal state before explicitly invoking `recycle()`.
Every generation, fused or not, repeats `host_wait` readiness before Core
publishes the active generation and enters the provider.

`completion_event` is exposed only when the backend advertises
`cuda_completion_event`, and only while a generation is active. The event is
generation-local: enqueue any device wait before `recycle()`, because a later
generation may re-record or replace the provider event.

The provider slot is adopted only after Core has published the slot in the
complete endpoint/binding/plan lifetime graph and before the provider may create
native state. Once `start()` crosses into the provider, Core conservatively owns
that generation even if an asynchronous exception crosses the call boundary.
Providers must convert failures after possible native launch into terminal slot
state. The retained generation can then be inspected, cancelled where possible,
and recycled without losing native ownership.

Slot creation publishes the Endpoint recovery edge before the binding blocker,
so an interruption cannot leave an inaccessible child that permanently blocks
its parent. Close uses closure as a retryable commit marker: it retains at least
one parent edge until that marker is visible, and a repeated `close()` finishes
any interrupted Core-only unlink/cleanup epilogue. Generation publication and
terminal failure/cancellation publication use the same commit-last rule. A
pre-provider interruption can therefore be recycled as a Core-only failed
generation without issuing provider `recycle()` against an idle native slot;
an interrupted terminal-error lookup is retried on the next observation.

This opt-in path is for repeated immutable transfers. `BoundTransfer.submit()`
remains the general concurrent path and continues to return an independent
`Work` for every submission. A TransferSlot never makes `Work` reusable.

## Work and ownership

`Work.test()` is request-local and returns whether the request is terminal.
`Work.state` distinguishes completed, failed, and cancelled work; `Work.error`
retains the failure. `wait(timeout)` raises typed timeout, cancellation, or
backend errors. Cancellation is best effort and does not roll back remote writes.

Core retains the plan, registrations, peer, ordering tokens, and notification
payload until Work is closed. Closing a live or referenced resource raises
`BusyError`. Close operations are idempotent after success. Required teardown is:

```text
Work -> BoundTransfer (when used) -> TransferPlan -> Peer / Registration -> Endpoint
TransferSlot -> BoundTransfer -> TransferPlan -> Peer / Registration -> Endpoint
```

`Endpoint.open_bound_transfers` is the recovery snapshot for bindings. A
binding cannot close while it has an open Work or TransferSlot, a plan cannot
close while it has an open binding, and an endpoint cannot close until all
bindings are closed. `Endpoint.open_transfer_slots` exposes idle, in-flight,
terminal, and cleanup-only slots for recovery after interruption.

One-shot READ/WRITE owns an internal plan and releases it when Work closes.
The provider SPI uses an explicit adoption handshake for data transfers and
standalone notifications: Core publishes the Work and its complete lifetime
graph first, then the provider must adopt a `BackendWork` before launching
native activity. A provider may raise before adoption only when no native work
remains; after adoption, launch errors must become terminal Work failures. This
closes the interruption window between native submission and Python
bookkeeping. `Endpoint.open_works` provides a recovery snapshot if an
asynchronous exception lands after submission but before caller code stores the
returned Work.

Code that must close that final caller-side adoption window may pass a unique
opaque `recovery_token=` to any Work-producing operation. Core stores the
object by identity before publishing the Work and exposes it unchanged through
`Work.recovery_token`. Recovery is deliberately a cold-path scan:

```python
attempt = object()
try:
    work = plan.submit_indices(
        "write",
        local_indices=local_indices,
        remote_indices=remote_indices,
        recovery_token=attempt,
    )
except BaseException:
    matches = [
        candidate
        for candidate in endpoint.open_works
        if candidate.recovery_token is attempt
    ]
    # zero: Core retained no provider activity; one: the lost Work;
    # more than one: an ambiguous token-uniqueness error.
    raise
```

Core never hashes the token, compares it, coerces it to Boolean, invokes it,
serializes it, or passes it to the provider. It is held strongly until
`Work.close()` removes every lifetime edge, then cleared as the close
epilogue's last commit. Use `is`, never `==`, and do not use the default `None`
for recovery matching. This is distinct from provider-facing `reuse_token`:
the reuse token identifies a cached native request shape; the recovery token
identifies one caller submission attempt.

Registration, peer import, and plan preparation use the same ownership rule:
Core publishes a resource wrapper, the provider adopts a retry-safe transaction
record before its first native mutation, and Core rolls it back on failure. If
rollback itself fails, the wrapper is cleanup-only and remains recoverable via
`Endpoint.open_registrations`, `open_peers`, or `open_plans`; `close()` retries
the provider cleanup.

Endpoint construction is the one boundary where Core cannot first publish a
recovery object: Python has not returned the `Endpoint` to its caller yet.
Provider factories must therefore catch `BaseException` and self-roll back all
native state created before they return. This is a provider obligation, not an
exactly-once guarantee that Core can enforce around arbitrary third-party
factory code.

## Progress and threading

`progress_mode` is `manual` or `background`. Background progress is provider
owned; Core does not create a hidden thread. `thread_mode` is `single`,
`caller_serialized`, `serialized`, or `multiple`:

- `single` is confined to the thread that created the Endpoint and has no Core
  locks;
- `caller_serialized` may migrate between threads, but the application must
  prevent overlap and reentry for the entire Endpoint resource graph, including
  Work/slot observation, progress, cancellation, snapshots, waits, and teardown;
- `serialized` lets calls migrate and makes Core provide that exclusion,
  including provider entry;
- `multiple` permits concurrent operations when the provider advertises that
  capability. Core publishes Work/lifetime reservations before releasing its
  lifecycle lock, so data submission and status, standalone notification send,
  destructive notification receive, and manual progress can overlap when their
  backend resources are independent. Destructive notification consumers remain
  serialized with each other. Endpoint close returns `BusyError` while an
  admitted provider call is active and never tears down its target concurrently.
  Providers still serialize private destructive state that cannot safely join
  that overlap.

Provider-owned background progress threads are outside the caller boundary.
An application must keep its external exclusion for the complete public call,
including a blocking `host_wait` or `await Work` iteration. Selecting
`caller_serialized` without that whole-graph contract is unsafe.

Manual progress is endpoint-local. Every participating process must call
`Endpoint.progress()` regularly, including an otherwise idle remote responder;
waiting on one endpoint cannot drive a different process's transport engine.

## Device ordering

READ and WRITE expose two explicit CUDA producer-ordering modes:

- `caller_ready` (default): the caller guarantees that selected local memory is
  ready before submission. No CUDA API is called.
- `host_wait`: pass exactly one real `torch.cuda.Stream` or recorded
  `torch.cuda.Event`. Core validates that every selected local Region is on one
  explicitly indexed CUDA device, records one event when a stream is supplied,
  then waits for that event on the host before posting the provider request.

The wait happens outside Core's state and provider locks. It never inserts a
device-wide synchronization, and providers must not add another implicit wait.
`host_wait` protects local memory only: readiness of a remote READ source still
belongs to the remote application's control protocol. Core checks capture on a
supplied stream before recording its event. A supplied ready event goes directly
to its authoritative `synchronize()` operation and does not query the unrelated
current stream's capture state; an event CUDA cannot synchronize still fails
through CUDA.
Native stream-attached, graph-safe transport submission would require a future,
separately advertised mode.

Terminal Work completion means the local allocation can be reused. A provider
may additionally advertise `cuda_completion_event`, in which case
`Work.completion_event` expresses that terminal point to CUDA. The initial NIXL
provider does not advertise such an event.

## Notifications

Capabilities distinguish two contracts that native integrations use
differently:

- `attached_notifications`: a notification attached to READ/WRITE is
  transfer-ordered and cannot become observable before the associated remote
  writes are visible.
- `standalone_notifications`: `send_notification` has the terminal meaning in
  `standalone_notification_completion`. NIXL advertises `local_accepted`: a
  completed Work means the native API accepted/copied the payload, not that the
  remote application observed it. Remote delivery still requires transport
  progress and receiver polling.

A backend may additionally advertise `synchronous_notification_send` and
implement `Endpoint.send_notification_sync(peer, payload)`. This additive fast
path returns `None` and creates no Core or provider `Work`; return still means
only `standalone_notification_completion`. For NIXL that is `local_accepted`,
not remote observation. An exception after provider entry can therefore be
outcome-ambiguous. Reliable control protocols must retain sequence IDs,
acknowledgement, deduplication, and retry. Callers that need inspectable local
ownership after interruption should keep using `send_notification()`.
In `multiple` mode Core uses a short peer-lifetime reservation and releases its
endpoint lifecycle lock before provider entry, so independent synchronous sends
retain provider/native concurrency. The reservation is a pooled strong token in
the send frame plus a close-visible weak identity: steady state allocates no new
token, and an interrupted cleanup leaves a dead identity that `Peer.close()`
prunes rather than a permanent false blocker. `Peer.close()` reports
`BusyError` only while a live reservation exists. Providers rely on this Core
pin and should not duplicate it on their hot SPI path. `serialized` continues
to exclude the complete call, while `single` and `caller_serialized` retain
their caller contracts.

The asynchronous `send_notification()` path applies the same lock-breadth rule
without another token allocation: its required `Work` is published with the
peer and payload lifetime anchors before provider entry. Progress and receive
use pooled close-visible call reservations, allocated only when concurrency
exceeds the warmed high-water mark. Receive calls take a dedicated consumer
lock because dequeue is destructive; that lock does not exclude data submit,
independent Work status, asynchronous sends, or provider-safe progress.

Received notifications carry either no source identity or the complete
`(endpoint_id, incarnation)` pair. Consumers must reject stale or unknown
incarnations rather than routing by a reusable display name alone.

`poll_notifications()` is an at-most-once receive boundary. The current SPI
allows a provider to dequeue before returning the batch, so an asynchronous
exception or allocation failure after dequeue can lose that batch. Protocols
that require reliable control messages must include an application sequence ID,
acknowledgement, deduplication, and retransmission. A future peek/token/ack SPI
would be required to provide a stronger receive transaction.

High-rate consumers should use
`poll_notification_batches(max_items=None)`. `max_items` counts payloads across
all returned batches; zero returns without entering the provider. Each
`NotificationBatch` contains a nonempty immutable `tuple[bytes, ...]` and either
a complete source identity or no identity. Repeated batches for one source are
legal and preserve that source's observation order. Batch boundaries have no
wire meaning, and no total order is promised between different sources.

Providers may advertise `grouped_notification_receive` when they retain their
native source grouping instead of first constructing one provider object per
payload. The public operation remains available without that capability through
a scalar compatibility implementation. It groups only adjacent notifications
with the same complete identity and keeps anonymous notifications separate.
Payload tuples that are already exact immutable bytes are retained by identity;
mutable inputs are snapshotted. Returned-value validation and public object
construction run after Core releases its endpoint state lock, while the
destructive provider call itself remains serialized with close and mutation.

`max_items` is an egress bound, not automatically a native-ingress memory bound.
A provider whose native API drains a complete transport batch before returning
must separately cap or document that temporary materialization. The current
NIXL provider bounds retained Python count and bytes and fails closed on
malformed/overflowed ingress, but the underlying NIXL `getNotifs` API does not
yet offer a bounded dequeue.

## Metadata and security boundary

`EndpointMetadata` uses bounded canonical JSON with base64 opaque provider
fields, a `{major, minor}` wire version, required-feature names, optional opaque
extensions, stable endpoint ID, and restart incarnation. Unknown outer fields,
missing or duplicate fields, unknown required features, malformed values, and
unsupported major versions are rejected. A higher minor version is accepted
only through understood required features and the extension container. Encoded
metadata, provider payloads, descriptors, registration counts, notification
payloads, and identity strings have explicit limits. Opaque capability-bearing
provider data is redacted from object representations. No pickle is used by
Core.

Callers with a stable logical identity may pass `endpoint_id=`. Core creates a
fresh incarnation for each Endpoint construction and rejects replacement
metadata until the old Peer is explicitly closed.

When the control plane already knows a peer identity, use
`import_peer(expected_identity=...)`, or the individual `expected_endpoint_id`,
`expected_name`, and `expected_incarnation` checks. Core performs these checks
before the provider can mutate native peer state.

`export_metadata(registrations=[...])` is a capability grant for exactly the
selected registrations. Providers must emit partial transport metadata or reject
the operation; silently exposing omitted registrations is not allowed.
In this initial contract each imported Peer is an immutable grant snapshot.
Extending or revoking that grant requires closing dependent plans and the Peer,
then importing a replacement snapshot. Incremental `Peer.update` semantics are
left open because merge, revocation, and in-flight-work ordering must be defined
together.

Metadata authenticity, confidentiality, replay protection, and application-level
authorization remain the control plane's responsibility. Metadata is a bearer
capability: do not send it over an unauthenticated channel or log its opaque
fields. Core's bounds and strict parser are defense in depth, not a secure
transport envelope.

The exporting endpoint cannot observe remote plans or work. The application
must therefore hold a registration lease until every peer has quiesced and
released work that may reference it, then communicate revocation before local
deregistration. Core's local lifetime anchors do not implement a distributed
lease protocol.

## Initial limitations

- private Python API and SPI; no stable ABI;
- CPU reference backend only for conformance;
- the NIXL provider has two-node GB200/UCX correctness and large-transfer
  performance coverage, but not full framework end-to-end GPU coverage;
- no atomic model activation, remote lease protocol, or rollback;
- no FILE/OBJ tier, distributed-checkpoint durability, or NIXL-EP token path;
- `host_wait` is a host-side gate, not a native GPUNETIO stream submission;
- no CUDA Graph or compiler integration and no CUDA completion event from NIXL;
- no promise that cancellation prevents partial remote visibility.
- full-catalog overlap checks conservatively reject compact strided regions
  whose bounding envelopes interleave; explicit indexed selections are checked
  exactly without expanding the whole catalog.
