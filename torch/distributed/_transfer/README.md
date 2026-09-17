# Transfer API — review MVP

Independent of ProcessGroup: move registered bytes between named peers using an
injected provider. No NIXL import/build dependency in Core. This private API is
experimental and deliberately not a production transport.

```python
import torch
from torch.distributed._transfer import Endpoint
from torch.distributed._transfer._reference import MemoryBackend

source, target = torch.arange(8, dtype=torch.uint8), torch.zeros(8, dtype=torch.uint8)
a, b = Endpoint(MemoryBackend()), Endpoint(MemoryBackend())
a.register_tensor(source)
b.register_tensor(target)
peer = a.connect(b.metadata())  # a real application exchanges this out of band
local = a.prepare([(source.data_ptr(), 4, 0, 4, 2)], kind="host")
remote = a.prepare([(target.data_ptr(), 4, 0, 4, 2)], kind="host", peer=peer)
work = a.submit("WRITE", local, [1, 0], remote, [0, 1])
work.wait()
work.close()
assert target.tolist() == [4, 5, 6, 7, 0, 1, 2, 3]
a.close()
b.close()
```

For NIXL, construct `Endpoint(nixl.torch_transfer.NixlBackend(native_agent))`.
The provider stays in NIXL. `MemoryBackend` is CPU-only, synchronous and in-process.

## Contract

- `register_tensor` retains a contiguous tensor. Raw `register` takes rows
  `(address, bytes, device)` and a strong owner; the caller guarantees validity.
- `prepare` caches a descriptor catalog; optional `(stride, count)` fields keep
  strided layouts compact. The provider validates indices/ranges/paired lengths.
- `submit` selects equally sized lists of indices. WRITE reads local memory;
  READ writes local memory. Destination spans must not overlap.
- All methods serialize host access with one endpoint lock. GPU ordering is
  separate: `caller_ready` trusts the caller; `host_wait` synchronizes the supplied
  CUDA event before posting. No CUDA graph or stream-ordered completion support.
- `Work.poll()` returns `pending`, `done` or `error`. `wait()` timeout does not
  cancel. Close succeeds only after `done`; pending/error work retains ownership.
- Close work, catalogs, peers, then registrations (or call endpoint.close).
  Registration teardown conservatively waits for ALL endpoint catalogs to close.
- Stop remote access before teardown. Applications own rendezvous, authentication,
  allocation lifetime, leases and acknowledgement protocols. Standalone `notify`
  means local acceptance, not remote acknowledgement. Bytes remain opaque.

## Deliberately deferred

Active cancellation, failure recovery, revocation, metadata schemas, capability
negotiation, native dispatch helpers, slots, graph capture and performance parity.
No asynchronous Python interruption guarantee. Native GPU/end-to-end framework
validation is required before expanding this prototype or claiming speed parity.
