from __future__ import annotations

import threading
import time
from typing import Any

import torch

from ._backend import Backend


# Dropping a Python Work reference must not free memory still used by DMA.
# Explicit close removes the root; there is deliberately no GC-driven abort.
_live_endpoints: set[Endpoint] = set()


class Endpoint:
    """One provider instance and its locally owned resources.

    Control-plane exchange, remote allocation lifetime and disjoint destination
    writes belong to the application. Never resize/reallocate registered memory.
    Host calls are serialized; this does NOT order GPU streams.
    """

    def __init__(self, backend: Backend):
        self._backend = backend
        self._lock = threading.RLock()
        self._registrations: set[Registration] = set()
        self._catalogs: set[Catalog] = set()
        self._works: set[Work] = set()
        self._peers: set[str] = set()
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("endpoint is closed")

    def register(self, regions: Any, *, kind: str, owner: Any) -> Registration:
        """Register raw regions; owner must keep ALL backing allocations alive."""
        if kind not in ("host", "cuda") or owner is None:
            raise ValueError("expected host/cuda memory and a strong owner")
        with self._lock:
            self._check_open()
            reg = Registration(self, self._backend.register(regions, kind), owner)
            self._registrations.add(reg)
            return reg

    def register_tensor(self, tensor: torch.Tensor) -> Registration:
        """Register the byte span of a contiguous CPU or CUDA tensor."""
        if tensor.device.type not in ("cpu", "cuda") or not tensor.is_contiguous():
            raise ValueError("expected a contiguous CPU or CUDA tensor")
        if tensor.numel() == 0:
            raise ValueError("empty registration")
        regions = [
            (
                tensor.data_ptr(),
                tensor.numel() * tensor.element_size(),
                tensor.device.index or 0,
            )
        ]
        return self.register(
            regions, kind="cuda" if tensor.is_cuda else "host", owner=tensor
        )

    def metadata(self) -> bytes:
        """Export provider metadata AFTER registration; exchange out of band."""
        with self._lock:
            self._check_open()
            return self._backend.metadata()

    def connect(self, metadata: bytes) -> str:
        with self._lock:
            self._check_open()
            peer = self._backend.connect(metadata)
            self._peers.add(peer)
            return peer

    def disconnect(self, peer: str) -> None:
        with self._lock:
            if any(c.peer == peer for c in self._catalogs):
                raise RuntimeError("close peer catalogs before disconnecting")
            if peer in self._peers:
                self._backend.disconnect(peer)
                self._peers.remove(peer)

    def prepare(self, regions: Any, *, kind: str, peer: str | None = None) -> Catalog:
        """Prepare a reusable descriptor catalog, local if peer is None.

        The provider checks registration coverage. The application supplies
        remote addresses/layout alongside metadata through its control plane.
        """
        with self._lock:
            self._check_open()
            if kind not in ("host", "cuda"):
                raise ValueError("expected host/cuda memory")
            if peer is not None and peer not in self._peers:
                raise ValueError("connect the peer first")
            catalog = Catalog(self, self._backend.prepare(regions, kind, peer), peer)
            self._catalogs.add(catalog)
            return catalog

    def submit(
        self,
        operation: str,
        local: Catalog,
        local_indices: Any,
        remote: Catalog,
        remote_indices: Any,
        *,
        notification: bytes = b"",
        ordering: str = "caller_ready",
        ready_event: Any = None,
    ) -> Work:
        """Submit READ (remote -> local) or WRITE (local -> remote).

        caller_ready: caller has completed source production and prior uses
        of the local destination. host_wait: synchronize the supplied CUDA
        event first. The remote owner must independently order remote memory.
        Neither mode makes completion stream-ordered or graph-capturable.
        """
        if operation not in ("READ", "WRITE"):
            raise ValueError("expected READ or WRITE")
        if ordering == "host_wait" and ready_event is not None:
            ready_event.synchronize()
        elif ordering != "caller_ready" or ready_event is not None:
            raise ValueError("host_wait requires an event; caller_ready takes none")
        with self._lock:
            self._check_open()
            if (
                local not in self._catalogs
                or remote not in self._catalogs
                or local.peer is not None
                or remote.peer is None
            ):
                raise ValueError("expected live local/remote catalogs on this endpoint")
            if len(local_indices) == 0 or len(local_indices) != len(remote_indices):
                raise ValueError("expected equal, nonempty index selections")
            handle = self._backend.create_work(
                operation,
                local._handle,
                local_indices,
                remote._handle,
                remote_indices,
                notification,
            )
            work = Work(self, handle, local, remote)
            self._works.add(work)
            _live_endpoints.add(self)
            # Retain the handle BEFORE posting. A post exception does not prove
            # that nothing was submitted; endpoint.close() can retry cleanup.
            work._state = self._backend.post(handle)
            return work

    def notify(self, peer: str, message: bytes) -> None:
        """Standalone notification: local acceptance, NOT remote acknowledgement."""
        with self._lock:
            self._check_open()
            if peer not in self._peers:
                raise ValueError("connect the peer first")
            self._backend.notify(peer, message)

    def notifications(self) -> dict[str, list[bytes]]:
        with self._lock:
            self._check_open()
            return self._backend.notifications()

    def close(self) -> None:
        """Close in dependency order; a failure retains resources for retry.

        The application must FIRST stop remote access to exported memory.
        This method is not a distributed revoke, drain or shutdown protocol.
        """
        with self._lock:
            for work in list(self._works):
                work.close()
            for catalog in list(self._catalogs):
                catalog.close()
            for peer in list(self._peers):
                self.disconnect(peer)
            for reg in list(self._registrations):
                reg.close()
            self._closed = True


class Registration:
    def __init__(self, endpoint: Endpoint, handle: Any, owner: Any):
        self._endpoint, self._handle, self._owner = endpoint, handle, owner

    def close(self) -> None:
        ep = self._endpoint
        with ep._lock:
            if self not in ep._registrations:
                return
            # Deliberately coarse for the MVP: no region-to-catalog dependency graph.
            if ep._catalogs:
                raise RuntimeError("close catalogs before deregistering memory")
            ep._backend.deregister(self._handle)
            ep._registrations.remove(self)
            self._owner = None


class Catalog:
    def __init__(self, endpoint: Endpoint, handle: Any, peer: str | None):
        self._endpoint, self._handle, self.peer = endpoint, handle, peer

    def close(self) -> None:
        ep = self._endpoint
        with ep._lock:
            if self not in ep._catalogs:
                return
            if any(self in (w._local, w._remote) for w in ep._works):
                raise RuntimeError("close dependent work before its catalog")
            ep._backend.release_catalog(self._handle)
            ep._catalogs.remove(self)


class Work:
    def __init__(
        self, endpoint: Endpoint, handle: Any, local: Catalog, remote: Catalog
    ):
        self._endpoint, self._handle = endpoint, handle
        self._local, self._remote = local, remote
        self._state, self._closed = "pending", False

    @property
    def state(self) -> str:
        """Last observed state without an extra provider query."""
        return self._state

    def poll(self) -> str:
        """Return pending/done/error. A polling exception is NOT completion."""
        ep = self._endpoint
        with ep._lock:
            if not self._closed and self._state == "pending":
                self._state = ep._backend.poll(self._handle)
            return self._state

    def wait(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() == "pending":
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("work is still live; timeout does not cancel it")
            time.sleep(0)
        if self._state != "done":
            raise RuntimeError("transfer failed or was cancelled")

    def close(self) -> None:
        """Release completed work; active cancellation is outside this MVP."""
        ep = self._endpoint
        with ep._lock:
            if self._closed:
                return
            if self.poll() != "done":
                raise RuntimeError("work is not DONE; keep polling or retain memory")
            ep._backend.release_work(self._handle)
            ep._works.remove(self)
            self._closed = True
            if not ep._works:
                _live_endpoints.discard(ep)
