# mypy: allow-untyped-defs
"""Backend-neutral asymmetric transfer API prototype.

The API is modeled on vLLM's NIXL connector shape: applications move opaque
metadata on their own side channel, load remote metadata explicitly, create
local and remote transfer views, submit READ/WRITE requests, and receive
request-local status plus optional notifications.

This is not wired into PyTorch distributed yet. It is a small, typed proposal
surface that can host NIXL, Mooncake-like engines, or other transfer runtimes
without requiring a fixed process group or symmetric allocation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence


class TransferOp(str, Enum):
    READ = "read"
    WRITE = "write"


class TransferState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class Capabilities:
    supports_read: bool = False
    supports_write: bool = False
    supports_notifications: bool = False
    supports_metadata_refresh: bool = False
    supports_device_view: bool = False
    supported_memory_types: tuple[str, ...] = ("cpu", "cuda")


@dataclass(frozen=True)
class MetadataBlob:
    """Opaque backend metadata plus optional structured hints."""

    backend: str
    agent_name: str
    payload: bytes
    attributes: Mapping[str, Any] = field(default_factory=dict)
    epoch: int = 0
    expires_at: float | None = None


@dataclass(frozen=True)
class RegisteredMemory:
    id: str
    name: str
    tensor: Any
    memory_type: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteAgent:
    name: str
    metadata: MetadataBlob
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransferView:
    """Local or remote slice prepared for a transfer request."""

    owner: RegisteredMemory | RemoteAgent
    name: str
    blocks: tuple[int, ...] | None = None
    offsets: tuple[int, ...] | None = None
    nbytes: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Notification:
    payload: bytes
    target: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransferRequest:
    op: TransferOp
    local: TransferView
    remote: TransferView
    notification: Notification | None = None
    timeout_s: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransferStatus:
    state: TransferState
    error_type: str | None = None
    error_message: str | None = None
    bytes_transferred: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.state == TransferState.SUCCESS


class TransferBackend(Protocol):
    def capabilities(self) -> Capabilities:
        ...

    def register_tensor(
        self,
        tensor: Any,
        *,
        name: str,
        memory_type: str | None,
        attributes: Mapping[str, Any],
    ) -> RegisteredMemory:
        ...

    def deregister(self, memory: RegisteredMemory) -> None:
        ...

    def export_metadata(
        self,
        memories: Sequence[RegisteredMemory] | None = None,
        *,
        include_connection_info: bool = True,
    ) -> MetadataBlob:
        ...

    def load_remote_metadata(self, metadata: MetadataBlob | bytes) -> RemoteAgent:
        ...

    def invalidate_remote(self, remote: RemoteAgent) -> None:
        ...

    def submit(self, request: TransferRequest, *, async_op: bool) -> Any:
        ...

    def check(self, handle: Any) -> TransferStatus:
        ...

    def wait(
        self,
        handle: Any,
        timeout_s: float | None = None,
        poll_interval_s: float = 0.001,
    ) -> TransferStatus:
        ...

    def get_notifications(self) -> list[Notification]:
        ...

    def close(self) -> None:
        ...


class TransferHandle:
    def __init__(
        self,
        backend: TransferBackend,
        raw_handle: Any,
        request: TransferRequest,
    ) -> None:
        self._backend = backend
        self._raw_handle = raw_handle
        self.request = request

    def status(self) -> TransferStatus:
        return self._backend.check(self._raw_handle)

    def done(self) -> bool:
        return self.status().state in (TransferState.SUCCESS, TransferState.FAILED)

    def wait(
        self,
        timeout_s: float | None = None,
        poll_interval_s: float = 0.001,
    ) -> TransferStatus:
        return self._backend.wait(self._raw_handle, timeout_s, poll_interval_s)


BackendFactory = Callable[[str, Optional[str], Mapping[str, Any]], TransferBackend]
_BACKENDS: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    key = name.lower()
    if key in _BACKENDS:
        raise ValueError(f"transfer backend already registered: {name}")
    _BACKENDS[key] = factory


def create_backend(
    backend: str,
    *,
    agent_name: str,
    device: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> TransferBackend:
    try:
        factory = _BACKENDS[backend.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown transfer backend: {backend}") from exc
    return factory(agent_name, device, options or {})


class Agent:
    """Local transfer endpoint.

    The application owns metadata transport. For example, vLLM can carry
    `MetadataBlob.payload` over ZMQ while preserving request IDs, block IDs, and
    lease state in its scheduler.
    """

    def __init__(
        self,
        name: str,
        *,
        backend: str,
        device: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.backend_name = backend
        self.device = device
        self._backend = create_backend(
            backend,
            agent_name=name,
            device=device,
            options=options or {},
        )

    def capabilities(self) -> Capabilities:
        return self._backend.capabilities()

    def register_tensor(
        self,
        tensor: Any,
        *,
        name: str,
        memory_type: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> RegisteredMemory:
        return self._backend.register_tensor(
            tensor,
            name=name,
            memory_type=memory_type,
            attributes=attributes or {},
        )

    def deregister(self, memory: RegisteredMemory) -> None:
        self._backend.deregister(memory)

    def export_metadata(
        self,
        memories: Sequence[RegisteredMemory] | None = None,
        *,
        include_connection_info: bool = True,
    ) -> MetadataBlob:
        return self._backend.export_metadata(
            memories,
            include_connection_info=include_connection_info,
        )

    def load_remote_metadata(self, metadata: MetadataBlob | bytes) -> RemoteAgent:
        return self._backend.load_remote_metadata(metadata)

    def invalidate_remote(self, remote: RemoteAgent) -> None:
        self._backend.invalidate_remote(remote)

    def view(
        self,
        memory: RegisteredMemory,
        *,
        name: str | None = None,
        blocks: Sequence[int] | None = None,
        offsets: Sequence[int] | None = None,
        nbytes: int | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TransferView:
        return TransferView(
            owner=memory,
            name=name or memory.name,
            blocks=tuple(blocks) if blocks is not None else None,
            offsets=tuple(offsets) if offsets is not None else None,
            nbytes=nbytes,
            attributes=attributes or {},
        )

    def remote_view(
        self,
        remote: RemoteAgent,
        name: str,
        *,
        blocks: Sequence[int] | None = None,
        offsets: Sequence[int] | None = None,
        nbytes: int | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TransferView:
        return TransferView(
            owner=remote,
            name=name,
            blocks=tuple(blocks) if blocks is not None else None,
            offsets=tuple(offsets) if offsets is not None else None,
            nbytes=nbytes,
            attributes=attributes or {},
        )

    def read(
        self,
        *,
        local: TransferView,
        remote: TransferView,
        notification: Notification | None = None,
        async_op: bool = True,
        timeout_s: float | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TransferHandle:
        return self.submit(
            TransferRequest(
                op=TransferOp.READ,
                local=local,
                remote=remote,
                notification=notification,
                timeout_s=timeout_s,
                attributes=attributes or {},
            ),
            async_op=async_op,
        )

    def write(
        self,
        *,
        local: TransferView,
        remote: TransferView,
        notification: Notification | None = None,
        async_op: bool = True,
        timeout_s: float | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TransferHandle:
        return self.submit(
            TransferRequest(
                op=TransferOp.WRITE,
                local=local,
                remote=remote,
                notification=notification,
                timeout_s=timeout_s,
                attributes=attributes or {},
            ),
            async_op=async_op,
        )

    def submit(self, request: TransferRequest, *, async_op: bool = True) -> TransferHandle:
        raw = self._backend.submit(request, async_op=async_op)
        return TransferHandle(self._backend, raw, request)

    def get_notifications(self) -> list[Notification]:
        return self._backend.get_notifications()

    def close(self) -> None:
        self._backend.close()


@dataclass
class _MockRawHandle:
    status: TransferStatus
    ready_at: float


class _MockBackend:
    """In-process backend for API smoke tests and examples."""

    def __init__(
        self,
        agent_name: str,
        device: str | None,
        options: Mapping[str, Any],
    ) -> None:
        self.agent_name = agent_name
        self.device = device
        self.options = dict(options)
        self.memories: dict[str, RegisteredMemory] = {}
        self.remotes: dict[str, RemoteAgent] = {}
        self.notifications: list[Notification] = []

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_read=True,
            supports_write=True,
            supports_notifications=True,
            supports_metadata_refresh=True,
            supports_device_view=False,
        )

    def register_tensor(
        self,
        tensor: Any,
        *,
        name: str,
        memory_type: str | None,
        attributes: Mapping[str, Any],
    ) -> RegisteredMemory:
        memory = RegisteredMemory(
            id=str(uuid.uuid4()),
            name=name,
            tensor=tensor,
            memory_type=memory_type,
            attributes=dict(attributes),
        )
        self.memories[memory.id] = memory
        return memory

    def deregister(self, memory: RegisteredMemory) -> None:
        self.memories.pop(memory.id, None)

    def export_metadata(
        self,
        memories: Sequence[RegisteredMemory] | None = None,
        *,
        include_connection_info: bool = True,
    ) -> MetadataBlob:
        memory_names = [m.name for m in memories] if memories is not None else [
            m.name for m in self.memories.values()
        ]
        payload = ",".join(memory_names).encode()
        return MetadataBlob(
            backend="mock",
            agent_name=self.agent_name,
            payload=payload,
            attributes={
                "device": self.device,
                "include_connection_info": include_connection_info,
                "memories": tuple(memory_names),
            },
        )

    def load_remote_metadata(self, metadata: MetadataBlob | bytes) -> RemoteAgent:
        if isinstance(metadata, bytes):
            metadata = MetadataBlob(
                backend="mock",
                agent_name=f"remote-{len(self.remotes)}",
                payload=metadata,
            )
        remote = RemoteAgent(name=metadata.agent_name, metadata=metadata)
        self.remotes[remote.name] = remote
        return remote

    def invalidate_remote(self, remote: RemoteAgent) -> None:
        self.remotes.pop(remote.name, None)

    def submit(self, request: TransferRequest, *, async_op: bool) -> _MockRawHandle:
        if request.notification is not None:
            self.notifications.append(request.notification)
        status = TransferStatus(
            state=TransferState.SUCCESS,
            bytes_transferred=request.local.nbytes or request.remote.nbytes,
            attributes={"op": request.op.value},
        )
        return _MockRawHandle(status=status, ready_at=time.monotonic())

    def check(self, handle: _MockRawHandle) -> TransferStatus:
        return handle.status

    def wait(
        self,
        handle: _MockRawHandle,
        timeout_s: float | None = None,
        poll_interval_s: float = 0.001,
    ) -> TransferStatus:
        return handle.status

    def get_notifications(self) -> list[Notification]:
        notifications = self.notifications
        self.notifications = []
        return notifications

    def close(self) -> None:
        self.memories.clear()
        self.remotes.clear()
        self.notifications.clear()


register_backend("mock", _MockBackend)
