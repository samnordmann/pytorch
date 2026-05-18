# mypy: allow-untyped-defs
"""Prototype backend-neutral transfer API.

This package is an RFC seed for asymmetric transfer engines such as NIXL and
Mooncake-like KV-cache connectors. It is intentionally pure Python and does not
register a production backend in PyTorch.
"""

from .api import (
    Agent,
    Capabilities,
    MetadataBlob,
    Notification,
    RegisteredMemory,
    RemoteAgent,
    TransferBackend,
    TransferHandle,
    TransferOp,
    TransferRequest,
    TransferState,
    TransferStatus,
    TransferView,
    create_backend,
    register_backend,
)
from .kv import KVCacheLayout, KVTransferPlan, kv_view

__all__ = [
    "Agent",
    "Capabilities",
    "KVCacheLayout",
    "KVTransferPlan",
    "MetadataBlob",
    "Notification",
    "RegisteredMemory",
    "RemoteAgent",
    "TransferBackend",
    "TransferHandle",
    "TransferOp",
    "TransferRequest",
    "TransferState",
    "TransferStatus",
    "TransferView",
    "create_backend",
    "kv_view",
    "register_backend",
]
