# mypy: allow-untyped-defs
"""KV-cache helpers for the transfer API prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .api import Agent, RegisteredMemory, TransferView


@dataclass(frozen=True)
class KVCacheLayout:
    """Structured hints carried beside opaque backend metadata."""

    block_size: int
    num_blocks: int
    block_lens: tuple[int, ...] = ()
    kv_cache_layout: str | None = None
    dtype: str | None = None
    device_id: int | None = None
    tp_size: int | None = None
    physical_blocks_per_logical_kv_block: int = 1
    attn_backend_name: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_attributes(self) -> dict[str, Any]:
        return {
            "kind": "kv_cache",
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "block_lens": self.block_lens,
            "kv_cache_layout": self.kv_cache_layout,
            "dtype": self.dtype,
            "device_id": self.device_id,
            "tp_size": self.tp_size,
            "physical_blocks_per_logical_kv_block": (
                self.physical_blocks_per_logical_kv_block
            ),
            "attn_backend_name": self.attn_backend_name,
            **dict(self.attributes),
        }


@dataclass(frozen=True)
class KVTransferPlan:
    """vLLM-shaped request metadata without depending on vLLM classes."""

    request_id: str
    local_blocks: tuple[int, ...]
    remote_blocks: tuple[int, ...]
    remote_engine_id: str | None = None
    remote_request_id: str | None = None
    remote_host: str | None = None
    remote_port: int | None = None
    tp_size: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_attributes(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "remote_engine_id": self.remote_engine_id,
            "remote_request_id": self.remote_request_id,
            "remote_host": self.remote_host,
            "remote_port": self.remote_port,
            "tp_size": self.tp_size,
            **dict(self.attributes),
        }


def kv_view(
    agent: Agent,
    memory: RegisteredMemory,
    *,
    blocks: Sequence[int],
    layout: KVCacheLayout,
    name: str = "kv-cache",
    nbytes: int | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> TransferView:
    merged = layout.to_attributes()
    if attributes:
        merged.update(attributes)
    return agent.view(
        memory,
        name=name,
        blocks=blocks,
        nbytes=nbytes,
        attributes=merged,
    )
