import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from ._errors import InvalidMetadataError, InvalidRegionError


__all__ = [
    "BackendNotification",
    "BackendNotificationBatch",
    "BackendRegion",
    "Capabilities",
    "CudaOrderingMode",
    "EndpointIdentity",
    "EndpointInfo",
    "EndpointMetadata",
    "Notification",
    "NotificationBatch",
    "NotificationCompletion",
    "ProgressMode",
    "RegistrationMetadata",
    "ThreadMode",
    "TransferOp",
    "WorkState",
]


class WorkState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransferOp(str, Enum):
    READ = "read"
    WRITE = "write"


class ProgressMode(str, Enum):
    MANUAL = "manual"
    BACKGROUND = "background"


class ThreadMode(str, Enum):
    """Concurrency contract for one Endpoint resource graph.

    ``SINGLE`` is confined to the creating thread. ``CALLER_SERIALIZED`` may
    migrate between threads, but the caller must prevent overlapping or
    reentrant operations across the endpoint and every derived resource.
    ``SERIALIZED`` asks Core to provide that exclusion. ``MULTIPLE`` permits
    concurrent calls and requires a provider that supports them.
    """

    SINGLE = "single"
    CALLER_SERIALIZED = "caller_serialized"
    SERIALIZED = "serialized"
    MULTIPLE = "multiple"


class CudaOrderingMode(str, Enum):
    """How transport access is ordered after CUDA producer work."""

    CALLER_READY = "caller_ready"
    HOST_WAIT = "host_wait"


class NotificationCompletion(str, Enum):
    """Meaning of terminal standalone-notification Work."""

    LOCAL_ACCEPTED = "local_accepted"
    REMOTE_DELIVERED = "remote_delivered"


@dataclass(frozen=True, slots=True)
class EndpointIdentity:
    """Stable logical endpoint plus one process-lifetime incarnation."""

    endpoint_id: str
    incarnation: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.endpoint_id, "endpoint_id")
        _require_nonempty_str(self.incarnation, "incarnation")


@dataclass(frozen=True, slots=True)
class Capabilities:
    read: bool = False
    write: bool = False
    attached_notifications: bool = False
    standalone_notifications: bool = False
    raw_spans: bool = False
    strided_regions: bool = False
    manual_progress: bool = False
    background_progress: bool = False
    cuda_completion_event: bool = False
    persistent_request_slots: bool = False
    fused_request_slot_start_poll: bool = False
    fused_request_slot_poll: bool = False
    cuda_ordering_modes: frozenset[CudaOrderingMode] = frozenset(
        {CudaOrderingMode.CALLER_READY}
    )
    standalone_notification_completion: NotificationCompletion | None = None
    deferred_successful_request_slot_recycle: bool = False
    # Appended to preserve positional construction compatibility for every
    # pre-existing capability field. New code should use keyword arguments.
    request_slot_notification_overrides: bool = False
    fused_prevalidated_submit_poll: bool = False
    fused_work_poll: bool = False
    grouped_notification_receive: bool = False
    synchronous_notification_send: bool = False

    def __post_init__(self) -> None:
        values = {
            "read": self.read,
            "write": self.write,
            "attached_notifications": self.attached_notifications,
            "standalone_notifications": self.standalone_notifications,
            "raw_spans": self.raw_spans,
            "strided_regions": self.strided_regions,
            "manual_progress": self.manual_progress,
            "background_progress": self.background_progress,
            "cuda_completion_event": self.cuda_completion_event,
            "persistent_request_slots": self.persistent_request_slots,
            "request_slot_notification_overrides": (
                self.request_slot_notification_overrides
            ),
            "fused_prevalidated_submit_poll": (self.fused_prevalidated_submit_poll),
            "fused_work_poll": self.fused_work_poll,
            "grouped_notification_receive": self.grouped_notification_receive,
            "synchronous_notification_send": self.synchronous_notification_send,
            "fused_request_slot_start_poll": self.fused_request_slot_start_poll,
            "fused_request_slot_poll": self.fused_request_slot_poll,
            "deferred_successful_request_slot_recycle": (
                self.deferred_successful_request_slot_recycle
            ),
        }
        for name, value in values.items():
            if not isinstance(value, bool):
                raise TypeError(f"capability {name} must be a bool")
        try:
            ordering_modes = frozenset(self.cuda_ordering_modes)
        except TypeError as error:
            raise TypeError("cuda_ordering_modes must be an iterable") from error
        if not ordering_modes or not all(
            isinstance(mode, CudaOrderingMode) for mode in ordering_modes
        ):
            raise TypeError("cuda_ordering_modes must contain CudaOrderingMode values")
        if CudaOrderingMode.CALLER_READY not in ordering_modes:
            raise ValueError("cuda_ordering_modes must include caller_ready")
        object.__setattr__(self, "cuda_ordering_modes", ordering_modes)
        if self.fused_request_slot_start_poll and not self.persistent_request_slots:
            raise ValueError(
                "fused_request_slot_start_poll requires persistent_request_slots"
            )
        if (
            self.request_slot_notification_overrides
            and not self.persistent_request_slots
        ):
            raise ValueError(
                "request_slot_notification_overrides requires "
                "persistent_request_slots"
            )
        if self.request_slot_notification_overrides and not self.attached_notifications:
            raise ValueError(
                "request_slot_notification_overrides requires " "attached_notifications"
            )
        if self.fused_request_slot_poll and not self.persistent_request_slots:
            raise ValueError(
                "fused_request_slot_poll requires persistent_request_slots"
            )
        if (
            self.deferred_successful_request_slot_recycle
            and not self.persistent_request_slots
        ):
            raise ValueError(
                "deferred_successful_request_slot_recycle requires "
                "persistent_request_slots"
            )
        completion = self.standalone_notification_completion
        if self.standalone_notifications and not isinstance(
            completion, NotificationCompletion
        ):
            raise TypeError(
                "notification-capable backends must declare standalone "
                "notification completion"
            )
        if not self.standalone_notifications and completion is not None:
            raise ValueError(
                "standalone notification completion requires notifications"
            )
        if self.grouped_notification_receive and not self.notifications:
            raise ValueError("grouped_notification_receive requires notifications")
        if self.synchronous_notification_send and not self.standalone_notifications:
            raise ValueError(
                "synchronous_notification_send requires standalone_notifications"
            )

    @property
    def notifications(self) -> bool:
        """Whether either attached or standalone notifications are available."""

        return self.attached_notifications or self.standalone_notifications


@dataclass(frozen=True, slots=True)
class EndpointInfo:
    backend: str
    endpoint_id: str
    name: str
    incarnation: str
    capabilities: Capabilities

    def __post_init__(self) -> None:
        _require_nonempty_str(self.backend, "backend")
        _require_nonempty_str(self.endpoint_id, "endpoint_id")
        _require_nonempty_str(self.name, "name")
        _require_nonempty_str(self.incarnation, "incarnation")
        if not isinstance(self.capabilities, Capabilities):
            raise TypeError("capabilities must be Capabilities")

    @property
    def identity(self) -> EndpointIdentity:
        return EndpointIdentity(self.endpoint_id, self.incarnation)


@dataclass(frozen=True, slots=True)
class BackendRegion:
    """Provider-facing region with a core-owned lifetime anchor.

    Providers consume ``backend_descriptor`` and must treat ``owner`` as opaque.
    For local regions the descriptor is the provider registration handle; for
    remote regions it is ``ImportedRegistration`` or ``ImportedRawMemory``.
    In an indexed plan, ``count`` compactly contributes that many independently
    selectable logical blocks, ordered by increasing stride displacement.
    """

    owner: object = field(repr=False)
    offset: int
    nbytes: int
    device: str
    memory_type: str
    backend_descriptor: object = field(repr=False)
    remote: bool = False
    stride: int | None = None
    count: int | None = None

    def __post_init__(self) -> None:
        if not _is_int(self.offset) or self.offset < 0:
            raise InvalidRegionError("offset must be a non-negative integer")
        if not _is_int(self.nbytes) or self.nbytes <= 0:
            raise InvalidRegionError("nbytes must be a positive integer")
        _require_nonempty_str(self.device, "device", InvalidRegionError)
        _require_nonempty_str(self.memory_type, "memory_type", InvalidRegionError)
        if not isinstance(self.remote, bool):
            raise InvalidRegionError("remote must be a bool")
        if (self.stride is None) != (self.count is None):
            raise InvalidRegionError("stride and count must be specified together")
        if self.stride is not None:
            if not _is_int(self.stride) or self.stride <= 0:
                raise InvalidRegionError("stride must be a positive integer")
            if (
                isinstance(self.count, bool)
                or not isinstance(self.count, int)
                or self.count <= 0
            ):
                raise InvalidRegionError("count must be a positive integer")
            if self.stride < self.nbytes:
                raise InvalidRegionError("stride must be at least nbytes")


@dataclass(frozen=True, slots=True)
class BackendNotification:
    MAX_PAYLOAD_BYTES: ClassVar[int] = 1024 * 1024

    payload: bytes = field(repr=False)
    source_endpoint_id: str | None = None
    source_incarnation: str | None = None

    def __post_init__(self) -> None:
        payload = _immutable_bytes(self.payload, "payload")
        if len(payload) > self.MAX_PAYLOAD_BYTES:
            raise ValueError("notification payload exceeds the wire size limit")
        object.__setattr__(self, "payload", payload)
        if self.source_endpoint_id is not None:
            _require_nonempty_str(self.source_endpoint_id, "source_endpoint_id")
        if self.source_incarnation is not None:
            _require_nonempty_str(self.source_incarnation, "source_incarnation")
        if (self.source_endpoint_id is None) != (self.source_incarnation is None):
            raise ValueError(
                "notification source_endpoint_id and source_incarnation must be "
                "specified together"
            )

    @property
    def source(self) -> EndpointIdentity | None:
        if self.source_endpoint_id is None or self.source_incarnation is None:
            return None
        return EndpointIdentity(self.source_endpoint_id, self.source_incarnation)


@dataclass(frozen=True, slots=True)
class BackendNotificationBatch:
    """Provider-facing notification group from one native source batch.

    Providers may return multiple batches for one source. Payload order is
    significant within and across that source's batches; no total order is
    promised between different sources.
    """

    MAX_PAYLOAD_BYTES: ClassVar[int] = BackendNotification.MAX_PAYLOAD_BYTES

    payloads: tuple[bytes, ...] = field(repr=False)
    total_nbytes: int = field(init=False)
    source_endpoint_id: str | None = None
    source_incarnation: str | None = None

    def __post_init__(self) -> None:
        payloads, total_nbytes = _immutable_notification_payloads(
            self.payloads, self.MAX_PAYLOAD_BYTES
        )
        if not payloads:
            raise ValueError("notification batch must contain at least one payload")
        object.__setattr__(self, "payloads", payloads)
        object.__setattr__(self, "total_nbytes", total_nbytes)
        if self.source_endpoint_id is not None:
            _require_nonempty_str(self.source_endpoint_id, "source_endpoint_id")
        if self.source_incarnation is not None:
            _require_nonempty_str(self.source_incarnation, "source_incarnation")
        if (self.source_endpoint_id is None) != (self.source_incarnation is None):
            raise ValueError(
                "notification source_endpoint_id and source_incarnation must be "
                "specified together"
            )

    @property
    def source(self) -> EndpointIdentity | None:
        if self.source_endpoint_id is None or self.source_incarnation is None:
            return None
        return EndpointIdentity(self.source_endpoint_id, self.source_incarnation)


@dataclass(frozen=True, slots=True)
class Notification:
    MAX_PAYLOAD_BYTES: ClassVar[int] = BackendNotification.MAX_PAYLOAD_BYTES

    payload: bytes = field(repr=False)
    source_endpoint_id: str | None = None
    source_incarnation: str | None = None

    def __post_init__(self) -> None:
        payload = _immutable_bytes(self.payload, "payload")
        if len(payload) > self.MAX_PAYLOAD_BYTES:
            raise ValueError("notification payload exceeds the wire size limit")
        object.__setattr__(self, "payload", payload)
        if self.source_endpoint_id is not None:
            _require_nonempty_str(self.source_endpoint_id, "source_endpoint_id")
        if self.source_incarnation is not None:
            _require_nonempty_str(self.source_incarnation, "source_incarnation")
        if (self.source_endpoint_id is None) != (self.source_incarnation is None):
            raise ValueError(
                "notification source_endpoint_id and source_incarnation must be "
                "specified together"
            )

    @property
    def source(self) -> EndpointIdentity | None:
        if self.source_endpoint_id is None or self.source_incarnation is None:
            return None
        return EndpointIdentity(self.source_endpoint_id, self.source_incarnation)


@dataclass(frozen=True, slots=True)
class NotificationBatch:
    """Public notification group preserving one provider source boundary."""

    MAX_PAYLOAD_BYTES: ClassVar[int] = BackendNotification.MAX_PAYLOAD_BYTES

    payloads: tuple[bytes, ...] = field(repr=False)
    source_endpoint_id: str | None = None
    source_incarnation: str | None = None

    def __post_init__(self) -> None:
        payloads, _ = _immutable_notification_payloads(
            self.payloads, self.MAX_PAYLOAD_BYTES
        )
        if not payloads:
            raise ValueError("notification batch must contain at least one payload")
        object.__setattr__(self, "payloads", payloads)
        if self.source_endpoint_id is not None:
            _require_nonempty_str(self.source_endpoint_id, "source_endpoint_id")
        if self.source_incarnation is not None:
            _require_nonempty_str(self.source_incarnation, "source_incarnation")
        if (self.source_endpoint_id is None) != (self.source_incarnation is None):
            raise ValueError(
                "notification source_endpoint_id and source_incarnation must be "
                "specified together"
            )

    @property
    def source(self) -> EndpointIdentity | None:
        if self.source_endpoint_id is None or self.source_incarnation is None:
            return None
        return EndpointIdentity(self.source_endpoint_id, self.source_incarnation)


def _notification_batch_from_backend(
    batch: BackendNotificationBatch,
) -> NotificationBatch:
    """Adopt one exact, already-validated provider batch without rescanning it."""

    if type(batch) is not BackendNotificationBatch:
        raise TypeError("expected an exact BackendNotificationBatch")
    result = object.__new__(NotificationBatch)
    object.__setattr__(result, "payloads", batch.payloads)
    object.__setattr__(result, "source_endpoint_id", batch.source_endpoint_id)
    object.__setattr__(result, "source_incarnation", batch.source_incarnation)
    return result


@dataclass(frozen=True, slots=True)
class RegistrationMetadata:
    registration_id: str
    name: str
    nbytes: int
    device: str
    memory_type: str
    backend_descriptor: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.registration_id, "registration_id", InvalidMetadataError
        )
        _require_nonempty_str(self.name, "name", InvalidMetadataError)
        if not _is_int(self.nbytes) or self.nbytes <= 0:
            raise InvalidMetadataError("registration nbytes must be positive")
        _require_nonempty_str(self.device, "device", InvalidMetadataError)
        _require_nonempty_str(self.memory_type, "memory_type", InvalidMetadataError)
        descriptor = _immutable_bytes(
            self.backend_descriptor, "backend_descriptor", InvalidMetadataError
        )
        object.__setattr__(self, "backend_descriptor", descriptor)


@dataclass(frozen=True, slots=True)
class EndpointMetadata:
    WIRE_MAJOR: ClassVar[int] = 1
    WIRE_MINOR: ClassVar[int] = 0
    KNOWN_REQUIRED_FEATURES: ClassVar[frozenset[str]] = frozenset()
    MAX_ENCODED_BYTES: ClassVar[int] = 64 * 1024 * 1024
    MAX_BACKEND_PAYLOAD_BYTES: ClassVar[int] = 32 * 1024 * 1024
    MAX_DESCRIPTOR_BYTES: ClassVar[int] = 4 * 1024 * 1024
    MAX_TOTAL_DESCRIPTOR_BYTES: ClassVar[int] = 32 * 1024 * 1024
    MAX_REGISTRATIONS: ClassVar[int] = 65_536
    MAX_REQUIRED_FEATURES: ClassVar[int] = 256
    MAX_EXTENSIONS: ClassVar[int] = 256
    MAX_EXTENSION_BYTES: ClassVar[int] = 4 * 1024 * 1024
    MAX_TOTAL_EXTENSION_BYTES: ClassVar[int] = 32 * 1024 * 1024
    MAX_STRING_BYTES: ClassVar[int] = 4_096

    wire_major: int
    wire_minor: int
    required_features: tuple[str, ...]
    backend: str
    endpoint_id: str
    name: str
    incarnation: str
    backend_payload: bytes = field(repr=False)
    registrations: tuple[RegistrationMetadata, ...]
    extensions: tuple[tuple[str, bytes], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not _is_int(self.wire_major) or self.wire_major != self.WIRE_MAJOR:
            raise InvalidMetadataError(
                f"unsupported endpoint metadata wire major: {self.wire_major!r}"
            )
        if not _is_int(self.wire_minor) or self.wire_minor < 0:
            raise InvalidMetadataError(
                f"invalid endpoint metadata wire minor: {self.wire_minor!r}"
            )
        try:
            required_features = tuple(self.required_features)
        except TypeError as error:
            raise InvalidMetadataError(
                "required_features must be an iterable"
            ) from error
        if not all(isinstance(item, str) and item for item in required_features):
            raise InvalidMetadataError(
                "required_features must contain non-empty strings"
            )
        if len(required_features) > self.MAX_REQUIRED_FEATURES:
            raise InvalidMetadataError("too many required endpoint metadata features")
        if any(
            _utf8_nbytes(item, "required feature") > self.MAX_STRING_BYTES
            for item in required_features
        ):
            raise InvalidMetadataError(
                "required endpoint metadata feature exceeds the wire size limit"
            )
        if len(required_features) != len(set(required_features)):
            raise InvalidMetadataError("required_features must be unique")
        unknown_features = set(required_features) - self.KNOWN_REQUIRED_FEATURES
        if unknown_features:
            raise InvalidMetadataError(
                "unsupported required endpoint metadata features: "
                f"{', '.join(sorted(unknown_features))}"
            )
        object.__setattr__(self, "required_features", tuple(sorted(required_features)))
        _require_nonempty_str(self.backend, "backend", InvalidMetadataError)
        _require_nonempty_str(self.endpoint_id, "endpoint_id", InvalidMetadataError)
        _require_nonempty_str(self.name, "name", InvalidMetadataError)
        _require_nonempty_str(self.incarnation, "incarnation", InvalidMetadataError)
        for field_name in ("backend", "endpoint_id", "name", "incarnation"):
            if (
                _utf8_nbytes(getattr(self, field_name), field_name)
                > self.MAX_STRING_BYTES
            ):
                raise InvalidMetadataError(f"{field_name} exceeds the wire size limit")
        payload = _immutable_bytes(
            self.backend_payload, "backend_payload", InvalidMetadataError
        )
        if len(payload) > self.MAX_BACKEND_PAYLOAD_BYTES:
            raise InvalidMetadataError("backend_payload exceeds the wire size limit")
        object.__setattr__(self, "backend_payload", payload)

        try:
            registrations = tuple(self.registrations)
        except TypeError as error:
            raise InvalidMetadataError("registrations must be an iterable") from error
        if not all(isinstance(item, RegistrationMetadata) for item in registrations):
            raise InvalidMetadataError(
                "registrations must contain RegistrationMetadata values"
            )
        registration_ids = [item.registration_id for item in registrations]
        names = [item.name for item in registrations]
        if len(registration_ids) != len(set(registration_ids)):
            raise InvalidMetadataError("registration_id values must be unique")
        if len(names) != len(set(names)):
            raise InvalidMetadataError("registration names must be unique")
        if len(registrations) > self.MAX_REGISTRATIONS:
            raise InvalidMetadataError("too many endpoint metadata registrations")
        descriptor_bytes = 0
        for registration in registrations:
            for field_name in (
                "registration_id",
                "name",
                "device",
                "memory_type",
            ):
                if (
                    _utf8_nbytes(
                        getattr(registration, field_name),
                        f"registration {field_name}",
                    )
                    > self.MAX_STRING_BYTES
                ):
                    raise InvalidMetadataError(
                        f"registration {field_name} exceeds the wire size limit"
                    )
            if len(registration.backend_descriptor) > self.MAX_DESCRIPTOR_BYTES:
                raise InvalidMetadataError(
                    "registration descriptor exceeds the wire size limit"
                )
            descriptor_bytes += len(registration.backend_descriptor)
        if descriptor_bytes > self.MAX_TOTAL_DESCRIPTOR_BYTES:
            raise InvalidMetadataError(
                "registration descriptors exceed the aggregate wire size limit"
            )
        object.__setattr__(self, "registrations", registrations)

        raw_extensions: Any = self.extensions
        if isinstance(raw_extensions, Mapping):
            extension_items = tuple(raw_extensions.items())
        else:
            try:
                extension_items = tuple(raw_extensions)
            except TypeError as error:
                raise InvalidMetadataError("extensions must be a mapping") from error
        if len(extension_items) > self.MAX_EXTENSIONS:
            raise InvalidMetadataError("too many endpoint metadata extensions")
        extensions: list[tuple[str, bytes]] = []
        extension_bytes = 0
        for item in extension_items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise InvalidMetadataError("extensions must contain key/value pairs")
            key, value = item
            if not isinstance(key, str) or not key:
                raise InvalidMetadataError("extension names must be non-empty strings")
            if _utf8_nbytes(key, "extension name") > self.MAX_STRING_BYTES:
                raise InvalidMetadataError("extension name exceeds the wire size limit")
            extension = _immutable_bytes(
                value, f"extension {key!r}", InvalidMetadataError
            )
            if len(extension) > self.MAX_EXTENSION_BYTES:
                raise InvalidMetadataError(
                    f"extension {key!r} exceeds the wire size limit"
                )
            extension_bytes += len(extension)
            extensions.append((key, extension))
        if extension_bytes > self.MAX_TOTAL_EXTENSION_BYTES:
            raise InvalidMetadataError(
                "endpoint metadata extensions exceed the aggregate wire size limit"
            )
        extension_names = [key for key, _ in extensions]
        if len(extension_names) != len(set(extension_names)):
            raise InvalidMetadataError("extension names must be unique")
        object.__setattr__(self, "extensions", tuple(sorted(extensions)))

    def __bytes__(self) -> bytes:
        registrations = [
            {
                "backend_descriptor": _encode_bytes(item.backend_descriptor),
                "device": item.device,
                "memory_type": item.memory_type,
                "name": item.name,
                "nbytes": item.nbytes,
                "registration_id": item.registration_id,
            }
            for item in self.registrations
        ]
        value = {
            "backend": self.backend,
            "backend_payload": _encode_bytes(self.backend_payload),
            "endpoint_id": self.endpoint_id,
            "extensions": {
                key: _encode_bytes(extension) for key, extension in self.extensions
            },
            "incarnation": self.incarnation,
            "name": self.name,
            "registrations": registrations,
            "required_features": list(self.required_features),
            "wire_version": {
                "major": self.wire_major,
                "minor": self.wire_minor,
            },
        }
        encoded = json.dumps(
            value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(encoded) > self.MAX_ENCODED_BYTES:
            raise InvalidMetadataError("endpoint metadata exceeds the wire size limit")
        return encoded

    def to_bytes(self) -> bytes:
        """Return the canonical wire representation."""

        return bytes(self)

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "EndpointMetadata":
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise InvalidMetadataError("endpoint metadata must be bytes-like")
        try:
            encoded_nbytes = data.nbytes if isinstance(data, memoryview) else len(data)
        except ValueError as error:
            raise InvalidMetadataError(
                "endpoint metadata buffer is unavailable"
            ) from error
        if encoded_nbytes > cls.MAX_ENCODED_BYTES:
            raise InvalidMetadataError("endpoint metadata exceeds the wire size limit")
        try:
            encoded = bytes(data)
            if len(encoded) > cls.MAX_ENCODED_BYTES:
                raise InvalidMetadataError(
                    "endpoint metadata exceeds the wire size limit"
                )
            value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_json_object)
        except (
            UnicodeDecodeError,
            ValueError,
            RecursionError,
            InvalidMetadataError,
        ) as error:
            if isinstance(error, InvalidMetadataError):
                raise
            raise InvalidMetadataError("endpoint metadata is not valid JSON") from error

        _require_object_keys(
            value,
            {
                "backend",
                "backend_payload",
                "endpoint_id",
                "extensions",
                "incarnation",
                "name",
                "registrations",
                "required_features",
                "wire_version",
            },
            "endpoint metadata",
        )
        wire_version = value["wire_version"]
        _require_object_keys(wire_version, {"major", "minor"}, "wire_version")
        required_features = value["required_features"]
        if not isinstance(required_features, list):
            raise InvalidMetadataError("required_features must be a JSON array")
        if len(required_features) > cls.MAX_REQUIRED_FEATURES:
            raise InvalidMetadataError("too many required endpoint metadata features")
        raw_extensions = value["extensions"]
        if not isinstance(raw_extensions, dict):
            raise InvalidMetadataError("extensions must be a JSON object")
        if len(raw_extensions) > cls.MAX_EXTENSIONS:
            raise InvalidMetadataError("too many endpoint metadata extensions")
        raw_registrations = value["registrations"]
        if not isinstance(raw_registrations, list):
            raise InvalidMetadataError("registrations must be a JSON array")
        if len(raw_registrations) > cls.MAX_REGISTRATIONS:
            raise InvalidMetadataError("too many endpoint metadata registrations")
        registrations = tuple(
            _registration_from_json(item, index)
            for index, item in enumerate(raw_registrations)
        )
        return cls(
            wire_major=wire_version["major"],
            wire_minor=wire_version["minor"],
            required_features=tuple(required_features),
            backend=value["backend"],
            endpoint_id=value["endpoint_id"],
            name=value["name"],
            incarnation=value["incarnation"],
            backend_payload=_decode_bytes(value["backend_payload"], "backend_payload"),
            registrations=registrations,
            extensions=tuple(
                (key, _decode_bytes(extension, f"extension {key!r}"))
                for key, extension in raw_extensions.items()
            ),
        )


def _registration_from_json(value: Any, index: int) -> RegistrationMetadata:
    context = f"registration at index {index}"
    _require_object_keys(
        value,
        {
            "backend_descriptor",
            "device",
            "memory_type",
            "name",
            "nbytes",
            "registration_id",
        },
        context,
    )
    return RegistrationMetadata(
        registration_id=value["registration_id"],
        name=value["name"],
        nbytes=value["nbytes"],
        device=value["device"],
        memory_type=value["memory_type"],
        backend_descriptor=_decode_bytes(
            value["backend_descriptor"], f"{context} backend_descriptor"
        ),
    )


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise InvalidMetadataError(f"duplicate JSON field: {key!r}")
        value[key] = item
    return value


def _require_object_keys(value: Any, expected: set[str], context: str) -> None:
    if not isinstance(value, dict):
        raise InvalidMetadataError(f"{context} must be a JSON object")
    keys = set(value)
    missing = expected - keys
    unknown = keys - expected
    if missing:
        raise InvalidMetadataError(
            f"{context} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise InvalidMetadataError(
            f"{context} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_bytes(value: Any, field: str) -> bytes:
    if not isinstance(value, str):
        raise InvalidMetadataError(f"{field} must be a base64 string")
    try:
        encoded = value.encode("ascii")
        return base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise InvalidMetadataError(f"{field} is not valid base64") from error


def _immutable_bytes(
    value: Any,
    field: str,
    error_type: type[Exception] = TypeError,
) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise error_type(f"{field} must be bytes-like")
    return bytes(value)


def _immutable_notification_payloads(
    value: Any,
    max_payload_bytes: int,
) -> tuple[tuple[bytes, ...], int]:
    """Snapshot payloads and count bytes during the required size validation."""

    try:
        payloads = value if type(value) is tuple else tuple(value)
    except TypeError as error:
        raise TypeError("payloads must be an iterable of bytes-like values") from error
    total_nbytes = 0
    for payload in payloads:
        if type(payload) is not bytes:
            for item in payloads:
                if not isinstance(item, (bytes, bytearray, memoryview)):
                    raise TypeError("payload must be bytes-like")
                reported_nbytes = (
                    item.nbytes if isinstance(item, memoryview) else len(item)
                )
                if reported_nbytes > max_payload_bytes:
                    raise ValueError("notification payload exceeds the wire size limit")
            immutable_payloads = tuple(
                _immutable_bytes(item, "payload") for item in payloads
            )
            total_nbytes = 0
            for item in immutable_payloads:
                item_nbytes = len(item)
                if item_nbytes > max_payload_bytes:
                    raise ValueError("notification payload exceeds the wire size limit")
                total_nbytes += item_nbytes
            return immutable_payloads, total_nbytes
        payload_nbytes = len(payload)
        if payload_nbytes > max_payload_bytes:
            raise ValueError("notification payload exceeds the wire size limit")
        total_nbytes += payload_nbytes
    return payloads, total_nbytes


def _require_nonempty_str(
    value: Any,
    field: str,
    error_type: type[Exception] = TypeError,
) -> None:
    if not isinstance(value, str) or not value:
        raise error_type(f"{field} must be a non-empty string")


def _utf8_nbytes(value: str, field: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise InvalidMetadataError(f"{field} is not valid Unicode") from error


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
