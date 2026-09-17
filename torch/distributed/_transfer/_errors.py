__all__ = [
    "BackendError",
    "BackendFailureError",
    "BackendMismatchError",
    "BackendNotFoundError",
    "BusyError",
    "ClosedError",
    "InvalidMetadataError",
    "InvalidRegionError",
    "TransferCancelledError",
    "TransferError",
    "TransferTimeoutError",
    "UnsupportedError",
]


class TransferError(RuntimeError):
    """Base exception for endpoint transfer failures."""


class BackendError(TransferError):
    """Base exception for transfer backend failures."""


class BackendNotFoundError(BackendError):
    """Raised when a requested backend is not registered."""


class BackendMismatchError(BackendError):
    """Raised when metadata belongs to a different backend."""


class BackendFailureError(BackendError):
    """Raised when a backend operation fails."""


class InvalidMetadataError(TransferError):
    """Raised when endpoint metadata is malformed or unsupported."""


class InvalidRegionError(TransferError):
    """Raised when a memory region is invalid."""


class UnsupportedError(TransferError):
    """Raised when a requested operation is unsupported."""


class ClosedError(TransferError):
    """Raised when an operation targets a closed resource."""


class BusyError(TransferError):
    """Raised when a resource cannot be changed while work is in flight."""


class TransferTimeoutError(TransferError):
    """Raised when transfer work exceeds its deadline."""


class TransferCancelledError(TransferError):
    """Raised when transfer work is cancelled."""
