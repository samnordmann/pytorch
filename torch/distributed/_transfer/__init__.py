"""Experimental, backend-independent point-to-point memory transfers.

This is a review prototype, not a stable or production-supported API.
"""

from ._api import Catalog, Endpoint, Registration, Work
from ._backend import Backend

__all__ = ["Backend", "Catalog", "Endpoint", "Registration", "Work"]
