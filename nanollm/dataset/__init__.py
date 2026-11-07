from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .data import get_batch


__all__ = ["get_batch"]


def __getattr__(name):
    if name == "get_batch":
        from .data import get_batch
        return get_batch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")