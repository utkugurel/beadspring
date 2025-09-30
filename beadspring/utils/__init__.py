"""Utility namespace for :mod:`beadspring`."""

from __future__ import annotations

from importlib import import_module

__all__ = []

_modules = [
    import_module(".build_architectures", __name__),
    import_module(".file_utils", __name__),
]

for _module in _modules:
    _names = getattr(_module, "__all__", None)
    if _names is None:
        _names = [name for name in dir(_module) if not name.startswith("_")]
    globals().update({name: getattr(_module, name) for name in _names})
    __all__.extend(_names)

__all__ = sorted(set(__all__))
