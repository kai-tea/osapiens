"""Compatibility wrapper for the Cini-owned label pipeline package."""

from importlib import import_module

_impl = import_module("cini.label_pipeline")

__path__ = _impl.__path__
__all__ = getattr(_impl, "__all__", [])

for _name in __all__:
    globals()[_name] = getattr(_impl, _name)
