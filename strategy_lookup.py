"""Lookup the module registry for an AI-generated strategy by slug.

Used by strategy_editor.py to inspect a generated strategy without having to
load its .py file again.
"""
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)


def get_generated_strategy_info(slug: str) -> dict[str, str]:
    """Return metadata about a generated strategy (source file, class name)
    without importing the code.

    Returns:
        A dict with keys 'slug', 'class_name', 'display_name', 'description',
        'created_at', 'file_path' and 'imported' (bool).
    """
    from ai_strategy_writer import _load_manifest
    try:
        meta = next(
            (e for e in _load_manifest() if e.get("slug") == slug), None
        ) or {}
    except Exception as _e:  # noqa: BLE001
        meta = {}
    import strategies.generated as gen  # may raise if package missing
    return {
        "slug": slug,
        "class_name": meta.get("class_name", ""),
        "display_name": meta.get("display_name", ""),
        "description": meta.get("description", ""),
        "created_at": meta.get("created_at", ""),
        "file_path": gen.__path__[0] if hasattr(gen, "__path__") else "",
    }


def load_generated_strategy_module(slug: str):
    """Import (or reload) the generated strategy module for the given slug.

    Returns:
        The loaded module, or None if the module cannot be imported.
    """
    try:
        modname = f"strategies.generated.{slug}"
        return importlib.import_module(modname)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not import generated module %r: %s", slug, exc)
        return None


def get_generated_strategy_class(slug: str):
    """Import a generated strategy by slug and return its class object.

    Raises KeyError if the module has no matching class.
    """
    mod = load_generated_strategy_module(slug)
    if mod is None:
        raise KeyError(f"No module for generated strategy '{slug}'")
    from ai_strategy_writer import _load_manifest as _lm2
    meta = next(
        (e for e in _lm2() if e.get("slug") == slug), None
    ) or {}
    class_name = meta.get("class_name") or slug
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise KeyError(
            f"No class named {class_name!r} in generated module '{slug}'"
        )
    return cls