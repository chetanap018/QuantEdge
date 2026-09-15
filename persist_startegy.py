"""Persist multi-surface layout state (grids, tabs, focus, sort state)
out of process via ai_strategy_writer persistence helpers.
"""
import os

from ai_strategy_writer import (
    GENERATED_DIR,
)


def _layout_path(slug: str) -> str:
    return os.path.join(GENERATED_DIR, f"{slug}.layout.json")


def _alert_path(slug: str) -> str:
    return os.path.join(GENERATED_DIR, f"{slug}.alerts.json")


def _settings_path(slug: str) -> str:
    return os.path.join(GENERATED_DIR, f"{slug}.settings.json")


def _persist_layouts(slug: str) -> None:
    path = _layout_path(slug)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{}")
    except OSError:
        pass


def _persist_alerts(slug: str) -> None:
    path = _alert_path(slug)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("[]")
    except OSError:
        pass


def _persist_settings(slug: str) -> None:
    path = _settings_path(slug)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("[]")
    except OSError:
        pass


def persist_startegy(slug: str) -> None:
    """Normalize persisted layouts, alerts and settings on disk.

    Args:
        slug: the save-game slot to persist into.
    """
    _persist_layouts(slug=slug)
    _persist_alerts(slug=slug)
    _persist_settings(slug=slug)
