"""Path resolution utilities — PyInstaller-aware."""

import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def get_base_dir() -> Path:
    """Return the base directory for bundled resources.

    In development: project root (4 levels up from this file).
    In frozen mode: sys._MEIPASS (PyInstaller extraction dir).
    """
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent.parent.parent


def get_frontend_dir() -> Path:
    """Return the path to the frontend directory."""
    return get_base_dir() / "frontend"


def get_env_path() -> Path | None:
    """Return .env path if it exists, checking multiple locations.

    Search order:
    1. Next to the executable (frozen mode)
    2. Current working directory
    3. User config dir (~/.config/hue-visualizer/.env)
    """
    from .persistence import _config_dir

    candidates: list[Path] = []

    if is_frozen():
        candidates.append(Path(sys.executable).parent / ".env")

    candidates.append(Path.cwd() / ".env")
    candidates.append(_config_dir() / ".env")

    for p in candidates:
        if p.exists():
            return p
    return None
