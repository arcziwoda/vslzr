"""Persistent configuration storage for bridge credentials.

Stores bridge config in a JSON file in the user's config directory.
Uses platformdirs for cross-platform paths. Thread-safe read/write operations.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

logger = logging.getLogger(__name__)

_APP_NAME = "hue-visualizer"
_CONFIG_FILE = "config.json"

_lock = threading.Lock()


def _config_dir() -> Path:
    """Return the config directory path (platform-aware via platformdirs).

    macOS:   ~/Library/Application Support/hue-visualizer
    Windows: C:/Users/<user>/AppData/Roaming/hue-visualizer
    Linux:   ~/.config/hue-visualizer (XDG)
    """
    return Path(user_config_dir(_APP_NAME))


def _config_path() -> Path:
    """Return the full path to the config file."""
    return _config_dir() / _CONFIG_FILE


def _read_raw() -> dict[str, Any]:
    """Read raw config dict from disk. Returns empty dict if file doesn't exist."""
    path = _config_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read config file {path}: {e}")
        return {}


def _write_raw(data: dict[str, Any]) -> None:
    """Write config dict to disk. Creates directory if needed."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Config saved to {path}")
    except OSError as e:
        logger.error(f"Failed to write config file {path}: {e}")
        raise


def load_bridge_config() -> dict[str, str | None]:
    """Load bridge configuration from persistent storage.

    Returns:
        Dict with keys: ip, username, clientkey, entertainment_area_id.
        Values are None if not set.
    """
    with _lock:
        data = _read_raw()
    bridge = data.get("bridge", {})
    return {
        "ip": bridge.get("ip"),
        "username": bridge.get("username"),
        "clientkey": bridge.get("clientkey"),
        "entertainment_area_id": bridge.get("entertainment_area_id"),
    }


def save_bridge_config(
    ip: str,
    username: str,
    clientkey: str,
    area_id: str | None = None,
) -> None:
    """Save bridge configuration to persistent storage.

    Args:
        ip: Bridge IP address.
        username: Hue API username.
        clientkey: Entertainment API client key.
        area_id: Entertainment area ID (optional).
    """
    with _lock:
        data = _read_raw()
        data["bridge"] = {
            "ip": ip,
            "username": username,
            "clientkey": clientkey,
            "entertainment_area_id": area_id,
        }
        _write_raw(data)


def clear_bridge_config() -> None:
    """Remove bridge configuration from persistent storage."""
    with _lock:
        data = _read_raw()
        data.pop("bridge", None)
        _write_raw(data)


def load_audio_device_preference() -> int | None:
    """Load preferred audio device index from persistent storage."""
    with _lock:
        data = _read_raw()
    return data.get("audio_device_index")


def save_audio_device_preference(device_index: int) -> None:
    """Save preferred audio device index to persistent storage."""
    with _lock:
        data = _read_raw()
        data["audio_device_index"] = device_index
        _write_raw(data)


def clear_audio_device_preference() -> None:
    """Remove audio device preference from persistent storage."""
    with _lock:
        data = _read_raw()
        data.pop("audio_device_index", None)
        _write_raw(data)


def get_config_path() -> str:
    """Return the config file path as a string (for diagnostics)."""
    return str(_config_path())
