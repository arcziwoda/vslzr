"""Desktop launcher — system tray / menu bar + browser auto-open.

Entry point for PyInstaller-packaged apps (Windows .exe, macOS .app).
Runs uvicorn in a background thread, opens the browser, and shows a
system tray icon (Windows) or menu bar icon (macOS).
"""

import logging
import multiprocessing
import os
import socket
import sys
import threading
import webbrowser

import uvicorn
from dotenv import load_dotenv

from hue_visualizer.core.paths import get_env_path, get_base_dir


def _fix_windowed_stdio():
    """Redirect stdout/stderr to devnull when running as a windowed PyInstaller app.

    PyInstaller with console=False (Windows) or .app bundle (macOS) sets
    sys.stdout/stderr to None, which crashes logging.StreamHandler and print().
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


def _get_log_path():
    """Return path to the log file in the user's config directory."""
    from hue_visualizer.core.persistence import _config_dir

    log_dir = _config_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "hue-visualizer.log"


def _setup_logging():
    """Configure logging with file output for desktop builds."""
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # Console handler (goes to devnull in windowed mode, but useful for dev)
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    # File handler — persists across sessions, rotates at 5 MB, keeps 3 backups
    log_path = _get_log_path()
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    for name in ("hue_visualizer", "uvicorn"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(console)
        lg.addHandler(file_handler)

    logging.getLogger(__name__).info(f"Log file: {log_path}")


def _find_available_port(host: str, preferred: int) -> int:
    """Return *preferred* if free, otherwise ask the OS for any available port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, preferred))
            return preferred
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _init_macos_app():
    """Initialize NSApplication for macOS .app bundles.

    When launched as a .app bundle, macOS needs the NSApplication to be
    explicitly set up with accessory policy before pystray can create
    status bar items. Without this, the icon silently fails to appear.
    """
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except ImportError:
        pass


def _create_tray_icon(url: str, server: uvicorn.Server):
    """Create and run the system tray icon (blocks on main thread)."""
    try:
        import pystray
        from PIL import Image
    except ImportError:
        # pystray/Pillow not available — just wait for server thread
        logging.getLogger(__name__).warning(
            "pystray not available — running without system tray"
        )
        return None

    icon_path = get_base_dir() / "assets" / "icon.png"
    if icon_path.exists():
        image = Image.open(icon_path)
    else:
        image = Image.new("RGB", (64, 64), (0, 255, 204))

    def on_open_browser(icon, item):
        webbrowser.open(url)

    def on_open_logs(icon, item):
        log_path = _get_log_path()
        if sys.platform == "win32":
            os.startfile(str(log_path.parent))
        elif sys.platform == "darwin":
            import subprocess

            subprocess.Popen(["open", str(log_path.parent)])

    def on_quit(icon, item):
        icon.stop()
        server.should_exit = True

    icon = pystray.Icon(
        "hue-visualizer",
        image,
        "Hue Visualizer",
        menu=pystray.Menu(
            pystray.MenuItem("Open in Browser", on_open_browser, default=True),
            pystray.MenuItem("Show Logs", on_open_logs),
            pystray.MenuItem("Quit", on_quit),
        ),
    )
    return icon


def main():
    _fix_windowed_stdio()

    env_path = get_env_path()
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    _setup_logging()

    from hue_visualizer.core.config import Settings

    settings = Settings()

    port = _find_available_port(settings.server_host, settings.server_port)

    from hue_visualizer.server.app import app

    config = uvicorn.Config(
        app,
        host=settings.server_host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Run uvicorn in a daemon thread so main thread can own the tray
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    url = f"http://localhost:{port}"

    # Open browser after a short delay for server startup
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    # System tray — blocks main thread until user quits
    _init_macos_app()
    icon = _create_tray_icon(url, server)
    if icon:
        icon.run()
    else:
        # No tray available — wait for server thread
        server_thread.join()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
