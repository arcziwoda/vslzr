"""Windows desktop launcher — system tray + browser auto-open.

Entry point for the PyInstaller-packaged Windows .exe.
Runs uvicorn in a background thread, opens the browser, and shows a system tray icon.
"""

import logging
import threading
import webbrowser

import uvicorn
from dotenv import load_dotenv

from hue_visualizer.core.paths import get_env_path, get_base_dir


def _setup_logging():
    """Configure logging (same as __main__.py)."""
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )

    for name in ("hue_visualizer", "uvicorn"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)


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

    def on_quit(icon, item):
        icon.stop()
        server.should_exit = True

    icon = pystray.Icon(
        "hue-visualizer",
        image,
        "Hue Visualizer",
        menu=pystray.Menu(
            pystray.MenuItem("Open in Browser", on_open_browser, default=True),
            pystray.MenuItem("Quit", on_quit),
        ),
    )
    return icon


def main():
    env_path = get_env_path()
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    _setup_logging()

    from hue_visualizer.core.config import Settings

    settings = Settings()

    from hue_visualizer.server.app import app

    config = uvicorn.Config(
        app,
        host=settings.server_host,
        port=settings.server_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Run uvicorn in a daemon thread so main thread can own the tray
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    url = f"http://localhost:{settings.server_port}"

    # Open browser after a short delay for server startup
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    # System tray — blocks main thread until user quits
    icon = _create_tray_icon(url, server)
    if icon:
        icon.run()
    else:
        # No tray available — wait for server thread
        server_thread.join()


if __name__ == "__main__":
    main()
