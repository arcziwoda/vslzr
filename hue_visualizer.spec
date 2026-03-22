# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Hue Visualizer Windows desktop app.

Build with: uv run pyinstaller hue_visualizer.spec --clean
"""

a = Analysis(
    ["src/hue_visualizer/desktop.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("frontend", "frontend"),
        ("assets", "assets"),
    ],
    hiddenimports=[
        # uvicorn internals (dynamic imports)
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.wsproto_impl",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        # FastAPI / Starlette
        "starlette.routing",
        "starlette.responses",
        "starlette.middleware",
        "multipart",
        # pystray (platform-specific backend loaded dynamically)
        "pystray._win32",
        # hue_visualizer modules
        "hue_visualizer",
        "hue_visualizer.server.app",
        "hue_visualizer.audio",
        "hue_visualizer.audio.capture",
        "hue_visualizer.audio.analyzer",
        "hue_visualizer.audio.beat_detector",
        "hue_visualizer.audio.section_detector",
        "hue_visualizer.bridge",
        "hue_visualizer.bridge.connection",
        "hue_visualizer.bridge.discovery",
        "hue_visualizer.bridge.entertainment_controller",
        "hue_visualizer.bridge.effects",
        "hue_visualizer.visualizer",
        "hue_visualizer.visualizer.engine",
        "hue_visualizer.visualizer.color_mapper",
        "hue_visualizer.visualizer.spatial",
        "hue_visualizer.visualizer.presets",
        "hue_visualizer.core",
        "hue_visualizer.core.config",
        "hue_visualizer.core.persistence",
        "hue_visualizer.core.paths",
        "hue_visualizer.core.exceptions",
        "hue_visualizer.utils",
        "hue_visualizer.utils.color_conversion",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HueVisualizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon="assets/icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="HueVisualizer",
)
