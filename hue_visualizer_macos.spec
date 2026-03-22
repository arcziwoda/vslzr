# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for VSLZR macOS .app bundle.

Build with: uv run pyinstaller hue_visualizer_macos.spec --clean
"""

from PyInstaller.utils.hooks import collect_submodules

# python-mbedtls is a Cython package — PyInstaller can't auto-detect its submodules
mbedtls_hidden = collect_submodules("mbedtls")

a = Analysis(
    ["src/hue_visualizer/desktop.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("frontend", "frontend"),
        ("assets", "assets"),
    ],
    hiddenimports=mbedtls_hidden + [
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
        # pystray macOS backend (AppKit, loaded dynamically)
        "pystray._darwin",
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
    name="VSLZR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX not commonly available on macOS
    console=False,
    icon="assets/icon.icns",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VSLZR",
)

app = BUNDLE(
    coll,
    name="VSLZR.app",
    icon="assets/icon.icns",
    bundle_identifier="com.arturkempinski.vslzr",
    info_plist={
        "CFBundleName": "VSLZR",
        "CFBundleDisplayName": "VSLZR",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1.0.0",
        # Menu bar only — no Dock icon
        "LSUIElement": True,
        # Microphone access (required by macOS for audio capture)
        "NSMicrophoneUsageDescription": (
            "VSLZR needs microphone access to capture audio "
            "for real-time music visualization on your Philips Hue lights."
        ),
        # High-DPI support
        "NSHighResolutionCapable": True,
        # Minimum macOS version (matches Python 3.11+ support)
        "LSMinimumSystemVersion": "12.0",
    },
)
