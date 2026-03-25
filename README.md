# VSLZR

Real-time music visualization for Philips Hue lights. Analyzes audio via FFT and multi-agent beat tracking, then drives the Hue Entertainment API for low-latency light control synchronized to your music.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-yellow.svg)](https://www.python.org/)
[![Hue Entertainment API](https://img.shields.io/badge/Hue-Entertainment%20API-orange.svg)](https://developers.meethue.com/)

![VSLZR](assets/demo.gif)

## Features

- **Beat detection** — SuperFlux onset detection with multi-agent PI-PLL tracking, per-band onset separation (kick/snare/hi-hat), predictive triggering, and coasting through breakdowns
- **7-band FFT analysis** — spectral decomposition with Mel filterbank, per-frequency brightness and color mapping in spatial modes
- **Hybrid visualization engine** — blends a slow generative layer (hue rotation + BPM-synced breathing) with a fast reactive layer based on audio energy
- **Genre presets** — tuned parameter sets for techno, house, drum & bass, and ambient
- **Spatial light mapping** — distribute frequencies across lights with uniform, frequency zones, wave, mirror, chase, and alternating modes
- **Strobe system** — auto-triggers on detected drops, with manual burst control and safe mode (2 Hz cap)
- **Section awareness** — detects drops, buildups, breakdowns, and sustains to modulate intensity and trigger effects automatically
- **Palette system** — curated genre palettes + algorithmic generation (complementary, triadic, analogous)
- **Intensity & effects size** — three intensity levels (intense/normal/chill) and adjustable effects spread across lights
- **Calibration mode** — visual beat alignment with adjustable latency compensation
- **Web control panel** — dark-themed UI with real-time spectrum visualization, genre/spatial/palette controls, and live light preview
- **Audio-only mode** — works without a Hue Bridge for development and preview (WebSocket-driven light preview in browser)
- **Cross-platform** — macOS (PyAudio) and Windows (WASAPI loopback via PyAudioWPatch), with standalone desktop builds and auto-update

## Downloads

Grab the latest build from [GitHub Releases](https://github.com/arcziwoda/vslzr/releases):

| Platform | Package | Notes |
|----------|---------|-------|
| **Windows** | `VSLZR-vX.X.X-windows-x64.zip` | Runs as system tray icon, opens browser automatically |
| **macOS** | `VSLZR-vX.X.X-macos-arm64.dmg` | Menu bar app. Not signed — right-click → Open on first launch |

## Running from Source

### Prerequisites

**macOS:**
```bash
brew install portaudio mbedtls@2
```

**Windows:** No system dependencies — Python wheels include everything. Requires **Python 3.12** (`python-mbedtls` has no 3.13+ wheels).

### Install & Run

```bash
git clone https://github.com/arcziwoda/vslzr.git
cd vslzr

# Requires uv — https://docs.astral.sh/uv/
uv sync

# Audio-only mode — no bridge config needed
uv run python -m hue_visualizer

# Tests
uv run pytest
```

Open **http://localhost:8080** — control panel with live audio visualization.

### Connect to Hue Bridge

1. Click **BRIDGE** in the UI
2. Follow the wizard — discovers bridge, pairs (press the link button), pick an entertainment area
3. Credentials saved automatically

## Architecture

```
Audio Input → PyAudio Capture (thread) → Ring Buffer
                                            ↓
                              FFT Analyzer (2048-pt Hann window)
                                            ↓
                              Beat Detector (SuperFlux + multi-agent PI-PLL)
                                            ↓
                    ┌───────────────────────────────────────────┐
                    │           Effect Engine (~50 Hz)          │
                    │                                           │
                    │   ColorMapper (spectral → HSV)            │
                    │   SpatialMapper (per-light distribution)  │
                    │   Beat flash + EMA smoothing + gamma      │
                    └──────────────┬────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ↓                             ↓
          WebSocket → Browser            DTLS → Hue Bridge → Lights
          (live visualization)              (~50 Hz streaming)
```

## Configuration

All settings are optional and loaded from `.env`. See [`.env.example`](.env.example) for the full list.

| Setting | Default | Description |
|---------|---------|-------------|
| `BRIDGE_IP` | — | Hue Bridge IP address |
| `HUE_USERNAME` | — | API authentication token |
| `HUE_CLIENTKEY` | — | Entertainment API client key |
| `ENTERTAINMENT_AREA_ID` | — | Entertainment area ID |
| `FPS_TARGET` | `50` | Light update rate (Hz) |
| `BEAT_COOLDOWN_MS` | `300` | Min interval between beats (ms) |
| `ATTACK_ALPHA` | `0.7` | EMA fast attack (0.5–0.8) |
| `RELEASE_ALPHA` | `0.1` | EMA slow release (0.05–0.15) |
| `BRIGHTNESS_GAMMA` | `2.2` | Perceptual gamma correction |
| `MAX_FLASH_HZ` | `3.0` | Epilepsy safety flash limit (Hz) |

## Tech Stack

- **Audio**: PyAudio / PyAudioWPatch, NumPy (FFT, Mel filterbank)
- **Networking**: Hue Entertainment API via DTLS (python-mbedtls)
- **Server**: FastAPI + uvicorn, WebSocket for real-time UI updates
- **Frontend**: Vanilla JS + Canvas (single-file, no build step)
- **Packaging**: uv, PyInstaller (Windows `.exe`, macOS `.app`), GitHub Actions CI

## License

[MIT](LICENSE)
