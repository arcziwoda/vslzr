# Hue Music Visualizer

Real-time music visualization for Philips Hue lights. Analyzes audio via FFT and beat detection, drives Hue Entertainment API streaming for low-latency light control.

## Architecture

```
[Mic/System Audio] → PyAudio → AudioCapture (thread) → ring buffer
                                    ↓
                        AudioAnalyzer (FFT 2048 Hann) → AudioFeatures
                                    ↓
                        BeatDetector (adaptive threshold + PLL) → BeatInfo
                                    ↓
                    AudioPipeline.process_all() — processes all buffered frames
                                    ↓
                        server/app.py audio_loop (~30 Hz)
                           ↓                    ↓
              WebSocket → Browser        EffectEngine.tick() → list[LightState]
                  (live viz UI)              ↓
                                ColorMapper + SpatialMapper + beat flash + EMA smoothing
                                             ↓
                                EntertainmentController → DTLS → Bridge → Lights (~50 Hz)
```

All audio processing and light control in Python backend. Web UI is a control panel + real-time visualization.

## Current Status

Full pipeline operational: audio → FFT → beat detection → hybrid effect engine → Hue Entertainment API + WebSocket → browser visualization. P0/P1/P2 backlog complete — STFT overlap, predictive beats, hybrid reactive-generative engine, section detection, per-band onsets, spatial effects, palettes, UI controls, safety improvements.

See `BACKLOG.md` for remaining items and research spec references.

## Development

### Package Management — uv only

```bash
uv add package-name          # Add dependency (NEVER edit pyproject.toml manually)
uv add --dev package-name    # Dev dependency
uv sync                      # Install all
uv run python <script>       # Run anything
uv run pytest                # Tests
```

### System Dependencies (macOS)

```bash
brew install portaudio       # PyAudio
brew install mbedtls@2       # Entertainment API (DTLS)
```

### Running

```bash
uv run python -m hue_visualizer                # Main app — server + audio + web UI on localhost:8080
uv run python scripts/test_audio.py            # Test audio pipeline (terminal)
uv run python scripts/test_entertainment.py    # Test Hue Entertainment API
```

## Project Structure

```
src/hue_visualizer/
├── __main__.py                # Entry point (uvicorn + dotenv)
├── core/
│   ├── config.py              # Pydantic Settings (all params from .env)
│   ├── exceptions.py          # Custom exception hierarchy
│   └── persistence.py         # State persistence (bridge config, presets)
├── bridge/
│   ├── connection.py          # HueBridge REST API wrapper
│   ├── discovery.py           # Bridge discovery & pairing
│   ├── entertainment_controller.py  # Entertainment API (DTLS streaming)
│   └── effects.py             # Tick-based effects (PulseEffect, BreatheEffect, StrobeEffect, FlashDecayEffect, ColorCycleEffect)
├── audio/
│   ├── capture.py             # PyAudio wrapper, threaded capture, ring buffer
│   ├── analyzer.py            # FFT (2048 Hann, 50% overlap), 7-band energies, Mel filterbank, spectral features
│   ├── beat_detector.py       # Adaptive beat detection, per-band onsets, BPM estimation, PLL
│   └── section_detector.py    # Section detection (drop/buildup/breakdown awareness)
├── visualizer/
│   ├── color_mapper.py        # Audio features → HSV (centroid→hue, RMS→brightness, flatness→saturation)
│   ├── spatial.py             # Per-light distribution (uniform, frequency_zones, wave, mirror)
│   ├── engine.py              # EffectEngine: orchestrates color+spatial+beat+smoothing+safety → LightState[]
│   └── presets.py             # Genre presets (techno, house, DnB, ambient) — parameter sets
├── server/
│   └── app.py                 # FastAPI + WebSocket + AudioPipeline + EffectEngine + EntertainmentController
└── utils/
    └── color_conversion.py    # RGB ↔ HSV ↔ XY (CIE) conversions

frontend/
└── index.html                 # Single-file web UI (dark industrial techno, canvas viz, vanilla JS)

scripts/                       # Setup & test scripts
docs/                          # Research documents
```

## Non-Obvious Design Decisions

- **Bridge optional**: EffectEngine always active — drives light preview on WebSocket even without bridge connection. Audio-only mode works without .env bridge vars
- **ColorMapper hue**: palette-driven with spectral centroid as ±20° offset modulator (deliberate deviation from research spec which uses centroid as primary hue driver)
- **SpatialMapper**: distribution logic lives in `EffectEngine._distribute()`, not in SpatialMapper itself
- **Hybrid engine**: GenerativeLayer always active (slow hue rotation + breathing), blended with reactive layer — quiet passages ~80% generative, loud ~80% reactive
- **Predictive beats**: PLL predicts next beat, engine fires early with confidence gating — reduces perceived latency
- **BeatDetector**: uses median (matching research spec) for adaptive threshold. Beat threshold is fully automatic (adaptive Parallelcube) — no manual sensitivity config
- **effects.py**: standalone building blocks (Pulse, Breathe, Strobe, FlashDecay, ColorCycle) — NOT integrated with EffectEngine
- **Genre presets**: `_apply_genre_preset()` updates pipeline + engine atomically, uses `set_base_attack_alpha()` to preserve intensity multiplier
- **Light send rate**: configurable via `fps_target` (default 50 Hz) — oversampling compensates for UDP packet loss

## Configuration (.env)

**Required for light control:**
- `BRIDGE_IP` — Hue Bridge IP
- `HUE_USERNAME` — API username (from `scripts/get_clientkey.py`)
- `HUE_CLIENTKEY` — Entertainment API client key
- `ENTERTAINMENT_AREA_ID` — Entertainment area ID (usually "1")

**Optional (with defaults — audio-only mode works without .env):**
- Audio: `SAMPLE_RATE=44100`, `BUFFER_SIZE=1024`, `FFT_SIZE=2048`
- Beat: `BEAT_COOLDOWN_MS=300` (threshold is fully automatic)
- Smoothing: `ATTACK_ALPHA=0.7`, `RELEASE_ALPHA=0.1`, `BRIGHTNESS_GAMMA=2.2`, `BASS_BOOST_FACTOR=2.0`
- Safety: `MAX_FLASH_HZ=3.0` (epilepsy limit)
- Server: `SERVER_HOST=0.0.0.0`, `SERVER_PORT=8080`

## Key Technical Constraints

- **Hue Entertainment API**: ~50 Hz send rate (configurable `fps_target`), ~12.5 FPS effective at bulbs
- **End-to-end latency**: 80-120ms typical (audio → light)
- **Safety**: Max 3 Hz flash rate, never strobe saturated red
- **Smoothing**: Asymmetric EMA — fast attack (α=0.5-0.8), slow release (α=0.05-0.15)
- **Beat detection**: Adaptive threshold (variance-based), 300ms min cooldown
- **Brightness**: Gamma-corrected (γ=2.2) for perceptual linearity (Weber-Fechner)
- **Bass**: Boosted 2× to compensate Fletcher-Munson hearing curve

## Testing

```bash
uv run pytest                              # All tests
uv run pytest tests/test_beat_detector.py  # Single module
uv run pytest -k "test_name"              # Single test by name
```

## Research

- `docs/cemplex_audio_lightning_research.md` — DSP, Hue protocol, perceptual science, architecture
- `docs/ilightshow_research.md` — iLightShow teardown: pre-computed beats, palette system, effects

## Backlog

See `BACKLOG.md` for remaining items with references to research specs.

## Documentation

Always use context7 MCP tools when code generation or library/API docs are needed.
