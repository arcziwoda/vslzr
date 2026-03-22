"""FastAPI server with WebSocket for real-time audio visualization and light control."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from ..audio import AudioCapture, AudioAnalyzer, AudioFeatures, BeatDetector, BeatInfo, BAND_NAMES, SectionDetector, SectionInfo
from ..bridge.discovery import (
    discover_bridge,
    create_entertainment_user,
    list_entertainment_areas,
    verify_connection,
)
from ..bridge.entertainment_controller import EntertainmentController
from ..core.config import Settings
from ..core.exceptions import BridgeDiscoveryError, BridgeConnectionError
from ..core.exceptions import AudioCaptureError
from ..core.persistence import (
    load_bridge_config, save_bridge_config, clear_bridge_config,
    load_audio_device_preference, save_audio_device_preference, clear_audio_device_preference,
)
from ..utils.color_conversion import hsv_to_rgb
from ..visualizer import EffectEngine
from ..visualizer.engine import INTENSITY_LEVELS, INTENSITY_MULTIPLIERS, INTENSITY_NORMAL
from ..visualizer.presets import PRESETS, PALETTES, generate_palette, PALETTE_ALGO_MODES

logger = logging.getLogger(__name__)

SPECTRUM_BINS = 64
WS_RATE_HZ = 30


from ..core.paths import get_frontend_dir

FRONTEND_DIR = get_frontend_dir()


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WebSocket connected ({len(self.active)} clients)")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info(f"WebSocket disconnected ({len(self.active)} clients)")

    async def broadcast(self, message: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.active:
                self.active.remove(ws)


class AudioPipeline:
    """Audio capture -> FFT analysis -> beat detection pipeline.

    Includes peak-hold buffer (Task 0.5): tracks max RMS, band energies,
    and spectral flux between output ticks so transient peaks are not lost.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self.capture = AudioCapture(
            sample_rate=settings.sample_rate,
            buffer_size=settings.buffer_size,
        )
        self.analyzer = AudioAnalyzer(
            sample_rate=settings.sample_rate,
            fft_size=settings.fft_size,
            bass_boost=settings.bass_boost_factor,
            hop_size=settings.buffer_size,
        )
        self.beat_detector = BeatDetector(
            sample_rate=settings.sample_rate,
            hop_size=settings.buffer_size,
            cooldown_ms=settings.beat_cooldown_ms,
            bpm_min=settings.bpm_min,
            bpm_max=settings.bpm_max,
        )
        self.section_detector = SectionDetector(
            sample_rate_hz=float(settings.sample_rate) / settings.buffer_size,
        )
        self.features = AudioFeatures()
        self.beat_info = BeatInfo()
        self.section_info = SectionInfo()
        self._pending_beat = False
        self._pending_beat_strength = 0.0

        # Per-band onset latching (Task 1.5): latch onsets between output ticks
        self._pending_kick: bool = False
        self._pending_snare: bool = False
        self._pending_hihat: bool = False
        self._peak_kick_energy: float = 0.0
        self._peak_snare_energy: float = 0.0
        self._peak_hihat_energy: float = 0.0

        # Peak-hold buffer: captures transient maxima between output ticks
        self._peak_rms: float = 0.0
        self._peak_band_energies: np.ndarray = np.zeros(7)
        self._peak_spectral_flux: float = 0.0
        self._peak_has_data: bool = False

    def start(self):
        self.capture.start()
        self._sync_sample_rate()
        logger.info("Audio pipeline started")

    def _sync_sample_rate(self):
        """Re-create analyzer/beat_detector if device sample rate differs from config."""
        actual_rate = self.capture._device_rate
        settings = self._settings
        if actual_rate != settings.sample_rate:
            logger.warning(
                f"Device sample rate ({actual_rate} Hz) differs from config "
                f"({settings.sample_rate} Hz) — re-initializing DSP components"
            )
            self.analyzer = AudioAnalyzer(
                sample_rate=actual_rate,
                fft_size=settings.fft_size,
                bass_boost=settings.bass_boost_factor,
                hop_size=settings.buffer_size,
            )
            self.beat_detector = BeatDetector(
                sample_rate=actual_rate,
                hop_size=settings.buffer_size,
                cooldown_ms=settings.beat_cooldown_ms,
                bpm_min=settings.bpm_min,
                bpm_max=settings.bpm_max,
            )
            self.section_detector = SectionDetector(
                sample_rate_hz=float(actual_rate) / settings.buffer_size,
            )

    def stop(self):
        self.capture.stop()
        logger.info("Audio pipeline stopped")

    def reset_analysis(self) -> None:
        """Reset all analysis state after device switch."""
        self.analyzer.reset()
        self.beat_detector.reset()
        self.section_detector.reset()
        self.features = AudioFeatures()
        self.beat_info = BeatInfo()
        self.section_info = SectionInfo()
        self._pending_beat = False
        self._pending_beat_strength = 0.0
        self._pending_kick = False
        self._pending_snare = False
        self._pending_hihat = False
        self._peak_kick_energy = 0.0
        self._peak_snare_energy = 0.0
        self._peak_hihat_energy = 0.0
        self._peak_rms = 0.0
        self._peak_band_energies = np.zeros(7)
        self._peak_spectral_flux = 0.0
        self._peak_has_data = False

    @property
    def is_running(self) -> bool:
        return self.capture.is_running

    def process_all(self) -> bool:
        """Process all buffered audio frames. Returns True if any were processed."""
        frames = self.capture.get_all_frames()
        for frame in frames:
            self.features = self.analyzer.analyze(frame)
            self.beat_info = self.beat_detector.detect(self.features)
            if self.beat_info.is_beat:
                self._pending_beat = True
                self._pending_beat_strength = max(
                    self._pending_beat_strength, self.beat_info.beat_strength
                )

            # Latch per-band onsets (Task 1.5)
            if self.beat_info.kick_onset:
                self._pending_kick = True
                self._peak_kick_energy = max(
                    self._peak_kick_energy, self.beat_info.kick_energy
                )
            if self.beat_info.snare_onset:
                self._pending_snare = True
                self._peak_snare_energy = max(
                    self._peak_snare_energy, self.beat_info.snare_energy
                )
            if self.beat_info.hihat_onset:
                self._pending_hihat = True
                self._peak_hihat_energy = max(
                    self._peak_hihat_energy, self.beat_info.hihat_energy
                )

            # Section detection: runs on every frame alongside beat detector
            self.section_info = self.section_detector.update(
                bass_energy=self.features.bass_energy,
                rms=self.features.rms,
                centroid=self.features.spectral_centroid,
                is_beat=self.beat_info.is_beat,
                bpm=self.beat_info.bpm,
            )

            # Update peak-hold buffer with per-frame maxima
            self._peak_rms = max(self._peak_rms, self.features.rms)
            self._peak_band_energies = np.maximum(
                self._peak_band_energies, self.features.band_energies
            )
            self._peak_spectral_flux = max(
                self._peak_spectral_flux, self.features.spectral_flux
            )
            self._peak_has_data = True

        return len(frames) > 0

    def consume_beat(self) -> tuple[bool, float]:
        """Return and clear pending beat flag."""
        had_beat = self._pending_beat
        strength = self._pending_beat_strength
        self._pending_beat = False
        self._pending_beat_strength = 0.0
        return had_beat, strength

    def consume_band_onsets(self) -> tuple[bool, bool, bool, float, float, float]:
        """Return and clear pending per-band onset flags (Task 1.5).

        Returns:
            (kick_onset, snare_onset, hihat_onset,
             kick_energy, snare_energy, hihat_energy)
        """
        kick = self._pending_kick
        snare = self._pending_snare
        hihat = self._pending_hihat
        kick_e = self._peak_kick_energy
        snare_e = self._peak_snare_energy
        hihat_e = self._peak_hihat_energy

        self._pending_kick = False
        self._pending_snare = False
        self._pending_hihat = False
        self._peak_kick_energy = 0.0
        self._peak_snare_energy = 0.0
        self._peak_hihat_energy = 0.0

        return kick, snare, hihat, kick_e, snare_e, hihat_e

    def consume_features(self) -> AudioFeatures:
        """Return peak-held features and reset the peak buffer.

        Returns features with the maximum RMS, band energies, and spectral
        flux observed since the last consume call, preserving the most recent
        spectrum and other scalar features for display.
        """
        if not self._peak_has_data:
            return self.features

        # Build features with peak values overlaid on the latest frame
        peak_features = AudioFeatures(
            band_energies=self._peak_band_energies.copy(),
            band_energies_raw=self.features.band_energies_raw,
            spectral_centroid=self.features.spectral_centroid,
            spectral_flux=self._peak_spectral_flux,
            spectral_rolloff=self.features.spectral_rolloff,
            spectral_flatness=self.features.spectral_flatness,
            rms=self._peak_rms,
            peak=self.features.peak,
            spectrum=self.features.spectrum,
        )

        # Reset peak buffer
        self._peak_rms = 0.0
        self._peak_band_energies = np.zeros(7)
        self._peak_spectral_flux = 0.0
        self._peak_has_data = False

        return peak_features


def _prepare_spectrum(spectrum_db: np.ndarray, n_bins: int = SPECTRUM_BINS) -> list[float]:
    """Downsample FFT spectrum to n_bins, normalized to 0-1."""
    if len(spectrum_db) == 0:
        return [0.0] * n_bins

    # Normalize dB to 0-1 range
    min_db, max_db = -80.0, 0.0
    normalized = np.clip((spectrum_db - min_db) / (max_db - min_db), 0, 1)

    # Log-spaced bin edges for better low-frequency resolution
    n_fft = len(normalized)
    edges = np.unique(
        np.logspace(0, np.log10(max(n_fft, 2)), n_bins + 1, dtype=int).clip(0, n_fft - 1)
    )

    result = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        result.append(float(np.max(normalized[s : max(e, s + 1)])))

    # Pad or trim to exact n_bins
    while len(result) < n_bins:
        result.append(0.0)
    return result[:n_bins]


def _light_states_to_preview(engine: EffectEngine) -> list[dict]:
    """Convert engine's current per-light smoothed state to RGB dicts for UI preview."""
    preview = []
    for light in engine._lights:
        r, g, b = hsv_to_rgb(light.hue, light.saturation, light.brightness)
        preview.append({"r": r, "g": g, "b": b})
    return preview


# --- Global state ---
manager = ConnectionManager()
pipeline: AudioPipeline | None = None
effect_engine: EffectEngine | None = None
entertainment_ctrl: EntertainmentController | None = None
settings: Settings | None = None
current_genre: str = "techno"
current_palette: str = "neon"
current_intensity: str = INTENSITY_NORMAL

# Bridge credentials currently in use (populated from .env or persistence or wizard)
_bridge_ip: str | None = None
_bridge_username: str | None = None
_bridge_clientkey: str | None = None
_bridge_area_id: str | None = None


async def audio_loop():
    """Background task: process audio, drive lights, broadcast to WebSocket clients.

    Uses target-based timing (Task 0.7) to prevent drift from processing time.
    Light send rate uses settings.fps_target (Task 0.2) for UDP oversampling.
    """
    ws_interval = 1.0 / WS_RATE_HZ
    # Task 0.2: Use fps_target from config (default 50 Hz) instead of hardcoded 25 Hz.
    # Oversampling at 50-60 Hz compensates for UDP packet loss; bridge decimates internally.
    light_interval = 1.0 / settings.fps_target if settings else 1.0 / 50
    last_light_tick = time.monotonic()

    while True:
        # Task 0.7: Target-based timing — compute next tick before processing
        next_ws_tick = time.monotonic() + ws_interval

        now = time.monotonic()

        if pipeline and pipeline.is_running:
            had_frames = pipeline.process_all()
            had_beat, beat_strength = pipeline.consume_beat()
            kick, snare, hihat, kick_e, snare_e, hihat_e = pipeline.consume_band_onsets()

            # Task 0.5: Consume peak-held features for this output tick
            output_features = pipeline.consume_features()

            # --- Effect engine tick (always, for preview + optional light output) ---
            dt_light = now - last_light_tick
            if effect_engine and dt_light >= light_interval:
                last_light_tick = now

                beat_for_engine = BeatInfo(
                    is_beat=had_beat,
                    bpm=pipeline.beat_info.bpm,
                    bpm_confidence=pipeline.beat_info.bpm_confidence,
                    beat_strength=beat_strength,
                    predicted_next_beat=pipeline.beat_info.predicted_next_beat,
                    time_since_beat=pipeline.beat_info.time_since_beat,
                    kick_onset=kick,
                    snare_onset=snare,
                    hihat_onset=hihat,
                    kick_energy=kick_e,
                    snare_energy=snare_e,
                    hihat_energy=hihat_e,
                )

                try:
                    light_states = effect_engine.tick(
                        output_features, beat_for_engine, dt_light,
                        section_info=pipeline.section_info,
                    )
                    # Task B.5: Send all light states as a batch instead of one-by-one
                    if entertainment_ctrl:
                        entertainment_ctrl.set_light_states_batch(light_states)
                except Exception as e:
                    logger.error(f"Light control error: {e}")

            # --- WebSocket broadcast ---
            if had_frames and manager.active:
                f = output_features
                b = pipeline.beat_info

                data = {
                    "type": "audio",
                    "spectrum": _prepare_spectrum(f.spectrum),
                    "bands": [round(v, 4) for v in f.band_energies_raw.tolist()],
                    "band_names": BAND_NAMES,
                    "beat": {
                        "is_beat": had_beat,
                        "bpm": round(b.bpm, 1),
                        "confidence": round(b.bpm_confidence, 2),
                        "strength": round(beat_strength, 2),
                    },
                    "rms": round(f.rms, 4),
                    "peak": round(f.peak, 4),
                    "spectral_centroid": round(f.spectral_centroid, 1),
                    "spectral_flatness": round(f.spectral_flatness, 3),
                }

                if effect_engine:
                    data["lights_active"] = (
                        entertainment_ctrl is not None
                        and entertainment_ctrl.is_connected
                    )
                    data["bridge_ip"] = _bridge_ip if entertainment_ctrl else None
                    data["spatial_mode"] = effect_engine.spatial_mapper.mode
                    data["light_preview"] = _light_states_to_preview(effect_engine)
                    data["genre"] = current_genre
                    data["palette"] = current_palette
                    data["color_mode"] = effect_engine.color_mapper.color_mode
                    data["reactive_weight"] = round(effect_engine.reactive_weight, 3)
                    data["energy_level"] = round(effect_engine.energy_level, 3)
                    data["section"] = {
                        "name": effect_engine.current_section.value,
                        "intensity": round(effect_engine.section_intensity, 3),
                    }
                    # Task 1.12: Intensity selector state
                    data["intensity_level"] = effect_engine.intensity_level
                    # Task 1.13: Effects size state
                    data["effects_size"] = round(effect_engine.effects_size, 2)
                    # Task 1.14: Light group info
                    data["light_groups"] = effect_engine.light_groups
                    # Task 2.5: Safe mode state
                    data["safe_mode"] = effect_engine.safe_mode
                    # Task 2.19: Saturation boost state
                    data["saturation_boost"] = round(effect_engine.saturation_boost, 2)
                    # Strobe state
                    data["strobe_enabled"] = effect_engine.strobe_enabled
                    data["strobe_active"] = effect_engine.strobe_active
                    # Calibration mode state
                    data["calibration_mode"] = effect_engine.calibration_mode
                    # Task 2.6: Calibration delay state
                    data["calibration_delay"] = round(effect_engine.calibration_delay_ms)
                    # Task 2.8: Brightness min/max state
                    data["brightness_min"] = round(effect_engine.brightness_min, 2)
                    data["brightness_max"] = round(effect_engine.brightness_max, 2)

                # Task B.10: Include control state so frontend can sync on (re)connect
                if pipeline:
                    data["bass_boost"] = round(pipeline.analyzer.bass_boost, 1)
                    # Audio device info for UI sync
                    dev = pipeline.capture.current_device_info
                    data["audio_device"] = dev["name"] if dev else None
                    data["audio_device_index"] = dev["index"] if dev else None
                    data["audio_error"] = pipeline.capture._last_error

                await manager.broadcast(json.dumps(data))

        # Task 0.7: Sleep only the remaining time until next tick
        sleep_time = next_ws_tick - time.monotonic()
        await asyncio.sleep(max(0, sleep_time))


def _resolve_bridge_credentials() -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve bridge credentials: .env overrides persistent config.

    Returns:
        (ip, username, clientkey, area_id) -- any may be None.
    """
    # Start with persistent config
    persisted = load_bridge_config()
    ip = persisted.get("ip")
    username = persisted.get("username")
    clientkey = persisted.get("clientkey")
    area_id = persisted.get("entertainment_area_id")

    # .env / environment variables override persistent storage
    if settings:
        if settings.bridge_ip:
            ip = settings.bridge_ip
        if settings.hue_username:
            username = settings.hue_username
        if settings.hue_clientkey:
            clientkey = settings.hue_clientkey
        if settings.entertainment_area_id:
            area_id = settings.entertainment_area_id

    return ip, username, clientkey, area_id


def _do_bridge_connect(
    ip: str, username: str, clientkey: str, area_id: str | None
) -> None:
    """Connect to bridge and wire up the effect engine.

    Mutates global state: entertainment_ctrl, effect_engine, _bridge_* vars.
    Does NOT touch the audio pipeline.
    """
    global entertainment_ctrl, effect_engine
    global _bridge_ip, _bridge_username, _bridge_clientkey, _bridge_area_id

    ctrl = EntertainmentController(
        bridge_ip=ip,
        username=username,
        clientkey=clientkey,
        entertainment_area_id=area_id,
    )
    ctrl.connect()
    entertainment_ctrl = ctrl

    _bridge_ip = ip
    _bridge_username = username
    _bridge_clientkey = clientkey
    _bridge_area_id = area_id

    num_lights = ctrl._num_lights or (settings.num_lights if settings else 6)

    if effect_engine:
        effect_engine.set_num_lights(num_lights)
        if ctrl.light_positions:
            effect_engine.set_light_positions(ctrl.light_positions)
            logger.info(
                f"Light positions from bridge applied: "
                f"{[round(p, 3) for p in ctrl.light_positions]}"
            )

    logger.info(f"Bridge connected: {ip}, {num_lights} lights")


def _do_bridge_disconnect() -> None:
    """Disconnect from bridge. Engine continues in preview mode."""
    global entertainment_ctrl

    if entertainment_ctrl:
        try:
            entertainment_ctrl.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting entertainment API: {e}")
        entertainment_ctrl = None
        logger.info("Bridge disconnected -- preview mode")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop audio pipeline, effect engine, and entertainment controller."""
    global pipeline, effect_engine, entertainment_ctrl, settings
    global _bridge_ip, _bridge_username, _bridge_clientkey, _bridge_area_id

    settings = Settings()

    pipeline = AudioPipeline(settings)

    # Resolve audio device: .env overrides persistent preference
    saved_device = load_audio_device_preference()
    device_index = settings.audio_device_index if settings.audio_device_index is not None else saved_device
    if device_index is not None:
        pipeline.capture.device_index = device_index
        logger.info(f"Using saved audio device index: {device_index}")

    try:
        pipeline.start()
    except Exception as e:
        logger.error(f"Failed to start audio capture: {e}")
        logger.warning("Running without audio -- connect a microphone and restart")

    # --- Effect engine (always created — drives UI preview even without bridge) ---
    num_lights = settings.num_lights

    # --- Resolve bridge credentials: .env overrides persistent config ---
    # Credentials are loaded into globals but we do NOT auto-connect.
    # The user connects manually from the UI (BRIDGE wizard → CONNECT).
    ip, username, clientkey, area_id = _resolve_bridge_credentials()
    _bridge_ip = ip
    _bridge_username = username
    _bridge_clientkey = clientkey
    _bridge_area_id = area_id

    if ip and username and clientkey:
        logger.info(f"Saved bridge config found: {ip} (area {area_id}) — connect from UI")
    else:
        logger.info("No bridge configured -- audio + preview mode")

    if not effect_engine:
        effect_engine = EffectEngine(
            num_lights=num_lights,
            gamma=settings.brightness_gamma,
            attack_alpha=settings.attack_alpha,
            release_alpha=settings.release_alpha,
            max_flash_hz=settings.max_flash_hz,
            spatial_mode=settings.spatial_mode,
            latency_compensation_ms=settings.latency_compensation_ms,
            predictive_confidence_threshold=settings.predictive_confidence_threshold,
            generative_hue_cycle_period=settings.generative_hue_cycle_period,
            generative_breathing_rate_hz=settings.generative_breathing_rate_hz,
            generative_breathing_min=settings.generative_breathing_min,
            generative_breathing_max=settings.generative_breathing_max,
        )

    # Task 2.6: Apply calibration delay from config
    if settings.calibration_delay_ms > 0:
        effect_engine.set_calibration_delay(settings.calibration_delay_ms)
    # Task 2.8: Apply brightness min/max from config
    if settings.brightness_min > 0:
        effect_engine.set_brightness_min(settings.brightness_min)
    if settings.brightness_max < 1.0:
        effect_engine.set_brightness_max(settings.brightness_max)

    # Task 1.15: Pass bridge light positions to spatial mapper when available
    if entertainment_ctrl and entertainment_ctrl.light_positions:
        effect_engine.set_light_positions(entertainment_ctrl.light_positions)
        logger.info(
            f"Light positions from bridge applied: "
            f"{[round(p, 3) for p in entertainment_ctrl.light_positions]}"
        )

    logger.info(
        f"Effect engine: {num_lights} lights, mode={settings.spatial_mode}, "
        f"latency_comp={settings.latency_compensation_ms}ms, "
        f"predictive_thresh={settings.predictive_confidence_threshold}, "
        f"generative_breathing={settings.generative_breathing_rate_hz}Hz"
    )

    # Task 0.6: Apply genre preset at startup so engine params match the preset
    _apply_genre_preset(current_genre)

    task = asyncio.create_task(audio_loop())
    logger.info(f"Server ready -- open http://localhost:{settings.server_port}")

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    if pipeline:
        pipeline.stop()
    _do_bridge_disconnect()


app = FastAPI(title="VSLZR", lifespan=lifespan)


@app.get("/")
async def index():
    """Serve the main visualization UI."""
    html_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                _handle_control(msg)
            except (json.JSONDecodeError, ValueError):
                pass
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Bridge setup wizard REST API
# ---------------------------------------------------------------------------


@app.get("/api/bridge/status")
async def bridge_status():
    """Return current bridge connection status and saved config info."""
    connected = entertainment_ctrl is not None and entertainment_ctrl.is_connected
    num_lights = entertainment_ctrl._num_lights if entertainment_ctrl else 0
    has_saved = bool(_bridge_ip and _bridge_username and _bridge_clientkey)
    return {
        "connected": connected,
        "bridge_ip": _bridge_ip if connected else None,
        "num_lights": num_lights,
        "area_id": _bridge_area_id if connected else None,
        "has_saved_config": has_saved,
        "saved_bridge_ip": _bridge_ip if has_saved else None,
        "saved_area_id": _bridge_area_id if has_saved else None,
    }


@app.post("/api/bridge/discover")
async def bridge_discover():
    """Discover a Hue Bridge on the local network."""
    try:
        ip = await asyncio.get_event_loop().run_in_executor(None, discover_bridge)
        return {"ip": ip}
    except BridgeDiscoveryError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/bridge/pair")
async def bridge_pair(body: dict | None = None):
    """Pair with the bridge (user must press the link button).

    Polls for up to 30 seconds waiting for the button press.

    Request body (optional):
        {"ip": "192.168.x.x"}   -- override discovered IP
    """
    ip = (body or {}).get("ip") or _bridge_ip
    if not ip:
        # Try auto-discover
        try:
            ip = await asyncio.get_event_loop().run_in_executor(None, discover_bridge)
        except BridgeDiscoveryError as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"No bridge IP provided and discovery failed: {e}"},
            )

    # Poll for ~30 seconds (15 attempts x 2s sleep)
    last_error = ""
    for attempt in range(15):
        try:
            username, clientkey = await asyncio.get_event_loop().run_in_executor(
                None, create_entertainment_user, ip
            )
            return {
                "username": username,
                "clientkey": clientkey,
                "bridge_ip": ip,
            }
        except BridgeConnectionError as e:
            last_error = str(e)
            if "button not pressed" in last_error.lower() or "link button" in last_error.lower():
                await asyncio.sleep(2)
                continue
            # Non-retryable error
            return JSONResponse(status_code=400, content={"error": last_error})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return JSONResponse(
        status_code=408,
        content={"error": "Timed out waiting for link button press. Please try again."},
    )


@app.get("/api/bridge/areas")
async def bridge_areas(ip: str | None = None, username: str | None = None):
    """List entertainment areas on the bridge.

    Query params:
        ip       -- bridge IP (defaults to current)
        username -- API username (defaults to current)
    """
    _ip = ip or _bridge_ip
    _user = username or _bridge_username
    if not _ip or not _user:
        return JSONResponse(
            status_code=400,
            content={"error": "Bridge IP and username required"},
        )

    try:
        areas = await asyncio.get_event_loop().run_in_executor(
            None, list_entertainment_areas, _ip, _user
        )
        return {"areas": areas}
    except BridgeConnectionError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/bridge/connect")
async def bridge_connect(body: dict | None = None):
    """Connect to bridge with the given or stored credentials.

    Request body (optional -- uses stored credentials if omitted):
        {"ip": "...", "username": "...", "clientkey": "...", "area_id": "1"}
    """
    global _bridge_ip, _bridge_username, _bridge_clientkey, _bridge_area_id

    b = body or {}
    ip = b.get("ip") or _bridge_ip
    username = b.get("username") or _bridge_username
    clientkey = b.get("clientkey") or _bridge_clientkey
    area_id = b.get("area_id") or _bridge_area_id

    if not ip or not username or not clientkey:
        return JSONResponse(
            status_code=400,
            content={"error": "Bridge IP, username and clientkey required"},
        )

    # Disconnect existing connection first
    if entertainment_ctrl and entertainment_ctrl.is_connected:
        _do_bridge_disconnect()

    try:
        await asyncio.get_event_loop().run_in_executor(
            None, _do_bridge_connect, ip, username, clientkey, area_id
        )
        num_lights = entertainment_ctrl._num_lights if entertainment_ctrl else 0
        return {
            "connected": True,
            "bridge_ip": ip,
            "num_lights": num_lights,
            "area_id": area_id,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/bridge/disconnect")
async def bridge_disconnect():
    """Disconnect from the bridge. Audio pipeline continues."""
    _do_bridge_disconnect()
    return {"connected": False}


@app.post("/api/bridge/save")
async def bridge_save():
    """Save current bridge credentials to persistent storage."""
    if not _bridge_ip or not _bridge_username or not _bridge_clientkey:
        return JSONResponse(
            status_code=400,
            content={"error": "No bridge credentials to save. Connect first."},
        )
    try:
        save_bridge_config(
            ip=_bridge_ip,
            username=_bridge_username,
            clientkey=_bridge_clientkey,
            area_id=_bridge_area_id,
        )
        return {"saved": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/api/bridge/config")
async def bridge_clear_config():
    """Clear saved bridge credentials from persistent storage."""
    try:
        clear_bridge_config()
        return {"cleared": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Audio device endpoints
# ---------------------------------------------------------------------------


@app.get("/api/audio/devices")
async def list_audio_devices():
    """List available audio input devices."""
    if not pipeline:
        return JSONResponse(status_code=503, content={"error": "Audio pipeline not initialized"})
    try:
        devices = await asyncio.get_event_loop().run_in_executor(
            None, pipeline.capture.list_devices
        )
        current = pipeline.capture.current_device_info
        return {
            "devices": devices,
            "current_device_index": current["index"] if current else None,
            "current_device_name": current["name"] if current else None,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@app.get("/api/diagnostics")
async def diagnostics():
    """Return paths and info useful for debugging."""
    from ..core.persistence import get_config_path

    info = {"config_path": get_config_path(), "log_path": None}

    try:
        from ..desktop import _get_log_path

        info["log_path"] = str(_get_log_path())
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# WebSocket control message handler
# ---------------------------------------------------------------------------


def _handle_control(msg: dict):
    """Apply control messages from the frontend."""
    if not pipeline:
        return

    t = msg.get("type")

    if t == "set_bass_boost":
        v = float(msg["value"])
        pipeline.analyzer.bass_boost = v
        logger.info(f"Bass boost -> {v}")

    elif t == "set_spatial_mode" and effect_engine:
        mode = msg.get("value", "frequency_zones")
        effect_engine.set_spatial_mode(mode)
        logger.info(f"Spatial mode -> {mode}")

    elif t == "set_genre":
        global current_genre
        genre = msg.get("value", "techno")
        current_genre = genre
        _apply_genre_preset(genre)

    elif t == "set_color_mode" and effect_engine:
        mode = msg.get("value", "palette")
        effect_engine.set_color_mode(mode)
        logger.info(f"Color mode -> {mode}")

    elif t == "set_palette":
        global current_palette
        name = msg.get("value", "neon")
        palette = PALETTES.get(name)
        if palette and effect_engine:
            current_palette = name
            effect_engine.set_palette(palette)
            logger.info(f"Palette -> {name}")

    elif t == "set_intensity":
        global current_intensity
        level = msg.get("value", INTENSITY_NORMAL)
        if level in INTENSITY_LEVELS and effect_engine:
            current_intensity = level
            effect_engine.set_intensity(level)
            logger.info(f"Intensity -> {level}")

    elif t == "set_effects_size" and effect_engine:
        v = float(msg["value"])
        effect_engine.set_effects_size(v)
        logger.info(f"Effects size -> {v}")

    elif t == "set_safe_mode" and effect_engine:
        enabled = bool(msg.get("value", False))
        effect_engine.set_safe_mode(enabled)
        logger.info(f"Safe mode -> {'ON' if enabled else 'OFF'}")

    elif t == "set_effects_size_preset" and effect_engine:
        # Convenience: accepts named presets "1L", "25%", "50%", "ALL"
        preset = msg.get("value", "ALL")
        if preset == "1L":
            size = 1.0 / max(effect_engine.num_lights, 1)
        elif preset == "25%":
            size = 0.25
        elif preset == "50%":
            size = 0.5
        else:
            size = 1.0
        effect_engine.set_effects_size(size)
        logger.info(f"Effects size preset -> {preset} ({size})")

    elif t == "set_palette_algo" and effect_engine:
        # Task 2.10: Algorithmic palette generation
        algo_mode = msg.get("mode", "complementary")
        base_hue = float(msg.get("base_hue", 200))
        if algo_mode in PALETTE_ALGO_MODES:
            palette = generate_palette(algo_mode, base_hue)
            current_palette = f"algo:{algo_mode}"
            effect_engine.set_palette(palette)
            logger.info(
                f"Algorithmic palette -> {algo_mode} base={base_hue:.0f} "
                f"hues={[round(h, 1) for h in palette]}"
            )

    elif t == "set_saturation" and effect_engine:
        # Task 2.19: Saturation slider
        v = float(msg.get("value", 1.0))
        effect_engine.set_saturation_boost(v)
        logger.info(f"Saturation boost -> {v:.2f}")

    elif t == "set_strobe_enabled" and effect_engine:
        enabled = bool(msg.get("value", False))
        effect_engine.set_strobe_enabled(enabled)
        logger.info(f"Auto strobe -> {'ON' if enabled else 'OFF'}")

    elif t == "trigger_flash" and effect_engine:
        effect_engine.trigger_manual_flash()
        logger.info("Manual flash triggered")

    elif t == "trigger_strobe" and effect_engine:
        effect_engine.trigger_manual_strobe()
        logger.info("Manual strobe burst triggered")

    elif t == "set_calibration_mode" and effect_engine:
        enabled = bool(msg.get("value", False))
        effect_engine.set_calibration_mode(enabled)

    elif t == "set_calibration_delay" and effect_engine:
        # Task 2.6: Manual calibration delay
        v = float(msg.get("value", 0))
        effect_engine.set_calibration_delay(v)
        logger.info(f"Calibration delay -> {v}ms (effective: {effect_engine.effective_latency_compensation_ms:.0f}ms)")

    elif t == "set_brightness_min" and effect_engine:
        # Task 2.8: Brightness floor
        v = float(msg.get("value", 0))
        effect_engine.set_brightness_min(v)
        logger.info(f"Brightness min -> {v:.2f}")

    elif t == "set_brightness_max" and effect_engine:
        # Task 2.8: Brightness cap
        v = float(msg.get("value", 1.0))
        effect_engine.set_brightness_max(v)
        logger.info(f"Brightness max -> {v:.2f}")

    elif t == "set_audio_device":
        device_index = msg.get("value")
        if device_index is not None:
            device_index = int(device_index)
        try:
            info = pipeline.capture.switch_device(device_index)
            pipeline._sync_sample_rate()
            pipeline.reset_analysis()
            if device_index is not None:
                save_audio_device_preference(device_index)
            else:
                clear_audio_device_preference()
            logger.info(f"Audio device -> {info.get('name')} (index={device_index})")
        except AudioCaptureError as e:
            logger.error(f"Failed to switch audio device: {e}")


def _apply_genre_preset(genre: str) -> None:
    """Apply a genre preset to pipeline + effect engine, including default palette.

    Task B.4: Uses public setter methods instead of accessing private attributes.
    Task 1.11: Genre preset now includes default_palette, applied atomically.
    """
    global current_palette

    preset = PRESETS.get(genre)
    if not preset:
        return

    if pipeline:
        pipeline.beat_detector.set_cooldown(preset.beat_cooldown_ms)
        pipeline.beat_detector.set_bpm_range(preset.bpm_min, preset.bpm_max)
        pipeline.beat_detector.reset()
        pipeline.analyzer.bass_boost = preset.bass_boost

    if effect_engine:
        # Task 1.12: Use set_base_attack_alpha so intensity multiplier is preserved
        effect_engine.set_base_attack_alpha(preset.attack_alpha)
        effect_engine.release_alpha = preset.release_alpha
        effect_engine.set_spatial_mode(preset.spatial_mode)
        effect_engine.set_flash_tau(preset.flash_tau)
        effect_engine.set_hue_drift_speed(preset.hue_drift_speed)
        effect_engine.set_strobe_frequency(preset.strobe_frequency)

        # Apply genre's default palette (Task 1.11)
        palette = PALETTES.get(preset.default_palette)
        if palette:
            current_palette = preset.default_palette
            effect_engine.set_palette(palette)

    logger.info(f"Genre preset -> {genre} (palette: {preset.default_palette})")
