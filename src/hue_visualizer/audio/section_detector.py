"""Section detection — real-time drop/buildup/breakdown classification.

Five-layer architecture:
1. Feature extraction (raw unnormalized values from AudioAnalyzer)
2. Dual-timescale EMA tracking (short ~1s, long ~8s) → exertion ratios
3. Six-signal weighted fusion → composite drop score
4. Variance-adaptive thresholding (Patin C)
5. Six-state machine with minimum dwell times

Key insight: exertion ratios (short EMA / long EMA) are inherently
gain-invariant. A kick lasting 50ms barely moves the 1s EMA, but a
2-second bass return saturates it. This timescale separation is the
foundation for distinguishing kicks from drops.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Section(str, Enum):
    """Musical section classification."""

    UNKNOWN = "unknown"  # Cold start, EMAs seeding
    QUIET = "quiet"  # Near-silence
    NORMAL = "normal"  # Default state
    BREAKDOWN = "breakdown"  # Low energy section
    BUILDUP = "buildup"  # Rising energy toward drop
    DROP = "drop"  # THE trigger state — strobe fires on entry
    SUSTAIN = "sustain"  # Continuation after DROP


@dataclass
class SectionInfo:
    """Section detection output for a single tick."""

    section: Section = Section.NORMAL
    confidence: float = 0.0  # 0-1
    intensity: float = 0.0  # 0-1, smoothed for effects
    beats_in_section: int = 0
    # Diagnostic fields (for analyze_track.py and tuning)
    drop_score: float = 0.0  # Weighted fusion score
    bass_exertion: float = 0.0  # Short/long bass ratio
    rms_exertion: float = 0.0  # Short/long RMS ratio
    adaptive_threshold: float = 0.0  # Current Patin C threshold


class SectionDetector:
    """Detects musical sections from audio features using dual-timescale
    EMA tracking, multi-signal fusion, and a six-state machine.

    Typical usage:
        detector = SectionDetector(sample_rate_hz=44100 / 1024)
        ...
        info = detector.update(
            bass_energy=features.bass_energy,
            rms=features.rms,
            centroid=features.spectral_centroid,
            is_beat=beat_info.is_beat,
            bpm=beat_info.bpm,
            rms_raw=features.rms_raw,
            spectral_flux=features.spectral_flux,
            spectral_flatness=features.spectral_flatness,
            band_energies=features.band_energies_unnorm,
        )
    """

    def __init__(self, sample_rate_hz: float = 43.07):
        self._sample_rate_hz = max(1.0, sample_rate_hz)

        # --- EMA alphas ---
        self._alpha_short: float = 0.047  # ~1s time constant
        self._alpha_long: float = 0.006  # ~8s time constant
        self._alpha_fast: float = 0.087  # ~0.5s, slope tracking
        self._alpha_drop_score: float = 0.15  # ~140ms smoothing

        # --- Patin C variance-adaptive threshold ---
        self._patin_c_max: float = 1.55
        self._patin_c_min: float = 1.0
        self._base_threshold: float = 0.40

        # --- Drop entry conditions ---
        self._emergency_override: float = 0.90
        self._bass_exertion_min: float = 1.3

        # --- Silence / pause ---
        self._silence_threshold: float = 0.01  # raw RMS
        self._pause_confirm_frames: int = 13  # ~300ms
        self._resume_lockout_frames: int = 43  # ~1s

        # --- Cold start ---
        self._cold_start_lockout: int = 86  # ~2s
        self._full_warmup: int = 344  # ~8s
        self._warmup_threshold_mult: float = 1.3

        # --- Song change ---
        self._song_change_cosine_thresh: float = 0.6
        self._song_change_confirm_sec: float = 2.0

        # --- Signal weights ---
        self._w_bass: float = 0.30
        self._w_broadband: float = 0.25
        self._w_centroid: float = 0.15
        self._w_flux: float = 0.15
        self._w_flatness: float = 0.10
        self._w_buildup: float = 0.05

        # --- Minimum dwell times (frames) ---
        self._min_dwell: dict[Section, int] = {
            Section.UNKNOWN: 86,
            Section.QUIET: 22,
            Section.BREAKDOWN: 43,
            Section.BUILDUP: 22,
            Section.DROP: 22,
            Section.SUSTAIN: 43,
            Section.NORMAL: 0,
        }

        # --- Initialize mutable state ---
        self._init_state()

    def _init_state(self) -> None:
        """Initialize / reset all mutable state."""
        # Dual-timescale EMAs
        self._ema_short_bass: float = 0.0
        self._ema_long_bass: float = 0.0
        self._ema_short_rms: float = 0.0
        self._ema_long_rms: float = 0.0
        self._ema_short_centroid: float = 0.0
        self._ema_long_centroid: float = 0.0
        self._ema_long_flux: float = 0.0
        self._ema_long_flatness: float = 0.0

        # Slope tracking
        self._prev_rms_raw: float = 0.0
        self._energy_slope_ema: float = 0.0

        # Drop scoring
        self._drop_score: float = 0.0
        self._drop_score_variance_ema: float = 0.0

        # State machine
        self._state: Section = Section.UNKNOWN
        self._frames_in_state: int = 0
        self._frame_count: int = 0
        self._total_beats: int = 0
        self._beats_in_section: int = 0

        # Pause / resume
        self._silence_frames: int = 0
        self._emas_frozen: bool = False
        self._pause_start_frame: int = 0
        self._resume_lockout_remaining: int = 0

        # Song change detection
        feature_window_size = int(self._sample_rate_hz * 4)  # 4s buffer
        self._feature_window: deque[np.ndarray] = deque(maxlen=max(feature_window_size, 10))
        self._prev_feature_avg: np.ndarray | None = None
        self._song_change_frames: int = 0
        self._song_change_threshold_boost: int = 0
        self._prev_bpm: float = 0.0

        # Buildup recency tracker (frames since we left BUILDUP)
        self._frames_since_buildup: int = 999

        # Smoothed output intensity
        self._smoothed_intensity: float = 0.0
        self._sustain_intensity: float = 0.0

        # Seeding flag
        self._seeded: bool = False

        # Confidence
        self._section_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        bass_energy: float,
        rms: float,
        centroid: float,
        is_beat: bool,
        bpm: float,
        now: float | None = None,
        *,
        rms_raw: float = 0.0,
        spectral_flux: float = 0.0,
        spectral_flatness: float = 0.0,
        band_energies: np.ndarray | None = None,
    ) -> SectionInfo:
        """Process one frame and classify the current musical section.

        Args:
            bass_energy: Normalized bass energy (kept for backward compat).
            rms: Normalized RMS (kept for backward compat).
            centroid: Spectral centroid in Hz.
            is_beat: Whether a beat was detected this frame.
            bpm: Current estimated BPM (0 if unknown).
            now: Current monotonic time.
            rms_raw: Raw unnormalized RMS (critical for gain-invariant detection).
            spectral_flux: Raw spectral flux.
            spectral_flatness: Spectral flatness 0-1.
            band_energies: 7-band unnormalized power sums (for bass EMA + song change).
        """
        if now is None:
            now = time.monotonic()

        self._frame_count += 1
        self._frames_in_state += 1

        # Beat counting
        if is_beat:
            self._total_beats += 1
            self._beats_in_section += 1

        # Track buildup recency
        if self._state != Section.BUILDUP:
            self._frames_since_buildup += 1

        # Compute raw bass from band_energies if available
        if band_energies is not None and len(band_energies) >= 2:
            bass_raw = float(band_energies[0] + band_energies[1]) / 2.0
        else:
            # Fallback: use rms_raw as bass proxy (degraded mode)
            bass_raw = rms_raw

        # --- 1. Pause detection ---
        if self._check_pause(rms_raw):
            return self._make_output()

        # --- 2. Resume lockout ---
        if self._resume_lockout_remaining > 0:
            self._resume_lockout_remaining -= 1
            return self._make_output()

        # --- 3. Seed EMAs on first real signal ---
        if not self._seeded:
            if rms_raw > self._silence_threshold:
                self._seed_emas(bass_raw, rms_raw, centroid, spectral_flux, spectral_flatness)
            return self._make_output()

        # --- 4. Update EMAs ---
        self._update_emas(bass_raw, rms_raw, centroid, spectral_flux, spectral_flatness)

        # --- 5. Energy slope ---
        self._compute_energy_slope(rms_raw)

        # --- 6. Exertion ratios ---
        bass_exertion, rms_exertion, centroid_ratio = self._compute_exertion_ratios()

        # --- 7. Drop score (6-signal fusion) ---
        drop_score = self._compute_drop_score(
            bass_exertion, rms_exertion, centroid_ratio,
            spectral_flux, spectral_flatness,
        )

        # --- 8. Adaptive threshold ---
        threshold = self._compute_adaptive_threshold()

        # --- 9. Song change detection ---
        self._check_song_change(band_energies, centroid, spectral_flatness, bpm)

        # --- 10. State machine ---
        new_state = self._transition_state(
            drop_score, threshold,
            bass_exertion, rms_exertion, centroid_ratio,
            rms_raw,
        )

        # Handle state transition
        if new_state != self._state:
            # Track buildup recency on exit
            if self._state == Section.BUILDUP:
                self._frames_since_buildup = 0
            # Capture intensity at DROP for SUSTAIN decay
            if new_state == Section.DROP:
                self._sustain_intensity = max(0.5, min(1.0, drop_score * 2.0))
            self._state = new_state
            self._frames_in_state = 0
            self._beats_in_section = 0

        # --- 11. Compute output intensity ---
        self._update_intensity(rms_exertion, drop_score)

        # --- 12. Build output ---
        self._section_confidence = min(1.0, drop_score / max(threshold, 0.01))
        return SectionInfo(
            section=self._state,
            confidence=self._section_confidence,
            intensity=self._smoothed_intensity,
            beats_in_section=self._beats_in_section,
            drop_score=self._drop_score,
            bass_exertion=bass_exertion,
            rms_exertion=rms_exertion,
            adaptive_threshold=threshold,
        )

    def reset(self) -> None:
        """Reset all state (cold restart)."""
        self._init_state()

    @property
    def current_section(self) -> Section:
        """Current detected section."""
        return self._state

    @property
    def beats_in_section(self) -> int:
        """Number of beats since the current section started."""
        return self._beats_in_section

    # ------------------------------------------------------------------
    # Layer 2: Dual-timescale EMA
    # ------------------------------------------------------------------

    def _seed_emas(
        self,
        bass_raw: float,
        rms_raw: float,
        centroid: float,
        flux: float,
        flatness: float,
    ) -> None:
        """Seed all EMAs with first frame values. Called once."""
        self._ema_short_bass = self._ema_long_bass = bass_raw
        self._ema_short_rms = self._ema_long_rms = rms_raw
        self._ema_short_centroid = self._ema_long_centroid = centroid
        self._ema_long_flux = max(flux, 1e-6)  # Avoid zero division
        self._ema_long_flatness = flatness
        self._prev_rms_raw = rms_raw
        self._seeded = True

    def _update_emas(
        self,
        bass_raw: float,
        rms_raw: float,
        centroid: float,
        flux: float,
        flatness: float,
    ) -> None:
        """Advance all dual-timescale EMAs. Skipped when frozen."""
        if self._emas_frozen:
            return

        a_s = self._alpha_short
        a_l = self._alpha_long

        # Bass: separate from broadband (sub_bass + bass power sum)
        self._ema_short_bass += a_s * (bass_raw - self._ema_short_bass)
        self._ema_long_bass += a_l * (bass_raw - self._ema_long_bass)

        # Broadband RMS
        self._ema_short_rms += a_s * (rms_raw - self._ema_short_rms)
        self._ema_long_rms += a_l * (rms_raw - self._ema_long_rms)

        # Spectral centroid
        self._ema_short_centroid += a_s * (centroid - self._ema_short_centroid)
        self._ema_long_centroid += a_l * (centroid - self._ema_long_centroid)

        # Flux and flatness (long-term only — used as reference baselines)
        self._ema_long_flux += a_l * (flux - self._ema_long_flux)
        self._ema_long_flatness += a_l * (flatness - self._ema_long_flatness)

    def _compute_energy_slope(self, rms_raw: float) -> None:
        """Compute fast EMA of frame-to-frame RMS delta (energy slope)."""
        delta = rms_raw - self._prev_rms_raw
        self._energy_slope_ema += self._alpha_fast * (delta - self._energy_slope_ema)
        self._prev_rms_raw = rms_raw

    def _compute_exertion_ratios(self) -> tuple[float, float, float]:
        """Compute short/long ratios for bass, RMS, and centroid.

        Returns:
            (bass_exertion, rms_exertion, centroid_ratio)
            All ≥ 0. A value of 1.0 means "at section average."
        """
        bass_ex = self._ema_short_bass / max(self._ema_long_bass, 1e-10)
        rms_ex = self._ema_short_rms / max(self._ema_long_rms, 1e-10)
        centroid_r = self._ema_short_centroid / max(self._ema_long_centroid, 1e-6)
        return bass_ex, rms_ex, centroid_r

    # ------------------------------------------------------------------
    # Layer 3: Six-signal weighted fusion
    # ------------------------------------------------------------------

    def _compute_drop_score(
        self,
        bass_ex: float,
        rms_ex: float,
        centroid_ratio: float,
        flux: float,
        flatness: float,
    ) -> float:
        """Compute composite drop score from six normalized signals."""
        # Individual signals, each clipped to [0, 1]
        bass_signal = _clip01((bass_ex - 1.0) / 2.0)
        energy_signal = _clip01((rms_ex - 1.0) / 1.5)
        centroid_signal = _clip01((1.0 - centroid_ratio) / 0.3)
        flux_signal = _clip01(
            (flux / max(self._ema_long_flux, 1e-6) - 2.0) / 3.0
        )
        flatness_signal = _clip01(1.0 - flatness * 5.0)

        # Buildup context bonus: full credit if in BUILDUP, partial if recent
        if self._state == Section.BUILDUP:
            buildup_bonus = 1.0
        elif self._frames_since_buildup < int(self._sample_rate_hz):
            # Decaying bonus within 1s of leaving BUILDUP
            buildup_bonus = 1.0 - self._frames_since_buildup / self._sample_rate_hz
        else:
            buildup_bonus = 0.0

        raw_score = (
            self._w_bass * bass_signal
            + self._w_broadband * energy_signal
            + self._w_centroid * centroid_signal
            + self._w_flux * flux_signal
            + self._w_flatness * flatness_signal
            + self._w_buildup * buildup_bonus
        )

        # Smooth with EMA
        self._drop_score += self._alpha_drop_score * (raw_score - self._drop_score)

        # Update variance for Patin C (after smoothing so variance tracks score stability)
        delta = raw_score - self._drop_score
        self._drop_score_variance_ema += 0.01 * (
            delta * delta - self._drop_score_variance_ema
        )

        return self._drop_score

    # ------------------------------------------------------------------
    # Layer 4: Variance-adaptive threshold (Patin C)
    # ------------------------------------------------------------------

    def _compute_adaptive_threshold(self) -> float:
        """Compute drop threshold using Patin's variance-adaptive C."""
        variance = self._drop_score_variance_ema
        c_range = self._patin_c_max - self._patin_c_min
        c = self._patin_c_max - (variance / 0.02) * c_range
        c = max(self._patin_c_min, min(self._patin_c_max, c))
        threshold = self._base_threshold / c

        # Warmup: elevated threshold during first ~8s
        if self._frame_count < self._full_warmup:
            threshold *= self._warmup_threshold_mult

        # Song change: elevated threshold for 4s after detection
        if self._song_change_threshold_boost > 0:
            threshold *= 1.5
            self._song_change_threshold_boost -= 1

        return threshold

    # ------------------------------------------------------------------
    # Layer 5: Six-state machine
    # ------------------------------------------------------------------

    def _transition_state(
        self,
        drop_score: float,
        threshold: float,
        bass_ex: float,
        rms_ex: float,
        centroid_ratio: float,
        rms_raw: float,
    ) -> Section:
        """Determine next state based on current signals and dwell time."""
        state = self._state

        # Enforce minimum dwell time
        if self._frames_in_state < self._min_dwell.get(state, 0):
            return state

        # UNKNOWN → NORMAL after cold start lockout
        if state == Section.UNKNOWN:
            if self._frame_count >= self._cold_start_lockout and self._seeded:
                return Section.NORMAL
            return Section.UNKNOWN

        # Any → QUIET on near-silence
        if rms_raw < self._silence_threshold * 2:
            if state != Section.QUIET:
                return Section.QUIET
            return Section.QUIET

        # QUIET → NORMAL when audio returns
        if state == Section.QUIET:
            if rms_raw >= self._silence_threshold * 2:
                return Section.NORMAL
            return Section.QUIET

        # Emergency override: any non-UNKNOWN/DROP → DROP
        if (
            state not in (Section.UNKNOWN, Section.DROP)
            and drop_score > self._emergency_override
            and bass_ex > 2.0
            and rms_ex > 1.8
        ):
            return Section.DROP

        # Normal DROP entry (not from DROP or SUSTAIN)
        if (
            state not in (Section.DROP, Section.SUSTAIN, Section.UNKNOWN)
            and drop_score > threshold
            and bass_ex > self._bass_exertion_min
        ):
            return Section.DROP

        # State-specific transitions
        if state == Section.NORMAL:
            if rms_ex < 0.7:
                return Section.BREAKDOWN
            return Section.NORMAL

        if state == Section.BREAKDOWN:
            if self._energy_slope_ema > 0.001 and centroid_ratio > 1.05:
                return Section.BUILDUP
            if rms_ex > 0.9:
                return Section.NORMAL
            return Section.BREAKDOWN

        if state == Section.BUILDUP:
            if rms_ex < 0.7:
                return Section.BREAKDOWN
            if self._energy_slope_ema < -0.001:
                return Section.NORMAL
            return Section.BUILDUP

        if state == Section.DROP:
            # After dwell, always transition to SUSTAIN
            return Section.SUSTAIN

        if state == Section.SUSTAIN:
            if rms_ex < 0.7:
                return Section.BREAKDOWN
            if drop_score < threshold * 0.5:
                return Section.NORMAL
            # Re-trigger DROP on new spike from SUSTAIN
            if drop_score > threshold and bass_ex > self._bass_exertion_min:
                return Section.DROP
            return Section.SUSTAIN

        return state

    # ------------------------------------------------------------------
    # Operational edge cases
    # ------------------------------------------------------------------

    def _check_pause(self, rms_raw: float) -> bool:
        """Detect pause (silence) and freeze EMAs. Returns True if paused."""
        if rms_raw < self._silence_threshold:
            self._silence_frames += 1
        else:
            if self._emas_frozen:
                self._handle_resume()
            self._silence_frames = 0
            return False

        if self._silence_frames >= self._pause_confirm_frames and not self._emas_frozen:
            self._emas_frozen = True
            self._pause_start_frame = self._frame_count

        return self._emas_frozen

    def _handle_resume(self) -> None:
        """Handle resume from pause — re-seed or full reset."""
        pause_duration_frames = self._frame_count - self._pause_start_frame
        self._emas_frozen = False
        self._silence_frames = 0

        if pause_duration_frames > 5 * self._sample_rate_hz:
            # Long pause (>5s): full reset
            self._seeded = False
            self._state = Section.UNKNOWN
            self._frames_in_state = 0
            self._frame_count = 0
            self._drop_score = 0.0
            self._drop_score_variance_ema = 0.0
            self._energy_slope_ema = 0.0
            self._smoothed_intensity = 0.0
            self._sustain_intensity = 0.0
        else:
            # Short pause: re-seed short EMAs from long (preserve song context)
            self._ema_short_bass = self._ema_long_bass
            self._ema_short_rms = self._ema_long_rms
            self._ema_short_centroid = self._ema_long_centroid

        self._resume_lockout_remaining = self._resume_lockout_frames

    def _check_song_change(
        self,
        band_energies: np.ndarray | None,
        centroid: float,
        flatness: float,
        bpm: float,
    ) -> None:
        """Detect song changes via spectral similarity and BPM shift."""
        if band_energies is None or len(band_energies) < 7:
            return

        # Build 9-element feature vector
        centroid_norm = centroid / 10000.0
        feature_vec = np.zeros(9)
        feature_vec[:7] = band_energies[:7]
        feature_vec[7] = centroid_norm
        feature_vec[8] = flatness
        self._feature_window.append(feature_vec)

        # Need 2 seconds of data
        window_size = int(self._sample_rate_hz * 2.0)
        if len(self._feature_window) < window_size:
            return

        # Compute current window average
        window_list = list(self._feature_window)[-window_size:]
        current_avg = np.mean(window_list, axis=0)

        if self._prev_feature_avg is not None:
            # Cosine similarity
            dot = float(np.dot(current_avg, self._prev_feature_avg))
            norm_a = float(np.linalg.norm(current_avg))
            norm_b = float(np.linalg.norm(self._prev_feature_avg))
            if norm_a > 1e-10 and norm_b > 1e-10:
                similarity = dot / (norm_a * norm_b)
            else:
                similarity = 1.0

            # BPM change check
            bpm_changed = (
                self._prev_bpm > 0
                and bpm > 0
                and abs(bpm - self._prev_bpm) / self._prev_bpm > 0.05
            )

            if similarity < self._song_change_cosine_thresh or bpm_changed:
                self._song_change_frames += 1
            else:
                self._song_change_frames = 0

            # Confirmed song change
            confirm_frames = int(self._song_change_confirm_sec * self._sample_rate_hz)
            if self._song_change_frames >= confirm_frames:
                # Decay long EMAs (retain 20%)
                self._ema_long_bass *= 0.2
                self._ema_long_rms *= 0.2
                self._ema_long_centroid *= 0.2
                self._ema_long_flux *= 0.2
                self._ema_long_flatness *= 0.2
                # Elevate threshold for 4 seconds
                self._song_change_threshold_boost = int(4.0 * self._sample_rate_hz)
                self._song_change_frames = 0

        self._prev_feature_avg = current_avg.copy()
        # Only update reference BPM when NOT accumulating song change evidence
        # (otherwise, _prev_bpm races to match current bpm on each frame)
        if self._song_change_frames == 0:
            self._prev_bpm = bpm

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _update_intensity(self, rms_exertion: float, drop_score: float) -> None:
        """Compute smoothed intensity from current state."""
        # Target intensity per state
        if self._state in (Section.UNKNOWN, Section.QUIET, Section.NORMAL):
            target = 0.0
        elif self._state == Section.BREAKDOWN:
            target = _clip01(1.0 - rms_exertion)
        elif self._state == Section.BUILDUP:
            target = _clip01(self._energy_slope_ema * 100.0)
        elif self._state == Section.DROP:
            target = _clip01(drop_score * 2.0)
        elif self._state == Section.SUSTAIN:
            self._sustain_intensity *= 0.98
            target = self._sustain_intensity
        else:
            target = 0.0

        # Asymmetric EMA: fast attack, slow release
        if target > self._smoothed_intensity:
            alpha = 0.3
        else:
            alpha = 0.05
        self._smoothed_intensity += alpha * (target - self._smoothed_intensity)
        self._smoothed_intensity = _clip01(self._smoothed_intensity)

    def _make_output(self) -> SectionInfo:
        """Build SectionInfo for early-return paths (pause, lockout, unseeded)."""
        return SectionInfo(
            section=self._state,
            confidence=self._section_confidence,
            intensity=self._smoothed_intensity,
            beats_in_section=self._beats_in_section,
            drop_score=self._drop_score,
            bass_exertion=0.0,
            rms_exertion=0.0,
            adaptive_threshold=0.0,
        )


def _clip01(x: float) -> float:
    """Clip value to [0, 1]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
