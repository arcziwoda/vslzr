"""Beat detection — energy-based detection, autocorrelation BPM, PLL tracking.

Session 5 rewrite: stable BPM via autocorrelation of onset function,
proper PLL with proportional period+phase correction, octave error
protection, confidence gating, and output smoothing.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .analyzer import AudioFeatures


@dataclass
class BeatInfo:
    """Beat detection results for a single frame."""

    is_beat: bool = False
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    beat_strength: float = 0.0  # 0-1, how strong the beat is
    predicted_next_beat: float = 0.0  # Timestamp of predicted next beat
    time_since_beat: float = 0.0  # Seconds since last beat

    # Per-band onsets (Task 1.5)
    kick_onset: bool = False  # Low band (20-250 Hz) — kicks
    snare_onset: bool = False  # Mid band (250-4000 Hz) — snares
    hihat_onset: bool = False  # High band (4-20 kHz) — hi-hats
    kick_energy: float = 0.0  # Current low band energy (0-1)
    snare_energy: float = 0.0  # Current mid band energy (0-1)
    hihat_energy: float = 0.0  # Current high band energy (0-1)


class BeatDetector:
    """Real-time beat detection with autocorrelation BPM and PLL tracking.

    Detection: energy-based bass threshold (adaptive, variance-scaled).
    BPM estimation: autocorrelation of onset function over ~4s window.
    Tracking: PLL with proportional phase+period correction.
    Safety: octave error protection, confidence gating, output smoothing.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_size: int = 1024,
        cooldown_ms: float = 300,
        bpm_min: float = 80.0,
        bpm_max: float = 180.0,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self._manual_cooldown_sec = cooldown_ms / 1000.0
        self.cooldown_sec = self._manual_cooldown_sec
        self.auto_cooldown = True  # Auto-adjust cooldown based on BPM
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max

        # Frame rate derived from audio params
        self._frame_rate = sample_rate / hop_size  # ~43 fps at 44100/1024
        self._frame_dur = 1.0 / self._frame_rate  # ~23ms

        # --- Beat detection ---
        # Energy history (~1.5 seconds for threshold)
        history_len = int(self._frame_rate * 1.5)
        self._bass_history: deque[float] = deque(maxlen=max(history_len, 30))

        # --- Flux-based onset detection ---
        # Log compression parameter (gamma ~ 100 for log(1 + gamma * |X|))
        self._flux_gamma: float = 100.0
        # Flux onset threshold history (~0.2 seconds for adaptive median)
        flux_thresh_len = max(int(self._frame_rate * 0.2), 5)
        self._flux_onset_history: deque[float] = deque(maxlen=flux_thresh_len)
        self._flux_threshold_mult: float = 1.5  # Multiplier over median for flux onset

        # Beat timing
        self._last_beat_time: float = 0.0

        # --- Autocorrelation BPM estimation ---
        # Onset function buffer (~4 seconds for autocorrelation)
        self._onset_buf_len = int(self._frame_rate * 4)
        self._onset_buffer: deque[float] = deque(maxlen=self._onset_buf_len)

        # Lag range for autocorrelation (bpm_min..bpm_max → period in frames)
        self._lag_min = max(1, int(self._frame_rate * 60.0 / self.bpm_max))
        self._lag_max = int(self._frame_rate * 60.0 / self.bpm_min)

        # Frame counter for periodic autocorrelation (~every 0.5s)
        self._frame_count: int = 0
        self._acorr_interval: int = max(1, int(self._frame_rate * 0.5))  # ~22 frames

        # Raw autocorrelation BPM (before PLL)
        self._raw_bpm: float = 0.0
        self._raw_confidence: float = 0.0

        # --- PLL (Phase-Locked Loop) ---
        self._pll_phase: float = 0.0  # 0=beat, 0.5=midpoint, wraps at 1.0
        self._pll_period: float = 0.0  # Seconds between beats (authoritative)
        self._pll_phase_alpha: float = 0.2  # Phase correction strength
        self._pll_period_alpha: float = 0.05  # Period correction strength

        # --- Output smoothing ---
        self._smooth_bpm: float = 0.0  # EMA-smoothed output BPM
        self._display_bpm: float = 0.0  # Hysteresis-filtered for display
        self._bpm_lock_alpha: float = 0.3  # Fast lock-on (>10 BPM jump)
        self._bpm_drift_alpha: float = 0.008  # Very slow drift for stability
        self._bpm_hysteresis: float = 1.5  # Don't update display for changes < this
        self._confidence: float = 0.0
        self._locked: bool = False  # True when confidence > 0.8

        # Confidence gate: hold last stable BPM below this threshold
        self._confidence_gate: float = 0.4
        self._stable_bpm: float = 0.0  # Last high-confidence BPM
        self._low_confidence_frames: int = 0  # How long confidence has been low
        self._stale_timeout_frames: int = int(self._frame_rate * 5)  # ~5 seconds

        # --- Prediction-ratio confidence (Task 2.2) ---
        # Track PLL predictions vs actual beat confirmations over ~10 seconds
        self._prediction_tolerance: float = 0.050  # ±50ms confirmation window
        # Each entry: (predicted_time, was_confirmed)
        prediction_window_size = int(10.0 * self.bpm_max / 60.0)  # ~30 predictions at 180 BPM
        self._prediction_window: deque[tuple[float, bool]] = deque(
            maxlen=max(prediction_window_size, 20)
        )
        self._prediction_confidence: float = 0.0

        # --- Per-band onset detection (Task 1.5) ---
        # Separate onset histories for low/mid/high bands (~1.5s each)
        band_history_len = max(history_len, 30)
        self._low_band_history: deque[float] = deque(maxlen=band_history_len)
        self._mid_band_history: deque[float] = deque(maxlen=band_history_len)
        self._high_band_history: deque[float] = deque(maxlen=band_history_len)

        # Per-band cooldown timers (shorter than main beat cooldown)
        self._last_kick_time: float = 0.0
        self._last_snare_time: float = 0.0
        self._last_hihat_time: float = 0.0

        # Per-band cooldown durations (seconds)
        self._kick_cooldown: float = 0.15  # 150ms — kicks are spaced out
        self._snare_cooldown: float = 0.12  # 120ms
        self._hihat_cooldown: float = 0.06  # 60ms — hi-hats can be rapid

    def detect(self, features: AudioFeatures, timestamp: float | None = None) -> BeatInfo:
        """Process one frame. Call once per hop (~43 Hz at 44100/1024).

        Args:
            features: Current audio analysis results.
            timestamp: Optional override for current time (for testing).
        """
        now = timestamp if timestamp is not None else time.monotonic()
        info = BeatInfo()

        # --- 1. Beat detection (energy-based, adaptive threshold) ---
        bass = features.bass_energy
        self._bass_history.append(bass)

        if len(self._bass_history) < 10:
            self._onset_buffer.append(0.0)
            return info

        history = np.array(self._bass_history)
        avg = float(np.median(history))
        variance = float(np.var(history))

        # Adaptive threshold (Parallelcube spec): variance 0 → 1.55, variance 0.02 → 1.25
        threshold = float(np.clip(
            1.55 - (variance / 0.02) * 0.30,
            1.25, 1.55,
        ))

        # --- Flux-based onset detection ---
        # Log compression: Gamma(X) = log(1 + gamma * |X|)
        raw_flux = features.spectral_flux
        compressed_flux = float(np.log1p(self._flux_gamma * raw_flux))
        self._flux_onset_history.append(compressed_flux)

        # Compute flux adaptive threshold (moving median over ~0.2s window)
        flux_beat = False
        flux_median = 0.0
        if len(self._flux_onset_history) >= 5:
            flux_median = float(np.median(list(self._flux_onset_history)))
            flux_beat = compressed_flux > flux_median * self._flux_threshold_mult

        # Onset function: combine energy and flux for autocorrelation BPM
        energy_onset = max(0.0, bass - avg * threshold)
        flux_onset = max(0.0, compressed_flux - flux_median) if flux_median > 0 else 0.0
        onset_val = energy_onset + 0.5 * flux_onset
        self._onset_buffer.append(onset_val)

        # Beat trigger: either energy or flux onset (cooldown still applies)
        energy_beat = bass > avg * threshold
        past_cooldown = (now - self._last_beat_time) >= self.cooldown_sec
        is_beat = (energy_beat or flux_beat) and past_cooldown

        if is_beat:
            info.is_beat = True
            # Beat strength: max of energy-based and flux-based strength
            energy_strength = (bass - avg * threshold) / (avg * threshold + 1e-6)
            flux_strength = (compressed_flux - flux_median * self._flux_threshold_mult) / (flux_median * self._flux_threshold_mult + 1e-6) if flux_median > 0 else 0.0
            info.beat_strength = float(np.clip(max(energy_strength, flux_strength), 0, 1))
            self._last_beat_time = now

            # --- Confirm pending predictions (Task 2.2) ---
            # Check if this beat matches any unconfirmed prediction within ±50ms
            for idx in range(len(self._prediction_window)):
                pred_time, confirmed = self._prediction_window[idx]
                if not confirmed and abs(now - pred_time) <= self._prediction_tolerance:
                    self._prediction_window[idx] = (pred_time, True)
                    break  # Only confirm the closest prediction

            # --- PLL correction on detected beat ---
            if self._pll_period > 0:
                # Phase error: how far from expected beat (phase=0)
                phase_error = self._pll_phase
                if phase_error > 0.5:
                    phase_error -= 1.0

                # Proportional phase correction (don't hard reset)
                self._pll_phase -= self._pll_phase_alpha * phase_error

                # Period correction: beat arrived early → shorten period
                # phase_error > 0 means beat is late → lengthen period
                timing_error_sec = phase_error * self._pll_period
                self._pll_period += self._pll_period_alpha * timing_error_sec

                # Clamp period to valid BPM range
                period_min = 60.0 / self.bpm_max
                period_max = 60.0 / self.bpm_min
                self._pll_period = max(period_min, min(period_max, self._pll_period))

        # --- 1b. Per-band onset detection (Task 1.5) ---
        info = self._detect_per_band_onsets(features, now, info)

        # --- 2. Advance PLL phase ---
        if self._pll_period > 0:
            old_phase = self._pll_phase
            self._pll_phase += self._frame_dur / self._pll_period
            # Detect phase wrap (predicted beat moment): phase crossing 1.0
            # means one full beat period has elapsed — PLL expects a beat now.
            if self._pll_phase >= 1.0:
                # Compute exact predicted time: interpolate within this frame
                # overshoot = how far past 1.0 we went
                overshoot = self._pll_phase - 1.0
                predicted_time = now - overshoot * self._pll_period
                self._pll_phase %= 1.0
                # Record prediction for confirmation tracking
                self._prediction_window.append((predicted_time, False))

        # --- 3. Autocorrelation BPM estimation (every ~0.5s) ---
        self._frame_count += 1
        buf_len = len(self._onset_buffer)
        if buf_len >= self._lag_max + 10 and self._frame_count % self._acorr_interval == 0:
            self._estimate_bpm_autocorrelation()

            # Seed PLL period from autocorrelation if PLL not yet tracking
            if self._pll_period == 0 and self._raw_bpm > 0:
                self._pll_period = 60.0 / self._raw_bpm

            # Nudge PLL period toward autocorrelation estimate (slow coupling)
            if self._pll_period > 0 and self._raw_bpm > 0:
                raw_period = 60.0 / self._raw_bpm
                period_diff = raw_period - self._pll_period
                self._pll_period += 0.03 * period_diff

        # --- 4. Output: combine PLL + confidence gating + smoothing ---
        if self._pll_period > 0:
            pll_bpm = 60.0 / self._pll_period
        else:
            pll_bpm = self._raw_bpm

        # --- Confidence: blend autocorrelation + prediction ratio (Task 2.2) ---
        # Compute prediction-ratio confidence from sliding window
        self._update_prediction_confidence(now)
        # Blend: 50% autocorrelation strength + 50% prediction accuracy
        self._confidence = 0.5 * self._raw_confidence + 0.5 * self._prediction_confidence
        self._locked = self._confidence > 0.8

        # Confidence gate: below threshold, hold stable BPM (with timeout)
        if self._confidence >= self._confidence_gate and pll_bpm > 0:
            self._low_confidence_frames = 0

            # Asymmetric BPM smoothing: fast lock-on, slow drift
            if self._smooth_bpm == 0:
                self._smooth_bpm = pll_bpm  # First valid estimate
            else:
                diff = abs(pll_bpm - self._smooth_bpm)
                # Large jump (>10 BPM) = new tempo, lock on fast
                alpha = self._bpm_lock_alpha if diff > 10 else self._bpm_drift_alpha
                self._smooth_bpm += alpha * (pll_bpm - self._smooth_bpm)

            self._stable_bpm = self._smooth_bpm
        else:
            self._low_confidence_frames += 1

            if self._low_confidence_frames > self._stale_timeout_frames:
                # Stale: confidence has been low for >5s — release the lock
                # Allow new estimates through even at low confidence
                if pll_bpm > 0 and self._confidence > 0.1:
                    self._smooth_bpm += self._bpm_lock_alpha * (pll_bpm - self._smooth_bpm)
                    self._stable_bpm = self._smooth_bpm
                else:
                    # No signal at all — decay BPM to zero
                    self._smooth_bpm *= 0.99
                    if self._smooth_bpm < 1.0:
                        self._smooth_bpm = 0.0
                        self._stable_bpm = 0.0
                        self._pll_period = 0.0
            elif self._stable_bpm > 0:
                # Within timeout: hold last stable value
                self._smooth_bpm = self._stable_bpm

        # Hysteresis: don't jitter the display BPM for small changes
        if self._display_bpm == 0 and self._smooth_bpm > 0:
            self._display_bpm = self._smooth_bpm
        elif abs(self._smooth_bpm - self._display_bpm) >= self._bpm_hysteresis:
            self._display_bpm = self._smooth_bpm
        elif self._smooth_bpm == 0:
            self._display_bpm = 0

        # Auto-cooldown: set cooldown to ~75% of beat period when BPM is known
        # (research minimum: 300ms refractory period)
        if self.auto_cooldown and self._display_bpm > 0:
            beat_period = 60.0 / self._display_bpm
            self.cooldown_sec = max(0.30, beat_period * 0.75)

        # Fill output
        info.bpm = round(self._display_bpm, 1)
        info.bpm_confidence = self._confidence
        info.time_since_beat = now - self._last_beat_time

        # Predict next beat from PLL
        if self._pll_period > 0 and self._last_beat_time > 0:
            info.predicted_next_beat = self._last_beat_time + self._pll_period

        return info

    def _detect_per_band_onsets(
        self, features: AudioFeatures, now: float, info: BeatInfo
    ) -> BeatInfo:
        """Detect per-band onsets for kick/snare/hi-hat separation.

        Band mapping from AudioFeatures.band_energies (7 bands):
        - Low (kicks):   sub_bass[0] + bass[1]          -> 20-250 Hz
        - Mid (snares):  low_mid[2] + mid[3] + upper_mid[4] -> 250-4000 Hz
        - High (hi-hats): presence[5] + brilliance[6]    -> 4-20 kHz
        """
        bands = features.band_energies

        # Compute aggregate band energies
        low_energy = float(bands[0] + bands[1]) / 2.0
        mid_energy = float(bands[2] + bands[3] + bands[4]) / 3.0
        high_energy = float(bands[5] + bands[6]) / 2.0

        # Store raw energies in BeatInfo for effect intensity scaling
        info.kick_energy = low_energy
        info.snare_energy = mid_energy
        info.hihat_energy = high_energy

        # Append to per-band histories
        self._low_band_history.append(low_energy)
        self._mid_band_history.append(mid_energy)
        self._high_band_history.append(high_energy)

        # Need enough history for adaptive threshold
        min_history = 10

        # --- Low band (kicks) ---
        if len(self._low_band_history) >= min_history:
            low_arr = np.array(self._low_band_history)
            low_median = float(np.median(low_arr))
            low_var = float(np.var(low_arr))
            # Adaptive threshold: lower threshold when variance is high
            # (like the main detector's Parallelcube spec)
            low_thresh = float(np.clip(
                1.5 - (low_var / 0.02) * 0.25, 1.25, 1.5
            ))
            past_cooldown = (now - self._last_kick_time) >= self._kick_cooldown
            if low_energy > low_median * low_thresh and past_cooldown:
                info.kick_onset = True
                self._last_kick_time = now

        # --- Mid band (snares) ---
        if len(self._mid_band_history) >= min_history:
            mid_arr = np.array(self._mid_band_history)
            mid_median = float(np.median(mid_arr))
            mid_var = float(np.var(mid_arr))
            mid_thresh = float(np.clip(
                1.5 - (mid_var / 0.02) * 0.25, 1.25, 1.5
            ))
            past_cooldown = (now - self._last_snare_time) >= self._snare_cooldown
            if mid_energy > mid_median * mid_thresh and past_cooldown:
                info.snare_onset = True
                self._last_snare_time = now

        # --- High band (hi-hats) ---
        if len(self._high_band_history) >= min_history:
            high_arr = np.array(self._high_band_history)
            high_median = float(np.median(high_arr))
            high_var = float(np.var(high_arr))
            high_thresh = float(np.clip(
                1.5 - (high_var / 0.02) * 0.25, 1.25, 1.5
            ))
            past_cooldown = (now - self._last_hihat_time) >= self._hihat_cooldown
            if high_energy > high_median * high_thresh and past_cooldown:
                info.hihat_onset = True
                self._last_hihat_time = now

        return info

    def _update_prediction_confidence(self, now: float) -> None:
        """Compute prediction confidence: ratio of confirmed to total predictions.

        Expired predictions (more than tolerance past their predicted time and
        not confirmed) are considered missed. Only counts predictions whose
        confirmation window has fully elapsed.

        Also retroactively confirms any prediction close to the last beat time.
        This handles the case where a prediction is recorded in the same frame
        as a beat (phase wrap happens after beat detection in the processing order).
        """
        if len(self._prediction_window) == 0:
            self._prediction_confidence = 0.0
            return

        # Retroactive confirmation: check if any unconfirmed prediction is close
        # to the most recent beat time (handles same-frame ordering issue)
        if self._last_beat_time > 0:
            for idx in range(len(self._prediction_window)):
                pred_time, was_confirmed = self._prediction_window[idx]
                if not was_confirmed and abs(self._last_beat_time - pred_time) <= self._prediction_tolerance:
                    self._prediction_window[idx] = (pred_time, True)

        confirmed = 0
        total = 0
        for pred_time, was_confirmed in self._prediction_window:
            # Only count predictions whose window has passed
            # (give tolerance after predicted time for late confirmations)
            if now > pred_time + self._prediction_tolerance:
                total += 1
                if was_confirmed:
                    confirmed += 1

        if total >= 3:
            self._prediction_confidence = confirmed / total
        else:
            # Not enough data yet -- fall back to autocorrelation only
            self._prediction_confidence = self._raw_confidence

    def _estimate_bpm_autocorrelation(self) -> None:
        """Estimate BPM via autocorrelation of the onset function buffer."""
        onset = np.array(self._onset_buffer)

        # Normalize: subtract mean, avoid correlating DC offset
        onset = onset - np.mean(onset)
        norm = np.sum(onset ** 2)
        if norm < 1e-10:
            self._raw_confidence = 0.0
            return

        # Autocorrelation for lags in BPM range
        lag_min = self._lag_min
        lag_max = min(self._lag_max, len(onset) - 1)

        if lag_min >= lag_max:
            return

        correlations = np.zeros(lag_max - lag_min + 1)
        for i, lag in enumerate(range(lag_min, lag_max + 1)):
            correlations[i] = np.sum(onset[lag:] * onset[:len(onset) - lag])

        # Normalize by zero-lag (energy)
        correlations /= (norm + 1e-10)

        # Find peak
        peak_idx = np.argmax(correlations)
        peak_lag = lag_min + peak_idx
        peak_val = correlations[peak_idx]

        if peak_val < 0.05:
            self._raw_confidence = 0.0
            return

        # Convert lag to BPM
        beat_period_sec = peak_lag * self._frame_dur
        candidate_bpm = 60.0 / beat_period_sec

        # --- Octave error protection ---
        candidate_bpm = self._fix_octave_errors(candidate_bpm, correlations, lag_min)

        self._raw_bpm = candidate_bpm
        self._raw_confidence = float(np.clip(peak_val * 2.5, 0, 1))

    def _fix_octave_errors(
        self, bpm: float, correlations: np.ndarray, lag_min: int
    ) -> float:
        """Fix octave errors: if BPM is outside range, try half or double."""
        if self.bpm_min <= bpm <= self.bpm_max:
            return bpm

        # Try double (if we detected half-time)
        if bpm * 2 <= self.bpm_max and bpm < self.bpm_min:
            return bpm * 2

        # Try half (if we detected double-time)
        if bpm / 2 >= self.bpm_min and bpm > self.bpm_max:
            return bpm / 2

        # Still outside? Pick the candidate from correlations that's in range
        for lag in range(len(correlations)):
            real_lag = lag_min + lag
            period = real_lag * self._frame_dur
            if period > 0:
                candidate = 60.0 / period
                if self.bpm_min <= candidate <= self.bpm_max:
                    if correlations[lag] > 0.05:
                        return candidate

        # Fallback: clamp
        return max(self.bpm_min, min(self.bpm_max, bpm))

    def reset(self) -> None:
        """Reset all state."""
        self._bass_history.clear()
        self._flux_onset_history.clear()
        self._onset_buffer.clear()
        self._frame_count = 0
        self._last_beat_time = 0.0
        self._raw_bpm = 0.0
        self._raw_confidence = 0.0
        self._pll_phase = 0.0
        self._pll_period = 0.0
        self._smooth_bpm = 0.0
        self._display_bpm = 0.0
        self._stable_bpm = 0.0
        self._confidence = 0.0
        self._locked = False
        self._low_confidence_frames = 0

        # Per-band onset state (Task 1.5)
        self._low_band_history.clear()
        self._mid_band_history.clear()
        self._high_band_history.clear()
        self._last_kick_time = 0.0
        self._last_snare_time = 0.0
        self._last_hihat_time = 0.0

        # Prediction-ratio confidence state (Task 2.2)
        self._prediction_window.clear()
        self._prediction_confidence = 0.0

    # --- Public setters for genre preset configuration ---

    def set_cooldown(self, ms: float) -> None:
        """Set the manual cooldown in milliseconds and re-enable auto-cooldown."""
        self._manual_cooldown_sec = ms / 1000.0
        self.cooldown_sec = self._manual_cooldown_sec
        self.auto_cooldown = True

    def set_bpm_range(self, bpm_min: float, bpm_max: float) -> None:
        """Set the BPM range for octave error protection and recompute lag bounds."""
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self._lag_min = max(1, int(self._frame_rate * 60.0 / bpm_max))
        self._lag_max = int(self._frame_rate * 60.0 / bpm_min)

    @property
    def current_bpm(self) -> float:
        return self._smooth_bpm

    @property
    def pll_phase(self) -> float:
        """Current position in beat cycle (0=beat, 0.5=midpoint)."""
        return self._pll_phase

    @property
    def is_locked(self) -> bool:
        """True when BPM tracking is high-confidence (>80%)."""
        return self._locked
