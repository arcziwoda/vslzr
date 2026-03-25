"""Beat detection — SuperFlux onset, generalized autocorrelation, multi-agent PI-PLL.

Architecture (based on beat_detection_research_2026_03.md):
- Onset: SuperFlux (Böck & Widmer 2013) — mel-domain log-compressed spectral flux
  with max-filtering across frequency bins. Reduces false positives by ~60%.
- BPM estimation: Generalized FFT autocorrelation (Percival & Tzanetakis 2014)
  with p=0.5 compression + Ellis 2007 perceptual weighting (log-Gaussian at 120 BPM).
- Beat tracking: Multi-agent PI-PLL (inspired by IBT, Oliveira et al. 2010).
  3-15 competing agents seeded from autocorrelation peaks, each with PI correction.
  Best agent drives output. Recovery from phase slips in 1-2 beats.
- Coasting: Tiered confidence decay without onset confirmation (4s full, 4-16s decay,
  16s+ maintain at 50%). Sidechain compression detection in mid-frequency bands.
- Safety: Octave error protection, adaptive cooldown, confidence gating.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .analyzer import AudioFeatures


@dataclass
class BeatAgent:
    """A competing beat tracking hypothesis with PI-PLL."""

    period: float  # Seconds between beats
    phase: float = 0.0  # 0=beat, 0.5=midpoint, wraps at 1.0
    integral_error: float = 0.0
    score: float = 1.0  # Accumulated confidence score
    consecutive_misses: int = 0


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

        # FFT-based autocorrelation padding (next power of 2 × 2 for linear correlation)
        self._fft_pad_len = int(2 ** np.ceil(np.log2(max(self._onset_buf_len, 64))))
        # Perceptual weighting: log-Gaussian prior centered at 120 BPM (Ellis 2007)
        self._perceptual_weights = self._compute_perceptual_weights()

        # Frame counter for periodic autocorrelation (~every 0.5s)
        self._frame_count: int = 0
        self._acorr_interval: int = max(1, int(self._frame_rate * 0.5))  # ~22 frames

        # Raw autocorrelation BPM (before PLL)
        self._raw_bpm: float = 0.0
        self._raw_confidence: float = 0.0

        # --- Multi-Agent PLL (Phase-Locked Loop) ---
        # Each agent is a competing beat hypothesis with its own PI-PLL
        self._agents: list[BeatAgent] = []
        self._max_agents: int = 15
        self._initial_agents: int = 5
        self._agent_kill_ratio: float = 0.8  # Kill below 80% of best score
        self._agent_max_misses: int = 8  # Kill after 8 consecutive misses
        self._agent_confirmation_window: float = 0.050  # ±50ms

        # PI-PLL gains (applied to all agents)
        self._pll_kp: float = 0.25  # Phase correction gain
        self._pll_period_alpha: float = 0.05  # Period correction strength
        self._pll_ki: float = 0.02  # Integral gain

        # Synced from best agent (for downstream consumers)
        self._pll_phase: float = 0.0
        self._pll_period: float = 0.0

        # --- Output smoothing ---
        self._smooth_bpm: float = 0.0  # EMA-smoothed output BPM
        self._display_bpm: float = 0.0  # Hysteresis-filtered for display
        self._bpm_drift_alpha: float = 0.08  # Light EMA (agents provide stability)
        self._bpm_hysteresis: float = 1.0  # Don't update display for changes < this
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

        # --- Coasting behavior (breakdowns, non-percussive sections) ---
        self._last_strong_onset_time: float = 0.0
        self._coasting: bool = False
        self._coast_confidence_mult: float = 1.0
        # Mid-frequency flux history for sidechain compression detection (~4s)
        self._mid_flux_history: deque[float] = deque(maxlen=int(self._frame_rate * 4))

        # --- Per-band onset detection ---
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

        # --- Flux-based onset detection (SuperFlux) ---
        # SuperFlux is already log-compressed + max-filtered in AudioAnalyzer
        compressed_flux = features.superflux_onset
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

            # Track strong onsets for coasting behavior
            if info.beat_strength > 0.3:
                self._last_strong_onset_time = now

            # --- Confirm pending predictions ---
            for idx in range(len(self._prediction_window)):
                pred_time, confirmed = self._prediction_window[idx]
                if not confirmed and abs(now - pred_time) <= self._prediction_tolerance:
                    self._prediction_window[idx] = (pred_time, True)
                    break

            # --- Multi-agent PLL correction on detected beat ---
            # Information gate: only correct agents on strong onsets
            # (weak onsets/noise don't disturb agent phase during coasting)
            if info.beat_strength > 0.2:
                self._correct_agents_on_beat(now)

        # --- 1b. Per-band onset detection ---
        info = self._detect_per_band_onsets(features, now, info)

        # --- 2. Advance all agent phases ---
        self._advance_agents(now)

        # --- 3. Autocorrelation BPM estimation (every ~0.5s) ---
        self._frame_count += 1
        buf_len = len(self._onset_buffer)
        if buf_len >= self._lag_max + 10 and self._frame_count % self._acorr_interval == 0:
            self._estimate_bpm_autocorrelation()
            self._seed_agents_from_autocorrelation()

        # --- 4. Sync best agent to output vars ---
        self._sync_best_agent()

        # --- 5. Output: combine PLL + confidence gating + smoothing ---
        if self._pll_period > 0:
            pll_bpm = 60.0 / self._pll_period
        else:
            pll_bpm = self._raw_bpm

        # --- Sidechain compression detection (mid-frequency periodic modulation) ---
        mid_flux = float(np.sum(features.band_energies[2:5]))
        self._mid_flux_history.append(mid_flux)
        sidechain_detected = self._detect_sidechain()

        # --- Confidence: blend autocorrelation + prediction ratio ---
        self._update_prediction_confidence(now)
        raw_confidence = 0.5 * self._raw_confidence + 0.5 * self._prediction_confidence

        # --- Coasting: tiered confidence decay without strong onsets ---
        self._update_coasting(now, sidechain_detected)
        self._confidence = raw_confidence * self._coast_confidence_mult
        self._locked = self._confidence > 0.8

        # BPM smoothing: light EMA on PLL output (multi-agent already provides stability)
        if pll_bpm > 0 and self._confidence > 0.1:
            self._low_confidence_frames = 0

            if self._smooth_bpm == 0:
                self._smooth_bpm = pll_bpm
            else:
                # Simple EMA: fast enough to track, slow enough to avoid frame jitter
                self._smooth_bpm += self._bpm_drift_alpha * (pll_bpm - self._smooth_bpm)

            self._stable_bpm = self._smooth_bpm
        else:
            self._low_confidence_frames += 1

            if self._low_confidence_frames > self._stale_timeout_frames:
                self._smooth_bpm *= 0.99
                if self._smooth_bpm < 1.0:
                    self._smooth_bpm = 0.0
                    self._stable_bpm = 0.0

        # Hysteresis: don't jitter the display BPM for small changes
        if self._display_bpm == 0 and self._smooth_bpm > 0:
            self._display_bpm = self._smooth_bpm
        elif abs(self._smooth_bpm - self._display_bpm) >= self._bpm_hysteresis:
            self._display_bpm = self._smooth_bpm
        elif self._smooth_bpm == 0:
            self._display_bpm = 0

        # Auto-cooldown: set cooldown to ~75% of beat period when BPM is known
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

    # --- Multi-agent PLL methods ---

    def _correct_agents_on_beat(self, now: float) -> None:
        """Correct all agents when a beat is detected. Agents whose predicted
        phase aligns with the beat get score boost + PI correction."""
        period_min = 60.0 / self.bpm_max
        period_max = 60.0 / self.bpm_min

        for agent in self._agents:
            if agent.period <= 0:
                continue

            # Phase error: how far from expected beat (phase=0)
            phase_error = agent.phase
            if phase_error > 0.5:
                phase_error -= 1.0

            time_error = abs(phase_error * agent.period)

            if time_error <= self._agent_confirmation_window:
                # Beat confirms this agent's prediction
                agent.score += 1.0
                agent.consecutive_misses = 0

                # Proportional phase correction
                agent.phase -= self._pll_kp * phase_error

                # Period correction
                timing_error_sec = phase_error * agent.period
                agent.period += self._pll_period_alpha * timing_error_sec

                # Integral correction (PI-PLL)
                agent.integral_error += phase_error
                agent.integral_error = max(-2.0, min(2.0, agent.integral_error))
                agent.period += self._pll_ki * agent.integral_error * agent.period

                # Clamp period
                agent.period = max(period_min, min(period_max, agent.period))

    def _advance_agents(self, now: float) -> None:
        """Advance phase of all agents by one frame. Record predictions from
        the best agent on phase wrap."""
        best_agent = max(self._agents, key=lambda a: a.score) if self._agents else None

        for agent in self._agents:
            if agent.period <= 0:
                continue

            agent.phase += self._frame_dur / agent.period
            # Slow integral leak
            agent.integral_error *= 0.999

            if agent.phase >= 1.0:
                # Phase wrap — agent expected a beat here
                overshoot = agent.phase - 1.0
                agent.phase %= 1.0

                # Only record prediction from best agent
                if agent is best_agent:
                    predicted_time = now - overshoot * agent.period
                    self._prediction_window.append((predicted_time, False))

                # If this agent expected a beat but none was detected recently,
                # increment misses (checked via time since last beat)
                time_since_beat = now - self._last_beat_time if self._last_beat_time > 0 else 999
                if time_since_beat > self._agent_confirmation_window:
                    agent.consecutive_misses += 1

    def _seed_agents_from_autocorrelation(self) -> None:
        """Seed new agents from top autocorrelation peaks. Prune weak agents."""
        if self._raw_bpm <= 0:
            return

        # Find top N peaks from the last autocorrelation run
        # Use the stored onset buffer to find multiple peaks
        onset = np.array(self._onset_buffer)
        onset = onset - np.mean(onset)
        energy = np.sum(onset ** 2)
        if energy < 1e-10:
            return

        lag_min = self._lag_min
        lag_max = min(self._lag_max, len(onset) - 1)
        if lag_min >= lag_max:
            return

        # FFT-based generalized autocorrelation (reuse the computation)
        pad_len = self._fft_pad_len * 2
        padded = np.zeros(pad_len)
        padded[:len(onset)] = onset
        spectrum = np.fft.rfft(padded)
        gac = np.fft.irfft(np.abs(spectrum) ** 0.5)
        zero_lag = gac[0]
        if zero_lag < 1e-10:
            return

        correlations = gac[lag_min:lag_max + 1] / zero_lag
        n_corr = len(correlations)
        n_weights = len(self._perceptual_weights)
        if n_corr <= n_weights:
            weighted = correlations * self._perceptual_weights[:n_corr]
        else:
            weighted = correlations.copy()
            weighted[:n_weights] *= self._perceptual_weights

        # Find top N peaks (local maxima)
        candidate_periods = []
        for i in range(1, len(weighted) - 1):
            if weighted[i] > weighted[i - 1] and weighted[i] > weighted[i + 1]:
                lag = lag_min + i
                period = lag * self._frame_dur
                candidate_periods.append((weighted[i], period))

        # Sort by score descending, take top N
        candidate_periods.sort(reverse=True)
        candidates = candidate_periods[:self._initial_agents]

        # Also include the primary BPM estimate
        primary_period = 60.0 / self._raw_bpm
        if not any(abs(p - primary_period) / primary_period < 0.05 for _, p in candidates):
            candidates.insert(0, (1.0, primary_period))

        # Seed new agents for candidates not already tracked
        for _, period in candidates:
            already_tracked = any(
                abs(a.period - period) / a.period < 0.05
                for a in self._agents if a.period > 0
            )
            if not already_tracked and len(self._agents) < self._max_agents:
                self._agents.append(BeatAgent(period=period))

        # Prune weak agents
        if self._agents:
            best_score = max(a.score for a in self._agents)
            self._agents = [
                a for a in self._agents
                if a.score >= best_score * self._agent_kill_ratio
                and a.consecutive_misses < self._agent_max_misses
            ]

        # Score decay: prevent old agents from accumulating unbounded scores
        for agent in self._agents:
            agent.score *= 0.95

    def _detect_sidechain(self) -> bool:
        """Detect sidechain compression pumping in mid-frequency bands.

        Sidechain compression creates periodic amplitude modulation on synth pads
        even when the kick drops out. Check for autocorrelation at the expected
        beat period in the mid-frequency flux.
        """
        if len(self._mid_flux_history) < self._frame_rate * 2:
            return False
        if self._pll_period <= 0:
            return False

        mf = np.array(self._mid_flux_history)
        mf = mf - np.mean(mf)
        mf_energy = np.sum(mf ** 2)
        if mf_energy < 1e-10:
            return False

        expected_lag = int(self._pll_period / self._frame_dur)
        if expected_lag <= 0 or expected_lag >= len(mf) - 1:
            return False

        # Autocorrelation at expected beat lag
        corr_at_lag = float(np.sum(mf[expected_lag:] * mf[:len(mf) - expected_lag]))
        sidechain_strength = corr_at_lag / mf_energy
        return bool(sidechain_strength > 0.3)

    def _update_coasting(self, now: float, sidechain_detected: bool) -> None:
        """Update coasting state: tiered confidence decay without strong onsets.

        Tiers:
        - 0-4s: full confidence, PLL freewheels normally
        - 4-16s: linear decay from 100% to 50%
        - 16s+: maintain internal tempo at 50% confidence (coasting)
        """
        if self._last_strong_onset_time <= 0:
            # No strong onset yet — don't coast, let warmup happen
            self._coast_confidence_mult = 1.0
            self._coasting = False
            return

        time_since_strong = now - self._last_strong_onset_time

        if time_since_strong <= 4.0 or sidechain_detected:
            self._coast_confidence_mult = 1.0
            self._coasting = False
        elif time_since_strong <= 16.0:
            # Linear decay from 100% to 50% over 12 seconds
            decay = 1.0 - 0.5 * (time_since_strong - 4.0) / 12.0
            self._coast_confidence_mult = max(0.5, decay)
            self._coasting = True
        else:
            self._coast_confidence_mult = 0.5
            self._coasting = True

    def _sync_best_agent(self) -> None:
        """Sync _pll_period and _pll_phase from the highest-scoring agent."""
        if self._agents:
            best = max(self._agents, key=lambda a: a.score)
            self._pll_period = best.period
            self._pll_phase = best.phase
        # If no agents, _pll_period stays at whatever it was (or 0)

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

    def _compute_perceptual_weights(self) -> np.ndarray:
        """Compute log-Gaussian perceptual weighting for autocorrelation lags.

        Centers prior at 120 BPM with σ=1.4 octaves (Ellis, JNMR 2007).
        Higher weights near 120 BPM, smoothly decaying for extreme tempi.
        """
        n_lags = self._lag_max - self._lag_min + 1
        if n_lags <= 0:
            return np.ones(1)

        weights = np.zeros(n_lags)
        center_bpm = 128.0  # Electronic music center (house/techno sweet spot)
        sigma_octaves = 1.6  # Wider spread to not penalize DnB/ambient extremes

        for i in range(n_lags):
            lag = self._lag_min + i
            period_sec = lag * self._frame_dur
            if period_sec > 0:
                bpm = 60.0 / period_sec
                if bpm > 0:
                    log2_ratio = np.log2(bpm / center_bpm)
                    weights[i] = np.exp(-0.5 * (log2_ratio / sigma_octaves) ** 2)

        # Normalize so max weight = 1
        w_max = np.max(weights)
        if w_max > 0:
            weights /= w_max

        return weights

    def _estimate_bpm_autocorrelation(self) -> None:
        """Estimate BPM via generalized FFT autocorrelation (Percival & Tzanetakis 2014).

        Uses GAC(τ) = IFFT(|FFT(x)|^p) with p=0.5 compression to sharpen peaks
        and reduce octave errors. Perceptual log-Gaussian weighting resolves
        ambiguity toward musically plausible tempi.
        """
        onset = np.array(self._onset_buffer)

        # Subtract mean to remove DC offset
        onset = onset - np.mean(onset)
        energy = np.sum(onset ** 2)
        if energy < 1e-10:
            self._raw_confidence = 0.0
            return

        lag_min = self._lag_min
        lag_max = min(self._lag_max, len(onset) - 1)
        if lag_min >= lag_max:
            return

        # Zero-pad for linear (non-circular) autocorrelation
        pad_len = self._fft_pad_len * 2
        padded = np.zeros(pad_len)
        padded[:len(onset)] = onset

        # Generalized autocorrelation: IFFT(|FFT(x)|^p) with p=0.5
        spectrum = np.fft.rfft(padded)
        gac = np.fft.irfft(np.abs(spectrum) ** 0.5)

        # Extract lag range and normalize by zero-lag
        zero_lag = gac[0]
        if zero_lag < 1e-10:
            self._raw_confidence = 0.0
            return

        correlations = gac[lag_min:lag_max + 1] / zero_lag

        # Apply perceptual weighting
        n_corr = len(correlations)
        n_weights = len(self._perceptual_weights)
        if n_corr <= n_weights:
            weighted = correlations * self._perceptual_weights[:n_corr]
        else:
            weighted = correlations.copy()
            weighted[:n_weights] *= self._perceptual_weights

        # Find peak
        peak_idx = np.argmax(weighted)
        peak_lag = lag_min + peak_idx
        peak_val = weighted[peak_idx]

        if peak_val < 0.01:
            self._raw_confidence = 0.0
            return

        # Convert lag to BPM
        beat_period_sec = peak_lag * self._frame_dur
        candidate_bpm = 60.0 / beat_period_sec

        # Octave error protection
        candidate_bpm = self._fix_octave_errors(candidate_bpm, weighted, lag_min)

        self._raw_bpm = candidate_bpm

        # Confidence: peak SNR (peak height relative to sidelobes)
        mean_corr = float(np.mean(weighted))
        std_corr = float(np.std(weighted))
        snr = (peak_val - mean_corr) / (std_corr + 1e-10)
        self._raw_confidence = float(np.clip(snr / 5.0, 0, 1))

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
        self._agents.clear()
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

        # Prediction-ratio confidence state
        self._prediction_window.clear()
        self._prediction_confidence = 0.0

        # Coasting state
        self._last_strong_onset_time = 0.0
        self._coasting = False
        self._coast_confidence_mult = 1.0
        self._mid_flux_history.clear()

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
        self._perceptual_weights = self._compute_perceptual_weights()
        # Kill agents outside new BPM range
        period_min = 60.0 / bpm_max
        period_max = 60.0 / bpm_min
        self._agents = [
            a for a in self._agents
            if period_min <= a.period <= period_max
        ]

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

    @property
    def is_coasting(self) -> bool:
        """True when beat tracker is freewheeling without onset confirmation."""
        return self._coasting
