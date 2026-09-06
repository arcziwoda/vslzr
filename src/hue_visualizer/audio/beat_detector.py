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
from dataclasses import dataclass, field

import numpy as np

from .analyzer import AudioFeatures


@dataclass
class BeatAgent:
    """A competing beat tracking hypothesis with PI-PLL.

    score is a leaky weighted hit count: decays by AGENT_SCORE_DECAY on every
    informative prediction (a phase wrap with a strong onset somewhere in the
    period) and gains the onset weight on every confirmed onset. A perfect agent
    settles at 1 / (1 - decay) = 10 times the mean onset weight. New agents
    inherit 0.9x the best score (IBT) and are protected from pruning for
    AGENT_GRACE_BEATS of their own period.
    """

    period: float  # Seconds between beats
    phase: float = 0.0  # 0=beat, 0.5=midpoint, wraps at 1.0
    integral_error: float = 0.0
    score: float = 1.0  # Leaky hit count, see class docstring
    consecutive_misses: int = 0
    born: float = 0.0  # Timestamp the agent was seeded
    # Confirmed onsets as (beat index, time) for the least-squares period fit
    confirmations: deque = field(default_factory=lambda: deque(maxlen=16))
    beat_index: int = 0
    last_confirmation: float = 0.0


AGENT_SCORE_DECAY = 0.9
AGENT_LS_MIN_POINTS = 6  # Confirmations needed before the LS period fit is used
AGENT_LS_GAIN = 0.3  # Pull of the agent period toward the LS estimate per confirmation
AGENT_GRACE_BEATS = 4.0
AGENT_SWITCH_MARGIN = 1.25  # Challenger must beat the current best by 25%


@dataclass
class BeatInfo:
    """Beat detection results for a single frame."""

    is_beat: bool = False  # Raw onset (energy or SuperFlux flux) past the raw cooldown
    # Onset where the trusted PLL expects a beat (own half-period cooldown, independent
    # of is_beat). Equals is_beat while no agent is trusted yet.
    is_metric_beat: bool = False
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
    """Real-time beat detection with autocorrelation BPM and multi-agent PLL tracking.

    Onsets: bass energy adaptive threshold OR SuperFlux median threshold.
    BPM estimation: generalized FFT autocorrelation of the onset function (~4 s).
    Tracking: competing PI-PLL agents seeded from autocorrelation peaks, phase-aligned
    to the last onset; the best agent (sticky, tempo-agreement tie-break) drives
    is_metric_beat and predicted_next_beat.
    Output streams: is_beat (raw onset + cooldown) and is_metric_beat (PLL-validated).
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

        # --- Onset detection function (ODF) ---
        # Two half-wave rectified increase detectors, each normalized by its own
        # running maximum, averaged into a 0..1 ODF:
        #   bass_diff = max(0, bass - max(bass[n-1], bass[n-2]))  (kick attacks;
        #              a level detector would re-fire on the decaying tail)
        #   superflux  = mel-domain log-compressed flux from AudioAnalyzer
        # The running maxima decay over ~12 s but never below 10% of a very slow
        # peak hold, so a long breakdown cannot inflate pad noise into onsets.
        history_len = int(self._frame_rate * 1.5)
        self._bass_prev: deque[float] = deque(maxlen=2)
        self._bass_diff_ref: float = 1e-6
        self._bass_diff_peak: float = 1e-6
        self._flux_ref: float = 1e-6
        self._flux_peak: float = 1e-6
        self._ref_decay: float = 0.998  # ~12 s time constant at 43 fps
        self._peak_decay: float = 0.99995  # ~8 min
        self._ref_floor_ratio: float = 0.1
        # ODF history (~1 s) for the adaptive threshold
        self._odf_history: deque[float] = deque(maxlen=max(int(self._frame_rate * 1.0), 10))
        self._odf_threshold_floor: float = 0.12
        self._odf_threshold_sigmas: float = 1.5

        # Beat timing
        self._last_beat_time: float = 0.0  # Last raw beat (is_beat)
        self._last_onset_time: float = 0.0  # Last onset of any kind (feeds agents)
        self._last_metric_beat_time: float = 0.0  # Last PLL-validated beat
        self._onset_min_gap: float = 0.060  # Same transient spans 2-3 frames

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

        # Raw autocorrelation BPM (before PLL) and the last weighted curve
        self._raw_bpm: float = 0.0
        self._raw_confidence: float = 0.0
        self._last_weighted_corr: np.ndarray | None = None
        self._last_corr_lag_min: int = self._lag_min
        # Tempo prior for agent ranking: median of recent estimates, so a single
        # glitched autocorrelation cycle cannot flip the best agent
        self._raw_bpm_recent: deque[float] = deque(maxlen=3)
        self._prior_bpm: float = 0.0

        # --- Multi-Agent PLL (Phase-Locked Loop) ---
        # Each agent is a competing beat hypothesis with its own PI-PLL
        self._agents: list[BeatAgent] = []
        self._best_agent: BeatAgent | None = None  # Sticky choice (hysteresis)
        self._max_agents: int = 15
        self._initial_agents: int = 5
        self._agent_kill_ratio: float = 0.8  # Kill below 80% of best score (after grace)
        self._agent_max_misses: int = 8  # Kill after 8 consecutive misses
        self._agent_confirmation_window: float = 0.050  # ±50ms: confirms + scores
        # Strong onsets inside this fraction of the period (but outside the
        # confirmation window) pull the phase without scoring, so an agent whose
        # period is slightly off re-captures the beat instead of losing lock.
        self._agent_capture_fraction: float = 0.25
        # "Strong" = kick-like relative to the loudest transient of the last
        # minutes (immune to the 12 s normalization creep during breakdowns).
        self._strong_onset_level: float = 0.4
        # Score at which an agent counts as fully reliable (a leaky score of 4
        # is ~40% weighted hit rate, which sparse patterns such as half-time
        # trap or two-step DnB reach; four-on-the-floor sits near 8-10).
        self._score_ref: float = 4.0
        self._trust_threshold: float = 0.5  # quality*agreement needed to gate onsets
        self._quality_held: float = 0.0  # last tracking quality measured with evidence
        self._quality_current: float = 0.0  # quality used this frame (live or held)
        # Without a strong onset in this window there is no evidence to seed,
        # re-rank or re-weight agents on (breakdowns, pauses)
        self._evidence_window: float = 3.0

        # PI-PLL gains (applied to all agents)
        self._pll_kp: float = 0.25  # Phase correction gain
        self._pll_period_alpha: float = 0.02  # Period correction strength
        self._pll_ki: float = 0.005  # Integral gain
        # Agents near the autocorrelation tempo are pulled gently toward it every
        # housekeeping cycle (coarse anchor); the least-squares fit over confirmed
        # onsets provides the fine period estimate
        self._period_anchor_gain: float = 0.02

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

        # --- 1. Onset detection function ---
        bass = features.bass_energy
        prev_bass_max = max(self._bass_prev) if self._bass_prev else bass
        self._bass_prev.append(bass)
        bass_diff = max(0.0, bass - prev_bass_max)
        flux = max(0.0, features.superflux_onset)

        # Normalization references: fast-decaying running max with a slow peak floor
        self._bass_diff_peak = max(bass_diff, self._bass_diff_peak * self._peak_decay)
        self._flux_peak = max(flux, self._flux_peak * self._peak_decay)
        self._bass_diff_ref = max(
            bass_diff,
            self._bass_diff_ref * self._ref_decay,
            self._ref_floor_ratio * self._bass_diff_peak,
        )
        self._flux_ref = max(
            flux,
            self._flux_ref * self._ref_decay,
            self._ref_floor_ratio * self._flux_peak,
        )
        bass_norm = bass_diff / (self._bass_diff_ref + 1e-9)
        flux_norm = flux / (self._flux_ref + 1e-9)
        onset_val = float(np.clip(0.5 * (bass_norm + flux_norm), 0.0, 1.0))
        # Absolute-ish strength against the slow peak: drives agent scoring,
        # phase capture and coasting, where breakdown pad pumping must not count
        strong_val = float(np.clip(
            0.5 * (bass_diff / (self._bass_diff_peak + 1e-9) + flux / (self._flux_peak + 1e-9)),
            0.0, 1.0,
        ))

        self._onset_buffer.append(onset_val)
        self._odf_history.append(onset_val)

        if len(self._odf_history) < 10:
            return info

        # Adaptive threshold: mean + k*std over ~1 s, with an absolute floor and
        # a relative margin so a flat ODF creeping up as the normalization
        # reference decays cannot register as a stream of onsets
        odf_arr = np.fromiter(self._odf_history, dtype=float, count=len(self._odf_history))
        odf_mean = float(np.mean(odf_arr))
        threshold = max(
            self._odf_threshold_floor,
            odf_mean + self._odf_threshold_sigmas * float(np.std(odf_arr)),
            1.2 * odf_mean,
        )

        # An onset is a rising edge of the ODF above threshold; the same transient
        # spans 2-3 frames (flux leads bass by one hop), so require a minimum gap.
        # Cooldowns are applied later, separately for the raw and metric streams.
        is_onset = (
            onset_val > threshold
            and (now - self._last_onset_time) >= self._onset_min_gap
        )

        if is_onset:
            self._last_onset_time = now
            strength = onset_val

            # Track strong (kick-like) onsets for coasting and miss counting
            if strong_val >= self._strong_onset_level:
                self._last_strong_onset_time = now

            # Confirm pending predictions (any onset within tolerance counts)
            for idx in range(len(self._prediction_window)):
                pred_time, confirmed = self._prediction_window[idx]
                if not confirmed and abs(now - pred_time) <= self._prediction_tolerance:
                    self._prediction_window[idx] = (pred_time, True)
                    break

            # Alignment with the best agent, evaluated BEFORE correction so it
            # reflects where the PLL expected the beat to land.
            aligned = self._onset_aligned_with_best()

            # Agents are confirmed with the onset strength as weight, so dense
            # weak onsets (hats, 16th bass) cannot outscore kicks. Without recent
            # evidence (breakdown, pause) weak onsets do not touch the agents at
            # all: the first strong onset re-engages them.
            if strong_val >= self._strong_onset_level or self._has_recent_evidence(now):
                self._correct_agents_on_beat(now, strong_val)

            # Raw beat stream: onset past the raw cooldown.
            if (now - self._last_beat_time) >= self.cooldown_sec:
                info.is_beat = True
                info.beat_strength = strength
                self._last_beat_time = now

            # Metric beat stream: onset where the trusted PLL expects a beat,
            # at most one per half period. Independent of the raw cooldown so a
            # syncopated hit that consumed the raw cooldown cannot swallow the
            # real kick. Falls back to the raw stream while no agent is trusted.
            if self._pll_trusted():
                metric_gap = 0.5 * self._best_agent.period
                if aligned and (now - self._last_metric_beat_time) >= metric_gap:
                    info.is_metric_beat = True
                    info.beat_strength = max(info.beat_strength, strength)
                    self._last_metric_beat_time = now
            elif info.is_beat:
                info.is_metric_beat = True
                self._last_metric_beat_time = now

        # --- 1b. Per-band onset detection ---
        info = self._detect_per_band_onsets(features, now, info)

        # --- 2. Advance all agent phases ---
        self._advance_agents(now)

        # --- 3. Autocorrelation BPM estimation + agent housekeeping (every ~0.5s) ---
        self._frame_count += 1
        buf_len = len(self._onset_buffer)
        if buf_len >= self._lag_max + 10 and self._frame_count % self._acorr_interval == 0:
            self._estimate_bpm_autocorrelation()
            if self._has_recent_evidence(now):
                self._raw_bpm_recent.append(self._raw_bpm)
                self._prior_bpm = float(np.median(self._raw_bpm_recent))
                self._anchor_agent_periods()
                self._seed_agents_from_autocorrelation(now)
                self._merge_duplicate_agents()
                self._prune_agents(now)

        # --- 4. Pick the best agent (with hysteresis) and sync output vars ---
        if self._has_recent_evidence(now):
            self._select_best_agent()
        self._sync_best_agent()

        # --- 5. Output: combine PLL + confidence gating + smoothing ---
        if self._pll_period > 0:
            pll_bpm = 60.0 / self._pll_period
        else:
            pll_bpm = self._raw_bpm

        # Sidechain compression detection (mid-frequency periodic modulation)
        mid_flux = float(np.sum(features.band_energies[2:5]))
        self._mid_flux_history.append(mid_flux)
        sidechain_detected = self._detect_sidechain()

        # Confidence = best agent quality x tempo agreement with the
        # autocorrelation x coasting multiplier. Quality is held at its last
        # evidence-backed value when strong onsets stop, so a breakdown decays
        # confidence on the coasting schedule. Prediction ratio and
        # autocorrelation SNR are kept as diagnostics only.
        self._update_prediction_confidence(now)
        self._update_coasting(now, sidechain_detected)
        if self._has_recent_evidence(now):
            quality = self._tracking_quality()
            self._quality_held = quality
        else:
            quality = self._quality_held
        self._quality_current = quality
        agreement = self._agent_agreement_factor()
        self._confidence = quality * agreement * self._coast_confidence_mult
        self._locked = self._confidence > 0.8

        # BPM smoothing: light EMA on PLL output (multi-agent already provides stability)
        if pll_bpm > 0 and self._confidence > 0.1:
            self._low_confidence_frames = 0

            if self._smooth_bpm == 0:
                self._smooth_bpm = pll_bpm
            else:
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

        # Raw cooldown: preset value is the floor, scaled up to 75% of the beat
        # period once the tempo is known.
        if self.auto_cooldown and self._display_bpm > 0:
            beat_period = 60.0 / self._display_bpm
            self.cooldown_sec = max(self._manual_cooldown_sec, beat_period * 0.75)

        # Fill output
        info.bpm = round(self._display_bpm, 1)
        info.bpm_confidence = self._confidence
        info.time_since_beat = now - self._last_beat_time

        # Predict the next beat from the best agent's phase, not from the last
        # raw onset: false onsets must not move the prediction. Agent phases
        # were already advanced by one frame above, so they describe the state
        # at now + frame_dur.
        best = self._best_agent
        if best is not None and best.period > 0:
            info.predicted_next_beat = (
                now + self._frame_dur + (1.0 - best.phase) * best.period
            )

        return info

    # --- Multi-agent PLL methods ---

    def _tracking_quality(self) -> float:
        """0..1: how reliable the best agent is (leaky score / reference)."""
        best = self._best_agent
        if best is None or best.period <= 0:
            return 0.0
        return float(np.clip(best.score / self._score_ref, 0.0, 1.0))

    def _pll_trusted(self) -> bool:
        """True when the best agent is reliable enough (live, or held while
        coasting) to gate onsets."""
        best = self._best_agent
        if best is None or best.period <= 0:
            return False
        return self._quality_current * self._agent_agreement_factor() >= self._trust_threshold

    def _has_recent_evidence(self, now: float) -> bool:
        """True if a strong onset happened within the evidence window."""
        return (
            self._last_strong_onset_time > 0
            and (now - self._last_strong_onset_time) <= self._evidence_window
        )

    def _onset_aligned_with_best(self) -> bool:
        """True if the current onset lands within the confirmation window of the
        best agent's expected beat. Must be called BEFORE _correct_agents_on_beat."""
        best = self._best_agent
        if best is None or best.period <= 0:
            return False
        phase_error = best.phase
        if phase_error > 0.5:
            phase_error -= 1.0
        return abs(phase_error * best.period) <= self._agent_confirmation_window

    def _correct_agents_on_beat(self, now: float, weight: float = 1.0) -> None:
        """Correct all agents when an onset is detected.

        Inside the confirmation window: score boost (scaled by onset weight),
        miss counter reset, PI phase/period correction. Strong onsets inside the
        wider capture window: the same PI correction without scoring, so a
        slightly mis-tuned agent re-acquires the beat (and fixes its period)
        instead of drifting out of lock.
        """
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

            confirmed = time_error <= self._agent_confirmation_window
            captured = (
                not confirmed
                and weight >= self._strong_onset_level
                and time_error <= self._agent_capture_fraction * agent.period
            )
            if not (confirmed or captured):
                continue

            if confirmed:
                # Onset confirms this agent's prediction
                agent.score += weight
                agent.consecutive_misses = 0
                self._refine_period_least_squares(agent, now)

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

    @staticmethod
    def _refine_period_least_squares(agent: BeatAgent, now: float) -> None:
        """Fit beat time vs beat index over the agent's recent confirmations and
        pull the period toward the slope. A 4 s autocorrelation window resolves
        tempo to ~0.5%; 16 confirmed beats resolve it to ~0.1%, which is what a
        20 s breakdown needs to freewheel without drifting off the grid."""
        if agent.last_confirmation > 0:
            steps = int(round((now - agent.last_confirmation) / agent.period))
            agent.beat_index += max(1, steps)
        agent.last_confirmation = now
        agent.confirmations.append((agent.beat_index, now))
        if len(agent.confirmations) < AGENT_LS_MIN_POINTS:
            return
        idx = np.fromiter((k for k, _ in agent.confirmations), dtype=float)
        times = np.fromiter((t for _, t in agent.confirmations), dtype=float)
        slope = float(np.polyfit(idx, times, 1)[0])
        if slope > 0 and abs(slope - agent.period) / agent.period < 0.05:
            agent.period += AGENT_LS_GAIN * (slope - agent.period)

    def _advance_agents(self, now: float) -> None:
        """Advance phase of all agents by one frame. On phase wrap (a prediction)
        decay the agent's score, count misses, and record the best agent's
        prediction for the prediction-ratio confidence."""
        for agent in self._agents:
            if agent.period <= 0:
                continue

            agent.phase += self._frame_dur / agent.period
            # Slow integral leak
            agent.integral_error *= 0.999

            if agent.phase >= 1.0:
                overshoot = agent.phase - 1.0
                agent.phase %= 1.0

                if agent is self._best_agent:
                    predicted_time = now - overshoot * agent.period
                    self._prediction_window.append((predicted_time, False))

                # A prediction is informative only if a strong onset happened
                # during its period: then the score leaks and, if that onset was
                # not near the prediction, it counts as a miss. Silence
                # (breakdown) neither decays nor penalises; the agent freewheels.
                # A late onset inside the window still confirms and resets.
                if self._last_strong_onset_time > 0:
                    since_strong = now - self._last_strong_onset_time
                    if since_strong <= agent.period:
                        agent.score *= AGENT_SCORE_DECAY
                        if since_strong > self._agent_confirmation_window:
                            agent.consecutive_misses += 1

    def _seed_agents_from_autocorrelation(self, now: float) -> None:
        """Seed new agents from the top autocorrelation peaks.

        New agents get their phase from a comb search over the onset buffer (the
        phase that collects the most onset energy at the candidate period), so
        a syncopated hit cannot anchor the only hypothesis at that tempo, and
        inherit 0.9x the best score (IBT) so they can compete with established
        agents. Agents are deduplicated on tempo AND phase: several phase
        hypotheses at one tempo are allowed and compete on score.
        """
        if self._raw_bpm <= 0 or self._last_weighted_corr is None:
            return

        weighted = self._last_weighted_corr
        lag_min = self._last_corr_lag_min

        # Top N local maxima
        candidate_periods = []
        for i in range(1, len(weighted) - 1):
            if weighted[i] > weighted[i - 1] and weighted[i] > weighted[i + 1]:
                period = (lag_min + self._interpolate_peak(weighted, i)) * self._frame_dur
                candidate_periods.append((weighted[i], period))
        candidate_periods.sort(reverse=True)
        candidates = candidate_periods[:self._initial_agents]

        # Always include the primary BPM estimate
        primary_period = 60.0 / self._raw_bpm
        if not any(abs(p - primary_period) / primary_period < 0.05 for _, p in candidates):
            candidates.insert(0, (1.0, primary_period))

        best_score = max((a.score for a in self._agents), default=0.0)
        inherited = 0.9 * best_score if self._agents else 1.0

        for _, period in candidates:
            if len(self._agents) >= self._max_agents:
                break

            phase = self._phase_from_onset_buffer(period)
            if phase is None:
                if self._last_onset_time <= 0 or (now - self._last_onset_time) >= 2.0 * period:
                    continue
                phase = ((now - self._last_onset_time) / period) % 1.0

            already_tracked = any(
                abs(a.period - period) / a.period < 0.05
                and self._phase_distance_sec(a.phase, phase, period)
                <= self._agent_confirmation_window
                for a in self._agents if a.period > 0
            )
            if already_tracked:
                continue

            self._agents.append(
                BeatAgent(period=period, phase=phase, score=inherited, born=now)
            )

    def _phase_from_onset_buffer(self, period_sec: float) -> float | None:
        """Phase (0=beat) at the newest buffer frame for the given period, chosen
        as the comb offset that collects the most onset energy in the buffer."""
        n = len(self._onset_buffer)
        p_frames = period_sec / self._frame_dur
        if p_frames < 2.0 or n < 2.0 * p_frames:
            return None
        buf = np.fromiter(self._onset_buffer, dtype=float, count=n)
        best_phi = 0
        best_sum = -1.0
        for phi in range(int(np.ceil(p_frames))):
            count = int((n - 1 - phi) / p_frames) + 1
            idx = np.round((n - 1 - phi) - np.arange(count) * p_frames).astype(int)
            idx = idx[idx >= 0]
            total = float(buf[idx].sum())
            if total > best_sum:
                best_sum = total
                best_phi = phi
        if best_sum <= 0.0:
            return None
        # The last beat was best_phi frames ago
        return (best_phi * self._frame_dur / period_sec) % 1.0

    @staticmethod
    def _phase_distance_sec(phase_a: float, phase_b: float, period: float) -> float:
        d = abs(phase_a - phase_b) % 1.0
        return min(d, 1.0 - d) * period

    def _anchor_agent_periods(self) -> None:
        """Pull agents within 5% of the autocorrelation tempo toward it."""
        if self._raw_bpm <= 0:
            return
        target = 60.0 / self._raw_bpm
        for agent in self._agents:
            if agent.period > 0 and abs(agent.period - target) / target < 0.05:
                agent.period += self._period_anchor_gain * (target - agent.period)

    def _merge_duplicate_agents(self) -> None:
        """Collapse agents that PI tracking has driven to the same tempo and phase
        (within 5% / one confirmation window) into the higher-scoring one, so the
        best-agent choice cannot flip between two copies of one hypothesis."""
        if len(self._agents) < 2:
            return
        ordered = sorted(self._agents, key=lambda a: a.score, reverse=True)
        if self._best_agent in ordered:
            ordered.remove(self._best_agent)
            ordered.insert(0, self._best_agent)
        kept: list[BeatAgent] = []
        for agent in ordered:
            duplicate = any(
                abs(k.period - agent.period) / k.period < 0.05
                and self._phase_distance_sec(k.phase, agent.phase, k.period)
                <= self._agent_confirmation_window
                for k in kept
            )
            if not duplicate:
                kept.append(agent)
        self._agents = kept

    def _prune_agents(self, now: float) -> None:
        """Drop agents with too many consecutive misses, or (once past their
        grace period) ranking below the kill ratio of the best ranked score."""
        if not self._agents:
            return

        # Rank with the same tempo prior as selection: a half/double-tempo agent
        # is confirmed on every one of its predictions and would otherwise
        # out-score (and kill) the correct agent whenever some beats lack onsets.
        ranked = {id(a): a.score * self._tempo_prior(a) for a in self._agents}
        best_ranked = max(ranked.values())
        kept = []
        for agent in self._agents:
            if agent.consecutive_misses >= self._agent_max_misses:
                continue
            in_grace = (now - agent.born) < AGENT_GRACE_BEATS * agent.period
            if not in_grace and ranked[id(agent)] < best_ranked * self._agent_kill_ratio:
                continue
            kept.append(agent)
        self._agents = kept

        if self._best_agent is not None and self._best_agent not in self._agents:
            self._best_agent = None

    def _select_best_agent(self) -> None:
        """Choose the output agent with hysteresis.

        Agents are ranked by score times a tempo prior centred on the
        autocorrelation estimate: the autocorrelation decides the gross tempo,
        agents track phase and fine period. The current best is kept while its
        ranked score is within AGENT_SWITCH_MARGIN of the top.
        """
        if not self._agents:
            self._best_agent = None
            return

        ranked = {id(a): a.score * self._tempo_prior(a) for a in self._agents}
        top_score = max(ranked.values())
        current = self._best_agent
        if (
            current is not None
            and current in self._agents
            and ranked[id(current)] * AGENT_SWITCH_MARGIN >= top_score
        ):
            return

        self._best_agent = max(self._agents, key=lambda a: ranked[id(a)])

    def _tempo_prior(self, agent: BeatAgent) -> float:
        """Weight for agent selection: 1.0 at the autocorrelation tempo, falling
        off as a log-Gaussian (sigma ~8%), floored so a much better-scoring agent
        can still win when the autocorrelation estimate is off."""
        if agent.period <= 0 or self._prior_bpm <= 0:
            return 1.0
        log_ratio = np.log2((60.0 / agent.period) / self._prior_bpm)
        return max(0.3, float(np.exp(-0.5 * (log_ratio / 0.08) ** 2)))

    def _agent_agreement_factor(self) -> float:
        """Confidence multiplier: 1.0 when the best agent matches the
        autocorrelation tempo, 0.8 for an octave relation, 0.5 otherwise."""
        best = self._best_agent
        if best is None or best.period <= 0 or self._prior_bpm <= 0:
            return 1.0
        ratio = (60.0 / best.period) / self._prior_bpm
        if abs(ratio - 1.0) < 0.05:
            return 1.0
        if abs(ratio - 2.0) < 0.1 or abs(ratio - 0.5) < 0.025:
            return 0.8
        return 0.5

    def _sync_best_agent(self) -> None:
        """Sync _pll_period and _pll_phase from the selected best agent."""
        best = self._best_agent
        if best is not None:
            self._pll_period = best.period
            self._pll_phase = best.phase
        # If no agents, _pll_period stays at whatever it was (or 0)

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
        self._last_weighted_corr = None
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

        # Keep the weighted curve for agent seeding
        self._last_weighted_corr = weighted
        self._last_corr_lag_min = lag_min

        # Find peak (sub-lag precision via parabolic interpolation)
        peak_idx = int(np.argmax(weighted))
        peak_val = weighted[peak_idx]

        if peak_val < 0.01:
            self._raw_confidence = 0.0
            return

        peak_lag = lag_min + self._interpolate_peak(weighted, peak_idx)

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

    @staticmethod
    def _interpolate_peak(curve: np.ndarray, idx: int) -> float:
        """Parabolic refinement of a local maximum position (fractional index)."""
        if idx <= 0 or idx >= len(curve) - 1:
            return float(idx)
        y0, y1, y2 = float(curve[idx - 1]), float(curve[idx]), float(curve[idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) < 1e-12:
            return float(idx)
        offset = 0.5 * (y0 - y2) / denom
        return idx + float(np.clip(offset, -0.5, 0.5))

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
        self._bass_prev.clear()
        self._bass_diff_ref = 1e-6
        self._bass_diff_peak = 1e-6
        self._flux_ref = 1e-6
        self._flux_peak = 1e-6
        self._odf_history.clear()
        self._onset_buffer.clear()
        self._frame_count = 0
        self._last_beat_time = 0.0
        self._last_onset_time = 0.0
        self._last_metric_beat_time = 0.0
        self._raw_bpm = 0.0
        self._raw_confidence = 0.0
        self._last_weighted_corr = None
        self._raw_bpm_recent.clear()
        self._prior_bpm = 0.0
        self._agents.clear()
        self._best_agent = None
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
        self._quality_held = 0.0
        self._quality_current = 0.0
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
        if self._best_agent is not None and self._best_agent not in self._agents:
            self._best_agent = None
        # Prediction window covers ~10 s at the new maximum tempo
        window_size = max(int(10.0 * bpm_max / 60.0), 20)
        self._prediction_window = deque(self._prediction_window, maxlen=window_size)

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
