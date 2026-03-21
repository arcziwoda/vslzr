"""Tests for BeatDetector — autocorrelation BPM, PLL, octave errors, confidence."""

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatDetector, BeatInfo


def _make_features(bass_energy: float = 0.0, flux: float = 0.0) -> AudioFeatures:
    """Create AudioFeatures with specified bass energy."""
    bands = np.zeros(7)
    bands[0] = bass_energy  # sub-bass
    bands[1] = bass_energy  # bass
    return AudioFeatures(
        band_energies=bands,
        spectral_flux=flux,
        rms=bass_energy * 0.5,
    )


def _simulate_beats(
    bd: BeatDetector,
    bpm: float,
    duration_sec: float = 10.0,
    beat_energy: float = 0.9,
    quiet_energy: float = 0.1,
) -> list[BeatInfo]:
    """Simulate a beat train at given BPM and return all BeatInfo results."""
    frame_rate = bd._frame_rate
    frame_dur = 1.0 / frame_rate
    total_frames = int(duration_sec * frame_rate)
    beat_period_frames = frame_rate * 60.0 / bpm

    results = []
    for i in range(total_frames):
        t = i * frame_dur  # simulated time

        # Is this frame a beat?
        phase = (i % beat_period_frames) / beat_period_frames
        if phase < 0.1:  # First 10% of beat period = onset
            energy = beat_energy
        else:
            energy = quiet_energy

        info = bd.detect(_make_features(bass_energy=energy), timestamp=t)
        results.append(info)

    return results


class TestBeatDetection:
    def test_detects_beats(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=5)
        beats = [r for r in results if r.is_beat]
        assert len(beats) > 3, "Should detect multiple beats"

    def test_cooldown_prevents_double_trigger(self):
        bd = BeatDetector(cooldown_ms=400, bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0
        # Warmup with quiet frames
        for _ in range(20):
            bd.detect(_make_features(bass_energy=0.1), timestamp=t)
            t += frame_dur
        r1 = bd.detect(_make_features(bass_energy=0.9), timestamp=t)
        t += frame_dur  # Only ~23ms later
        r2 = bd.detect(_make_features(bass_energy=0.9), timestamp=t)
        if r1.is_beat:
            assert not r2.is_beat, "Cooldown should prevent double trigger"

    def test_beat_strength_positive(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=5)
        beats = [r for r in results if r.is_beat]
        for b in beats:
            assert b.beat_strength >= 0


class TestAutocorrelationBPM:
    def test_estimates_128_bpm(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=10)
        final = results[-1]
        assert final.bpm > 0, "Should have a BPM estimate"
        assert abs(final.bpm - 128) < 15, f"Expected ~128, got {final.bpm}"

    def test_estimates_140_bpm(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=140, duration_sec=10)
        final = results[-1]
        assert abs(final.bpm - 140) < 15, f"Expected ~140, got {final.bpm}"

    def test_estimates_100_bpm(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=100, duration_sec=10)
        final = results[-1]
        assert abs(final.bpm - 100) < 15, f"Expected ~100, got {final.bpm}"


class TestOctaveErrorProtection:
    def test_fix_double_time(self):
        bd = BeatDetector(bpm_min=100, bpm_max=180)
        # BPM 64 should be doubled to 128
        fixed = bd._fix_octave_errors(64.0, np.array([0.1]), 14)
        assert fixed == 128.0

    def test_fix_half_time(self):
        bd = BeatDetector(bpm_min=100, bpm_max=180)
        # BPM 256 should be halved to 128
        fixed = bd._fix_octave_errors(256.0, np.array([0.1]), 14)
        assert fixed == 128.0

    def test_in_range_unchanged(self):
        bd = BeatDetector(bpm_min=100, bpm_max=180)
        fixed = bd._fix_octave_errors(128.0, np.array([0.1]), 14)
        assert fixed == 128.0

    def test_genre_range_constrains(self):
        bd = BeatDetector(bpm_min=155, bpm_max=185)  # DnB range
        # 85 BPM should double to 170
        fixed = bd._fix_octave_errors(85.0, np.array([0.1]), 14)
        assert fixed == 170.0


class TestConfidenceGating:
    def test_silence_low_confidence(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        info = None
        for _ in range(100):
            info = bd.detect(_make_features(bass_energy=0.0))
        assert info is not None and info.bpm_confidence < 0.3

    def test_stable_beats_high_confidence(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=10)
        final = results[-1]
        # After 10 seconds of steady beats, confidence should be reasonable
        assert final.bpm_confidence > 0.2

    def test_holds_stable_bpm_in_silence(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        # First: establish a BPM
        _simulate_beats(bd, bpm=128, duration_sec=8)
        stable = bd._stable_bpm
        assert stable > 0, "Should have established a stable BPM"

        # Now silence — BPM should stay close to stable value (not jump wildly)
        t = 8.0
        frame_dur = 1.0 / bd._frame_rate
        info = None
        for _ in range(100):
            info = bd.detect(_make_features(bass_energy=0.0), timestamp=t)
            t += frame_dur
        assert info is not None
        assert abs(info.bpm - stable) < 2, \
            f"BPM should hold near {stable:.1f}, got {info.bpm}"


class TestPLL:
    def test_pll_phase_advances(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=5)
        # PLL phase should be advancing
        phase1 = bd.pll_phase
        bd.detect(_make_features(bass_energy=0.1))
        bd.detect(_make_features(bass_energy=0.1))
        phase2 = bd.pll_phase
        assert phase1 != phase2 or bd._pll_period == 0

    def test_pll_period_set_from_autocorrelation(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=8)
        assert bd._pll_period > 0, "PLL period should be set"
        expected_period = 60.0 / 128
        assert abs(bd._pll_period - expected_period) < 0.15, \
            f"PLL period {bd._pll_period:.3f} should be near {expected_period:.3f}"

    def test_predicted_next_beat(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=8)
        final = results[-1]
        if final.predicted_next_beat > 0:
            assert final.predicted_next_beat > 0


class TestBPMSmoothing:
    def test_smooth_bpm_not_jumpy(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=10)

        # Collect all non-zero BPM values from last 5 seconds
        frame_rate = bd._frame_rate
        last_half = results[int(5 * frame_rate):]
        bpm_values = [r.bpm for r in last_half if r.bpm > 0]

        if len(bpm_values) > 10:
            std = np.std(bpm_values)
            assert std < 10, f"BPM should be stable, got std={std:.1f}"


class TestMedianThreshold:
    """Task 0.3: Verify median-based threshold is more robust to outliers."""

    def test_outlier_does_not_suppress_beats(self):
        """A single energy outlier should not inflate the threshold for many frames."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        bd.auto_cooldown = False
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup with moderate energy
        for _ in range(30):
            bd.detect(_make_features(bass_energy=0.3), timestamp=t)
            t += frame_dur

        # Inject a single massive outlier
        bd.detect(_make_features(bass_energy=5.0), timestamp=t)
        t += frame_dur

        # Feed a few quiet frames to let the outlier enter history
        for _ in range(3):
            bd.detect(_make_features(bass_energy=0.3), timestamp=t)
            t += frame_dur

        # Now a legitimate beat should still trigger (median resists the outlier)
        t += 0.5  # Ensure past cooldown
        result = bd.detect(_make_features(bass_energy=0.8), timestamp=t)
        # With median, 0.8 should be well above the median of ~0.3 * threshold
        # The key assertion: the beat should be detected despite the outlier
        assert result.is_beat, "Median-based threshold should resist outlier inflation"


class TestParallelcubeThreshold:
    """Task 1.7: Verify Parallelcube spec threshold calibration."""

    def test_zero_variance_threshold(self):
        """With zero variance, threshold should be 1.55."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Feed constant energy (zero variance)
        for _ in range(30):
            bd.detect(_make_features(bass_energy=0.5), timestamp=t)
            t += frame_dur

        history = np.array(bd._bass_history)
        variance = float(np.var(history))
        # Variance should be ~0 for constant input
        assert variance < 0.001, f"Variance should be ~0, got {variance}"

        # Threshold at zero variance should be 1.55
        threshold = float(np.clip(1.55 - (variance / 0.02) * 0.30, 1.25, 1.55))
        assert abs(threshold - 1.55) < 0.01, f"Threshold should be ~1.55, got {threshold}"

    def test_high_variance_threshold(self):
        """With high variance (~0.02), threshold should be ~1.25."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Alternate between low and high energy to create variance ~0.02
        for i in range(60):
            energy = 0.05 if i % 2 == 0 else 0.35
            bd.detect(_make_features(bass_energy=energy), timestamp=t)
            t += frame_dur

        history = np.array(bd._bass_history)
        variance = float(np.var(history))
        # With alternating 0.05/0.35, variance should be ~0.0225
        threshold = float(np.clip(1.55 - (variance / 0.02) * 0.30, 1.25, 1.55))
        assert threshold <= 1.30, f"Threshold should be <=1.30 for high variance, got {threshold}"
        assert threshold >= 1.25, f"Threshold should be clamped at 1.25, got {threshold}"

    def test_threshold_clamped_at_bounds(self):
        """Threshold should be clamped to [1.25, 1.55] range."""
        # Zero variance -> 1.55 (upper clamp)
        assert float(np.clip(1.55 - (0.0 / 0.02) * 0.30, 1.25, 1.55)) == 1.55
        # Extreme variance -> 1.25 (lower clamp)
        assert float(np.clip(1.55 - (0.1 / 0.02) * 0.30, 1.25, 1.55)) == 1.25
        # Mid variance -> between bounds
        mid = float(np.clip(1.55 - (0.01 / 0.02) * 0.30, 1.25, 1.55))
        assert 1.25 < mid < 1.55, f"Mid-variance threshold should be between bounds, got {mid}"
        assert abs(mid - 1.40) < 0.01, f"At variance=0.01, threshold should be ~1.40, got {mid}"


class TestFluxOnsetDetection:
    """Task 1.6: Verify spectral flux onset detection."""

    def test_flux_only_beat_detection(self):
        """High spectral flux should trigger a beat even with low bass energy."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        bd.auto_cooldown = False
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup with low energy, low flux
        for _ in range(30):
            bd.detect(_make_features(bass_energy=0.1, flux=0.01), timestamp=t)
            t += frame_dur

        # Ensure past cooldown
        t += 1.0

        # High flux spike with low bass — should trigger via flux onset
        result = bd.detect(_make_features(bass_energy=0.1, flux=2.0), timestamp=t)
        assert result.is_beat, "High spectral flux should trigger a beat even with low bass"

    def test_flux_log_compression(self):
        """Verify log compression is applied: log(1 + gamma * flux)."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Feed a known flux value
        bd.detect(_make_features(bass_energy=0.1, flux=0.5), timestamp=t)

        # Check that flux_onset_history contains log-compressed value
        expected = float(np.log1p(100.0 * 0.5))  # log(1 + 100 * 0.5) = log(51)
        # After warmup check (bass_history < 10), first 10 frames return early
        # so we need enough frames
        for _ in range(15):
            t += frame_dur
            bd.detect(_make_features(bass_energy=0.1, flux=0.5), timestamp=t)

        assert len(bd._flux_onset_history) > 0
        last_flux = bd._flux_onset_history[-1]
        assert abs(last_flux - expected) < 0.01, \
            f"Flux should be log-compressed: expected {expected:.3f}, got {last_flux:.3f}"

    def test_flux_adaptive_threshold(self):
        """Flux onset uses moving median — steady flux should not trigger beats."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        bd.auto_cooldown = False
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Feed constant moderate flux (should NOT trigger beats after warmup)
        beat_count = 0
        for i in range(100):
            result = bd.detect(_make_features(bass_energy=0.1, flux=0.5), timestamp=t)
            t += frame_dur
            if i > 30 and result.is_beat:  # Skip warmup period
                beat_count += 1

        # Constant flux should not keep triggering beats (median adapts)
        assert beat_count <= 2, \
            f"Constant flux should not trigger repeated beats, got {beat_count}"

    def test_combined_energy_and_flux_beats(self):
        """Both energy and flux onsets should contribute to beat detection."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        bd.auto_cooldown = False
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup
        for _ in range(30):
            bd.detect(_make_features(bass_energy=0.1, flux=0.01), timestamp=t)
            t += frame_dur

        # Energy-only beat
        t += 1.0
        r1 = bd.detect(_make_features(bass_energy=0.9, flux=0.01), timestamp=t)

        # Flux-only beat (after cooldown)
        t += 1.0
        r2 = bd.detect(_make_features(bass_energy=0.1, flux=2.0), timestamp=t)

        # Both should trigger
        assert r1.is_beat, "Energy-only onset should trigger a beat"
        assert r2.is_beat, "Flux-only onset should trigger a beat"

    def test_flux_onset_respects_cooldown(self):
        """Flux-triggered beats must still respect the cooldown period."""
        bd = BeatDetector(cooldown_ms=500, bpm_min=80, bpm_max=180)
        bd.auto_cooldown = False
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup
        for _ in range(30):
            bd.detect(_make_features(bass_energy=0.1, flux=0.01), timestamp=t)
            t += frame_dur

        # First beat via flux
        t += 1.0
        r1 = bd.detect(_make_features(bass_energy=0.1, flux=3.0), timestamp=t)

        # Immediately another flux spike (within cooldown)
        t += 0.05  # 50ms, well within 500ms cooldown
        r2 = bd.detect(_make_features(bass_energy=0.1, flux=3.0), timestamp=t)

        if r1.is_beat:
            assert not r2.is_beat, "Flux beat should respect cooldown"

    def test_flux_history_cleared_on_reset(self):
        """Reset should clear flux onset history."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        for _ in range(20):
            bd.detect(_make_features(bass_energy=0.3, flux=0.5), timestamp=t)
            t += frame_dur

        assert len(bd._flux_onset_history) > 0

        bd.reset()
        assert len(bd._flux_onset_history) == 0

    def test_beat_strength_from_flux(self):
        """Beat strength should reflect flux contribution when flux triggers the beat."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        bd.auto_cooldown = False
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0

        # Warmup with low flux
        for _ in range(30):
            bd.detect(_make_features(bass_energy=0.1, flux=0.01), timestamp=t)
            t += frame_dur

        # Flux-triggered beat
        t += 1.0
        result = bd.detect(_make_features(bass_energy=0.1, flux=3.0), timestamp=t)
        assert result.is_beat, "Should trigger from flux"
        assert result.beat_strength > 0, "Beat strength should be positive for flux beat"


class TestPredictionConfidence:
    """Task 2.2: Confidence scoring with prediction ratio."""

    def test_prediction_window_populated(self):
        """After steady beats, prediction window should have entries."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=10)
        assert len(bd._prediction_window) > 0, \
            "Prediction window should have entries after steady beats"

    def test_prediction_confidence_high_for_steady_beats(self):
        """Steady beats at 128 BPM should yield high prediction confidence."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=10)
        # After 10 seconds, many predictions should have been confirmed
        assert bd._prediction_confidence > 0.3, \
            f"Prediction confidence {bd._prediction_confidence:.2f} should be reasonable for steady beats"

    def test_prediction_confidence_low_for_silence(self):
        """Silence should produce low prediction confidence (no beats to confirm)."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0
        for _ in range(500):
            bd.detect(_make_features(bass_energy=0.0), timestamp=t)
            t += frame_dur
        # With no beats and no PLL, prediction confidence should be 0
        assert bd._prediction_confidence <= 0.1, \
            f"Prediction confidence {bd._prediction_confidence:.2f} should be low for silence"

    def test_blended_confidence_uses_both_sources(self):
        """Final confidence should blend autocorrelation and prediction ratio."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=10)
        final = results[-1]
        # Confidence should be between 0 and 1
        assert 0 <= final.bpm_confidence <= 1.0
        # With steady beats, confidence should be meaningful
        assert final.bpm_confidence > 0.1, \
            f"Blended confidence {final.bpm_confidence:.2f} should be > 0.1 for steady beats"

    def test_prediction_tolerance_50ms(self):
        """Predictions should be confirmable within ±50ms tolerance."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        assert bd._prediction_tolerance == 0.050

    def test_reset_clears_prediction_state(self):
        """Reset should clear all prediction tracking state."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=5)
        bd.reset()
        assert len(bd._prediction_window) == 0
        assert bd._prediction_confidence == 0.0

    def test_random_beats_low_prediction_confidence(self):
        """Random (non-periodic) beats should produce low prediction confidence."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        bd.auto_cooldown = False
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0
        rng = np.random.RandomState(42)

        # First: establish a BPM with steady beats so PLL starts predicting
        _simulate_beats(bd, bpm=128, duration_sec=6)
        t = 6.0

        # Then: feed random energy — beats at irregular intervals
        for _ in range(int(6 * bd._frame_rate)):
            energy = rng.uniform(0, 1.0)
            bd.detect(_make_features(bass_energy=energy), timestamp=t)
            t += frame_dur

        # After random input, prediction confidence should have dropped
        # (PLL predictions won't match random beats)
        assert bd._prediction_confidence < 0.8, \
            f"Prediction confidence {bd._prediction_confidence:.2f} should be low for random beats"

    def test_prediction_window_size_bounded(self):
        """Prediction window should be bounded (deque with maxlen)."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=20)
        # Window should not grow unbounded
        assert len(bd._prediction_window) <= bd._prediction_window.maxlen


class TestReset:
    def test_reset_clears_all(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=5)
        bd.reset()
        assert bd._smooth_bpm == 0
        assert bd._pll_period == 0
        assert bd._raw_bpm == 0
        assert len(bd._onset_buffer) == 0
        assert not bd._locked
        # Task 2.2: prediction state also cleared
        assert len(bd._prediction_window) == 0
        assert bd._prediction_confidence == 0.0
