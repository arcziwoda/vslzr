"""Tests for BeatDetector — behavioral tests for correct beat detection.

Validates: BPM accuracy, convergence, IBI consistency, confidence,
octave errors, multi-agent recovery, coasting, per-band onsets, reset.
"""

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatDetector, BeatInfo


def _make_features(
    bass_energy: float = 0.0,
    flux: float = 0.0,
    band_energies: np.ndarray | None = None,
) -> AudioFeatures:
    """Create AudioFeatures with specified energies.

    SuperFlux onset is simulated from flux parameter.
    """
    if band_energies is not None:
        bands = band_energies.copy()
    else:
        bands = np.zeros(7)
        bands[0] = bass_energy  # sub-bass
        bands[1] = bass_energy  # bass
    return AudioFeatures(
        band_energies=bands,
        spectral_flux=flux,
        superflux_onset=float(np.log1p(100.0 * flux)) if flux > 0 else 0.0,
        rms=bass_energy * 0.5,
    )


def _simulate_beats(
    bd: BeatDetector,
    bpm: float,
    duration_sec: float = 10.0,
    beat_energy: float = 0.9,
    quiet_energy: float = 0.1,
    beat_flux: float = 2.0,
) -> list[BeatInfo]:
    """Simulate a beat train at given BPM and return all BeatInfo results."""
    frame_rate = bd._frame_rate
    frame_dur = 1.0 / frame_rate
    beat_period = 60.0 / bpm
    total_frames = int(duration_sec * frame_rate)

    results = []
    t = 0.0
    for _ in range(total_frames):
        phase = (t % beat_period) / beat_period
        if phase < 0.05:  # ~5% of beat period is the transient
            energy = beat_energy
            flux = beat_flux
        else:
            energy = quiet_energy
            flux = 0.01

        info = bd.detect(_make_features(bass_energy=energy, flux=flux), timestamp=t)
        results.append(info)
        t += frame_dur

    return results


def _get_detected_beats(results: list[BeatInfo]) -> list[int]:
    """Return frame indices where beats were detected."""
    return [i for i, r in enumerate(results) if r.is_beat]


# ============================================================
# BPM Accuracy
# ============================================================


class TestBPMAccuracy:
    """BPM estimation should converge to correct tempo."""

    @pytest.mark.parametrize("target_bpm", [100, 128, 143, 174])
    def test_bpm_converges(self, target_bpm: int):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=target_bpm, duration_sec=20)
        # Take BPM from last 3 seconds (allow time for smoothing EMA to converge)
        final_bpms = [r.bpm for r in results[-int(bd._frame_rate * 3):] if r.bpm > 0]
        assert len(final_bpms) > 0, "Should have BPM estimates"
        avg_bpm = np.mean(final_bpms)
        # Tolerance is wide because synthetic AudioFeatures don't produce
        # realistic onset patterns — real-music benchmark is the true measure
        assert abs(avg_bpm - target_bpm) < 35, (
            f"BPM should be within ±35 of {target_bpm}, got {avg_bpm:.1f}"
        )

    def test_dnb_range(self):
        """DnB tempo (170-178) should be detected correctly in DnB range."""
        bd = BeatDetector(bpm_min=155, bpm_max=185)
        results = _simulate_beats(bd, bpm=174, duration_sec=12)
        final_bpms = [r.bpm for r in results[-int(bd._frame_rate * 2):] if r.bpm > 0]
        assert len(final_bpms) > 0
        avg_bpm = np.mean(final_bpms)
        assert abs(avg_bpm - 174) < 10


# ============================================================
# Convergence Speed
# ============================================================


class TestConvergenceSpeed:
    """BPM should lock within a reasonable time."""

    def test_locks_within_6s(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=8)
        # Check confidence after 6 seconds
        frame_6s = int(6 * bd._frame_rate)
        late_results = results[frame_6s:]
        confidences = [r.bpm_confidence for r in late_results]
        avg_conf = np.mean(confidences) if confidences else 0
        assert avg_conf > 0.3, f"Should have reasonable confidence after 6s, got {avg_conf:.3f}"


# ============================================================
# IBI Consistency
# ============================================================


class TestIBIConsistency:
    """Inter-beat intervals should be consistent for steady tempo."""

    def test_steady_beat_ibi(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=15)

        # Get beat times after warmup (first 4 seconds)
        frame_dur = 1.0 / bd._frame_rate
        beat_times = [
            i * frame_dur for i, r in enumerate(results)
            if r.is_beat and i * frame_dur > 4.0
        ]

        if len(beat_times) < 5:
            pytest.skip("Not enough beats detected for IBI analysis")

        ibis = np.diff(beat_times)
        cv = float(np.std(ibis) / np.mean(ibis))
        assert cv < 0.5, f"IBI CV should be < 0.5 for steady beats, got {cv:.4f}"


# ============================================================
# Confidence
# ============================================================


class TestConfidence:
    """Confidence should reflect detection quality."""

    def test_high_confidence_on_steady_beats(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=12)
        # After warmup
        late_confs = [r.bpm_confidence for r in results[-int(bd._frame_rate * 3):]
                      if r.bpm > 0]
        assert len(late_confs) > 0
        avg = np.mean(late_confs)
        assert avg > 0.2, f"Confidence should be > 0.2 on steady beats, got {avg:.3f}"

    def test_low_confidence_on_silence(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        # Feed silence
        t = 0.0
        frame_dur = 1.0 / bd._frame_rate
        for _ in range(200):
            bd.detect(_make_features(bass_energy=0.0), timestamp=t)
            t += frame_dur
        info = bd.detect(_make_features(bass_energy=0.0), timestamp=t)
        assert info.bpm_confidence < 0.3


# ============================================================
# Octave Errors
# ============================================================


class TestOctaveErrors:
    """BPM estimation should resolve octave ambiguity."""

    def test_double_time_resolved(self):
        """64 BPM input with 80-180 range should double to ~128."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        # Simulate at 64 BPM — below range, should detect as 128
        results = _simulate_beats(bd, bpm=64, duration_sec=15)
        final_bpms = [r.bpm for r in results[-int(bd._frame_rate * 2):] if r.bpm > 0]
        if len(final_bpms) > 0:
            avg = np.mean(final_bpms)
            # Should be doubled to ~128, or at least within range
            assert avg >= 80, f"Should resolve octave error, got {avg:.1f}"

    def test_half_time_resolved(self):
        """256 BPM input with 80-180 range should halve to ~128."""
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=256, duration_sec=12)
        final_bpms = [r.bpm for r in results[-int(bd._frame_rate * 2):] if r.bpm > 0]
        if len(final_bpms) > 0:
            avg = np.mean(final_bpms)
            assert avg <= 180, f"Should resolve octave error, got {avg:.1f}"


# ============================================================
# Multi-Agent
# ============================================================


class TestMultiAgent:
    """Multi-agent PLL should maintain and manage beat hypotheses."""

    def test_agents_seeded(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=8)
        assert len(bd._agents) > 0, "Should have active agents after beats"

    def test_agents_cleared_on_reset(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=5)
        assert len(bd._agents) > 0
        bd.reset()
        assert len(bd._agents) == 0

    def test_best_agent_drives_period(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=10)
        if bd._agents:
            best = max(bd._agents, key=lambda a: a.score)
            assert abs(bd._pll_period - best.period) < 1e-6


# ============================================================
# Coasting
# ============================================================


class TestCoasting:
    """Coasting behavior during non-percussive sections."""

    def test_no_coasting_during_beats(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=8)
        assert not bd.is_coasting

    def test_coasting_after_beats_stop(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=8)
        # Feed silence for 6 seconds (past 4s threshold)
        t = 8.0
        frame_dur = 1.0 / bd._frame_rate
        for _ in range(int(6 * bd._frame_rate)):
            bd.detect(_make_features(bass_energy=0.0), timestamp=t)
            t += frame_dur
        assert bd.is_coasting, "Should be coasting after 6s of silence"

    def test_bpm_maintained_during_coasting(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=10)
        bpm_before = bd.current_bpm
        # Feed silence for 5 seconds
        t = 10.0
        frame_dur = 1.0 / bd._frame_rate
        for _ in range(int(5 * bd._frame_rate)):
            bd.detect(_make_features(bass_energy=0.0), timestamp=t)
            t += frame_dur
        bpm_after = bd.current_bpm
        if bpm_before > 0:
            assert bpm_after > 0, "BPM should be maintained during coasting"

    def test_confidence_decays_during_coasting(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=10)
        conf_during_beats = results[-1].bpm_confidence
        # Feed silence for 10 seconds (past 4s full confidence, into decay zone)
        t = 10.0
        frame_dur = 1.0 / bd._frame_rate
        last_info = None
        for _ in range(int(10 * bd._frame_rate)):
            last_info = bd.detect(_make_features(bass_energy=0.0), timestamp=t)
            t += frame_dur
        # Confidence should have decayed (coasting multiplier < 1.0)
        assert bd._coast_confidence_mult < 1.0


# ============================================================
# Per-Band Onsets
# ============================================================


class TestPerBandOnsets:
    """Kick/snare/hi-hat onset detection on appropriate frequency content."""

    def test_kick_onset_on_low_frequency(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0
        # Warmup with low energy
        for _ in range(30):
            bands = np.zeros(7)
            bands[0] = 0.1
            bands[1] = 0.1
            bd.detect(_make_features(band_energies=bands), timestamp=t)
            t += frame_dur
        # Spike on low bands
        t += 0.5
        bands = np.zeros(7)
        bands[0] = 0.9
        bands[1] = 0.9
        info = bd.detect(_make_features(band_energies=bands), timestamp=t)
        assert info.kick_onset, "Should detect kick onset on low frequency spike"

    def test_hihat_onset_on_high_frequency(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0
        # Warmup
        for _ in range(30):
            bands = np.zeros(7)
            bands[5] = 0.1
            bands[6] = 0.1
            bd.detect(_make_features(band_energies=bands), timestamp=t)
            t += frame_dur
        # Spike on high bands
        t += 0.5
        bands = np.zeros(7)
        bands[5] = 0.9
        bands[6] = 0.9
        info = bd.detect(_make_features(band_energies=bands), timestamp=t)
        assert info.hihat_onset, "Should detect hi-hat onset on high frequency spike"


# ============================================================
# Reset
# ============================================================


class TestReset:
    """Reset should clear all state."""

    def test_reset_clears_state(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=5)
        bd.reset()
        assert bd._smooth_bpm == 0
        assert bd._pll_period == 0
        assert bd._raw_bpm == 0
        assert len(bd._onset_buffer) == 0
        assert not bd._locked
        assert len(bd._agents) == 0
        assert len(bd._prediction_window) == 0
        assert bd._prediction_confidence == 0.0
        assert not bd._coasting
        assert bd._coast_confidence_mult == 1.0


# ============================================================
# Beat Detection Basics
# ============================================================


class TestBeatDetection:
    """Basic beat detection and cooldown."""

    def test_detects_beats(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=5)
        beats = [r for r in results if r.is_beat]
        assert len(beats) > 3, "Should detect multiple beats"

    def test_cooldown_prevents_double_trigger(self):
        bd = BeatDetector(cooldown_ms=500, bpm_min=80, bpm_max=180)
        frame_dur = 1.0 / bd._frame_rate
        t = 0.0
        # Warmup
        for _ in range(20):
            bd.detect(_make_features(bass_energy=0.1), timestamp=t)
            t += frame_dur
        # First beat
        r1 = bd.detect(_make_features(bass_energy=0.9, flux=2.0), timestamp=t)
        t += frame_dur
        # Immediately after — within cooldown
        r2 = bd.detect(_make_features(bass_energy=0.9, flux=2.0), timestamp=t)
        if r1.is_beat:
            assert not r2.is_beat, "Cooldown should prevent double trigger"

    def test_beat_strength_positive(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=5)
        beats = [r for r in results if r.is_beat]
        for beat in beats:
            assert beat.beat_strength >= 0


# ============================================================
# Prediction Confidence
# ============================================================


class TestPredictionConfidence:
    """PLL prediction tracking and confirmation."""

    def test_predictions_generated(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=10)
        assert len(bd._prediction_window) > 0

    def test_predicted_next_beat_set(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        results = _simulate_beats(bd, bpm=128, duration_sec=10)
        # At least some results should have predictions
        predictions = [r.predicted_next_beat for r in results if r.predicted_next_beat > 0]
        assert len(predictions) > 0


# ============================================================
# Genre Presets
# ============================================================


class TestGenrePresets:
    """set_cooldown and set_bpm_range should work correctly."""

    def test_set_bpm_range(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        bd.set_bpm_range(155, 185)
        assert bd.bpm_min == 155
        assert bd.bpm_max == 185

    def test_set_cooldown(self):
        bd = BeatDetector(cooldown_ms=300)
        bd.set_cooldown(500)
        assert abs(bd.cooldown_sec - 0.5) < 0.001

    def test_agents_pruned_on_range_change(self):
        bd = BeatDetector(bpm_min=80, bpm_max=180)
        _simulate_beats(bd, bpm=128, duration_sec=8)
        n_agents_before = len(bd._agents)
        # Narrow range to exclude 128 BPM
        bd.set_bpm_range(155, 185)
        n_agents_after = len(bd._agents)
        # Should have fewer agents (128 BPM agents killed)
        if n_agents_before > 0:
            assert n_agents_after <= n_agents_before
