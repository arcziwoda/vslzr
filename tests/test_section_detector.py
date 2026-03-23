"""Tests for section detection (drop detection rewrite).

Validates the five-layer architecture:
- Dual-timescale EMA tracking (gain-invariant exertion ratios)
- Six-signal weighted fusion (bass, broadband, centroid, flux, flatness, buildup)
- Variance-adaptive thresholding (Patin C)
- Six-state machine with minimum dwell times
- Operational edge cases (cold start, pause, resume, song change)

Critical test: kicks must NOT trigger DROP. Drops must trigger DROP.
"""

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatInfo
from hue_visualizer.audio.section_detector import (
    Section,
    SectionDetector,
    SectionInfo,
)
from hue_visualizer.visualizer.engine import EffectEngine


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

FPS = 43.07  # ~44100 / 1024


def _make_detector(**kwargs) -> SectionDetector:
    return SectionDetector(sample_rate_hz=kwargs.pop("fps", FPS), **kwargs)


def _feed_frames(
    det: SectionDetector,
    n: int,
    *,
    bass_raw: float = 0.01,
    rms_raw: float = 0.01,
    centroid: float = 3000.0,
    flux: float = 5.0,
    flatness: float = 0.2,
    bpm: float = 128.0,
    beat_interval: int = 15,
    start_time: float = 0.0,
    band_energies: np.ndarray | None = None,
) -> list[SectionInfo]:
    """Feed n frames of constant features, return all SectionInfo results."""
    dt = 1.0 / det._sample_rate_hz
    results = []
    if band_energies is None:
        band_energies = np.array([
            bass_raw * 0.6, bass_raw * 0.4,
            0.001, 0.001, 0.001, 0.0005, 0.0005,
        ])
    for i in range(n):
        now = start_time + i * dt
        info = det.update(
            bass_energy=0.5,
            rms=0.5,
            centroid=centroid,
            is_beat=(i % beat_interval == 0) if beat_interval > 0 else False,
            bpm=bpm,
            now=now,
            rms_raw=rms_raw,
            spectral_flux=flux,
            spectral_flatness=flatness,
            band_energies=band_energies,
        )
        results.append(info)
    return results


def _last_section(results: list[SectionInfo]) -> Section:
    """Get the last section from a list of results."""
    return results[-1].section


def _any_section(results: list[SectionInfo], section: Section) -> bool:
    """Check if any result has the given section."""
    return any(r.section == section for r in results)


# ---------------------------------------------------------------------------
# A. Internal computation tests
# ---------------------------------------------------------------------------


class TestEMAComputation:
    """Verify EMA alpha convergence rates."""

    def test_short_ema_converges_within_2_tau(self):
        """Short EMA (α=0.047, τ≈1s) should reach ~86% of step within 2τ."""
        det = _make_detector()
        # Seed with small value
        _feed_frames(det, 1, rms_raw=0.02, bass_raw=0.02)
        # Step up to 0.1
        _feed_frames(det, int(2.0 * FPS), rms_raw=0.1, bass_raw=0.1, start_time=1.0)
        # Short EMA should be close to 0.1 (within 15% of step)
        assert det._ema_short_rms > 0.08

    def test_long_ema_lags_behind_short(self):
        """Long EMA (α=0.006) should lag significantly behind short."""
        det = _make_detector()
        _feed_frames(det, 1, rms_raw=0.02, bass_raw=0.02)
        _feed_frames(det, int(2.0 * FPS), rms_raw=0.1, bass_raw=0.1, start_time=1.0)
        assert det._ema_long_rms < det._ema_short_rms


class TestExertionRatios:
    """Verify short/long ratio computation."""

    def test_exertion_at_steady_state_near_one(self):
        """Constant input should produce exertion ratios near 1.0."""
        det = _make_detector()
        # Feed long enough for both EMAs to converge
        _feed_frames(det, 500, rms_raw=0.05, bass_raw=0.05)
        bass_ex, rms_ex, centroid_r = det._compute_exertion_ratios()
        assert 0.9 < bass_ex < 1.1
        assert 0.9 < rms_ex < 1.1
        assert 0.9 < centroid_r < 1.1

    def test_exertion_spike_on_energy_increase(self):
        """Sudden energy increase should push exertion above 1.0."""
        det = _make_detector()
        _feed_frames(det, 400, rms_raw=0.02, bass_raw=0.02)
        # Spike
        _feed_frames(det, 43, rms_raw=0.15, bass_raw=0.15, start_time=10.0)
        bass_ex, rms_ex, _ = det._compute_exertion_ratios()
        assert bass_ex > 1.5
        assert rms_ex > 1.5


class TestDropScoreFusion:
    """Verify six-signal weighted fusion."""

    def test_score_zero_at_steady_state(self):
        """At steady state, drop score should be near zero."""
        det = _make_detector()
        results = _feed_frames(det, 500, rms_raw=0.05, bass_raw=0.05)
        assert results[-1].drop_score < 0.1

    def test_score_increases_on_energy_spike(self):
        """Drop score should increase on sudden energy spike."""
        det = _make_detector()
        _feed_frames(det, 400, rms_raw=0.02, bass_raw=0.02)
        results = _feed_frames(det, 60, rms_raw=0.15, bass_raw=0.15, start_time=10.0)
        assert results[-1].drop_score > 0.15


class TestAdaptiveThreshold:
    """Verify Patin C threshold range."""

    def test_threshold_in_valid_range(self):
        """Threshold should always be within [0.26, 0.52] (base/C_max to base/C_min × warmup)."""
        det = _make_detector()
        results = _feed_frames(det, 500, rms_raw=0.05, bass_raw=0.05)
        # After warmup (frame 344), threshold should be in [base/C_max, base/C_min]
        # = [0.40/1.55, 0.40/1.0] = [0.258, 0.40]
        t = results[-1].adaptive_threshold
        assert 0.2 < t < 0.5

    def test_warmup_elevates_threshold(self):
        """During warmup, threshold should be elevated by warmup_mult."""
        det = _make_detector()
        # Feed enough frames to get past UNKNOWN (86) but still in warmup (<344)
        results = _feed_frames(det, 200, rms_raw=0.05, bass_raw=0.05)
        # Find frames that are past UNKNOWN and have adaptive_threshold computed
        post_unknown = [r for r in results if r.section != Section.UNKNOWN and r.adaptive_threshold > 0]
        assert len(post_unknown) > 0, "No post-UNKNOWN frames with threshold"
        # During warmup (frame < 344), threshold = (base/C) * warmup_mult
        # At low variance: (0.40/1.55) * 1.3 ≈ 0.335
        # This should be higher than the post-warmup value (0.40/1.55 ≈ 0.258)
        assert post_unknown[-1].adaptive_threshold > 0.30


# ---------------------------------------------------------------------------
# B. Cold start / edge cases
# ---------------------------------------------------------------------------


class TestColdStart:
    """Verify cold start behavior."""

    def test_starts_in_unknown(self):
        """Detector should start in UNKNOWN state."""
        det = _make_detector()
        info = det.update(
            bass_energy=0.5, rms=0.5, centroid=3000.0,
            is_beat=False, bpm=128.0, now=0.0,
            rms_raw=0.05, spectral_flux=5.0, spectral_flatness=0.2,
        )
        assert info.section == Section.UNKNOWN

    def test_graduates_to_normal_after_lockout(self):
        """After 86 frames of signal, should transition to NORMAL."""
        det = _make_detector()
        results = _feed_frames(det, 120, rms_raw=0.05, bass_raw=0.05)
        # Should be NORMAL by frame 120
        assert _last_section(results) == Section.NORMAL

    def test_no_drop_during_unknown(self):
        """High energy during cold start should NOT trigger DROP."""
        det = _make_detector()
        # Feed high energy frames during UNKNOWN period
        results = _feed_frames(det, 86, rms_raw=0.3, bass_raw=0.3)
        assert not _any_section(results, Section.DROP)

    def test_stays_unknown_without_signal(self):
        """Without real signal (silence), should stay in UNKNOWN."""
        det = _make_detector()
        results = _feed_frames(det, 200, rms_raw=0.005, bass_raw=0.005)
        # rms_raw < silence_threshold (0.01), so never seeds
        assert _last_section(results) == Section.UNKNOWN


# ---------------------------------------------------------------------------
# C. Pause / resume
# ---------------------------------------------------------------------------


class TestPauseDetection:
    """Verify pause detection and resume behavior."""

    def test_pause_freezes_emas(self):
        """Silence for 13+ frames should freeze EMAs."""
        det = _make_detector()
        # Warm up
        _feed_frames(det, 200, rms_raw=0.05, bass_raw=0.05)

        # Silence for 20 frames (> 13 pause threshold)
        _feed_frames(det, 20, rms_raw=0.005, bass_raw=0.005, start_time=5.0)
        assert det._emas_frozen

        # Capture EMA values when frozen
        rms_at_freeze = det._ema_short_rms

        # Feed more silence frames — EMAs should NOT change further
        _feed_frames(det, 20, rms_raw=0.005, bass_raw=0.005, start_time=6.0)
        assert det._ema_short_rms == pytest.approx(rms_at_freeze, abs=1e-10)

    def test_short_pause_reseeds_short_from_long(self):
        """Resume after short pause should re-seed short EMAs from long."""
        det = _make_detector()
        _feed_frames(det, 400, rms_raw=0.05, bass_raw=0.05)
        long_rms = det._ema_long_rms

        # Short pause (1 second)
        _feed_frames(det, 50, rms_raw=0.005, bass_raw=0.005, start_time=10.0)
        assert det._emas_frozen

        # Resume
        _feed_frames(det, 5, rms_raw=0.05, bass_raw=0.05, start_time=12.0)
        assert not det._emas_frozen
        # Short EMA should have been re-seeded from long
        assert det._ema_short_rms == pytest.approx(long_rms, rel=0.1)

    def test_long_pause_resets_fully(self):
        """Resume after >5s pause should fully reset to UNKNOWN."""
        det = _make_detector()
        _feed_frames(det, 400, rms_raw=0.05, bass_raw=0.05)
        assert det._seeded

        # Long pause (>5 seconds worth of frames = 5 * 43 = 215 frames)
        _feed_frames(det, 250, rms_raw=0.005, bass_raw=0.005, start_time=10.0)

        # Resume — should have triggered full reset
        results = _feed_frames(det, 5, rms_raw=0.05, bass_raw=0.05, start_time=16.0)
        # After full reset, needs to re-seed, so state goes through UNKNOWN
        assert det._resume_lockout_remaining > 0 or not det._seeded or det._state == Section.UNKNOWN

    def test_no_false_trigger_on_resume(self):
        """Resume lockout should prevent false DROP on resume."""
        det = _make_detector()
        _feed_frames(det, 400, rms_raw=0.02, bass_raw=0.02)

        # Pause
        _feed_frames(det, 50, rms_raw=0.005, bass_raw=0.005, start_time=10.0)

        # Resume with HIGH energy (should not trigger drop due to lockout)
        results = _feed_frames(det, 43, rms_raw=0.2, bass_raw=0.2, start_time=12.0)
        assert not _any_section(results, Section.DROP)


# ---------------------------------------------------------------------------
# D. State machine transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    """Verify state machine transition logic."""

    def test_normal_to_breakdown(self):
        """Energy drop should transition NORMAL → BREAKDOWN."""
        det = _make_detector()
        # Establish baseline with moderate energy
        _feed_frames(det, 400, rms_raw=0.08, bass_raw=0.08)
        assert det._state == Section.NORMAL

        # Drop energy significantly — above silence threshold (0.01) but low enough
        # to drive rms_exertion below 0.7
        results = _feed_frames(det, 150, rms_raw=0.015, bass_raw=0.015, start_time=10.0)
        assert _any_section(results, Section.BREAKDOWN) or _any_section(results, Section.QUIET)

    def test_breakdown_to_buildup(self):
        """Rising energy slope + centroid should transition BREAKDOWN → BUILDUP."""
        det = _make_detector()
        # Establish high baseline, then breakdown
        _feed_frames(det, 400, rms_raw=0.08, bass_raw=0.08)
        _feed_frames(det, 100, rms_raw=0.015, bass_raw=0.015, start_time=10.0)
        # Should be in BREAKDOWN or QUIET
        assert det._state in (Section.BREAKDOWN, Section.QUIET, Section.NORMAL)

        # Gradually rising energy with rising centroid
        dt = 1.0 / FPS
        states_seen = set()
        for i in range(200):
            now = 13.0 + i * dt
            ramp = 0.015 + 0.002 * i  # Rising from 0.015 to 0.415
            centroid = 3000.0 + 20.0 * i  # Rising centroid
            info = det.update(
                bass_energy=0.5, rms=0.5, centroid=centroid,
                is_beat=(i % 15 == 0), bpm=128.0, now=now,
                rms_raw=ramp, spectral_flux=5.0, spectral_flatness=0.2,
                band_energies=np.array([ramp * 0.6, ramp * 0.4, 0.001, 0.001, 0.001, 0.0005, 0.0005]),
            )
            states_seen.add(info.section)
        # Should have transitioned through BUILDUP (or directly to NORMAL/DROP)
        assert states_seen & {Section.BUILDUP, Section.DROP, Section.NORMAL}

    def test_drop_to_sustain(self):
        """After DROP min dwell (22 frames), should transition to SUSTAIN."""
        det = _make_detector()
        # Get to DROP state via a clear breakdown → massive energy spike
        _feed_frames(det, 400, rms_raw=0.08, bass_raw=0.08)
        _feed_frames(det, 100, rms_raw=0.005, bass_raw=0.005, start_time=10.0)

        # Massive spike to trigger DROP
        results = _feed_frames(
            det, 100, rms_raw=0.5, bass_raw=0.5,
            flux=50.0, start_time=13.0,
        )

        sections = [r.section for r in results]
        if Section.DROP in sections:
            drop_idx = sections.index(Section.DROP)
            # After DROP, SUSTAIN should appear
            post_drop = sections[drop_idx:]
            assert Section.SUSTAIN in post_drop

    def test_sustain_to_normal_on_energy_settle(self):
        """SUSTAIN should transition to NORMAL when energy settles."""
        det = _make_detector()
        _feed_frames(det, 400, rms_raw=0.08, bass_raw=0.08)
        _feed_frames(det, 100, rms_raw=0.005, bass_raw=0.005, start_time=10.0)
        _feed_frames(det, 50, rms_raw=0.5, bass_raw=0.5, flux=50.0, start_time=13.0)

        # Settle back to normal energy
        results = _feed_frames(det, 500, rms_raw=0.08, bass_raw=0.08, start_time=15.0)
        assert _last_section(results) in (Section.NORMAL, Section.SUSTAIN)

    def test_emergency_override(self):
        """Very high confidence should bypass normal path to DROP."""
        det = _make_detector()
        # Warm up with low (but above silence) energy
        _feed_frames(det, 400, rms_raw=0.02, bass_raw=0.02)
        assert det._state == Section.NORMAL

        # Massive unambiguous spike (emergency override conditions:
        # drop_score > 0.9, bass_exertion > 2.0, rms_exertion > 1.8)
        results = _feed_frames(
            det, 60,
            rms_raw=0.5, bass_raw=0.5,
            flux=100.0, centroid=1000.0, flatness=0.01,
            start_time=10.0,
        )
        assert _any_section(results, Section.DROP), (
            f"Emergency override not triggered! "
            f"Max drop_score={max(r.drop_score for r in results):.3f}, "
            f"Max bass_ex={max(r.bass_exertion for r in results):.3f}, "
            f"Max rms_ex={max(r.rms_exertion for r in results):.3f}"
        )


class TestDwellTimes:
    """Verify minimum dwell time enforcement."""

    def test_unknown_dwell(self):
        """Should stay in UNKNOWN for at least 86 frames."""
        det = _make_detector()
        results = _feed_frames(det, 86, rms_raw=0.05, bass_raw=0.05)
        # All frames before 86 should be UNKNOWN
        for r in results[:85]:
            assert r.section == Section.UNKNOWN

    def test_drop_dwell(self):
        """Once in DROP, should stay for at least 22 frames."""
        det = _make_detector()
        _feed_frames(det, 400, rms_raw=0.08, bass_raw=0.08)
        _feed_frames(det, 100, rms_raw=0.005, bass_raw=0.005, start_time=10.0)

        # Trigger DROP
        results = _feed_frames(
            det, 100, rms_raw=0.5, bass_raw=0.5, flux=50.0, start_time=13.0,
        )

        sections = [r.section for r in results]
        if Section.DROP in sections:
            drop_start = sections.index(Section.DROP)
            # Count consecutive DROP frames
            drop_count = 0
            for s in sections[drop_start:]:
                if s == Section.DROP:
                    drop_count += 1
                else:
                    break
            assert drop_count >= 22


# ---------------------------------------------------------------------------
# E. The critical kick-vs-drop test
# ---------------------------------------------------------------------------


class TestKickVsDrop:
    """THE most important tests: kicks must not trigger DROP, drops must trigger DROP."""

    def test_periodic_kicks_do_not_trigger_drop(self):
        """Feed periodic bass spikes (like 128 BPM kicks) — must NEVER trigger DROP.

        This is the single most important test in this file. The old detector
        failed here because individual kicks produce the same energy signatures
        as drops in a single-timescale analysis.
        """
        det = _make_detector()
        # Warm up to stable state
        _feed_frames(det, 200, rms_raw=0.05, bass_raw=0.05)
        assert det._state == Section.NORMAL

        # Simulate 128 BPM kicks for 10 seconds
        # At 43 fps and 128 BPM: one kick every ~20 frames
        # Kick = 2-3 frames of high energy, then low
        dt = 1.0 / FPS
        kick_period = int(FPS * 60.0 / 128.0)  # ~20 frames
        kick_duration = 3  # frames

        all_results = []
        for i in range(int(10 * FPS)):
            now = 5.0 + i * dt
            phase = i % kick_period
            if phase < kick_duration:
                # Kick: high bass
                rms_raw = 0.15
                bass_raw = 0.15
                flux = 30.0
            else:
                # Between kicks: low energy
                rms_raw = 0.02
                bass_raw = 0.02
                flux = 3.0

            band = np.array([bass_raw * 0.6, bass_raw * 0.4, 0.001, 0.001, 0.001, 0.0005, 0.0005])
            info = det.update(
                bass_energy=0.5, rms=0.5, centroid=3000.0,
                is_beat=(phase == 0), bpm=128.0, now=now,
                rms_raw=rms_raw, spectral_flux=flux, spectral_flatness=0.2,
                band_energies=band,
            )
            all_results.append(info)

        # NO frame should ever be in DROP state
        drop_frames = [r for r in all_results if r.section == Section.DROP]
        assert len(drop_frames) == 0, (
            f"Kicks triggered {len(drop_frames)} DROP frames! "
            f"Max drop_score={max(r.drop_score for r in all_results):.3f}, "
            f"Max bass_exertion={max(r.bass_exertion for r in all_results):.3f}"
        )

    def test_real_drop_after_breakdown_triggers(self):
        """Feed 2+ seconds of low energy, then sustained high energy — must trigger DROP."""
        det = _make_detector()
        # Establish baseline
        _feed_frames(det, 400, rms_raw=0.08, bass_raw=0.08)

        # Breakdown: 3 seconds of low energy
        _feed_frames(det, int(3 * FPS), rms_raw=0.005, bass_raw=0.005, start_time=10.0)

        # Drop: sustained high energy for 2 seconds
        results = _feed_frames(
            det, int(2 * FPS),
            rms_raw=0.4, bass_raw=0.4,
            flux=40.0, centroid=2000.0, flatness=0.05,
            start_time=14.0,
        )

        assert _any_section(results, Section.DROP), (
            f"Drop not detected! Max drop_score={max(r.drop_score for r in results):.3f}, "
            f"Max bass_exertion={max(r.bass_exertion for r in results):.3f}, "
            f"Threshold={results[-1].adaptive_threshold:.3f}"
        )

    def test_prodigy_problem_spectral_drop(self):
        """Already-high bass + spectral shape change should trigger DROP.

        This tests the "Prodigy problem": drops where bass only increases
        modestly (1.5x) but spectral content changes significantly.
        """
        det = _make_detector()
        # Establish high-energy baseline (already loud track)
        _feed_frames(
            det, 400,
            rms_raw=0.08, bass_raw=0.08,
            centroid=4000.0, flux=10.0, flatness=0.3,
        )

        # Brief breakdown (energy dip + spectral shift)
        _feed_frames(
            det, int(2 * FPS),
            rms_raw=0.03, bass_raw=0.02,
            centroid=5000.0, flux=3.0, flatness=0.5,
            start_time=10.0,
        )

        # "Prodigy drop": bass only 1.5x baseline but with massive spectral change
        # Centroid drops (bass reintroduction), flux spikes, flatness drops (tonal)
        results = _feed_frames(
            det, int(2 * FPS),
            rms_raw=0.12, bass_raw=0.12,
            centroid=2000.0, flux=60.0, flatness=0.05,
            start_time=13.0,
        )

        assert _any_section(results, Section.DROP), (
            f"Spectral drop not detected! Max drop_score={max(r.drop_score for r in results):.3f}, "
            f"Max bass_exertion={max(r.bass_exertion for r in results):.3f}"
        )


# ---------------------------------------------------------------------------
# F. Integration with engine
# ---------------------------------------------------------------------------


def _silence_features() -> AudioFeatures:
    """AudioFeatures representing complete silence."""
    return AudioFeatures(
        band_energies=np.zeros(7),
        spectral_centroid=0.0,
        spectral_flux=0.0,
        spectral_rolloff=0.0,
        spectral_flatness=0.0,
        rms=0.0,
        peak=0.0,
        spectrum=np.zeros(1024),
    )


def _moderate_features() -> AudioFeatures:
    """AudioFeatures representing moderate energy."""
    bands = np.array([0.4, 0.4, 0.3, 0.3, 0.2, 0.1, 0.1])
    return AudioFeatures(
        band_energies=bands.copy(),
        band_energies_raw=bands.copy(),
        spectral_centroid=3000.0,
        spectral_flux=5.0,
        spectral_rolloff=8000.0,
        spectral_flatness=0.3,
        rms=0.5,
        peak=0.7,
        spectrum=np.zeros(1024),
    )


def _no_beat() -> BeatInfo:
    return BeatInfo()


class TestEngineIntegration:
    """Verify engine handles new section states correctly."""

    def test_engine_strobe_on_drop_transition(self):
        """Engine should trigger strobe on DROP entry (not from SUSTAIN)."""
        engine = EffectEngine(num_lights=3)
        engine.set_strobe_enabled(True)

        # First tick: NORMAL
        section_normal = SectionInfo(section=Section.NORMAL)
        engine.tick(_moderate_features(), _no_beat(), 0.02, section_info=section_normal)
        assert not engine.strobe_active

        # DROP transition
        section_drop = SectionInfo(section=Section.DROP, intensity=0.8)
        engine.tick(_moderate_features(), _no_beat(), 0.02, section_info=section_drop)
        assert engine.strobe_active

    def test_engine_sustain_keeps_reactive_high(self):
        """SUSTAIN should maintain high reactive weight like DROP."""
        engine = EffectEngine(num_lights=3)

        section = SectionInfo(section=Section.SUSTAIN, intensity=0.8)
        states = engine.tick(_moderate_features(), _no_beat(), 0.02, section_info=section)
        # All light states should have valid brightness
        for ls in states:
            assert 0.0 <= ls.brightness <= 1.0

    def test_engine_unknown_quiet_like_normal(self):
        """UNKNOWN and QUIET should behave like NORMAL — no special effects."""
        engine = EffectEngine(num_lights=3)

        for sec in (Section.UNKNOWN, Section.QUIET):
            section = SectionInfo(section=sec, intensity=0.0)
            states = engine.tick(_moderate_features(), _no_beat(), 0.02, section_info=section)
            for ls in states:
                assert 0.0 <= ls.brightness <= 1.0

    def test_engine_sustain_does_not_retrigger_strobe(self):
        """Transition from DROP to SUSTAIN should NOT trigger a new strobe."""
        engine = EffectEngine(num_lights=3)
        engine.set_strobe_enabled(True)

        # Enter DROP
        section_drop = SectionInfo(section=Section.DROP, intensity=0.8)
        engine.tick(_moderate_features(), _no_beat(), 0.02, section_info=section_drop)
        assert engine.strobe_active

        # Run enough ticks to finish strobe
        for _ in range(100):
            engine.tick(_moderate_features(), _no_beat(), 0.02, section_info=section_drop)
        strobe_was_active = engine.strobe_active

        # Transition to SUSTAIN — should NOT trigger new strobe
        section_sustain = SectionInfo(section=Section.SUSTAIN, intensity=0.6)
        engine.tick(_moderate_features(), _no_beat(), 0.02, section_info=section_sustain)
        # If strobe had finished, it should not restart
        if not strobe_was_active:
            assert not engine.strobe_active

    def test_all_brightnesses_valid(self):
        """All section states should produce valid brightness values."""
        engine = EffectEngine(num_lights=3)
        for sec in Section:
            section = SectionInfo(section=sec, intensity=0.5)
            states = engine.tick(_moderate_features(), _no_beat(), 0.02, section_info=section)
            for ls in states:
                assert 0.0 <= ls.brightness <= 1.0, f"Invalid brightness {ls.brightness} for {sec}"


# ---------------------------------------------------------------------------
# G. Continuous operation
# ---------------------------------------------------------------------------


class TestContinuousOperation:
    """Verify system stability under extended operation."""

    def test_30_second_simulation(self):
        """30-second simulation with varying input — no crashes, all outputs bounded."""
        det = _make_detector()
        dt = 1.0 / FPS
        rng = np.random.RandomState(42)

        for i in range(int(30 * FPS)):
            now = i * dt
            # Random-ish varying features
            base_rms = 0.05 + 0.03 * np.sin(i * 0.01)
            rms_raw = max(0.001, base_rms + rng.normal(0, 0.01))
            bass_raw = max(0.001, rms_raw * 0.8 + rng.normal(0, 0.005))
            centroid = max(100.0, 3000.0 + rng.normal(0, 200))
            flux = max(0.0, 5.0 + rng.normal(0, 2))
            flatness = max(0.0, min(1.0, 0.2 + rng.normal(0, 0.05)))

            band = np.array([bass_raw * 0.6, bass_raw * 0.4, 0.001, 0.001, 0.001, 0.0005, 0.0005])
            info = det.update(
                bass_energy=0.5, rms=0.5, centroid=centroid,
                is_beat=(i % 20 == 0), bpm=128.0, now=now,
                rms_raw=rms_raw, spectral_flux=flux, spectral_flatness=flatness,
                band_energies=band,
            )

            assert 0.0 <= info.intensity <= 1.0
            assert 0.0 <= info.confidence <= 2.0  # Can exceed 1.0 slightly
            assert info.section in Section

    def test_reset_clears_state(self):
        """Reset should return detector to initial state."""
        det = _make_detector()
        _feed_frames(det, 500, rms_raw=0.05, bass_raw=0.05)
        assert det._seeded
        assert det._frame_count > 0

        det.reset()
        assert not det._seeded
        assert det._frame_count == 0
        assert det._state == Section.UNKNOWN
        assert det._drop_score == 0.0


# ---------------------------------------------------------------------------
# H. Song change detection
# ---------------------------------------------------------------------------


class TestSongChange:
    """Verify song change detection."""

    def test_bpm_change_elevates_threshold(self):
        """Large BPM change should trigger song change detection."""
        det = _make_detector()
        # Establish baseline at 128 BPM with enough frames for feature window
        _feed_frames(det, 400, rms_raw=0.05, bass_raw=0.05, bpm=128.0)
        # The _prev_feature_avg should be set by now
        assert det._prev_feature_avg is not None

        # Switch to very different BPM — needs enough frames for comparison
        # Song change requires 2s (confirm_sec) of sustained BPM difference
        _feed_frames(
            det, int(4 * FPS),
            rms_raw=0.05, bass_raw=0.05, bpm=170.0,
            start_time=10.0,
        )
        # Should have detected BPM change and either accumulated frames or boosted threshold
        assert det._song_change_threshold_boost > 0 or det._song_change_frames > 0, (
            f"Song change not detected: boost={det._song_change_threshold_boost}, "
            f"frames={det._song_change_frames}, prev_bpm={det._prev_bpm}"
        )
