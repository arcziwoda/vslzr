"""Regression tests for the beat detector review findings (docs/review/2026-09).

Feature-level simulations: ideal kick pulses in the bass bands plus a SuperFlux
spike, optionally with a syncopated percussive hit. These are the failure modes
that were observed before the rewrite and must not come back.
"""

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatDetector

SR = 44100
HOP = 1024
FRAME_DUR = HOP / SR


def _frame(kick: bool, perc: bool = False) -> AudioFeatures:
    bass = 1.4 if kick else 0.3
    high = 0.9 if perc else 0.2
    f = AudioFeatures()
    f.band_energies = np.array([bass, bass, 0.3, 0.3, 0.3, high, high])
    f.superflux_onset = 8.0 if kick else (6.0 if perc else 1.0)
    return f


def _run(
    detector: BeatDetector,
    bpm: float,
    seconds: float,
    synco_phase: float | None = None,
) -> dict:
    """Feed a kick train at `bpm`; optionally add a hit at `synco_phase` of each beat."""
    period = 60.0 / bpm
    out = {"raw": [], "metric": [], "predicted": [], "bpm": [], "conf": []}
    for i in range(int(seconds / FRAME_DUR)):
        t = i * FRAME_DUR
        phase = (t % period) / period
        kick = phase < 0.06
        perc = synco_phase is not None and synco_phase <= phase < synco_phase + 0.06
        info = detector.detect(_frame(kick, perc), timestamp=t)
        if info.is_beat:
            out["raw"].append(t)
        if info.is_metric_beat:
            out["metric"].append(t)
        if t > 5.0:
            out["bpm"].append(info.bpm)
            out["conf"].append(info.bpm_confidence)
            if info.predicted_next_beat > 0:
                out["predicted"].append((t, info.predicted_next_beat))
    return out


def _grid_error_ms(times: list[float], bpm: float, after: float = 5.0) -> np.ndarray:
    period = 60.0 / bpm
    ts = np.array([x for x in times if x > after])
    return (ts - np.round(ts / period) * period) * 1000.0


class TestTempoLock:
    """F1: agents seeded at an arbitrary phase let a wrong-tempo agent win."""

    @pytest.mark.parametrize("bpm", [100, 125, 130, 174])
    def test_reported_bpm_matches_ideal_kicks_with_default_range(self, bpm):
        det = BeatDetector(SR, HOP)
        res = _run(det, bpm, 30)
        median_bpm = float(np.median(res["bpm"]))
        assert abs(median_bpm - bpm) < 2.0, f"expected ~{bpm}, got {median_bpm:.1f}"

    def test_confidence_reaches_predictive_threshold(self):
        det = BeatDetector(SR, HOP)
        res = _run(det, 130, 30)
        assert float(np.mean(res["conf"][-200:])) >= 0.6


class TestMetricStream:
    """F3: a hit at 0.75 of the beat must not steal the beat from the kick."""

    def test_metric_beats_land_on_the_kick_despite_syncopation(self):
        det = BeatDetector(SR, HOP)
        res = _run(det, 130, 30, synco_phase=0.75)
        err = _grid_error_ms(res["metric"], 130)
        assert len(err) >= 40
        # Within one hop of the grid (onsets are stamped at frame start)
        assert abs(float(np.mean(err))) < 25.0
        assert float(np.std(err)) < 15.0

    def test_metric_stream_is_independent_of_raw_cooldown(self):
        det = BeatDetector(SR, HOP)
        res = _run(det, 130, 30, synco_phase=0.75)
        expected = int(25.0 / (60.0 / 130))
        n_metric = len([t for t in res["metric"] if t > 5.0])
        assert abs(n_metric - expected) <= 2

    def test_no_metric_beats_between_kicks(self):
        det = BeatDetector(SR, HOP)
        res = _run(det, 130, 30, synco_phase=0.75)
        err = _grid_error_ms(res["metric"], 130)
        assert np.all(np.abs(err) < 70.0)


class TestPrediction:
    """F5: predictions come from the PLL phase, not the last raw onset."""

    def test_predictions_land_on_the_grid(self):
        det = BeatDetector(SR, HOP)
        res = _run(det, 130, 30, synco_phase=0.75)
        preds = np.array([p for t, p in res["predicted"] if t > 10.0])
        err = _grid_error_ms(list(preds), 130, after=0.0)
        assert abs(float(np.mean(err))) < 25.0
        assert float(np.std(err)) < 25.0

    def test_prediction_is_in_the_future(self):
        det = BeatDetector(SR, HOP)
        res = _run(det, 130, 20)
        assert all(p > t for t, p in res["predicted"])


class TestCooldown:
    """F4: the preset cooldown is a floor, not overridden by the auto cooldown."""

    def test_short_preset_cooldown_is_honored(self):
        det = BeatDetector(SR, HOP, cooldown_ms=150, bpm_min=155, bpm_max=185)
        _run(det, 174, 20)
        # 75% of the 174 BPM period is 259 ms, above the 150 ms preset floor
        assert det.cooldown_sec == pytest.approx(0.75 * 60.0 / 174, abs=0.02)

    def test_long_preset_cooldown_is_honored(self):
        det = BeatDetector(SR, HOP, cooldown_ms=600, bpm_min=50, bpm_max=125)
        _run(det, 100, 20)
        assert det.cooldown_sec == pytest.approx(0.6, abs=1e-6)


class TestCoasting:
    """Silence must not kill the agents or collapse confidence immediately."""

    def test_agent_survives_a_breakdown_and_confidence_decays_slowly(self):
        det = BeatDetector(SR, HOP)
        period = 60.0 / 130
        confs = []
        for i in range(int(40 / FRAME_DUR)):
            t = i * FRAME_DUR
            in_breakdown = 15.0 <= t < 35.0
            kick = not in_breakdown and (t % period) / period < 0.06
            info = det.detect(_frame(kick), timestamp=t)
            if 20.0 <= t < 21.0 or 34.0 <= t < 35.0:
                confs.append((t, info.bpm_confidence))
        early = np.mean([c for t, c in confs if t < 21.0])
        late = np.mean([c for t, c in confs if t >= 34.0])
        assert det._best_agent is not None
        assert abs(60.0 / det._best_agent.period - 130) < 3.0
        assert early >= 0.6  # 5 s into the breakdown predictive is still active
        assert 0.4 <= late <= early  # tiered decay, floored at 50% of held quality


class TestAgentTakeover:
    """A freshly seeded agent inherits 0.9x the best score. Under timing jitter
    the autocorrelation estimate wobbles by one lag (130 -> 123 BPM); the tempo
    prior then favoured the fresh seed and it took the output for several beats
    with no confirmations of its own."""

    def test_fresh_seed_cannot_dethrone_an_established_agent(self):
        det = BeatDetector(SR, HOP)
        _run(det, 130, 15)
        best = det._best_agent
        assert best is not None and abs(60.0 / best.period - 130) < 2.0

        # Autocorrelation wobbles to 123 BPM and seeds an agent there with an
        # inherited score above the current best
        det._prior_bpm = 123.0
        from hue_visualizer.audio.beat_detector import BeatAgent
        det._agents.append(
            BeatAgent(period=60.0 / 123.0, phase=0.3, score=best.score * 1.5, born=15.0)
        )
        det._select_best_agent()
        det._prune_agents(15.0)
        assert det._best_agent is best
        assert best in det._agents

    def test_metric_stream_survives_timing_jitter(self):
        rng = np.random.default_rng(0)
        det = BeatDetector(SR, HOP)
        period = 60.0 / 130
        metric = []
        kick_times: dict[int, float] = {}
        for i in range(int(30 / FRAME_DUR)):
            t = i * FRAME_DUR
            beat_idx = round(t / period)
            if beat_idx not in kick_times:
                kick_times[beat_idx] = beat_idx * period + rng.normal(0.0, 0.012)
            kick = 0.0 <= t - kick_times[beat_idx] < FRAME_DUR
            info = det.detect(_frame(kick), timestamp=t)
            if info.is_metric_beat and t > 5.0:
                metric.append(t)
        expected = int(25.0 / period)
        assert len(metric) >= expected - 3
