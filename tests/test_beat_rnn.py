"""Learned beat activation (audio/beat_rnn.py) and its use as the onset source."""

import numpy as np
import pytest
from scipy import signal

from hue_visualizer.audio.analyzer import AudioAnalyzer, AudioFeatures
from hue_visualizer.audio.beat_detector import BeatDetector
from hue_visualizer.audio.beat_rnn import MODEL_PATH, BeatActivationRNN, StreamingResampler

SR = 44100
HOP = 1024

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="beat RNN model missing")


def _click_train(bpm: float, seconds: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Kick-like clicks (decaying 60 Hz sine + noise burst) on a beat grid."""
    rng = np.random.default_rng(seed)
    audio = np.zeros(int(sr * seconds), dtype=np.float32)
    period = 60.0 / bpm
    n = int(sr * 0.12)
    t = np.arange(n) / sr
    hit = np.sin(2 * np.pi * 60.0 * t) * np.exp(-t / 0.05)
    hit[: int(sr * 0.004)] += 0.5 * rng.standard_normal(int(sr * 0.004))
    beat = 0.0
    while beat < seconds:
        i = int(beat * sr)
        audio[i:i + n] += hit[: len(audio) - i]
        beat += period
    return (0.8 * audio / np.max(np.abs(audio))).astype(np.float32)


class TestActivation:
    def test_peaks_on_beats_and_quiet_between(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from synth_audio import techno_130

        rnn = BeatActivationRNN(SR)
        audio, beats, _ = techno_130(duration=20.0)
        act = rnn.process_offline(audio, chunk=HOP)
        fps = rnn.fps
        period = 60.0 / 130.0
        beats = beats[(beats > 3.0) & (beats < 19.0)]
        on = [act[int((b - 0.05) * fps):int((b + 0.05) * fps) + 1].max() for b in beats]
        off = [act[int((b + 0.5 * period - 0.05) * fps):int((b + 0.5 * period + 0.05) * fps) + 1].max()
               for b in beats]
        assert np.median(on) > 0.5
        assert np.median(off) < 0.1

    def test_streaming_matches_offline_regardless_of_chunking(self):
        audio = _click_train(130.0, 6.0)
        a = BeatActivationRNN(SR).process_offline(audio, chunk=1024)
        b = BeatActivationRNN(SR).process_offline(audio, chunk=333)
        n = min(len(a), len(b))
        assert n > 500
        np.testing.assert_allclose(a[:n], b[:n], atol=1e-6)

    def test_frame_timing(self):
        rnn = BeatActivationRNN(SR)
        out = rnn.push(np.zeros(HOP, dtype=np.float32))
        # Frame i is centred on sample i * 441; a frame needs 1024 samples after
        # its centre, so the first hop yields frames centred up to sample 0
        assert [round(t, 4) for t, _ in out] == [0.0]
        out = rnn.push(np.zeros(HOP, dtype=np.float32))
        assert len(out) == 2
        assert out[0][0] == pytest.approx(441 / SR)

    def test_48k_input_matches_44k(self):
        audio = _click_train(126.0, 8.0)
        a44 = BeatActivationRNN(44100).process_offline(audio, chunk=1024)
        audio48 = signal.resample_poly(audio, 160, 147).astype(np.float32)
        a48 = BeatActivationRNN(48000).process_offline(audio48, chunk=1024)
        n = min(len(a44), len(a48))
        assert np.corrcoef(a44[:n], a48[:n])[0, 1] > 0.99


class TestResampler:
    def test_matches_resample_poly_up_to_integer_delay(self):
        rng = np.random.default_rng(1)
        x = signal.sosfilt(signal.butter(4, 15000 / 24000, output="sos"),
                           rng.standard_normal(48000)).astype(np.float32)
        rs = StreamingResampler(48000, 44100)
        y = np.concatenate([rs.process(x[i:i + 1000]) for i in range(0, len(x), 1000)])
        ref = signal.resample_poly(x, 147, 160)
        d = rs.delay_out
        n = min(len(y) - d, len(ref))
        err = np.sqrt(np.mean((y[d:d + n] - ref[:n]) ** 2) / np.mean(ref[:n] ** 2))
        assert len(y) == pytest.approx(len(ref), abs=2)
        assert err < 5e-3


class TestDetectorOnActivation:
    def test_rnn_source_locks_tempo_and_confirms_metric_beats(self):
        det = BeatDetector(SR, HOP, onset_source="rnn")
        frame_dur = HOP / SR
        period = 60.0 / 130.0
        metric = []
        for i in range(int(30 / frame_dur)):
            t = i * frame_dur
            phase = (t % period) / period
            f = AudioFeatures()
            f.beat_activation = 0.9 if phase < 0.05 else 0.02
            info = det.detect(f, timestamp=t)
            if info.is_metric_beat and t > 5.0:
                metric.append(t)
        assert abs(info.bpm - 130.0) < 2.0
        assert info.bpm_confidence >= 0.6
        assert abs(len(metric) - int(25.0 / period)) <= 2

    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError):
            BeatDetector(SR, HOP, onset_source="magic")


class TestAnalyzerIntegration:
    def test_analyzer_fills_activation_when_enabled(self):
        analyzer = AudioAnalyzer(sample_rate=SR, hop_size=HOP, use_beat_rnn=True)
        assert analyzer.beat_rnn_available
        audio = _click_train(128.0, 4.0)
        peak = 0.0
        for i in range(0, len(audio) - HOP, HOP):
            peak = max(peak, analyzer.analyze(audio[i:i + HOP]).beat_activation)
        assert peak > 0.5

    def test_analyzer_without_model_leaves_zero(self):
        analyzer = AudioAnalyzer(sample_rate=SR, hop_size=HOP)
        assert not analyzer.beat_rnn_available
        f = analyzer.analyze(_click_train(128.0, 1.0)[:HOP])
        assert f.beat_activation == 0.0
