"""Tests for AudioAnalyzer."""

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioAnalyzer, AudioFeatures


@pytest.fixture
def analyzer():
    return AudioAnalyzer(sample_rate=44100, fft_size=2048, bass_boost=2.0)


def _sine(freq: float, duration_samples: int = 2048, sr: int = 44100) -> np.ndarray:
    t = np.arange(duration_samples) / sr
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


class TestAudioAnalyzer:
    def test_silence_returns_zeros(self, analyzer):
        frame = np.zeros(1024, dtype=np.float32)
        f = analyzer.analyze(frame)
        assert f.rms == 0.0 or f.rms < 0.01
        assert f.peak < 0.01

    def test_sine_wave_has_energy(self, analyzer):
        frame = _sine(440, 2048)
        f = analyzer.analyze(frame)
        assert f.peak > 0.1
        assert f.spectral_centroid > 0

    def test_bass_sine_centroid_low(self, analyzer):
        frame = _sine(100, 2048)
        f = analyzer.analyze(frame)
        assert f.spectral_centroid < 500

    def test_treble_sine_centroid_high(self, analyzer):
        frame = _sine(8000, 2048)
        f = analyzer.analyze(frame)
        assert f.spectral_centroid > 3000

    def test_band_energies_shape(self, analyzer):
        frame = _sine(440, 2048)
        f = analyzer.analyze(frame)
        assert f.band_energies.shape == (7,)
        assert all(0 <= v <= 2.0 for v in f.band_energies)

    def test_spectral_flux_increases_on_change(self, analyzer):
        # First frame: silence
        analyzer.analyze(np.zeros(2048, dtype=np.float32))
        # Second frame: loud sine -> high flux
        f = analyzer.analyze(_sine(1000, 2048))
        assert f.spectral_flux > 0

    def test_auto_gain_normalizes_rms(self, analyzer):
        # Feed consistent quiet signal
        for _ in range(100):
            f = analyzer.analyze(np.random.randn(1024).astype(np.float32) * 0.01)
        # Now a louder frame should normalize near 1.0
        f = analyzer.analyze(np.random.randn(1024).astype(np.float32) * 0.1)
        assert f.rms > 0.5

    def test_spectrum_is_db_scale(self, analyzer):
        f = analyzer.analyze(_sine(440, 2048))
        assert len(f.spectrum) > 0
        # dB values should be negative for most bins
        assert np.mean(f.spectrum) < 0

    def test_reset_clears_state(self, analyzer):
        analyzer.analyze(_sine(440, 2048))
        analyzer.reset()
        assert not np.any(analyzer._stft_buffer)
        assert analyzer._prev_magnitude is None
        assert len(analyzer._rms_history) == 0


class TestMelFilterbank:
    """Task 2.1: Mel filterbank with 32 perceptually-spaced triangular filters."""

    def test_mel_energies_shape(self, analyzer):
        """mel_energies should be a 32-element array."""
        f = analyzer.analyze(_sine(440, 2048))
        assert f.mel_energies.shape == (32,)

    def test_mel_energies_nonnegative(self, analyzer):
        """All Mel band energies should be >= 0."""
        f = analyzer.analyze(_sine(1000, 2048))
        assert np.all(f.mel_energies >= 0)

    def test_mel_filterbank_matrix_shape(self, analyzer):
        """Filterbank matrix should be (32, fft_size // 2 + 1)."""
        expected_shape = (32, analyzer.fft_size // 2 + 1)
        assert analyzer._mel_filterbank.shape == expected_shape

    def test_mel_filterbank_triangular_filters(self, analyzer):
        """Each filter should have a triangular shape with peak of 1.0."""
        fb = analyzer._mel_filterbank
        for i in range(fb.shape[0]):
            row = fb[i]
            nonzero = row[row > 0]
            if len(nonzero) > 0:
                # Peak value should be 1.0 (center of triangle)
                assert abs(np.max(row) - 1.0) < 1e-10, \
                    f"Filter {i} peak should be 1.0, got {np.max(row)}"
                # All values should be in [0, 1]
                assert np.all(row >= 0) and np.all(row <= 1.0 + 1e-10)

    def test_mel_filters_overlap(self, analyzer):
        """Adjacent Mel filters should overlap (shared bins)."""
        fb = analyzer._mel_filterbank
        overlap_count = 0
        for i in range(fb.shape[0] - 1):
            active_i = set(np.where(fb[i] > 0)[0])
            active_next = set(np.where(fb[i + 1] > 0)[0])
            if active_i & active_next:
                overlap_count += 1
        # Most adjacent filters should overlap (low-frequency ones may not
        # if they are narrow, but most should)
        assert overlap_count > fb.shape[0] * 0.5, \
            f"Expected most filters to overlap, only {overlap_count}/{fb.shape[0]-1} do"

    def test_mel_filters_cover_full_range(self, analyzer):
        """Filterbank should cover bins from low to high frequency."""
        fb = analyzer._mel_filterbank
        # Aggregate: which bins have at least one filter active
        active_bins = np.any(fb > 0, axis=0)
        active_indices = np.where(active_bins)[0]
        # Should start near the beginning and reach near the end
        assert active_indices[0] <= 5, \
            f"First active bin {active_indices[0]} should be near 0"
        assert active_indices[-1] >= fb.shape[1] * 0.8, \
            f"Last active bin {active_indices[-1]} should be near end ({fb.shape[1]})"

    def test_bass_sine_mel_energy_in_low_bands(self, analyzer):
        """A 100 Hz sine should excite low Mel bands, not high ones."""
        # Feed a few frames so auto-gain settles
        for _ in range(5):
            analyzer.analyze(_sine(100, 2048))
        f = analyzer.analyze(_sine(100, 2048))
        # Most energy should be in the first few Mel bands
        low_sum = float(np.sum(f.mel_energies[:8]))
        high_sum = float(np.sum(f.mel_energies[24:]))
        assert low_sum > high_sum, \
            f"Bass sine: low Mel sum ({low_sum:.3f}) should exceed high ({high_sum:.3f})"

    def test_treble_sine_mel_energy_in_high_bands(self, analyzer):
        """A 10 kHz sine should excite high Mel bands, not low ones."""
        for _ in range(5):
            analyzer.analyze(_sine(10000, 2048))
        f = analyzer.analyze(_sine(10000, 2048))
        low_sum = float(np.sum(f.mel_energies[:8]))
        high_sum = float(np.sum(f.mel_energies[24:]))
        assert high_sum > low_sum, \
            f"Treble sine: high Mel sum ({high_sum:.3f}) should exceed low ({low_sum:.3f})"

    def test_mel_auto_gain_normalization(self, analyzer):
        """Mel energies should be auto-gain normalized (values approach 1.0)."""
        # Feed steady signal for many frames
        for _ in range(50):
            analyzer.analyze(_sine(440, 2048))
        f = analyzer.analyze(_sine(440, 2048))
        # At least one Mel band should be close to 1.0 after auto-gain
        assert np.max(f.mel_energies) > 0.5, \
            f"Max Mel energy {np.max(f.mel_energies):.3f} should approach 1.0 after auto-gain"

    def test_silence_mel_energies_near_zero(self, analyzer):
        """Silence should produce near-zero Mel energies."""
        f = analyzer.analyze(np.zeros(1024, dtype=np.float32))
        assert np.max(f.mel_energies) < 0.01

    def test_mel_default_in_audio_features(self):
        """Default AudioFeatures should have 32-element zero mel_energies."""
        f = AudioFeatures()
        assert f.mel_energies.shape == (32,)
        assert np.all(f.mel_energies == 0)

    def test_reset_clears_mel_state(self, analyzer):
        """Reset should reinitialize mel_max to small values."""
        for _ in range(20):
            analyzer.analyze(_sine(440, 2048))
        analyzer.reset()
        # After reset, mel_max should be back to initial small value
        assert np.all(analyzer._mel_max < 1e-5)

    def test_mel_spacing_is_perceptual(self, analyzer):
        """Lower Mel bands should span fewer Hz than higher ones (Mel spacing)."""
        fb = analyzer._mel_filterbank
        n_fft_bins = fb.shape[1]
        freq_per_bin = analyzer.sample_rate / analyzer.fft_size

        # Find center bin of each filter (bin with max value)
        centers_hz = []
        for i in range(fb.shape[0]):
            center_bin = np.argmax(fb[i])
            centers_hz.append(center_bin * freq_per_bin)

        # Check that spacing increases: first 8 bands should be closer together
        # than last 8 bands (Mel property)
        if len(centers_hz) >= 16:
            low_spacing = centers_hz[8] - centers_hz[0]
            high_spacing = centers_hz[-1] - centers_hz[-9]
            assert high_spacing > low_spacing, \
                f"Mel spacing: high bands ({high_spacing:.0f} Hz) should span more than low ({low_spacing:.0f} Hz)"


class TestAnalyzerReviewRegressions:
    """Regressions for the Sept 2026 analyzer review findings."""

    def test_band_slices_do_not_overlap(self):
        for sr in (44100, 48000):
            analyzer = AudioAnalyzer(sample_rate=sr, fft_size=2048, hop_size=1024)
            slices = list(analyzer._band_slices.values())
            for (s0, e0), (s1, e1) in zip(slices, slices[1:]):
                assert e0 <= s1, f"bands share bin {e0 - 1} at {sr} Hz"
                assert e0 > s0

    def test_stft_window_is_full_for_small_hops(self):
        sr = 44100
        t = np.arange(sr) / sr
        sine = (0.8 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        peaks = {}
        for hop in (1024, 512, 256):
            analyzer = AudioAnalyzer(sample_rate=sr, fft_size=2048, hop_size=hop)
            for i in range(0, 8192, hop):
                features = analyzer.analyze(sine[i:i + hop])
            peaks[hop] = float(np.max(features.spectrum))
        assert abs(peaks[512] - peaks[1024]) < 0.5
        assert abs(peaks[256] - peaks[1024]) < 0.5

    def test_steady_level_rms_is_not_stretched_to_full_scale(self):
        rng = np.random.default_rng(0)
        analyzer = AudioAnalyzer(sample_rate=44100, fft_size=2048, hop_size=1024)
        values = []
        for _ in range(300):
            frame = (0.1 * rng.standard_normal(1024)).astype(np.float32)
            values.append(analyzer.analyze(frame).rms)
        values = np.array(values[20:])
        assert values.max() - values.min() < 0.5
        assert values.std() < 0.12
