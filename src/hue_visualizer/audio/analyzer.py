"""FFT analyzer — frequency analysis, band energies, spectral features."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# 7-band frequency ranges (Hz)
FREQUENCY_BANDS = {
    "sub_bass": (20, 60),
    "bass": (60, 250),
    "low_mid": (250, 500),
    "mid": (500, 2000),
    "upper_mid": (2000, 4000),
    "presence": (4000, 6000),
    "brilliance": (6000, 20000),
}

BAND_NAMES = list(FREQUENCY_BANDS.keys())


@dataclass
class AudioFeatures:
    """All extracted audio features for a single frame."""

    # Per-band energy (7 bands, normalized 0-1, bass-boosted for effects)
    band_energies: np.ndarray = field(default_factory=lambda: np.zeros(7))
    # Per-band energy WITHOUT bass boost (for UI visualization)
    band_energies_raw: np.ndarray = field(default_factory=lambda: np.zeros(7))

    # Mel-spaced band energies (32 bands, normalized 0-1)
    mel_energies: np.ndarray = field(default_factory=lambda: np.zeros(32))

    # Spectral features
    spectral_centroid: float = 0.0  # Hz — "brightness" of sound
    spectral_flux: float = 0.0  # Rate of spectral change
    spectral_rolloff: float = 0.0  # Frequency below which 85% energy lives
    spectral_flatness: float = 0.0  # 0=tonal, 1=noise-like
    superflux_onset: float = 0.0  # SuperFlux onset strength (log-compressed, max-filtered)

    # Amplitude
    rms: float = 0.0  # Root mean square energy (normalized 0-1)
    rms_raw: float = 0.0  # Raw RMS before sliding-window normalization
    peak: float = 0.0  # Peak amplitude

    # Per-band power sums before auto-gain normalization (truly unnormalized)
    band_energies_unnorm: np.ndarray = field(default_factory=lambda: np.zeros(7))

    # Raw spectrum for visualization
    spectrum: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def bass_energy(self) -> float:
        """Combined sub-bass + bass energy."""
        return float(self.band_energies[0] + self.band_energies[1]) / 2.0

    @property
    def mid_energy(self) -> float:
        """Combined low-mid + mid + upper-mid energy."""
        return float(np.mean(self.band_energies[2:5]))

    @property
    def high_energy(self) -> float:
        """Combined presence + brilliance energy."""
        return float(np.mean(self.band_energies[5:7]))

    @property
    def bass_energy_raw(self) -> float:
        """Combined sub-bass + bass from unnormalized band power sums."""
        return float(self.band_energies_unnorm[0] + self.band_energies_unnorm[1]) / 2.0


class AudioAnalyzer:
    """
    Real-time FFT analyzer with frequency band extraction and spectral features.

    Uses Hann-windowed FFT with configurable size. Extracts 7 frequency band
    energies, spectral centroid/flux/rolloff/flatness, and RMS.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        fft_size: int = 2048,
        bass_boost: float = 2.0,
        hop_size: int = 1024,
    ):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.bass_boost = bass_boost
        self.hop_size = hop_size

        # Pre-compute Hann window
        self._window = np.hanning(fft_size)

        # Pre-compute frequency bin edges for each band
        self._band_slices = self._compute_band_slices()

        # Frequency array for centroid/rolloff calculations
        self._freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

        # Previous frame for STFT 50% overlap
        self._prev_frame: np.ndarray | None = None

        # Previous spectrum for flux calculation
        self._prev_magnitude: np.ndarray | None = None

        # Running normalization: track max energy per band over ~5 seconds
        self._band_max = np.ones(7) * 1e-6
        self._band_max_decay = 0.995  # Slow decay for auto-gain

        # --- Mel filterbank (32 bands, 20 Hz to 20 kHz) ---
        self._n_mel_bands = 32
        self._mel_filterbank = self._compute_mel_filterbank(
            n_filters=self._n_mel_bands,
            f_min=20.0,
            f_max=min(20000.0, sample_rate / 2.0),
        )
        self._mel_max = np.ones(self._n_mel_bands) * 1e-6
        self._mel_max_decay = 0.995  # Same decay as 7-band system

        # --- SuperFlux onset detection (Böck & Widmer, DAFx 2013) ---
        self._prev_mel_magnitude: np.ndarray | None = None

        # Sliding-window RMS normalization (~5 seconds at audio frame rate)
        self._rms_window_size = int(5.0 * sample_rate / hop_size)  # ~215 frames at 44100/1024
        self._rms_history: deque[float] = deque(maxlen=self._rms_window_size)
        self._rms_floor = 1e-4  # Minimum RMS to avoid division by zero in silence

    def analyze(self, frame: np.ndarray) -> AudioFeatures:
        """
        Analyze a single audio frame.

        Args:
            frame: Normalized float32 audio samples in [-1, 1].
                   Length can be any size; will be zero-padded or truncated to fft_size.

        Returns:
            AudioFeatures with all extracted features.
        """
        features = AudioFeatures()

        # Amplitude features (from raw frame, before windowing)
        raw_rms = float(np.sqrt(np.mean(frame**2)))
        features.rms_raw = raw_rms
        features.peak = float(np.max(np.abs(frame)))

        # Sliding-window RMS normalization (volume-independent)
        self._rms_history.append(raw_rms)

        if len(self._rms_history) > 10:
            rms_min = min(self._rms_history)
            rms_max = max(self._rms_history)
            rms_range = max(rms_max - rms_min, self._rms_floor)
            features.rms = max(0.0, min(1.0, (raw_rms - rms_min) / rms_range))
        else:
            features.rms = raw_rms

        # STFT with 50% overlap: concatenate previous frame with current frame
        # to get a full fft_size window. On first call, zero-pad for compatibility.
        if self._prev_frame is not None:
            stft_frame = np.concatenate([self._prev_frame[-self.fft_size + len(frame):], frame])
        else:
            stft_frame = np.zeros(self.fft_size)
            stft_frame[-len(frame):] = frame

        # Store current frame for next overlap
        self._prev_frame = frame.copy()

        # Truncate or pad to exact fft_size (safety for variable-length inputs)
        if len(stft_frame) < self.fft_size:
            padded = np.zeros(self.fft_size)
            padded[-len(stft_frame):] = stft_frame
            stft_frame = padded
        elif len(stft_frame) > self.fft_size:
            stft_frame = stft_frame[-self.fft_size:]

        # Apply Hann window
        windowed = stft_frame * self._window

        # FFT → magnitude spectrum (power spectrum for energy calculations)
        fft_complex = np.fft.rfft(windowed)
        magnitude = np.abs(fft_complex)
        power = magnitude**2

        # Store magnitude spectrum for visualization (dBFS scale, normalized by N/2)
        magnitude_db = 20 * np.log10(magnitude / (self.fft_size / 2) + 1e-10)
        features.spectrum = magnitude_db

        # Band energies
        raw_energies = np.zeros(7)
        for i, (_, (start, end)) in enumerate(self._band_slices.items()):
            band_power = power[start:end]
            if len(band_power) > 0:
                raw_energies[i] = np.sum(band_power)

        # Store truly unnormalized band power sums (for section detection)
        features.band_energies_unnorm = raw_energies.copy()

        # Auto-gain normalization per band (before bass boost so boost is visible)
        self._band_max = np.maximum(
            raw_energies, self._band_max * self._band_max_decay
        )
        normalized = raw_energies / (self._band_max + 1e-10)

        # Store raw (pre-boost) for UI visualization
        features.band_energies_raw = normalized.copy()

        # Apply bass boost AFTER normalization (Fletcher-Munson compensation)
        # This makes bass stronger relative to other bands for the effect engine
        normalized[0] = min(normalized[0] * self.bass_boost, 1.5)  # sub-bass
        normalized[1] = min(normalized[1] * self.bass_boost, 1.5)  # bass
        features.band_energies = normalized

        # --- Mel filterbank energies (32 perceptually-spaced bands) ---
        raw_mel = self._mel_filterbank @ power
        self._mel_max = np.maximum(raw_mel, self._mel_max * self._mel_max_decay)
        features.mel_energies = raw_mel / (self._mel_max + 1e-10)

        # --- SuperFlux onset detection (Böck & Widmer 2013) ---
        # Mel-domain magnitude (not power) with log compression
        mel_magnitude = self._mel_filterbank @ magnitude
        log_mel = np.log1p(100.0 * mel_magnitude)

        if self._prev_mel_magnitude is not None:
            prev_log_mel = np.log1p(100.0 * self._prev_mel_magnitude)
            # Max filter (size=3) across frequency bins — suppresses vibrato
            padded = np.pad(prev_log_mel, (1, 1), mode='edge')
            max_prev = np.maximum(
                np.maximum(padded[:-2], padded[1:-1]),
                padded[2:]
            )
            # Half-wave rectified spectral flux
            diff = log_mel - max_prev
            features.superflux_onset = float(np.sum(np.maximum(0, diff)))

        self._prev_mel_magnitude = mel_magnitude.copy()

        # Spectral centroid: weighted mean frequency
        mag_sum = np.sum(magnitude)
        if mag_sum > 1e-10:
            features.spectral_centroid = float(
                np.sum(self._freqs * magnitude) / mag_sum
            )

        # Spectral flux: sum of positive magnitude differences
        if self._prev_magnitude is not None:
            diff = magnitude - self._prev_magnitude
            features.spectral_flux = float(np.sum(np.maximum(0, diff)))
        self._prev_magnitude = magnitude.copy()

        # Spectral rolloff: freq below which 85% of energy is concentrated
        cumulative = np.cumsum(power)
        total = cumulative[-1] if len(cumulative) > 0 else 0
        if total > 0:
            rolloff_idx = np.searchsorted(cumulative, 0.85 * total)
            features.spectral_rolloff = float(
                self._freqs[min(rolloff_idx, len(self._freqs) - 1)]
            )

        # Spectral flatness: geometric mean / arithmetic mean
        mag_positive = magnitude[magnitude > 0]
        if len(mag_positive) > 10:
            log_mean = np.mean(np.log(mag_positive + 1e-10))
            geo_mean = np.exp(log_mean)
            arith_mean = np.mean(mag_positive)
            features.spectral_flatness = float(
                np.clip(geo_mean / (arith_mean + 1e-10), 0, 1)
            )

        return features

    def _compute_band_slices(self) -> dict[str, tuple[int, int]]:
        """Pre-compute FFT bin index ranges for each frequency band."""
        slices = {}
        freq_per_bin = self.sample_rate / self.fft_size

        for band_name, (low_hz, high_hz) in FREQUENCY_BANDS.items():
            start_bin = max(1, round(low_hz / freq_per_bin))
            end_bin = min(self.fft_size // 2, round(high_hz / freq_per_bin) + 1)
            slices[band_name] = (start_bin, end_bin)

        return slices

    def _compute_mel_filterbank(
        self,
        n_filters: int = 32,
        f_min: float = 20.0,
        f_max: float = 20000.0,
    ) -> np.ndarray:
        """Pre-compute Mel filterbank matrix (n_filters x n_fft_bins).

        Creates overlapping triangular filters evenly spaced on the Mel scale.
        Standard Mel formula: mel = 2595 * log10(1 + f / 700).

        Args:
            n_filters: Number of Mel bands.
            f_min: Lowest filter center frequency (Hz).
            f_max: Highest filter center frequency (Hz), capped at Nyquist.

        Returns:
            Filter matrix of shape (n_filters, fft_size // 2 + 1).
        """
        n_fft_bins = self.fft_size // 2 + 1

        # Hz -> Mel conversion
        def hz_to_mel(f: float) -> float:
            return 2595.0 * np.log10(1.0 + f / 700.0)

        # Mel -> Hz conversion
        def mel_to_hz(m: float) -> float:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        # Create n_filters + 2 evenly spaced points in Mel scale
        # (extra 2 are the boundary points for the first and last filters)
        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
        hz_points = np.array([mel_to_hz(m) for m in mel_points])

        # Convert Hz points to FFT bin indices
        bin_points = np.round(hz_points * self.fft_size / self.sample_rate).astype(int)
        bin_points = np.clip(bin_points, 0, n_fft_bins - 1)

        # Build triangular filter matrix
        filterbank = np.zeros((n_filters, n_fft_bins))

        for i in range(n_filters):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising slope: left -> center
            if center > left:
                for j in range(left, center + 1):
                    filterbank[i, j] = (j - left) / (center - left)

            # Falling slope: center -> right
            if right > center:
                for j in range(center, right + 1):
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def reset(self) -> None:
        """Reset internal state (previous spectrum, normalization)."""
        self._prev_frame = None
        self._prev_magnitude = None
        self._prev_mel_magnitude = None
        self._band_max = np.ones(7) * 1e-6
        self._mel_max = np.ones(self._n_mel_bands) * 1e-6
        self._rms_history.clear()
