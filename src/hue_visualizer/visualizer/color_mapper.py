"""Color mapper — maps audio features to brightness and saturation.

Two color modes:
- **palette** (default): genre palette drives base hue, centroid is ±20° offset modulator
- **centroid**: log-scaled spectral centroid maps directly to hue (100Hz→red 0° to 10kHz→violet 300°)

Both modes share:
- RMS energy → brightness (gamma-corrected, Weber-Fechner)
- Spectral flatness → saturation (tonal=vivid, noise=pastel)
- Spectral flux → color change speed (EMA alpha for hue smoothing)
"""

import math

from ..audio.analyzer import AudioFeatures

# Centroid-to-hue mapping constants (log scale)
_CENTROID_MIN_HZ = 100.0
_CENTROID_MAX_HZ = 10000.0
_CENTROID_HUE_RANGE = 300.0  # 0° (red) to 300° (violet)
_LOG_CENTROID_MIN = math.log(_CENTROID_MIN_HZ)
_LOG_CENTROID_MAX = math.log(_CENTROID_MAX_HZ)
_LOG_CENTROID_RANGE = _LOG_CENTROID_MAX - _LOG_CENTROID_MIN

# Valid color modes
COLOR_MODE_PALETTE = "palette"
COLOR_MODE_CENTROID = "centroid"
COLOR_MODES = (COLOR_MODE_PALETTE, COLOR_MODE_CENTROID)


def centroid_to_hue(centroid_hz: float) -> float:
    """Map spectral centroid frequency to hue using log scale.

    100 Hz → 0° (red/warm), 10 kHz → 300° (violet/cool).
    Clamped to [0, 300]. Bass-heavy music produces warm colors,
    treble-heavy music produces cool colors.
    """
    if centroid_hz <= _CENTROID_MIN_HZ:
        return 0.0
    if centroid_hz >= _CENTROID_MAX_HZ:
        return _CENTROID_HUE_RANGE
    t = (math.log(centroid_hz) - _LOG_CENTROID_MIN) / _LOG_CENTROID_RANGE
    return t * _CENTROID_HUE_RANGE


class ColorMapper:
    """Maps audio features to brightness, saturation, and hue offset (or direct hue).

    In **palette mode** (default): returns (hue_offset, saturation, brightness)
    where hue_offset is a ±20° centroid-driven modulation of the palette hue.

    In **centroid mode**: returns (centroid_hue, saturation, brightness)
    where centroid_hue is a direct 0-300° mapping from spectral centroid.
    """

    def __init__(
        self,
        gamma: float = 2.2,
        saturation_alpha: float = 0.15,
        color_mode: str = COLOR_MODE_PALETTE,
        saturation_boost: float = 1.0,
    ):
        self.gamma = gamma
        self.saturation_alpha = saturation_alpha
        self._color_mode = color_mode if color_mode in COLOR_MODES else COLOR_MODE_PALETTE

        # Saturation multiplier (Task 2.19): 0.0 = grayscale, 1.0 = full color
        self._saturation_boost: float = max(0.0, min(1.0, saturation_boost))

        self._saturation = 0.8
        self._brightness = 0.0
        self._hue_offset = 0.0  # palette mode: centroid-driven, ±20°
        self._centroid_hue = 180.0  # centroid mode: smoothed direct hue
        self._flux_ema: float = 0.0

    @property
    def color_mode(self) -> str:
        """Current color mode: 'palette' or 'centroid'."""
        return self._color_mode

    def set_color_mode(self, mode: str) -> None:
        """Switch color mode. Valid values: 'palette', 'centroid'."""
        if mode in COLOR_MODES:
            self._color_mode = mode

    @property
    def saturation_boost(self) -> float:
        """Current saturation multiplier (0.0 = grayscale, 1.0 = full)."""
        return self._saturation_boost

    def set_saturation_boost(self, value: float) -> None:
        """Set saturation multiplier (Task 2.19).

        Args:
            value: Multiplier 0.0 to 1.0. 0.0 = grayscale, 0.5 = pastel,
                   1.0 = full saturation (default).
        """
        self._saturation_boost = max(0.0, min(1.0, value))

    def map(self, features: AudioFeatures) -> tuple[float, float, float]:
        """Map audio features to color values.

        Returns:
            In palette mode: (hue_offset -20..+20, saturation 0-1, brightness 0-1)
            In centroid mode: (centroid_hue 0-300, saturation 0-1, brightness 0-1)
        """
        centroid = features.spectral_centroid

        # Flux-driven smoothing speed (shared by both modes)
        flux = features.spectral_flux
        self._flux_ema = _ema(self._flux_ema, flux, 0.15)
        flux_normalized = min(1.0, self._flux_ema / 50.0)

        if self._color_mode == COLOR_MODE_CENTROID:
            # --- Centroid mode: direct hue mapping ---
            target_hue = centroid_to_hue(centroid)
            # Flux-driven smoothing: faster changes when spectral flux is high
            hue_alpha = 0.05 + flux_normalized * 0.25  # 0.05..0.30
            self._centroid_hue = _smooth_hue_range(
                self._centroid_hue, target_hue, hue_alpha, max_val=360.0
            )
            hue_result = self._centroid_hue
        else:
            # --- Palette mode: ±20° offset ---
            if centroid > 1.0:
                centroid_clamped = max(100.0, min(10000.0, centroid))
                t = (math.log10(centroid_clamped) - 2.0) / 2.0  # 0..1
                target_offset = (t - 0.5) * 40  # -20..+20
            else:
                target_offset = 0.0
            offset_alpha = 0.03 + flux_normalized * 0.2  # 0.03..0.23
            self._hue_offset = _ema(self._hue_offset, target_offset, offset_alpha)
            hue_result = self._hue_offset

        # --- Saturation from spectral flatness (inverse) ---
        energy_boost = min(0.15, features.rms * 0.3)
        target_sat = max(0.25, min(1.0, 1.0 - features.spectral_flatness * 0.7 + energy_boost))
        self._saturation = _ema(self._saturation, target_sat, self.saturation_alpha)

        # Apply saturation multiplier (Task 2.19): scales output saturation.
        # Smoothed internally first, then multiplied, so the multiplier acts
        # as a clean scaling factor that doesn't interfere with EMA dynamics.
        final_saturation = self._saturation * self._saturation_boost

        # --- Brightness from RMS (gamma-corrected, no EMA — engine handles smoothing) ---
        self._brightness = min(1.0, features.rms ** (1.0 / self.gamma))

        return (hue_result, final_saturation, self._brightness)

    def reset(self) -> None:
        self._saturation = 0.8
        self._brightness = 0.0
        self._hue_offset = 0.0
        self._centroid_hue = 180.0
        self._flux_ema = 0.0


def _ema(current: float, target: float, alpha: float) -> float:
    """Exponential moving average."""
    return alpha * target + (1.0 - alpha) * current


def _smooth_hue(current: float, target: float, alpha: float) -> float:
    """Smooth hue with shortest-path interpolation on circular 0-360 scale."""
    diff = target - current
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return (current + alpha * diff) % 360


def _smooth_hue_range(
    current: float, target: float, alpha: float, max_val: float = 360.0
) -> float:
    """Smooth a value on a circular scale with shortest-path interpolation.

    Like _smooth_hue but works on any circular range [0, max_val).
    Used for centroid hue smoothing where the range is [0, 300] but
    wrapping is still desired for smooth transitions.
    """
    half = max_val / 2.0
    diff = target - current
    if diff > half:
        diff -= max_val
    elif diff < -half:
        diff += max_val
    result = current + alpha * diff
    if result < 0:
        result += max_val
    elif result >= max_val:
        result -= max_val
    return result
