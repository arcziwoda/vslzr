"""
Color conversion utilities for Hue lights.

Provides conversion between different color spaces:
- RGB (standard 0-255)
- HSV (Hue: 0-360, Saturation: 0-1, Value: 0-1)
- XY (CIE 1931 color space used by Hue)
"""

import colorsys
from typing import Tuple


def rgb_to_xy(r: int, g: int, b: int) -> Tuple[float, float]:
    """
    Convert RGB to CIE XY color space (used by Hue lights).

    This implements the conversion algorithm recommended by Philips for
    the Hue bulb color gamut.

    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)

    Returns:
        Tuple of (x, y) coordinates in CIE color space (0.0 - 1.0)

    Example:
        >>> rgb_to_xy(255, 0, 0)  # Red
        (0.675, 0.322)
        >>> rgb_to_xy(0, 255, 0)  # Green
        (0.408, 0.517)
        >>> rgb_to_xy(0, 0, 255)  # Blue
        (0.167, 0.04)
    """
    # Normalize to 0-1 range
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    # Apply gamma correction
    r_gamma = _apply_gamma(r_norm)
    g_gamma = _apply_gamma(g_norm)
    b_gamma = _apply_gamma(b_norm)

    # Convert to XYZ using Wide RGB D65 conversion formula
    X = r_gamma * 0.664511 + g_gamma * 0.154324 + b_gamma * 0.162028
    Y = r_gamma * 0.283881 + g_gamma * 0.668433 + b_gamma * 0.047685
    Z = r_gamma * 0.000088 + g_gamma * 0.072310 + b_gamma * 0.986039

    # Avoid division by zero
    total = X + Y + Z
    if total == 0:
        # Return neutral white point for black
        return (0.3127, 0.3290)

    # Convert XYZ to xy
    x = X / total
    y = Y / total

    return (x, y)


def _rgb_float_to_xy(r: float, g: float, b: float) -> Tuple[float, float]:
    """Convert normalized RGB floats (0-1) to CIE XY, bypassing int quantization."""
    r_gamma = _apply_gamma(r)
    g_gamma = _apply_gamma(g)
    b_gamma = _apply_gamma(b)

    X = r_gamma * 0.664511 + g_gamma * 0.154324 + b_gamma * 0.162028
    Y = r_gamma * 0.283881 + g_gamma * 0.668433 + b_gamma * 0.047685
    Z = r_gamma * 0.000088 + g_gamma * 0.072310 + b_gamma * 0.986039

    total = X + Y + Z
    if total == 0:
        return (0.3127, 0.3290)

    return (X / total, Y / total)


def hsv_to_xy(h: float, s: float, v: float) -> Tuple[float, float]:
    """
    Convert HSV to CIE XY color space.

    Args:
        h: Hue (0.0 - 360.0 degrees)
        s: Saturation (0.0 - 1.0)
        v: Value/Brightness (0.0 - 1.0)

    Returns:
        Tuple of (x, y) coordinates in CIE color space

    Example:
        >>> hsv_to_xy(0, 1.0, 1.0)  # Red
        (0.675, 0.322)
        >>> hsv_to_xy(120, 1.0, 1.0)  # Green
        (0.408, 0.517)
    """
    # Convert HSV to RGB (float 0-1)
    h_norm = h / 360.0
    r, g, b = colorsys.hsv_to_rgb(h_norm, s, v)

    # Use float path directly — avoids int truncation to 0-255
    return _rgb_float_to_xy(r, g, b)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """
    Convert HSV to RGB.

    Args:
        h: Hue (0.0 - 360.0 degrees)
        s: Saturation (0.0 - 1.0)
        v: Value/Brightness (0.0 - 1.0)

    Returns:
        Tuple of (r, g, b) values (0-255)
    """
    h_norm = h / 360.0
    r, g, b = colorsys.hsv_to_rgb(h_norm, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert RGB to HSV.

    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)

    Returns:
        Tuple of (h, s, v) where:
        - h: Hue (0.0 - 360.0)
        - s: Saturation (0.0 - 1.0)
        - v: Value (0.0 - 1.0)
    """
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    return (h * 360.0, s, v)


def _apply_gamma(value: float) -> float:
    """
    Apply gamma correction for RGB to XYZ conversion.

    Args:
        value: Normalized RGB value (0.0 - 1.0)

    Returns:
        Gamma-corrected value
    """
    if value > 0.04045:
        return ((value + 0.055) / 1.055) ** 2.4
    else:
        return value / 12.92


# Common color presets in XY space
class ColorPresets:
    """Common color presets in CIE XY space."""

    # Basic colors
    RED = (0.675, 0.322)
    GREEN = (0.408, 0.517)
    BLUE = (0.167, 0.04)
    YELLOW = (0.475, 0.499)
    CYAN = (0.158, 0.237)
    MAGENTA = (0.385, 0.155)
    WHITE = (0.3127, 0.3290)
    WARM_WHITE = (0.45, 0.41)
    COOL_WHITE = (0.31, 0.33)

    # Orange/Amber tones
    ORANGE = (0.585, 0.388)
    AMBER = (0.568, 0.417)

    # Purple/Violet tones
    PURPLE = (0.271, 0.163)
    VIOLET = (0.245, 0.123)

    # Pink tones
    PINK = (0.465, 0.244)
    HOT_PINK = (0.535, 0.275)

    @classmethod
    def get_by_name(cls, name: str) -> Tuple[float, float]:
        """
        Get color preset by name (case-insensitive).

        Args:
            name: Color name (e.g., "red", "blue", "warm_white")

        Returns:
            Tuple of (x, y) coordinates

        Raises:
            ValueError: If color name not found
        """
        name_upper = name.upper()
        if hasattr(cls, name_upper):
            return getattr(cls, name_upper)
        raise ValueError(f"Color preset '{name}' not found")
