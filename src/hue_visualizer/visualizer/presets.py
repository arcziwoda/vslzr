"""Genre presets and color palettes.

Each genre has a default palette that is applied when the preset is selected.
Users can still override the palette manually via the palette buttons.

Algorithmic palette generation (Task 2.10):
  - complementary: 2 colors 180 degrees apart on the color wheel
  - triadic: 3 colors 120 degrees apart
  - analogous: 3 colors 30 degrees apart (tight cluster)
"""

from dataclasses import dataclass


# --- Color Palettes (can be selected independently, but genres have defaults) ---

PALETTES: dict[str, tuple[float, ...]] = {
    # General palettes
    "neon": (280.0, 240.0, 0.0, 320.0),       # deep purple, midnight blue, blood red, magenta
    "warm": (35.0, 15.0, 55.0, 5.0),           # amber, coral, gold, red-orange
    "cool": (200.0, 220.0, 260.0, 180.0),      # ice blue, sky, lavender, teal
    "fire": (0.0, 20.0, 40.0, 350.0),          # red, orange, amber, crimson
    "forest": (120.0, 90.0, 60.0, 150.0),      # green, lime, yellow-green, seafoam
    "ocean": (200.0, 180.0, 240.0, 160.0),     # blue, cyan, violet, aqua
    "sunset": (15.0, 35.0, 300.0, 340.0),      # coral, amber, pink, rose
    "monochrome": (240.0, 230.0, 250.0, 220.0),  # blue shades — subtle variation
    # Genre-specific palettes (from research)
    "techno": (280.0, 240.0, 0.0),             # deep purple, midnight blue, blood red
    "house": (35.0, 15.0, 180.0),              # warm amber, coral, cyan
    "dnb": (120.0, 60.0, 210.0),               # neon green, yellow, electric blue
    "ambient": (200.0, 270.0, 220.0),          # ice blue, lavender, soft sky
}

DEFAULT_PALETTE = "neon"

# --- Algorithmic palette generation (Task 2.10) ---

PALETTE_ALGO_MODES = ("complementary", "triadic", "analogous")


def generate_palette(mode: str, base_hue: float) -> tuple[float, ...]:
    """Generate a harmonious palette algorithmically from a base hue.

    Args:
        mode: One of 'complementary', 'triadic', or 'analogous'.
        base_hue: Base hue in degrees (0-360). Will be normalized.

    Returns:
        Tuple of hue values (degrees, 0-360). Length depends on mode:
        - complementary: 4 hues (base, base+180, plus two intermediate accents)
        - triadic: 3 hues (base, base+120, base+240)
        - analogous: 3 hues (base-30, base, base+30)

    Raises:
        ValueError: If mode is not a valid algorithm name.
    """
    if mode not in PALETTE_ALGO_MODES:
        raise ValueError(
            f"Unknown palette algorithm '{mode}'. "
            f"Valid modes: {PALETTE_ALGO_MODES}"
        )

    base_hue = base_hue % 360.0

    if mode == "complementary":
        # Two opposite hues plus two intermediate accent hues (at +/-90)
        # for a richer 4-color palette. The accents are offset slightly
        # (80 and 260 instead of 90 and 270) for a more pleasing result.
        return (
            base_hue,
            (base_hue + 180.0) % 360.0,
            (base_hue + 80.0) % 360.0,
            (base_hue + 260.0) % 360.0,
        )

    if mode == "triadic":
        # Three equally spaced hues — high contrast, balanced.
        return (
            base_hue,
            (base_hue + 120.0) % 360.0,
            (base_hue + 240.0) % 360.0,
        )

    # analogous: tight cluster of hues for a cohesive, harmonious look.
    return (
        (base_hue - 30.0) % 360.0,
        base_hue,
        (base_hue + 30.0) % 360.0,
    )


# --- Genre Presets (beat detection, smoothing, spatial + default palette) ---

@dataclass(frozen=True)
class GenrePreset:
    """Parameter set for a music genre, including default color palette."""

    name: str
    # Beat detection
    beat_cooldown_ms: float
    bass_boost: float
    # BPM range (octave error protection)
    bpm_min: float
    bpm_max: float
    # Smoothing
    attack_alpha: float
    release_alpha: float
    # Spatial
    spatial_mode: str
    # Flash: exponential decay time constant (seconds). Lower = snappier.
    flash_tau: float
    # Hue drift speed (degrees/second for generative base)
    hue_drift_speed: float
    # Default palette name (key into PALETTES dict)
    default_palette: str = "neon"


PRESETS: dict[str, GenrePreset] = {
    "techno": GenrePreset(
        name="techno",
        beat_cooldown_ms=300,
        bass_boost=2.5,
        bpm_min=120.0,
        bpm_max=150.0,
        attack_alpha=0.8,
        release_alpha=0.15,
        spatial_mode="frequency_zones",
        flash_tau=0.20,  # Snappy — 200ms decay
        hue_drift_speed=4.0,
        default_palette="techno",  # deep purple, midnight blue, blood red
    ),
    "house": GenrePreset(
        name="house",
        beat_cooldown_ms=320,
        bass_boost=2.0,
        bpm_min=115.0,
        bpm_max=135.0,
        attack_alpha=0.65,
        release_alpha=0.12,
        spatial_mode="frequency_zones",
        flash_tau=0.30,  # Warm glow — 300ms decay
        hue_drift_speed=8.0,
        default_palette="house",  # warm amber, coral, cyan
    ),
    "dnb": GenrePreset(
        name="dnb",
        beat_cooldown_ms=300,
        bass_boost=3.0,
        bpm_min=155.0,
        bpm_max=185.0,
        attack_alpha=0.9,
        release_alpha=0.15,
        spatial_mode="wave",
        flash_tau=0.15,  # Very snappy — 150ms for fast beats
        hue_drift_speed=10.0,
        default_palette="dnb",  # neon green, yellow, electric blue
    ),
    "ambient": GenrePreset(
        name="ambient",
        beat_cooldown_ms=500,
        bass_boost=1.5,
        bpm_min=60.0,
        bpm_max=120.0,
        attack_alpha=0.3,
        release_alpha=0.05,
        spatial_mode="uniform",
        flash_tau=0.50,  # Slow gentle pulse — 500ms
        hue_drift_speed=2.0,
        default_palette="ambient",  # ice blue, lavender, soft sky
    ),
}

DEFAULT_GENRE = "techno"
