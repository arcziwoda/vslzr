"""Genre presets and color palettes.

Each genre has a default palette that is applied when the preset is selected.
Users can still override the palette manually via the palette buttons.

Algorithmic palette generation (Task 2.10):
  - complementary: 2 colors 180 degrees apart on the color wheel
  - triadic: 3 colors 120 degrees apart
  - analogous: 3 colors 30 degrees apart (tight cluster)

Research basis for genre presets:
  - Palmer, Schloss, Xu & Prado-León (2013, PNAS): music-color via shared emotion
  - Spence (2011): crossmodal correspondences (louder→brighter, pitch→brightness)
  - Marks (1974): pitch-brightness mapping
  - Ward, Huckstep & Tsakanikos (2006): synesthesia pitch→hue
  - Stupacher, Hove & Janata (2016): spectral flux 0-200 Hz predicts groove
  - Vroomen & Keetels (2010): AV temporal binding window 100-300 ms
  - EMA coefficients derived for 50 Hz update: τ = -20ms / ln(1-α)
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
    # Genre-specific palettes (research-backed — Palmer et al. 2013, club conventions)
    "techno": (220.0, 240.0, 260.0, 0.0),      # deep azure, blue, blue-violet, red accent
    "house": (300.0, 330.0, 30.0, 270.0),      # magenta, rose/pink, amber/gold, purple
    "dnb": (120.0, 180.0, 240.0, 280.0),       # neon green, cyan, blue, violet
    "ambient": (240.0, 270.0, 200.0, 30.0),    # deep blue, purple, teal, warm amber accent
    "trap": (0.0, 270.0, 30.0, 330.0),         # red, purple, orange/gold, pink/rose
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
    # Strobe frequency in Hz (auto-strobe burst speed)
    strobe_frequency: float = 6.0


PRESETS: dict[str, GenrePreset] = {
    "techno": GenrePreset(
        name="techno",
        beat_cooldown_ms=250,       # catches every kick, filters 8th-note hats
        bass_boost=1.4,             # mid-bass punch (40-100 Hz), not sub-dominant
        bpm_min=118.0,
        bpm_max=150.0,
        attack_alpha=0.85,          # τ≈10.5ms — near-instant, matches kick A=0ms
        release_alpha=0.08,         # τ≈240ms — 52% of beat interval, distinct pulses
        spatial_mode="mirror",      # Berghain-style symmetric minimalism
        flash_tau=0.12,             # clean isolated pulses, 2% residual at next beat
        hue_drift_speed=4.0,        # full rotation ~90s, matches section length
        default_palette="techno",   # deep azure/blue/violet + red accent
        strobe_frequency=2.5,       # aggressive but below 3 Hz safety
    ),
    "house": GenrePreset(
        name="house",
        beat_cooldown_ms=280,       # groove-oriented, filters shaker 16ths
        bass_boost=1.2,             # warm round bass, balanced with mids/vocals
        bpm_min=110.0,
        bpm_max=132.0,
        attack_alpha=0.70,          # τ≈16.6ms — softer than techno, breathes with groove
        release_alpha=0.07,         # τ≈276ms — warm sustained glow between kicks
        spatial_mode="wave",        # flowing wash, disco mirror-ball heritage
        flash_tau=0.15,             # warm tail, 4% residual at next beat
        hue_drift_speed=12.0,       # colorful, matches faster harmonic rhythm
        default_palette="house",    # magenta, rose, amber/gold, purple — disco heritage
        strobe_frequency=1.5,       # gentle, groove-friendly
    ),
    "dnb": GenrePreset(
        name="dnb",
        beat_cooldown_ms=150,       # catches kick+snare two-step (min gap ~172ms)
        bass_boost=1.6,             # massive sub from Reese/modulated basslines
        bpm_min=155.0,
        bpm_max=185.0,
        attack_alpha=0.92,          # τ≈7.9ms — fastest, tracks 20-40 transients/bar
        release_alpha=0.14,         # τ≈133ms — sharp staccato, 7% residual at next hit
        spatial_mode="frequency_zones",  # stratified: sub, mid-bass processing, hats
        flash_tau=0.08,             # razor-sharp pulses, 1.3% at next quarter note
        hue_drift_speed=18.0,       # fast rotation ~20s, matches high energy
        default_palette="dnb",      # neon green, cyan, blue, violet — laser convention
        strobe_frequency=3.0,       # max safe — DnB drops demand intensity
    ),
    "ambient": GenrePreset(
        name="ambient",
        beat_cooldown_ms=600,       # only major events trigger, filters micro-transients
        bass_boost=0.8,             # below neutral — timbral, not percussive
        bpm_min=50.0,
        bpm_max=125.0,
        attack_alpha=0.15,          # τ≈123ms — glacial fade-in, mirrors pad attacks
        release_alpha=0.007,        # τ≈2850ms — ~3s decay, luminous respiration
        spatial_mode="wave",        # slow immersive drift, Eliasson-style
        flash_tau=0.80,             # slow bloom and fade, 29% at 1s
        hue_drift_speed=1.5,        # full rotation ~4min, glacial harmonic evolution
        default_palette="ambient",  # deep blue, purple, teal, warm amber accent
        strobe_frequency=0.0,       # no strobe — antithetical to ambient
    ),
    "trap": GenrePreset(
        name="trap",
        beat_cooldown_ms=350,       # half-time feel (~857ms between main hits)
        bass_boost=2.0,             # highest — 808 sub-bass is the genre identity
        bpm_min=60.0,              # locks PLL to half-time felt tempo
        bpm_max=90.0,              # prevents double-time lock at 120-180
        attack_alpha=0.80,          # τ≈12.4ms — snaps to 808 transient
        release_alpha=0.05,         # τ≈390ms — sustains with 808 decay (300ms-2s)
        spatial_mode="mirror",      # center-focused mix (808/kick/vocals mono)
        flash_tau=0.25,             # heavy sustained impact, visual 808 "boom"
        hue_drift_speed=6.0,        # moderate, matches verse/hook section pace
        default_palette="trap",     # red, purple, orange/gold, pink — high power
        strobe_frequency=2.0,       # powerful but spacious, not frantic
    ),
}

DEFAULT_GENRE = "techno"
