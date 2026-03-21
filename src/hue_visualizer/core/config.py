"""Configuration management using Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Hue Bridge Configuration (optional — audio-only mode works without these)
    bridge_ip: str | None = Field(
        default=None,
        description="IP address of the Philips Hue Bridge",
        examples=["192.168.1.100"],
    )
    hue_username: str | None = Field(
        default=None,
        description="Authentication token obtained from bridge pairing",
        examples=["1234567890abcdef"],
    )
    hue_clientkey: str | None = Field(
        default=None,
        description="Entertainment API client key from bridge pairing",
    )
    entertainment_area_id: str | None = Field(
        default=None,
        description="ID of the pre-configured entertainment area",
        examples=["1"],
    )

    # Visualizer
    num_lights: int = Field(
        default=6,
        description="Number of lights for effect engine (used for preview when bridge not connected)",
        ge=1,
        le=20,
    )
    spatial_mode: str = Field(
        default="frequency_zones",
        description="Spatial distribution mode: uniform, frequency_zones, wave, mirror, chase",
    )

    # Audio Configuration
    audio_device_index: int | None = Field(
        default=None,
        description="PyAudio input device index (None = system default). "
        "Use GET /api/audio/devices to list available devices.",
    )
    buffer_size: int = Field(
        default=1024,
        description="Audio buffer size / hop size (samples per frame)",
        ge=256,
        le=4096,
    )
    sample_rate: int = Field(
        default=44100,
        description="Audio sample rate in Hz",
    )
    fft_size: int = Field(
        default=2048,
        description="FFT window size (typically 2x buffer_size for overlap)",
        ge=512,
        le=8192,
    )

    # Performance Configuration
    fps_target: int = Field(
        default=50,
        description="Target frames per second for light updates (bridge caps at 25Hz)",
        ge=25,
        le=60,
    )

    # Beat Detection Configuration
    beat_cooldown_ms: int = Field(
        default=300,
        description="Minimum inter-beat interval in milliseconds",
        ge=50,
        le=1000,
    )
    bpm_min: float = Field(
        default=80.0,
        description="Minimum expected BPM (octave error protection)",
        ge=30.0,
        le=200.0,
    )
    bpm_max: float = Field(
        default=180.0,
        description="Maximum expected BPM (octave error protection)",
        ge=60.0,
        le=300.0,
    )

    # Smoothing Configuration
    attack_alpha: float = Field(
        default=0.7,
        description="EMA alpha for rising values (fast attack, 0.5-0.8)",
        ge=0.1,
        le=1.0,
    )
    release_alpha: float = Field(
        default=0.1,
        description="EMA alpha for falling values (slow release, 0.05-0.15)",
        ge=0.01,
        le=0.5,
    )
    brightness_gamma: float = Field(
        default=2.2,
        description="Gamma correction for brightness mapping (Weber-Fechner, 2.0-2.5)",
        ge=1.0,
        le=4.0,
    )
    bass_boost_factor: float = Field(
        default=2.0,
        description="Bass energy multiplier to compensate Fletcher-Munson curve (1.5-3.0)",
        ge=1.0,
        le=5.0,
    )

    # Predictive Beat Triggering
    latency_compensation_ms: float = Field(
        default=80.0,
        description="Latency compensation for predictive beat triggering in milliseconds. "
        "Fires light commands this many ms before the predicted beat to compensate "
        "for end-to-end system latency (audio → DTLS → bridge → bulb). "
        "Set to 0 to disable predictive triggering.",
        ge=0.0,
        le=200.0,
    )
    predictive_confidence_threshold: float = Field(
        default=0.6,
        description="Minimum PLL confidence required to use predictive triggering. "
        "Below this threshold, falls back to reactive beats only.",
        ge=0.0,
        le=1.0,
    )

    # Generative Layer Configuration (Task 1.1)
    generative_hue_cycle_period: float = Field(
        default=45.0,
        description="Period in seconds for full hue rotation in the generative layer (30-60s typical)",
        ge=5.0,
        le=120.0,
    )
    generative_breathing_rate_hz: float = Field(
        default=0.25,
        description="Breathing oscillation frequency in Hz (0.15-0.35 typical)",
        ge=0.05,
        le=1.0,
    )
    generative_breathing_min: float = Field(
        default=0.20,
        description="Minimum brightness for generative breathing effect (0-1)",
        ge=0.0,
        le=0.5,
    )
    generative_breathing_max: float = Field(
        default=0.80,
        description="Maximum brightness for generative breathing effect (0-1)",
        ge=0.3,
        le=1.0,
    )

    # Calibration delay (Task 2.6)
    calibration_delay_ms: float = Field(
        default=0.0,
        description="Manual calibration delay in milliseconds (-200 to 600). "
        "Positive = lights fire earlier (compensate for late lights). "
        "Negative = lights fire later (compensate for early lights, e.g. loopback without bridge). "
        "Adds to the predictive beat latency compensation.",
        ge=-200.0,
        le=600.0,
    )

    # Brightness min/max (Task 2.8)
    brightness_min: float = Field(
        default=0.0,
        description="Minimum brightness floor (0-1). Lights never go below this value.",
        ge=0.0,
        le=1.0,
    )
    brightness_max: float = Field(
        default=1.0,
        description="Maximum brightness cap (0-1). Lights never exceed this value.",
        ge=0.0,
        le=1.0,
    )

    # Safety Configuration
    max_flash_hz: float = Field(
        default=3.0,
        description="Maximum flash rate in Hz (epilepsy safety limit)",
        ge=1.0,
        le=5.0,
    )

    # Web Server Configuration
    server_host: str = Field(
        default="0.0.0.0",
        description="Web server bind address",
    )
    server_port: int = Field(
        default=8080,
        description="Web server port",
        ge=1024,
        le=65535,
    )
