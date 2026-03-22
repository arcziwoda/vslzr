"""Custom exceptions for the VSLZR application."""


class HueVisualizerError(Exception):
    """Base exception for all VSLZR errors."""

    pass


class BridgeConnectionError(HueVisualizerError):
    """Raised when connection to the Hue Bridge fails."""

    pass


class BridgeDiscoveryError(HueVisualizerError):
    """Raised when bridge discovery fails."""

    pass


class EntertainmentAPIError(HueVisualizerError):
    """Raised when Entertainment API operations fail."""

    pass


class AudioCaptureError(HueVisualizerError):
    """Raised when audio capture fails."""

    pass


class ConfigurationError(HueVisualizerError):
    """Raised when configuration is invalid or missing."""

    pass
