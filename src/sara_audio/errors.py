"""Domain-specific failures surfaced by the command-line interface."""


class SaraAudioError(Exception):
    """Base exception for expected extraction failures."""


class DependencyUnavailableError(SaraAudioError):
    """Raised when an optional runtime dependency has not been installed."""


class UnsupportedAudioError(SaraAudioError):
    """Raised when input audio cannot be decoded by the available backend."""


class RegistryValidationError(SaraAudioError):
    """Raised when the 92-feature registry is incomplete or malformed."""
