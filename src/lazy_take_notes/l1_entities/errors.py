"""Domain error types."""


class ModelResolutionError(Exception):
    """Raised when a whisper model cannot be resolved to a local path."""


class DigestFailedError(Exception):
    """Raised when a digest cycle fails."""
