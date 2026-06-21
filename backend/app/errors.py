"""
errors.py — Application-specific exceptions.

These let the infrastructure layer signal precise failure conditions that the
API layer (main.py) can translate into the right HTTP status without leaking
vendor-specific exception types (qdrant_client.*, requests.*) into the routes.
"""


class AppError(Exception):
    """Base class for all application-raised errors."""


class ConfigError(AppError):
    """
    A required configuration value is missing or invalid.

    Raised, for example, when QDRANT_URL is not set — there is no longer an
    in-memory fallback, so a running Qdrant server is mandatory.
    """


class QdrantUnavailableError(AppError):
    """
    The Qdrant server could not be reached or returned an unexpected response.

    The API layer maps this to HTTP 503 with the message
    "Qdrant server not working" instead of silently degrading to memory.
    """
