"""Custom exceptions for the application."""

class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass

class EmbeddingError(Exception):
    """Raised when embedding operations fail."""
    pass

class LLMError(Exception):
    """Raised when LLM operations fail."""
    pass

class ValidationError(Exception):
    """Raised when request validation fails."""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass