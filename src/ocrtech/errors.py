"""Project-specific exceptions."""

from __future__ import annotations


class OcrTechError(Exception):
    """Base class for all expected ocrtech failures."""


class DataValidationError(OcrTechError):
    """Raised when a dataset row, manifest, or schema is invalid."""


class EngineUnavailableError(OcrTechError):
    """Raised when an optional OCR engine is requested but unavailable."""


class ParseError(OcrTechError):
    """Raised when document parsing cannot complete."""


class BenchmarkError(OcrTechError):
    """Raised when a benchmark configuration is invalid."""


class ValidationError(OcrTechError):
    """Raised when a SOTA validation configuration is invalid."""


class PreflightError(OcrTechError):
    """Raised when a claim preflight configuration is invalid."""


class ClaimReviewError(OcrTechError):
    """Raised when a claim review bundle is invalid."""


class RemoteAuditError(OcrTechError):
    """Raised when a remote host audit cannot run or is misconfigured."""
