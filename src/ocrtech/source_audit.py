"""Source audits for real-world validation data."""

from __future__ import annotations

import csv
import hashlib
import http.client
import json
import logging
import random
import re
import shutil
import socket
import time
from dataclasses import dataclass, field
from html import escape as html_escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from .errors import DataValidationError
from .limbu_legacy import DEFAULT_LIMBU_LEGACY_MAP, LimbuLegacyConverter
from .manifest import ManifestEntry, load_manifest, sha256_file, sha256_text, write_manifest


DEFAULT_GORKHAPATRA_CATEGORY_URLS = (
    "https://gorkhapatraonline.com/categories/new-nepal",
    "https://gorkhapatraonline.com/categories/new-nepal?page=2",
)

DEFAULT_GORKHAPATRA_EPAPER_URL = (
    "https://epaper.gorkhapatraonline.com/epaper-list?"
    "pdfdaterange1=2026-06-09&pdfdaterange2=2026-06-16&slug=gorkhapatra"
)

GORKHAPATRA_ONLINE_BASE = "https://gorkhapatraonline.com"
GORKHAPATRA_EPAPER_BASE = "https://epaper.gorkhapatraonline.com"
DEFAULT_NAYA_NEPAL_PUBLICATION_URL = "https://gorkhapatraonline.com/publications/naya-nepal"
LANGUAGE_PAGE_MARKERS = ("भाषा पृष्ठ", "efiff k[i7", "bhasa pristha", "bhasa prishtha")
LOGGER = logging.getLogger(__name__)
_NETWORK_RETRY_DELAYS_SECONDS = (2.0, 8.0, 30.0)
_NETWORK_RETRY_JITTER_SECONDS = 0.5
# The Gorkhapatra epaper WAF intermittently resets connections from bare tool
# user agents (verified 2026-07-05: "ocrtech-source-audit/0.1" gets TCP resets
# while browser UAs pass), so present a browser-style UA with the tool name
# kept as a suffix for honest server logs.
_SOURCE_AUDIT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 ocrtech-source-audit/0.1"
)
_TRANSIENT_NETWORK_EXCEPTIONS = (
    URLError,
    ConnectionResetError,
    TimeoutError,
    socket.timeout,
    http.client.HTTPException,
)
_T = TypeVar("_T")
NEPALI_MONTHS = {
    "बैशाख": 1,
    "वैशाख": 1,
    "जेठ": 2,
    "असार": 3,
    "साउन": 4,
    "भदौ": 5,
    "भाद्र": 5,
    "असोज": 6,
    "कार्तिक": 7,
    "कात्तिक": 7,
    "मंसिर": 8,
    "मङ्सिर": 8,
    "पुस": 9,
    "पुष": 9,
    "माघ": 10,
    "फागुन": 11,
    "चैत": 12,
}


@dataclass(slots=True)
class DownloadedArtifact:
    url: str
    path: str
    sha256: str
    size_bytes: int
    kind: str
    warnings: list[str] = field(default_factory=list)
    pdf_audit: dict[str, Any] | None = None
    reused_from_disk: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "url": self.url,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
            "warnings": self.warnings,
        }
        if self.pdf_audit is not None:
            payload["pdf_audit"] = self.pdf_audit
        if self.reused_from_disk:
            payload["reused_from_disk"] = True
        return payload


@dataclass(slots=True)
class GorkhapatraArticleAsset:
    article_url: str | None
    image_url: str | None
    title: str
    language: str | None
    source_page: str
    downloaded_image: DownloadedArtifact | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_url": self.article_url,
            "image_url": self.image_url,
            "title": self.title,
            "language": self.language,
            "source_page": self.source_page,
            "downloaded_image": self.downloaded_image.to_dict() if self.downloaded_image else None,
        }


@dataclass(slots=True)
class GorkhapatraEpaperAsset:
    viewer_url: str
    direct_pdf_url: str
    source_page: str
    date_label: str | None = None
    thumbnail_url: str | None = None
    downloaded_pdf: DownloadedArtifact | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewer_url": self.viewer_url,
            "direct_pdf_url": self.direct_pdf_url,
            "source_page": self.source_page,
            "date_label": self.date_label,
            "thumbnail_url": self.thumbnail_url,
            "downloaded_pdf": self.downloaded_pdf.to_dict() if self.downloaded_pdf else None,
        }


@dataclass(slots=True)
class GorkhapatraSourceAudit:
    category_sources: list[str]
    epaper_sources: list[str]
    article_count: int
    epaper_count: int
    language_counts: dict[str, int]
    downloaded_count: int
    articles: list[GorkhapatraArticleAsset]
    epapers: list[GorkhapatraEpaperAsset]
    warnings: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category_sources": self.category_sources,
            "epaper_sources": self.epaper_sources,
            "article_count": self.article_count,
            "epaper_count": self.epaper_count,
            "language_counts": self.language_counts,
            "downloaded_count": self.downloaded_count,
            "warnings": self.warnings,
            "articles": [article.to_dict() for article in self.articles],
            "epapers": [epaper.to_dict() for epaper in self.epapers],
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


@dataclass(slots=True)
class GorkhapatraPackSummary:
    manifest_path: str
    sample_count: int
    article_image_count: int
    epaper_page_count: int
    language_counts: dict[str, int]
    reference_status_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "sample_count": self.sample_count,
            "article_image_count": self.article_image_count,
            "epaper_page_count": self.epaper_page_count,
            "language_counts": self.language_counts,
            "reference_status_counts": self.reference_status_counts,
            "warnings": self.warnings,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


@dataclass(slots=True)
class NayaNepalPublicationItem:
    article_url: str | None
    image_url: str | None
    title: str
    language: str | None
    date_label: str | None
    date_key: str | None
    source_page: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_url": self.article_url,
            "image_url": self.image_url,
            "title": self.title,
            "language": self.language,
            "date_label": self.date_label,
            "date_key": self.date_key,
            "source_page": self.source_page,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageHit:
    page_index: int
    marker: str
    text_chars: int
    text_preview: str
    rendered_page_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "marker": self.marker,
            "text_chars": self.text_chars,
            "text_preview": self.text_preview,
            "rendered_page_path": self.rendered_page_path,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageAlignment:
    publication_item: NayaNepalPublicationItem
    epaper: GorkhapatraEpaperAsset | None
    pdf_path: str | None
    pdf_sha256: str | None
    page_hits: list[GorkhapatraLanguagePageHit]
    alignment_status: str
    claim_evidence_eligible: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "publication_item": self.publication_item.to_dict(),
            "epaper": self.epaper.to_dict() if self.epaper else None,
            "pdf_path": self.pdf_path,
            "pdf_sha256": self.pdf_sha256,
            "page_hits": [hit.to_dict() for hit in self.page_hits],
            "alignment_status": self.alignment_status,
            "claim_evidence_eligible": self.claim_evidence_eligible,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageAudit:
    publication_sources: list[str]
    epaper_sources: list[str]
    publication_item_count: int
    aligned_count: int
    language_page_hit_count: int
    status_counts: dict[str, int]
    language_counts: dict[str, int]
    alignments: list[GorkhapatraLanguagePageAlignment]
    warnings: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "publication_sources": self.publication_sources,
            "epaper_sources": self.epaper_sources,
            "publication_item_count": self.publication_item_count,
            "aligned_count": self.aligned_count,
            "language_page_hit_count": self.language_page_hit_count,
            "status_counts": self.status_counts,
            "language_counts": self.language_counts,
            "warnings": self.warnings,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "alignments": [alignment.to_dict() for alignment in self.alignments],
        }


@dataclass(slots=True)
class GorkhapatraLanguagePagePackSummary:
    manifest_path: str
    sample_count: int
    copied_page_count: int
    language_counts: dict[str, int]
    reference_status_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "sample_count": self.sample_count,
            "copied_page_count": self.copied_page_count,
            "language_counts": self.language_counts,
            "reference_status_counts": self.reference_status_counts,
            "warnings": self.warnings,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageReviewSummary:
    manifest_path: str
    review_json_path: str
    review_csv_path: str
    review_md_path: str
    sample_count: int
    pending_review_count: int
    claim_evidence_eligible_count: int
    language_counts: dict[str, int]
    page_candidate_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "review_json_path": self.review_json_path,
            "review_csv_path": self.review_csv_path,
            "review_md_path": self.review_md_path,
            "sample_count": self.sample_count,
            "pending_review_count": self.pending_review_count,
            "claim_evidence_eligible_count": self.claim_evidence_eligible_count,
            "language_counts": self.language_counts,
            "page_candidate_counts": self.page_candidate_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageFinalizeSummary:
    manifest_path: str
    summary_json_path: str
    summary_md_path: str
    source_manifest: str
    review_csv_path: str
    sample_count: int
    accepted_review_count: int
    skipped_review_count: int
    language_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "manifest_path": self.manifest_path,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "source_manifest": self.source_manifest,
            "review_csv_path": self.review_csv_path,
            "sample_count": self.sample_count,
            "accepted_review_count": self.accepted_review_count,
            "skipped_review_count": self.skipped_review_count,
            "language_counts": self.language_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageReferenceTemplateSummary:
    output_dir: str
    summary_json_path: str
    summary_md_path: str
    index_csv_path: str
    source_manifest: str
    review_csv_path: str | None
    sample_count: int
    skipped_count: int
    language_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "index_csv_path": self.index_csv_path,
            "source_manifest": self.source_manifest,
            "review_csv_path": self.review_csv_path,
            "sample_count": self.sample_count,
            "skipped_count": self.skipped_count,
            "language_counts": self.language_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageReviewerBundleSummary:
    output_dir: str
    summary_json_path: str
    summary_md_path: str
    index_csv_path: str
    source_manifest: str
    review_csv_path: str
    reference_template_dir: str | None
    sample_count: int
    skipped_count: int
    copied_image_count: int
    copied_reference_count: int
    language_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "index_csv_path": self.index_csv_path,
            "source_manifest": self.source_manifest,
            "review_csv_path": self.review_csv_path,
            "reference_template_dir": self.reference_template_dir,
            "sample_count": self.sample_count,
            "skipped_count": self.skipped_count,
            "copied_image_count": self.copied_image_count,
            "copied_reference_count": self.copied_reference_count,
            "language_counts": self.language_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageTranscriptionWorkOrderSummary:
    output_dir: str
    summary_json_path: str
    summary_md_path: str
    index_csv_path: str
    transcription_html_path: str
    source_manifest: str
    review_csv_path: str
    sample_count: int
    blocked_count: int
    verified_count: int
    language_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "index_csv_path": self.index_csv_path,
            "transcription_html_path": self.transcription_html_path,
            "source_manifest": self.source_manifest,
            "review_csv_path": self.review_csv_path,
            "sample_count": self.sample_count,
            "blocked_count": self.blocked_count,
            "verified_count": self.verified_count,
            "language_counts": self.language_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePagePdfTextSummary:
    output_dir: str
    summary_json_path: str
    summary_md_path: str
    index_csv_path: str
    spans_csv_path: str
    source_manifest: str
    sample_count: int
    extracted_sample_count: int
    span_count: int
    target_span_count: int
    font_count: int
    embedded_font_count: int
    language_counts: dict[str, int]
    font_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.extracted_sample_count > 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "index_csv_path": self.index_csv_path,
            "spans_csv_path": self.spans_csv_path,
            "source_manifest": self.source_manifest,
            "sample_count": self.sample_count,
            "extracted_sample_count": self.extracted_sample_count,
            "span_count": self.span_count,
            "target_span_count": self.target_span_count,
            "font_count": self.font_count,
            "embedded_font_count": self.embedded_font_count,
            "language_counts": self.language_counts,
            "font_counts": self.font_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageAssistedReferenceSummary:
    output_dir: str
    summary_json_path: str
    summary_md_path: str
    index_csv_path: str
    source_manifest: str
    review_csv_path: str
    engine: str
    sample_count: int
    assisted_count: int
    failed_count: int
    skipped_count: int
    language_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and self.failed_count == 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "index_csv_path": self.index_csv_path,
            "source_manifest": self.source_manifest,
            "review_csv_path": self.review_csv_path,
            "engine": self.engine,
            "sample_count": self.sample_count,
            "assisted_count": self.assisted_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "language_counts": self.language_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageOcrSidecarSummary:
    output_dir: str
    summary_json_path: str
    summary_md_path: str
    index_csv_path: str
    source_manifest: str
    review_csv_path: str
    engine: str
    in_place: bool
    sample_count: int
    sidecar_count: int
    failed_count: int
    skipped_count: int
    language_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and self.failed_count == 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "index_csv_path": self.index_csv_path,
            "source_manifest": self.source_manifest,
            "review_csv_path": self.review_csv_path,
            "engine": self.engine,
            "in_place": self.in_place,
            "sample_count": self.sample_count,
            "sidecar_count": self.sidecar_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "language_counts": self.language_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageVerificationBundleSummary:
    output_dir: str
    summary_json_path: str
    summary_md_path: str
    index_csv_path: str
    review_html_path: str
    source_manifest: str
    review_csv_path: str
    assisted_references_dir: str
    sample_count: int
    line_count: int
    crop_count: int
    missing_bbox_count: int
    skipped_count: int
    language_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and self.line_count > 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "index_csv_path": self.index_csv_path,
            "review_html_path": self.review_html_path,
            "source_manifest": self.source_manifest,
            "review_csv_path": self.review_csv_path,
            "assisted_references_dir": self.assisted_references_dir,
            "sample_count": self.sample_count,
            "line_count": self.line_count,
            "crop_count": self.crop_count,
            "missing_bbox_count": self.missing_bbox_count,
            "skipped_count": self.skipped_count,
            "language_counts": self.language_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageVerificationApplySummary:
    output_dir: str
    summary_json_path: str
    summary_md_path: str
    updated_review_csv_path: str
    references_dir: str
    source_manifest: str
    review_csv_path: str
    verification_csv_path: str
    assisted_references_dir: str
    sample_count: int
    verified_reference_count: int
    blocked_count: int
    reviewed_line_count: int
    dropped_line_count: int
    language_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verified_reference_count > 0 and self.blocked_count == 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "updated_review_csv_path": self.updated_review_csv_path,
            "references_dir": self.references_dir,
            "source_manifest": self.source_manifest,
            "review_csv_path": self.review_csv_path,
            "verification_csv_path": self.verification_csv_path,
            "assisted_references_dir": self.assisted_references_dir,
            "sample_count": self.sample_count,
            "verified_reference_count": self.verified_reference_count,
            "blocked_count": self.blocked_count,
            "reviewed_line_count": self.reviewed_line_count,
            "dropped_line_count": self.dropped_line_count,
            "language_counts": self.language_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageVerificationCsvAuditSummary:
    verification_csv_path: str
    output_dir: str
    summary_json_path: str
    summary_md_path: str
    sample_count: int
    line_count: int
    ready_line_count: int
    dropped_line_count: int
    blocked_line_count: int
    status_counts: dict[str, int]
    language_counts: dict[str, int]
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and self.line_count > 0 and self.blocked_line_count == 0 and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "verification_csv_path": self.verification_csv_path,
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "sample_count": self.sample_count,
            "line_count": self.line_count,
            "ready_line_count": self.ready_line_count,
            "dropped_line_count": self.dropped_line_count,
            "blocked_line_count": self.blocked_line_count,
            "status_counts": self.status_counts,
            "language_counts": self.language_counts,
            "issues": self.issues,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageVerificationSplitSummary:
    verification_csv_path: str
    output_dir: str
    summary_json_path: str
    summary_md_path: str
    index_csv_path: str
    batch_size: int
    sample_count: int
    line_count: int
    batch_count: int
    status_counts: dict[str, int]
    language_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.line_count > 0 and self.batch_count > 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "verification_csv_path": self.verification_csv_path,
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "index_csv_path": self.index_csv_path,
            "batch_size": self.batch_size,
            "sample_count": self.sample_count,
            "line_count": self.line_count,
            "batch_count": self.batch_count,
            "status_counts": self.status_counts,
            "language_counts": self.language_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageVerificationMergeSummary:
    source_verification_csv_path: str
    batches_dir: str
    output_dir: str
    merged_csv_path: str
    summary_json_path: str
    summary_md_path: str
    source_line_count: int
    merged_line_count: int
    batch_file_count: int
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.source_line_count > 0 and self.merged_line_count == self.source_line_count and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "source_verification_csv_path": self.source_verification_csv_path,
            "batches_dir": self.batches_dir,
            "output_dir": self.output_dir,
            "merged_csv_path": self.merged_csv_path,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "source_line_count": self.source_line_count,
            "merged_line_count": self.merged_line_count,
            "batch_file_count": self.batch_file_count,
            "issues": self.issues,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageVerificationAssignmentSummary:
    split_index_csv_path: str
    output_dir: str
    assignment_csv_path: str
    summary_json_path: str
    summary_md_path: str
    batch_count: int
    assigned_count: int
    reviewer_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.batch_count > 0 and self.assigned_count == self.batch_count and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "split_index_csv_path": self.split_index_csv_path,
            "output_dir": self.output_dir,
            "assignment_csv_path": self.assignment_csv_path,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "batch_count": self.batch_count,
            "assigned_count": self.assigned_count,
            "reviewer_counts": self.reviewer_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GorkhapatraLanguagePageReviewAuditSummary:
    source_manifest: str
    review_csv_path: str
    summary_json_path: str
    summary_md_path: str
    sample_count: int
    group_count: int
    accepted_count: int
    rejected_count: int
    unresolved_count: int
    duplicate_accept_group_count: int
    missing_reference_count: int
    verified_reference_count: int
    require_verified_references: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "source_manifest": self.source_manifest,
            "review_csv_path": self.review_csv_path,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "sample_count": self.sample_count,
            "group_count": self.group_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "unresolved_count": self.unresolved_count,
            "duplicate_accept_group_count": self.duplicate_accept_group_count,
            "missing_reference_count": self.missing_reference_count,
            "verified_reference_count": self.verified_reference_count,
            "require_verified_references": self.require_verified_references,
            "issues": self.issues,
            "warnings": self.warnings,
        }


class _GorkhapatraCategoryParser(HTMLParser):
    def __init__(self, *, base_url: str, source_page: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.source_page = source_page
        self.assets: list[GorkhapatraArticleAsset] = []
        self._href_stack: list[str | None] = []
        self._link_text_stack: list[list[str]] = []
        self._title_links: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "a":
            href = attr.get("href")
            self._href_stack.append(urljoin(self.base_url, href) if href else None)
            self._link_text_stack.append([])
            return
        if tag != "img":
            return
        alt = (attr.get("alt") or "").strip()
        if "भाषा" not in alt:
            return
        src = attr.get("src")
        article_url = self._nearest_article_url()
        self.assets.append(
            GorkhapatraArticleAsset(
                article_url=article_url,
                image_url=urljoin(self.base_url, src) if src else None,
                title=alt,
                language=_extract_language_label(alt),
                source_page=self.source_page,
            )
        )

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href_stack:
            href = self._href_stack.pop()
            text_chunks = self._link_text_stack.pop() if self._link_text_stack else []
            text = " ".join(" ".join(text_chunks).split())
            if href and re.search(r"/news/\d+", href) and "भाषा" in text:
                self._title_links[text] = href

    def handle_data(self, data: str) -> None:
        if self._link_text_stack:
            self._link_text_stack[-1].append(data)

    def _nearest_article_url(self) -> str | None:
        for href in reversed(self._href_stack):
            if href and re.search(r"/news/\d+", href):
                return href
        return None

    def finalize_assets(self) -> list[GorkhapatraArticleAsset]:
        for asset in self.assets:
            if asset.article_url is None:
                asset.article_url = self._title_links.get(" ".join(asset.title.split()))
        return self.assets


class _GorkhapatraEpaperParser(HTMLParser):
    def __init__(self, *, base_url: str, source_page: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.source_page = source_page
        self.assets: list[GorkhapatraEpaperAsset] = []
        self._current_viewer: str | None = None
        self._last_thumbnail: str | None = None
        self._capture_date = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "a":
            href = attr.get("href")
            if href and "/pdf?file=" in href:
                self._current_viewer = urljoin(self.base_url, href)
            return
        if tag == "img":
            src = attr.get("src")
            if src and "/uploads/image/" in src:
                self._last_thumbnail = urljoin(self.base_url, src)
            return
        if tag == "span" and "date" in (attr.get("class") or ""):
            self._capture_date = True

    def handle_data(self, data: str) -> None:
        if not self._capture_date or not self._current_viewer:
            return
        date_label = " ".join(data.split())
        direct_pdf = _direct_epaper_pdf_url(self._current_viewer)
        self.assets.append(
            GorkhapatraEpaperAsset(
                viewer_url=self._current_viewer,
                direct_pdf_url=direct_pdf,
                source_page=self.source_page,
                date_label=date_label or None,
                thumbnail_url=self._last_thumbnail,
            )
        )
        self._current_viewer = None
        self._last_thumbnail = None
        self._capture_date = False

    def handle_endtag(self, tag: str) -> None:
        if tag == "span":
            self._capture_date = False


class _NayaNepalPublicationParser(HTMLParser):
    def __init__(self, *, base_url: str, source_page: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.source_page = source_page
        self.items: list[NayaNepalPublicationItem] = []
        self._href_stack: list[str | None] = []
        self._current_index: int | None = None
        self._capture_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "a":
            href = attr.get("href")
            self._href_stack.append(urljoin(self.base_url, href) if href else None)
            return
        if tag == "img":
            alt = (attr.get("alt") or "").strip()
            if "भाषा" not in alt:
                return
            src = attr.get("src")
            self.items.append(
                NayaNepalPublicationItem(
                    article_url=self._nearest_article_url(),
                    image_url=urljoin(self.base_url, src) if src else None,
                    title=alt,
                    language=_extract_language_label(alt),
                    date_label=None,
                    date_key=None,
                    source_page=self.source_page,
                )
            )
            self._current_index = len(self.items) - 1
            return
        if tag in {"h2", "h3"} and self._current_index is not None:
            self._capture_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href_stack:
            self._href_stack.pop()
        if tag in {"h2", "h3"}:
            self._capture_title = False

    def handle_data(self, data: str) -> None:
        if self._current_index is None:
            return
        text = " ".join(data.split())
        if not text:
            return
        item = self.items[self._current_index]
        date_key = _nepali_date_key(text)
        if date_key and item.date_key is None:
            item.date_label = _clean_date_label(text)
            item.date_key = date_key
        if self._capture_title and "भाषा" in text:
            item.title = text
            item.language = _extract_language_label(text)

    def _nearest_article_url(self) -> str | None:
        for href in reversed(self._href_stack):
            if href and re.search(r"/news/\d+", href):
                return href
        return None


def audit_gorkhapatra_source(
    output_dir: str | Path,
    *,
    category_urls: list[str] | None = None,
    epaper_urls: list[str] | None = None,
    category_html_paths: list[str | Path] | None = None,
    epaper_html_paths: list[str | Path] | None = None,
    download_assets: bool = False,
    max_article_images: int = 0,
    max_epaper_pdfs: int = 0,
    timeout_seconds: float = 30.0,
) -> GorkhapatraSourceAudit:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    category_sources = list(category_urls or ())
    epaper_sources = list(epaper_urls or ())
    category_inputs = _read_html_inputs(category_sources, category_html_paths or [], timeout_seconds=timeout_seconds, output_dir=out, prefix="category")
    epaper_inputs = _read_html_inputs(epaper_sources, epaper_html_paths or [], timeout_seconds=timeout_seconds, output_dir=out, prefix="epaper")
    articles: list[GorkhapatraArticleAsset] = []
    for source_name, html_text in category_inputs:
        parser = _GorkhapatraCategoryParser(base_url=GORKHAPATRA_ONLINE_BASE, source_page=source_name)
        parser.feed(html_text)
        articles.extend(parser.finalize_assets())
    epapers: list[GorkhapatraEpaperAsset] = []
    for source_name, html_text in epaper_inputs:
        parser = _GorkhapatraEpaperParser(base_url=GORKHAPATRA_EPAPER_BASE, source_page=source_name)
        parser.feed(html_text)
        epapers.extend(parser.assets)
    articles = _dedupe_articles(articles)
    epapers = _dedupe_epapers(epapers)
    if download_assets:
        _download_article_images(articles, out, max_count=max_article_images, timeout_seconds=timeout_seconds, warnings=warnings)
        _download_epaper_pdfs(epapers, out, max_count=max_epaper_pdfs, timeout_seconds=timeout_seconds, warnings=warnings)
    language_counts: dict[str, int] = {}
    for article in articles:
        if article.language:
            language_counts[article.language] = language_counts.get(article.language, 0) + 1
    downloaded_count = sum(1 for article in articles if article.downloaded_image) + sum(1 for epaper in epapers if epaper.downloaded_pdf)
    audit = GorkhapatraSourceAudit(
        category_sources=[source for source, _ in category_inputs],
        epaper_sources=[source for source, _ in epaper_inputs],
        article_count=len(articles),
        epaper_count=len(epapers),
        language_counts=dict(sorted(language_counts.items())),
        downloaded_count=downloaded_count,
        articles=articles,
        epapers=epapers,
        warnings=warnings,
    )
    _write_gorkhapatra_audit(audit, out)
    return audit


def prepare_gorkhapatra_pack(
    audit_json_path: str | Path,
    output_dir: str | Path,
    *,
    include_article_images: bool = True,
    include_epaper_pages: bool = True,
    max_article_images: int = 0,
    max_epaper_pdfs: int = 0,
    max_pages_per_pdf: int = 2,
    split: str = "eval",
    dataset_name: str = "gorkhapatra-naya-nepal",
) -> GorkhapatraPackSummary:
    audit_path = Path(audit_json_path)
    if not audit_path.is_file():
        raise DataValidationError(f"Gorkhapatra source audit JSON does not exist: {audit_path}")
    if max_pages_per_pdf < 1:
        raise DataValidationError("max_pages_per_pdf must be at least 1")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    out = Path(output_dir)
    inputs_dir = out / "inputs"
    refs_dir = out / "references"
    manifests_dir = out / "manifests"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ManifestEntry] = []
    warnings: list[str] = []
    article_count = 0
    epaper_page_count = 0
    if include_article_images:
        for article_index, article in enumerate(payload.get("articles") or [], start=1):
            if max_article_images and article_count >= max_article_images:
                break
            downloaded = article.get("downloaded_image") if isinstance(article, dict) else None
            if not isinstance(downloaded, dict) or not downloaded.get("path"):
                continue
            source_path = _resolve_audit_artifact_path(str(downloaded["path"]), audit_path)
            if not source_path.is_file():
                warnings.append(f"missing downloaded article image for article {article_index}: {source_path}")
                continue
            sample_id = f"gorkhapatra-article-{article_index:04d}"
            target_path = inputs_dir / f"{sample_id}{source_path.suffix.lower() or '.jpg'}"
            shutil.copy2(source_path, target_path)
            reference_path = refs_dir / f"{sample_id}.ref.json"
            language = _clean_optional_str(article.get("language"))
            source_url = _clean_optional_str(article.get("article_url")) or _clean_optional_str(article.get("image_url"))
            _write_pending_reference(
                reference_path,
                sample_id=sample_id,
                source_url=source_url,
                language=language,
                source_kind="gorkhapatra_article_image",
            )
            image_hash = sha256_file(target_path)
            entries.append(
                ManifestEntry(
                    sample_id=sample_id,
                    dataset=dataset_name,
                    split=split,
                    image_path=str(target_path),
                    text="",
                    sha256=image_hash,
                    metadata={
                        "benchmark_name": "gorkhapatra-naya-nepal",
                        "source": "gorkhapatra",
                        "source_kind": "article_image",
                        "source_url": source_url,
                        "source_image_url": article.get("image_url"),
                        "language": language,
                        "script": "unknown",
                        "input_format": "image",
                        "document_type": "article_image",
                        "reference_path": str(reference_path),
                        "reference_status": "pending_manual_label",
                        "claim_evidence_eligible": False,
                        "slices": _gorkhapatra_slices(language, "article_image"),
                        "text_sha256": sha256_text(""),
                        "sample_sha256": sha256_text(f"{image_hash}\n"),
                    },
                )
            )
            article_count += 1
    if include_epaper_pages:
        epapers = payload.get("epapers") or []
        rendered_pdf_count = 0
        for epaper_index, epaper in enumerate(epapers, start=1):
            if max_epaper_pdfs and rendered_pdf_count >= max_epaper_pdfs:
                break
            downloaded = epaper.get("downloaded_pdf") if isinstance(epaper, dict) else None
            if not isinstance(downloaded, dict) or not downloaded.get("path"):
                continue
            source_path = _resolve_audit_artifact_path(str(downloaded["path"]), audit_path)
            if not source_path.is_file():
                warnings.append(f"missing downloaded epaper PDF for epaper {epaper_index}: {source_path}")
                continue
            try:
                rendered_pages = _render_pdf_pages(source_path, inputs_dir, epaper_index=epaper_index, max_pages=max_pages_per_pdf)
            except DataValidationError as exc:
                warnings.append(str(exc))
                continue
            rendered_pdf_count += 1
            for page_index, target_path in rendered_pages:
                sample_id = f"gorkhapatra-epaper-{epaper_index:04d}-page-{page_index + 1:04d}"
                reference_path = refs_dir / f"{sample_id}.ref.json"
                source_url = _clean_optional_str(epaper.get("direct_pdf_url")) or _clean_optional_str(epaper.get("viewer_url"))
                _write_pending_reference(
                    reference_path,
                    sample_id=sample_id,
                    source_url=source_url,
                    language=None,
                    source_kind="gorkhapatra_epaper_page",
                )
                image_hash = sha256_file(target_path)
                entries.append(
                    ManifestEntry(
                        sample_id=sample_id,
                        dataset=dataset_name,
                        split=split,
                        image_path=str(target_path),
                        text="",
                        sha256=image_hash,
                        metadata={
                            "benchmark_name": "gorkhapatra-naya-nepal",
                            "source": "gorkhapatra",
                            "source_kind": "epaper_page",
                            "source_url": source_url,
                            "viewer_url": epaper.get("viewer_url"),
                            "date_label": epaper.get("date_label"),
                            "pdf_sha256": downloaded.get("sha256"),
                            "pdf_page_index": page_index,
                            "language": None,
                            "script": "unknown",
                            "input_format": "image",
                            "document_type": "epaper_page",
                            "reference_path": str(reference_path),
                            "reference_status": "pending_manual_label",
                            "claim_evidence_eligible": False,
                            "slices": _gorkhapatra_slices(None, "epaper_page"),
                            "text_sha256": sha256_text(""),
                            "sample_sha256": sha256_text(f"{image_hash}\n"),
                        },
                    )
                )
                epaper_page_count += 1
    manifest_path = manifests_dir / "gorkhapatra-pack.jsonl"
    write_manifest(entries, manifest_path)
    summary = _gorkhapatra_pack_summary(
        manifest_path,
        entries,
        article_image_count=article_count,
        epaper_page_count=epaper_page_count,
        warnings=warnings,
    )
    _write_gorkhapatra_pack_summary(summary, out)
    return summary


def audit_gorkhapatra_language_pages(
    output_dir: str | Path,
    *,
    publication_urls: list[str] | None = None,
    epaper_urls: list[str] | None = None,
    publication_html_paths: list[str | Path] | None = None,
    epaper_html_paths: list[str | Path] | None = None,
    download_pdfs: bool = False,
    render_pages: bool = False,
    max_publication_items: int = 0,
    max_epaper_pdfs: int = 0,
    timeout_seconds: float = 30.0,
) -> GorkhapatraLanguagePageAudit:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    publication_sources = list(publication_urls or ())
    epaper_sources = list(epaper_urls or ())
    publication_inputs = _read_html_inputs(
        publication_sources,
        publication_html_paths or [],
        timeout_seconds=timeout_seconds,
        output_dir=out,
        prefix="naya-nepal-publication",
    )
    epaper_inputs = _read_html_inputs(epaper_sources, epaper_html_paths or [], timeout_seconds=timeout_seconds, output_dir=out, prefix="epaper")
    publication_items: list[NayaNepalPublicationItem] = []
    for source_name, html_text in publication_inputs:
        parser = _NayaNepalPublicationParser(base_url=GORKHAPATRA_ONLINE_BASE, source_page=source_name)
        parser.feed(html_text)
        publication_items.extend(parser.items)
    publication_items = _dedupe_publication_items(publication_items)
    if max_publication_items:
        publication_items = publication_items[:max_publication_items]
    epapers: list[GorkhapatraEpaperAsset] = []
    for source_name, html_text in epaper_inputs:
        parser = _GorkhapatraEpaperParser(base_url=GORKHAPATRA_EPAPER_BASE, source_page=source_name)
        parser.feed(html_text)
        epapers.extend(parser.assets)
    epapers = _dedupe_epapers(epapers)
    epapers_by_date = {_nepali_date_key(epaper.date_label or ""): epaper for epaper in epapers if _nepali_date_key(epaper.date_label or "")}
    alignments: list[GorkhapatraLanguagePageAlignment] = []
    downloaded_pdf_count = 0
    pdf_cache: dict[str, DownloadedArtifact] = {}
    for index, item in enumerate(publication_items, start=1):
        item_warnings: list[str] = []
        if not item.date_key:
            alignments.append(_language_page_alignment(item, None, None, None, [], "missing_publication_date", item_warnings))
            continue
        epaper = epapers_by_date.get(item.date_key)
        if epaper is None:
            alignments.append(_language_page_alignment(item, None, None, None, [], "no_matching_epaper_date", item_warnings))
            continue
        if not download_pdfs:
            alignments.append(_language_page_alignment(item, epaper, None, None, [], "matched_epaper_not_downloaded", item_warnings))
            continue
        if max_epaper_pdfs and downloaded_pdf_count >= max_epaper_pdfs and epaper.direct_pdf_url not in pdf_cache:
            alignments.append(_language_page_alignment(item, epaper, None, None, [], "matched_epaper_not_downloaded_limit", item_warnings))
            continue
        try:
            artifact = pdf_cache.get(epaper.direct_pdf_url)
            if artifact is None:
                target = out / "epapers" / f"epaper-{item.date_key}.pdf"
                artifact = _download_or_reuse_epaper_pdf(epaper.direct_pdf_url, target, timeout_seconds=timeout_seconds)
                pdf_cache[epaper.direct_pdf_url] = artifact
                downloaded_pdf_count += 1
            epaper.downloaded_pdf = artifact
            hits = _find_language_page_hits(Path(artifact.path), out / "language-pages", render_pages=render_pages)
            status = "language_page_found" if hits else "matched_epaper_no_language_page_marker"
            alignments.append(_language_page_alignment(item, epaper, artifact.path, artifact.sha256, hits, status, item_warnings))
        except Exception as exc:  # noqa: BLE001 - recorded as provenance warning, not ignored.
            warning = f"{type(exc).__name__}: {exc}"
            item_warnings.append(warning)
            warnings.append(f"failed to inspect {epaper.direct_pdf_url}: {warning}")
            alignments.append(_language_page_alignment(item, epaper, None, None, [], "pdf_inspection_failed", item_warnings))
    audit = _gorkhapatra_language_page_audit(
        publication_sources=[source for source, _ in publication_inputs],
        epaper_sources=[source for source, _ in epaper_inputs],
        publication_items=publication_items,
        alignments=alignments,
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_audit(audit, out)
    return audit


def prepare_gorkhapatra_language_page_pack(
    language_page_audit_json_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "eval",
    dataset_name: str = "gorkhapatra-language-pages",
    max_samples: int = 0,
) -> GorkhapatraLanguagePagePackSummary:
    audit_path = Path(language_page_audit_json_path)
    if not audit_path.is_file():
        raise DataValidationError(f"Gorkhapatra language-page audit JSON does not exist: {audit_path}")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    out = Path(output_dir)
    inputs_dir = out / "inputs"
    refs_dir = out / "references"
    manifests_dir = out / "manifests"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ManifestEntry] = []
    warnings: list[str] = []
    copied_count = 0
    for alignment_index, alignment in enumerate(payload.get("alignments") or [], start=1):
        if max_samples and len(entries) >= max_samples:
            break
        if not isinstance(alignment, dict):
            continue
        publication_item = alignment.get("publication_item")
        if not isinstance(publication_item, dict):
            warnings.append(f"alignment {alignment_index} missing publication_item")
            continue
        language = _clean_optional_str(publication_item.get("language"))
        date_key = _clean_optional_str(publication_item.get("date_key")) or "unknown-date"
        page_hits = alignment.get("page_hits")
        if not isinstance(page_hits, list) or not page_hits:
            warnings.append(f"alignment {alignment_index} has no rendered page hits")
            continue
        for hit_index, hit in enumerate(page_hits, start=1):
            if max_samples and len(entries) >= max_samples:
                break
            if not isinstance(hit, dict):
                continue
            rendered_path_value = _clean_optional_str(hit.get("rendered_page_path"))
            if not rendered_path_value:
                warnings.append(f"alignment {alignment_index} page hit {hit_index} has no rendered_page_path")
                continue
            source_path = _resolve_audit_artifact_path(rendered_path_value, audit_path)
            if not source_path.is_file():
                warnings.append(f"alignment {alignment_index} page hit {hit_index} missing rendered page: {source_path}")
                continue
            page_index = int(hit.get("page_index") or 0)
            sample_id = f"gorkhapatra-language-page-{alignment_index:04d}-{hit_index:02d}"
            target_path = inputs_dir / f"{sample_id}{source_path.suffix.lower() or '.png'}"
            shutil.copy2(source_path, target_path)
            reference_path = refs_dir / f"{sample_id}.ref.json"
            source_url = _alignment_source_url(alignment)
            _write_pending_reference(
                reference_path,
                sample_id=sample_id,
                source_url=source_url,
                language=language,
                source_kind="gorkhapatra_language_page_candidate",
            )
            image_hash = sha256_file(target_path)
            entries.append(
                ManifestEntry(
                    sample_id=sample_id,
                    dataset=dataset_name,
                    split=split,
                    image_path=str(target_path),
                    text="",
                    sha256=image_hash,
                    metadata={
                        "benchmark_name": "gorkhapatra-naya-nepal-language-pages",
                        "source": "gorkhapatra",
                        "source_kind": "language_page_candidate",
                        "source_url": source_url,
                        "article_url": publication_item.get("article_url"),
                        "source_image_url": publication_item.get("image_url"),
                        "candidate_language": language,
                        "language": language,
                        "date_label": publication_item.get("date_label"),
                        "date_key": date_key,
                        "pdf_path": alignment.get("pdf_path"),
                        "pdf_sha256": alignment.get("pdf_sha256"),
                        "pdf_page_index": page_index,
                        "pdf_page_number": page_index + 1,
                        "language_page_marker": hit.get("marker"),
                        "alignment_status": alignment.get("alignment_status"),
                        "page_disambiguation_status": "candidate_requires_manual_review",
                        "script": "unknown",
                        "input_format": "image",
                        "document_type": "language_page_candidate",
                        "reference_path": str(reference_path),
                        "reference_status": "pending_manual_label",
                        "claim_evidence_eligible": False,
                        "slices": _gorkhapatra_slices(language, "language_page_candidate"),
                        "text_sha256": sha256_text(""),
                        "sample_sha256": sha256_text(f"{image_hash}\n{alignment.get('pdf_sha256') or ''}\n{page_index}\n"),
                    },
                )
            )
            copied_count += 1
    manifest_path = manifests_dir / "gorkhapatra-language-page-pack.jsonl"
    write_manifest(entries, manifest_path)
    summary = _gorkhapatra_language_page_pack_summary(
        manifest_path,
        entries,
        copied_page_count=copied_count,
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_pack_summary(summary, out)
    return summary


def prepare_gorkhapatra_language_page_review(
    manifest_path: str | Path,
    output_dir: str | Path,
) -> GorkhapatraLanguagePageReviewSummary:
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    entries = load_manifest(source_manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    language_counts: dict[str, int] = {}
    page_candidate_counts: dict[str, int] = {}
    for entry in entries:
        metadata = entry.metadata or {}
        source_kind = metadata.get("source_kind")
        if source_kind != "language_page_candidate":
            warnings.append(f"{entry.sample_id} source_kind is not language_page_candidate: {source_kind}")
        candidate_language = _clean_optional_str(metadata.get("candidate_language")) or _clean_optional_str(metadata.get("language"))
        if candidate_language:
            language_counts[candidate_language] = language_counts.get(candidate_language, 0) + 1
        date_key = _clean_optional_str(metadata.get("date_key")) or "unknown-date"
        page_number = metadata.get("pdf_page_number")
        page_key = f"{date_key}:page-{page_number if page_number is not None else 'unknown'}"
        page_candidate_counts[page_key] = page_candidate_counts.get(page_key, 0) + 1
        rows.append(
            {
                "sample_id": entry.sample_id,
                "candidate_language": candidate_language,
                "date_label": metadata.get("date_label"),
                "date_key": metadata.get("date_key"),
                "pdf_page_index": metadata.get("pdf_page_index"),
                "pdf_page_number": metadata.get("pdf_page_number"),
                "image_path": entry.image_path,
                "reference_path": metadata.get("reference_path"),
                "article_url": metadata.get("article_url"),
                "source_image_url": metadata.get("source_image_url"),
                "source_url": metadata.get("source_url"),
                "pdf_path": metadata.get("pdf_path"),
                "pdf_sha256": metadata.get("pdf_sha256"),
                "language_page_marker": metadata.get("language_page_marker"),
                "alignment_status": metadata.get("alignment_status"),
                "page_disambiguation_status": metadata.get("page_disambiguation_status"),
                "reference_status": metadata.get("reference_status"),
                "claim_evidence_eligible": bool(metadata.get("claim_evidence_eligible")),
                "review_decision": "",
                "reviewer": "",
                "reviewed_at": "",
                "notes": "",
            }
        )
    json_path = out / "gorkhapatra-language-page-review.json"
    csv_path = out / "gorkhapatra-language-page-review.csv"
    md_path = out / "gorkhapatra-language-page-review.md"
    summary = GorkhapatraLanguagePageReviewSummary(
        manifest_path=str(source_manifest),
        review_json_path=str(json_path),
        review_csv_path=str(csv_path),
        review_md_path=str(md_path),
        sample_count=len(rows),
        pending_review_count=sum(1 for row in rows if row["page_disambiguation_status"] == "candidate_requires_manual_review"),
        claim_evidence_eligible_count=sum(1 for row in rows if row["claim_evidence_eligible"]),
        language_counts=dict(sorted(language_counts.items())),
        page_candidate_counts=dict(sorted(page_candidate_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_review(summary, rows)
    return summary


def prepare_gorkhapatra_language_page_reference_templates(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    review_csv_path: str | Path | None = None,
    accepted_only: bool = False,
) -> GorkhapatraLanguagePageReferenceTemplateSummary:
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    review_csv = Path(review_csv_path) if review_csv_path is not None else None
    if review_csv is not None and not review_csv.is_file():
        raise DataValidationError(f"Gorkhapatra language-page review CSV does not exist: {review_csv}")
    review_rows = {row.get("sample_id"): row for row in _read_language_page_review_rows(review_csv)} if review_csv is not None else {}
    entries = load_manifest(source_manifest)
    out = Path(output_dir)
    refs_dir = out / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    skipped_count = 0
    language_counts: dict[str, int] = {}
    for entry in entries:
        metadata = dict(entry.metadata or {})
        review_row = review_rows.get(entry.sample_id, {})
        decision = _clean_optional_str(review_row.get("review_decision"))
        if accepted_only and decision != "accept_language_page":
            skipped_count += 1
            continue
        language = _clean_optional_str(review_row.get("candidate_language")) or _clean_optional_str(metadata.get("candidate_language")) or _clean_optional_str(metadata.get("language"))
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        template_path = refs_dir / f"{entry.sample_id}.ref.json"
        template_payload = _language_page_reference_template(entry, metadata=metadata, review_row=review_row, language=language)
        template_path.write_text(json.dumps(template_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index_rows.append(
            {
                "sample_id": entry.sample_id,
                "candidate_language": language,
                "review_decision": decision,
                "image_path": entry.image_path,
                "template_path": str(template_path),
                "source_reference_path": metadata.get("reference_path"),
                "pdf_page_number": metadata.get("pdf_page_number"),
                "date_key": metadata.get("date_key"),
            }
        )
    if not index_rows:
        warnings.append("no reference templates written")
    summary = GorkhapatraLanguagePageReferenceTemplateSummary(
        output_dir=str(out),
        summary_json_path=str(out / "gorkhapatra-language-page-reference-templates.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-reference-templates.md"),
        index_csv_path=str(out / "gorkhapatra-language-page-reference-templates.csv"),
        source_manifest=str(source_manifest),
        review_csv_path=str(review_csv) if review_csv is not None else None,
        sample_count=len(index_rows),
        skipped_count=skipped_count,
        language_counts=dict(sorted(language_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_reference_templates_summary(summary, index_rows)
    return summary


def prepare_gorkhapatra_language_page_reviewer_bundle(
    manifest_path: str | Path,
    review_csv_path: str | Path,
    output_dir: str | Path,
    *,
    reference_template_dir: str | Path | None = None,
) -> GorkhapatraLanguagePageReviewerBundleSummary:
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    review_csv = Path(review_csv_path)
    if not review_csv.is_file():
        raise DataValidationError(f"Gorkhapatra language-page review CSV does not exist: {review_csv}")
    template_dir = Path(reference_template_dir) if reference_template_dir is not None else None
    if template_dir is not None and not template_dir.is_dir():
        raise DataValidationError(f"Gorkhapatra reference template directory does not exist: {template_dir}")

    entries = {entry.sample_id: entry for entry in load_manifest(source_manifest)}
    review_rows = _read_language_page_review_rows(review_csv)
    out = Path(output_dir)
    images_dir = out / "images"
    references_dir = out / "references"
    images_dir.mkdir(parents=True, exist_ok=True)
    references_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    skipped_count = 0
    copied_image_count = 0
    copied_reference_count = 0
    language_counts: dict[str, int] = {}

    for row in review_rows:
        if _clean_optional_str(row.get("review_decision")) != "accept_language_page":
            skipped_count += 1
            continue
        sample_id = _clean_optional_str(row.get("sample_id"))
        if not sample_id:
            warnings.append("accepted review row missing sample_id")
            continue
        entry = entries.get(sample_id)
        if entry is None:
            warnings.append(f"{sample_id}: accepted review row not present in source manifest")
            continue
        metadata = entry.metadata or {}
        language = _clean_optional_str(row.get("candidate_language")) or _clean_optional_str(metadata.get("candidate_language")) or _clean_optional_str(metadata.get("language"))
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        image_path = _resolve_manifest_entry_path(entry.image_path, source_manifest=source_manifest)
        if not image_path.is_file():
            warnings.append(f"{sample_id}: image_path does not exist: {image_path}")
            continue
        image_target = images_dir / f"{sample_id}{image_path.suffix.lower() or '.png'}"
        shutil.copy2(image_path, image_target)
        copied_image_count += 1

        reference_source = _reviewer_bundle_reference_source(
            sample_id,
            row=row,
            metadata=metadata,
            source_manifest=source_manifest,
            review_csv=review_csv,
            template_dir=template_dir,
        )
        reference_target = references_dir / f"{sample_id}.ref.json"
        if reference_source is None:
            reference_target.write_text(
                json.dumps(_language_page_reference_template(entry, metadata=dict(metadata), review_row=row, language=language), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(reference_source, reference_target)
        copied_reference_count += 1
        index_rows.append(
            {
                "sample_id": sample_id,
                "candidate_language": language,
                "image_path": str(image_target),
                "reference_path": str(reference_target),
                "source_image_path": str(image_path),
                "source_reference_path": str(reference_source) if reference_source is not None else "",
                "article_url": metadata.get("article_url"),
                "source_url": metadata.get("source_url"),
                "date_key": metadata.get("date_key"),
                "pdf_page_number": metadata.get("pdf_page_number"),
                "reference_status": "draft",
                "claim_evidence_eligible": False,
            }
        )

    if not index_rows:
        warnings.append("no accepted review rows bundled")
    summary = GorkhapatraLanguagePageReviewerBundleSummary(
        output_dir=str(out),
        summary_json_path=str(out / "gorkhapatra-language-page-reviewer-bundle.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-reviewer-bundle.md"),
        index_csv_path=str(out / "gorkhapatra-language-page-reviewer-bundle.csv"),
        source_manifest=str(source_manifest),
        review_csv_path=str(review_csv),
        reference_template_dir=str(template_dir) if template_dir is not None else None,
        sample_count=len(index_rows),
        skipped_count=skipped_count,
        copied_image_count=copied_image_count,
        copied_reference_count=copied_reference_count,
        language_counts=dict(sorted(language_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_reviewer_bundle_summary(summary, index_rows)
    return summary


def prepare_gorkhapatra_language_page_transcription_work_order(
    manifest_path: str | Path,
    review_csv_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_languages: list[str] | None = None,
) -> GorkhapatraLanguagePageTranscriptionWorkOrderSummary:
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    review_csv = Path(review_csv_path)
    if not review_csv.is_file():
        raise DataValidationError(f"Gorkhapatra language-page review CSV does not exist: {review_csv}")
    selected_languages = _normalized_language_filter(candidate_languages)
    entries = {entry.sample_id: entry for entry in load_manifest(source_manifest)}
    review_rows = _read_language_page_review_rows(review_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    language_counts: dict[str, int] = {}
    blocked_count = 0
    verified_count = 0
    skipped_count = 0
    for row in review_rows:
        if _clean_optional_str(row.get("review_decision")) != "accept_language_page":
            skipped_count += 1
            continue
        sample_id = _clean_optional_str(row.get("sample_id"))
        if not sample_id:
            warnings.append("accepted review row missing sample_id")
            continue
        entry = entries.get(sample_id)
        if entry is None:
            warnings.append(f"{sample_id}: accepted review row not present in source manifest")
            continue
        language = _entry_or_review_candidate_language(entry, row)
        if selected_languages and language not in selected_languages:
            skipped_count += 1
            continue
        metadata = entry.metadata or {}
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        image_path = _resolve_manifest_entry_path(entry.image_path, source_manifest=source_manifest)
        reference_path_value = _clean_optional_str(row.get("reference_path")) or _clean_optional_str(metadata.get("reference_path"))
        reference_path = _resolve_review_reference_path(reference_path_value, source_manifest=source_manifest, review_csv=review_csv) if reference_path_value else None
        status = "blocked"
        blocker = "accepted row missing reference_path"
        if reference_path is not None:
            try:
                _load_verified_language_page_reference(reference_path, sample_id=sample_id)
                status = "verified"
                blocker = ""
                verified_count += 1
            except DataValidationError as exc:
                blocker = str(exc)
        if status != "verified":
            blocked_count += 1
        rows.append(
            {
                "sample_id": sample_id,
                "candidate_language": language,
                "status": status,
                "blocker": blocker,
                "image_path": str(image_path),
                "image_exists": image_path.is_file(),
                "reference_path": str(reference_path) if reference_path is not None else "",
                "reference_exists": reference_path.is_file() if reference_path is not None else False,
                "date_key": metadata.get("date_key"),
                "date_label": metadata.get("date_label"),
                "pdf_page_number": metadata.get("pdf_page_number"),
                "article_url": metadata.get("article_url"),
                "source_image_url": metadata.get("source_image_url"),
                "source_url": metadata.get("source_url"),
                "required_fields": "text; reading_order[].id/text; tables[]; figures[]; metadata.reference_status=verified; metadata.claim_evidence_eligible=true",
                "next_command": (
                    "uv run ocrtech audit-gorkhapatra-language-page-review "
                    "<manifest> <review_csv> "
                    f"--candidate-language {language or '<language>'} "
                    "--require-verified-references --out <audit_dir>"
                ),
            }
        )

    if not rows:
        warnings.append("no accepted language-page rows selected for transcription work order")
    summary = GorkhapatraLanguagePageTranscriptionWorkOrderSummary(
        output_dir=str(out),
        summary_json_path=str(out / "gorkhapatra-language-page-transcription-work-order.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-transcription-work-order.md"),
        index_csv_path=str(out / "gorkhapatra-language-page-transcription-work-order.csv"),
        transcription_html_path=str(out / "gorkhapatra-language-page-transcription-dashboard.html"),
        source_manifest=str(source_manifest),
        review_csv_path=str(review_csv),
        sample_count=len(rows),
        blocked_count=blocked_count,
        verified_count=verified_count,
        language_counts=dict(sorted(language_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_transcription_work_order(summary, rows, skipped_count=skipped_count)
    return summary


def extract_gorkhapatra_language_page_pdf_text(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_languages: list[str] | None = None,
    target_font: list[str] | None = None,
) -> GorkhapatraLanguagePagePdfTextSummary:
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    selected_languages = _normalized_language_filter(candidate_languages)
    requested_target_fonts = [_clean_optional_str(value).lower() for value in (target_font or []) if _clean_optional_str(value)]
    try:
        import fitz
    except Exception as exc:  # noqa: BLE001 - PyMuPDF is optional in some environments.
        raise DataValidationError(f"PyMuPDF is required for PDF-native text extraction: {exc}") from exc

    out = Path(output_dir)
    samples_dir = out / "samples"
    fonts_dir = out / "fonts"
    out.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    fonts_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(source_manifest)
    index_rows: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    language_counts: dict[str, int] = {}
    font_counts: dict[str, int] = {}
    embedded_fonts_seen: set[tuple[str, int]] = set()
    extracted_sample_count = 0
    target_span_count = 0

    for entry in entries:
        metadata = entry.metadata or {}
        language = _entry_or_review_candidate_language(entry, None)
        if selected_languages and language not in selected_languages:
            continue
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        pdf_path_value = _clean_optional_str(metadata.get("pdf_path"))
        if not pdf_path_value:
            warnings.append(f"{entry.sample_id}: manifest metadata has no pdf_path")
            continue
        pdf_path = _resolve_manifest_entry_path(pdf_path_value, source_manifest=source_manifest)
        if not pdf_path.is_file():
            warnings.append(f"{entry.sample_id}: PDF does not exist: {pdf_path}")
            continue
        try:
            page_index = int(metadata.get("pdf_page_index"))
        except (TypeError, ValueError) as exc:
            warnings.append(f"{entry.sample_id}: metadata pdf_page_index is missing or invalid")
            continue

        try:
            document = fitz.open(pdf_path)
        except Exception as exc:  # noqa: BLE001 - row-level PDF failures are recorded.
            warnings.append(f"{entry.sample_id}: could not open PDF {pdf_path}: {exc}")
            continue
        try:
            if page_index < 0 or page_index >= document.page_count:
                warnings.append(f"{entry.sample_id}: pdf_page_index {page_index} outside PDF page count {document.page_count}")
                continue
            page = document[page_index]
            sample_target_fonts = _target_font_needles_for_language(language, requested_target_fonts)
            spans = _extract_pdf_text_spans(page, sample_target_fonts=sample_target_fonts)
            fonts = _extract_pdf_page_fonts(document, page, fonts_dir / entry.sample_id)
        except Exception as exc:  # noqa: BLE001 - row-level PDF failures are recorded.
            warnings.append(f"{entry.sample_id}: could not extract page text/font metadata: {exc}")
            continue
        finally:
            document.close()

        sample_target_spans = sum(1 for span in spans if span["is_target_font"])
        target_span_count += sample_target_spans
        extracted_sample_count += 1
        for span in spans:
            font = str(span["font"])
            font_counts[font] = font_counts.get(font, 0) + 1
            row = {
                "sample_id": entry.sample_id,
                "candidate_language": language or "",
                "date_key": metadata.get("date_key") or "",
                "pdf_path": str(pdf_path),
                "pdf_page_index": page_index,
                "pdf_page_number": page_index + 1,
                **span,
                "bbox": json.dumps(span["bbox"], ensure_ascii=False),
                "codepoints": " ".join(span["codepoints"]),
            }
            span_rows.append(row)
        for font in fonts:
            embedded_fonts_seen.add((entry.sample_id, int(font["xref"])))
        sample_payload = {
            "sample_id": entry.sample_id,
            "candidate_language": language,
            "date_key": metadata.get("date_key"),
            "date_label": metadata.get("date_label"),
            "article_url": metadata.get("article_url"),
            "source_url": metadata.get("source_url"),
            "pdf_path": str(pdf_path),
            "pdf_sha256": metadata.get("pdf_sha256"),
            "pdf_page_index": page_index,
            "pdf_page_number": page_index + 1,
            "target_font_needles": sample_target_fonts,
            "spans": spans,
            "fonts": fonts,
            "notes": [
                "PDF-native extraction preserves the PDF text layer exactly as exposed by PyMuPDF.",
                "Some Gorkhapatra language pages use embedded legacy fonts, so raw text may require font/encoding conversion before it is Unicode reference text.",
            ],
        }
        sample_json = samples_dir / f"{entry.sample_id}.pdf-text.json"
        sample_txt = samples_dir / f"{entry.sample_id}.pdf-text.txt"
        sample_json.write_text(json.dumps(sample_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        sample_txt.write_text(_pdf_spans_plain_text(spans) + "\n", encoding="utf-8")
        index_rows.append(
            {
                "sample_id": entry.sample_id,
                "candidate_language": language or "",
                "date_key": metadata.get("date_key") or "",
                "pdf_path": str(pdf_path),
                "pdf_page_number": page_index + 1,
                "span_count": len(spans),
                "target_span_count": sample_target_spans,
                "font_count": len(fonts),
                "sample_json_path": str(sample_json),
                "sample_text_path": str(sample_txt),
                "raw_text_preview": _pdf_spans_plain_text(spans)[:240].replace("\n", " "),
            }
        )

    if not index_rows:
        warnings.append("no PDF text extracted from selected language-page manifest rows")
    summary = GorkhapatraLanguagePagePdfTextSummary(
        output_dir=str(out),
        summary_json_path=str(out / "gorkhapatra-language-page-pdf-text.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-pdf-text.md"),
        index_csv_path=str(out / "gorkhapatra-language-page-pdf-text.csv"),
        spans_csv_path=str(out / "gorkhapatra-language-page-pdf-spans.csv"),
        source_manifest=str(source_manifest),
        sample_count=len([entry for entry in entries if not selected_languages or _entry_or_review_candidate_language(entry, None) in selected_languages]),
        extracted_sample_count=extracted_sample_count,
        span_count=len(span_rows),
        target_span_count=target_span_count,
        font_count=len(font_counts),
        embedded_font_count=len(embedded_fonts_seen),
        language_counts=dict(sorted(language_counts.items())),
        font_counts=dict(sorted(font_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_pdf_text_summary(summary, index_rows, span_rows)
    return summary


def prepare_gorkhapatra_language_page_assisted_references(
    manifest_path: str | Path,
    review_csv_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_languages: list[str] | None = None,
    engine: str = "sidecar",
    tesseract_language: str = "nep+eng",
    overwrite: bool = False,
) -> GorkhapatraLanguagePageAssistedReferenceSummary:
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    review_csv = Path(review_csv_path)
    if not review_csv.is_file():
        raise DataValidationError(f"Gorkhapatra language-page review CSV does not exist: {review_csv}")
    selected_languages = _normalized_language_filter(candidate_languages)
    entries = {entry.sample_id: entry for entry in load_manifest(source_manifest)}
    review_rows = _read_language_page_review_rows(review_csv)

    try:
        from .engines import create_engine
    except Exception as exc:  # noqa: BLE001 - optional engine dependencies.
        raise DataValidationError(f"could not import OCR engine adapters: {exc}") from exc
    engine_kwargs: dict[str, Any] = {}
    if engine in {"tesseract", "tess"}:
        engine_kwargs["language"] = tesseract_language
    try:
        ocr_engine = create_engine(engine, **engine_kwargs)
    except Exception as exc:  # noqa: BLE001 - engine availability is runtime-specific.
        raise DataValidationError(f"could not initialize OCR engine {engine}: {exc}") from exc

    out = Path(output_dir)
    references_dir = out / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    language_counts: dict[str, int] = {}
    assisted_count = 0
    failed_count = 0
    skipped_count = 0

    for row in review_rows:
        if _clean_optional_str(row.get("review_decision")) != "accept_language_page":
            skipped_count += 1
            continue
        sample_id = _clean_optional_str(row.get("sample_id"))
        if not sample_id:
            warnings.append("accepted review row missing sample_id")
            failed_count += 1
            continue
        entry = entries.get(sample_id)
        if entry is None:
            warnings.append(f"{sample_id}: accepted review row not present in source manifest")
            failed_count += 1
            continue
        language = _entry_or_review_candidate_language(entry, row)
        if selected_languages and language not in selected_languages:
            skipped_count += 1
            continue
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        image_path = _resolve_manifest_entry_path(entry.image_path, source_manifest=source_manifest)
        target = references_dir / f"{sample_id}.ref.json"
        if target.exists() and not overwrite:
            warnings.append(f"{sample_id}: assisted reference exists and --overwrite was not set: {target}")
            skipped_count += 1
            continue
        status = "assisted"
        error = ""
        line_count = 0
        confidence: float | None = None
        try:
            output = ocr_engine.recognize(image_path)
            payload = _assisted_language_page_reference(entry, row=row, language=language, engine_name=engine, engine_output=output)
            line_count = len(payload["reading_order"])
            confidence = payload["metadata"].get("ocr_average_confidence")
            assisted_count += 1
        except Exception as exc:  # noqa: BLE001 - row-level OCR failures should be auditable.
            status = "failed"
            error = str(exc)
            failed_count += 1
            payload = _language_page_reference_template(entry, metadata=dict(entry.metadata or {}), review_row=row, language=language)
            payload["metadata"].update(
                {
                    "source_kind": "gorkhapatra_language_page_assisted_reference_draft",
                    "assisted_ocr_engine": engine,
                    "assisted_ocr_error": error,
                    "machine_generated": True,
                    "requires_human_review": True,
                }
            )
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index_rows.append(
            {
                "sample_id": sample_id,
                "candidate_language": language,
                "status": status,
                "error": error,
                "image_path": str(image_path),
                "reference_path": str(target),
                "ocr_line_count": line_count,
                "ocr_average_confidence": confidence,
                "reference_status": "draft",
                "claim_evidence_eligible": False,
            }
        )

    if not index_rows:
        warnings.append("no assisted references written")
    summary = GorkhapatraLanguagePageAssistedReferenceSummary(
        output_dir=str(out),
        summary_json_path=str(out / "gorkhapatra-language-page-assisted-references.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-assisted-references.md"),
        index_csv_path=str(out / "gorkhapatra-language-page-assisted-references.csv"),
        source_manifest=str(source_manifest),
        review_csv_path=str(review_csv),
        engine=engine,
        sample_count=len(index_rows),
        assisted_count=assisted_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        language_counts=dict(sorted(language_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_assisted_references_summary(summary, index_rows)
    return summary


_PDF_NATIVE_LEGACY_CONVERTERS: dict[str, dict[str, Any]] = {
    "limbu-namdhinggo": {
        "map_path": DEFAULT_LIMBU_LEGACY_MAP,
        "required_codepoint_ranges": ["1900-194F"],
        "language_aliases": {"लिम्बू", "लिम्बु", "limbu", "Limbu"},
    },
}


def _bbox_points_to_image_xywh(
    bbox_points: Any,
    scale_x: float,
    scale_y: float,
) -> list[float] | None:
    """Convert a PyMuPDF ``[x0, y0, x1, y1]`` point bbox to image-pixel ``[x, y, w, h]``."""
    if not (isinstance(bbox_points, list | tuple) and len(bbox_points) == 4):
        return None
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in bbox_points):
        return None
    x0, y0, x1, y1 = (float(value) for value in bbox_points)
    x = x0 * scale_x
    y = y0 * scale_y
    width = (x1 - x0) * scale_x
    height = (y1 - y0) * scale_y
    if width <= 0 or height <= 0:
        return None
    return [round(x, 3), round(y, 3), round(width, 3), round(height, 3)]


def _image_pdf_point_scale(pdf_path: Path, page_index: int, image_path: Path) -> tuple[float, float]:
    """Pixels-per-PDF-point scale of the rendered page image, computed from real geometry."""
    try:
        from PIL import Image
    except ImportError as exc:  # noqa: BLE001 - optional dependency.
        raise DataValidationError("PDF-native reference scaling requires Pillow. Install ocr-tech[eval].") from exc
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        raise DataValidationError(f"PDF-native reference scaling requires PyMuPDF: {type(exc).__name__}: {exc}") from exc
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    with fitz.open(pdf_path) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise DataValidationError(f"pdf page index {page_index} out of range for {pdf_path}")
        rect = document[page_index].rect
        page_width = float(rect.width)
        page_height = float(rect.height)
    if page_width <= 0 or page_height <= 0:
        raise DataValidationError(f"pdf page {page_index} has non-positive dimensions in {pdf_path}")
    return image_width / page_width, image_height / page_height


def _pdf_native_language_page_reference(
    entry: ManifestEntry,
    *,
    row: dict[str, Any],
    language: str | None,
    pdf_text_sample: dict[str, Any],
    converter: LimbuLegacyConverter,
    converter_id: str,
    map_path: Path,
    scale_x: float,
    scale_y: float,
) -> tuple[dict[str, Any], int, int, int]:
    """Build a non-claim-ready draft reference whose reading order is legacy→Unicode spans.

    Returns ``(payload, converted_span_count, flagged_span_count, unmapped_codepoint_count)``.
    Each kept reading-order item carries the converted Unicode as ``text`` so the existing
    verification bundle pre-fills it for human review instead of demanding transcription.
    """
    metadata = dict(entry.metadata or {})
    spans = pdf_text_sample.get("spans")
    if not isinstance(spans, list):
        raise DataValidationError(f"pdf-text sample for {entry.sample_id} has no spans array")
    reading_order: list[dict[str, Any]] = []
    text_lines: list[str] = []
    converted = 0
    flagged = 0
    unmapped_total = 0
    for span in spans:
        if not isinstance(span, dict) or not span.get("is_target_font"):
            continue
        conversion = converter.convert(str(span.get("raw_text") or ""))
        unicode_text = conversion.unicode_text.strip()
        if not unicode_text:
            continue
        converted += 1
        span_index = span.get("span_index")
        line_id = f"span-{int(span_index):04d}" if isinstance(span_index, int) else f"span-{converted:04d}"
        bbox_xywh = _bbox_points_to_image_xywh(span.get("bbox"), scale_x, scale_y)
        has_unmapped = bool(conversion.unmapped_codepoints)
        unmapped_total += len(conversion.unmapped_codepoints)
        review_note = ""
        if has_unmapped:
            flagged += 1
            review_note = "legacy conversion left unmapped codepoints: " + " ".join(conversion.unmapped_codepoints)
        char_count = max(len(unicode_text), 1)
        confidence = round(1.0 - len(conversion.unmapped_codepoints) / char_count, 4)
        reading_order.append(
            {
                "id": line_id,
                "type": "pdf_native_line",
                "text": unicode_text,
                "bbox": bbox_xywh,
                "page_index": 0,
                "confidence": confidence,
                "needs_review": True,
                "has_unmapped_codepoints": has_unmapped,
                "unmapped_codepoints": conversion.unmapped_codepoints,
                "legacy_text": conversion.legacy_text,
                "review_note": review_note,
                "font": span.get("font"),
                "block_index": span.get("block_index"),
                "line_index": span.get("line_index"),
                "span_index": span_index,
            }
        )
        text_lines.append(unicode_text)
    payload = {
        "text": "\n".join(text_lines),
        "reading_order": reading_order,
        "tables": [],
        "figures": [],
        "metadata": {
            "sample_id": entry.sample_id,
            "source_kind": "gorkhapatra_language_page_pdf_native_reference_draft",
            "assisted_source": "pdf_native_legacy_conversion",
            "source_url": metadata.get("source_url"),
            "image_path": entry.image_path,
            "candidate_language": language,
            "language": language,
            "script": metadata.get("script") or "unknown",
            "date_key": metadata.get("date_key"),
            "pdf_page_number": pdf_text_sample.get("pdf_page_number") or metadata.get("pdf_page_number"),
            "language_page_marker": metadata.get("language_page_marker"),
            "review_decision": row.get("review_decision"),
            "reviewer": row.get("reviewer"),
            "reviewed_at": row.get("reviewed_at"),
            "reference_status": "draft",
            "claim_evidence_eligible": False,
            "machine_generated": True,
            "requires_human_review": True,
            "legacy_converter": converter_id,
            "legacy_map_path": str(map_path),
            "image_point_scale_x": round(scale_x, 6),
            "image_point_scale_y": round(scale_y, 6),
            "pdf_native_span_count": converted,
            "pdf_native_unmapped_span_count": flagged,
            "pdf_native_unmapped_codepoint_count": unmapped_total,
            "required_for_finalization": [
                "spot-check pre-filled Unicode against the page crop",
                "scrutinize spans flagged has_unmapped_codepoints",
                "set review_status per line (accept/corrected/drop) in the verification CSV",
                "add tables and figures where present",
                "set metadata.reference_status=verified only after human review",
                "set metadata.claim_evidence_eligible=true only after human review",
            ],
        },
    }
    return payload, converted, flagged, unmapped_total


def prepare_gorkhapatra_language_page_pdf_native_references(
    manifest_path: str | Path,
    review_csv_path: str | Path,
    pdf_text_dir: str | Path,
    output_dir: str | Path,
    *,
    candidate_languages: list[str] | None = None,
    converter_id: str = "limbu-namdhinggo",
    map_path: str | Path | None = None,
    overwrite: bool = False,
) -> GorkhapatraLanguagePageAssistedReferenceSummary:
    """Draft references from PDF-native legacy→Unicode conversion instead of OCR.

    Output drafts are non-claim-ready (``claim_evidence_eligible=false``) and feed the
    existing ``prepare-…-verification-bundle`` → ``apply-…-verification-bundle`` chain,
    so every claim gate (per-line human review, script-range validation, structural
    review declarations) is preserved. The only change versus the OCR-assisted path is
    that the draft reading order carries deterministic font-map Unicode text.
    """
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    review_csv = Path(review_csv_path)
    if not review_csv.is_file():
        raise DataValidationError(f"Gorkhapatra language-page review CSV does not exist: {review_csv}")
    pdf_text_root = Path(pdf_text_dir)
    samples_dir = pdf_text_root / "samples"
    if not samples_dir.is_dir():
        raise DataValidationError(f"pdf-text samples directory does not exist: {samples_dir}")
    converter_spec = _PDF_NATIVE_LEGACY_CONVERTERS.get(converter_id)
    if converter_spec is None:
        known = ", ".join(sorted(_PDF_NATIVE_LEGACY_CONVERTERS)) or "<none>"
        raise DataValidationError(f"unknown legacy converter id {converter_id!r}; known: {known}")
    resolved_map_path = Path(map_path) if map_path is not None else Path(converter_spec["map_path"])
    converter = LimbuLegacyConverter.from_map_file(resolved_map_path)

    selected_languages = _normalized_language_filter(candidate_languages)
    entries = {entry.sample_id: entry for entry in load_manifest(source_manifest)}
    review_rows = _read_language_page_review_rows(review_csv)
    out = Path(output_dir)
    references_dir = out / "references"
    references_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    language_counts: dict[str, int] = {}
    converted_count = 0
    failed_count = 0
    skipped_count = 0

    for row in review_rows:
        if _clean_optional_str(row.get("review_decision")) != "accept_language_page":
            skipped_count += 1
            continue
        sample_id = _clean_optional_str(row.get("sample_id"))
        if not sample_id:
            warnings.append("accepted review row missing sample_id")
            failed_count += 1
            continue
        entry = entries.get(sample_id)
        if entry is None:
            warnings.append(f"{sample_id}: accepted review row not present in source manifest")
            failed_count += 1
            continue
        language = _entry_or_review_candidate_language(entry, row)
        if selected_languages and language not in selected_languages:
            skipped_count += 1
            continue
        sample_json_path = samples_dir / f"{sample_id}.pdf-text.json"
        if not sample_json_path.is_file():
            warnings.append(f"{sample_id}: pdf-text sample not found: {sample_json_path}")
            failed_count += 1
            continue
        target = references_dir / f"{sample_id}.ref.json"
        if target.exists() and not overwrite:
            warnings.append(f"{sample_id}: pdf-native reference exists and --overwrite was not set: {target}")
            skipped_count += 1
            continue
        try:
            pdf_text_sample = _load_json_object(sample_json_path, "pdf-text sample")
            image_path = _resolve_manifest_entry_path(entry.image_path, source_manifest=source_manifest)
            if not image_path.is_file():
                raise DataValidationError(f"image_path does not exist: {image_path}")
            pdf_path = _resolve_pdf_text_pdf_path(pdf_text_sample, source_manifest=source_manifest)
            page_index = _pdf_text_page_index(pdf_text_sample)
            scale_x, scale_y = _image_pdf_point_scale(pdf_path, page_index, image_path)
            payload, span_count, flagged, unmapped_total = _pdf_native_language_page_reference(
                entry,
                row=row,
                language=language,
                pdf_text_sample=pdf_text_sample,
                converter=converter,
                converter_id=converter_id,
                map_path=resolved_map_path,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            if span_count == 0:
                raise DataValidationError("no target-font spans converted to Unicode")
        except DataValidationError as exc:
            warnings.append(f"{sample_id}: {exc}")
            failed_count += 1
            continue
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        converted_count += 1
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        index_rows.append(
            {
                "sample_id": sample_id,
                "candidate_language": language,
                "status": "converted",
                "error": "",
                "image_path": str(image_path),
                "reference_path": str(target),
                "ocr_line_count": span_count,
                "ocr_average_confidence": None,
                "pdf_native_span_count": span_count,
                "pdf_native_unmapped_span_count": flagged,
                "pdf_native_unmapped_codepoint_count": unmapped_total,
                "reference_status": "draft",
                "claim_evidence_eligible": False,
            }
        )

    if not index_rows:
        warnings.append("no pdf-native references written")
    summary = GorkhapatraLanguagePageAssistedReferenceSummary(
        output_dir=str(out),
        summary_json_path=str(out / "gorkhapatra-language-page-pdf-native-references.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-pdf-native-references.md"),
        index_csv_path=str(out / "gorkhapatra-language-page-pdf-native-references.csv"),
        source_manifest=str(source_manifest),
        review_csv_path=str(review_csv),
        engine=f"pdf-native:{converter_id}",
        sample_count=len(index_rows),
        assisted_count=converted_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        language_counts=dict(sorted(language_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_assisted_references_summary(summary, index_rows)
    return summary


def _resolve_pdf_text_pdf_path(pdf_text_sample: dict[str, Any], *, source_manifest: Path) -> Path:
    raw = _clean_optional_str(pdf_text_sample.get("pdf_path"))
    if not raw:
        raise DataValidationError("pdf-text sample missing pdf_path")
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    manifest_relative = source_manifest.parent / raw
    if manifest_relative.is_file():
        return manifest_relative
    raise DataValidationError(f"pdf_path does not exist: {raw}")


def _pdf_text_page_index(pdf_text_sample: dict[str, Any]) -> int:
    index = pdf_text_sample.get("pdf_page_index")
    if isinstance(index, int) and not isinstance(index, bool):
        return index
    page_number = pdf_text_sample.get("pdf_page_number")
    if isinstance(page_number, int) and not isinstance(page_number, bool) and page_number > 0:
        return page_number - 1
    raise DataValidationError("pdf-text sample missing pdf_page_index/pdf_page_number")


@dataclass
class GorkhapatraLanguagePageRecognizerEvalSummary:
    output_dir: str
    manifest_path: str
    summary_json_path: str
    summary_md_path: str
    source_manifest: str
    references_dir: str
    sample_count: int
    line_count: int
    crop_count: int
    skipped_count: int
    claim_eligible_crop_count: int
    language_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.crop_count > 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "source_manifest": self.source_manifest,
            "references_dir": self.references_dir,
            "sample_count": self.sample_count,
            "line_count": self.line_count,
            "crop_count": self.crop_count,
            "skipped_count": self.skipped_count,
            "claim_eligible_crop_count": self.claim_eligible_crop_count,
            "language_counts": self.language_counts,
            "warnings": self.warnings,
        }


def _reading_order_bbox_to_pixels(bbox: Any) -> list[int] | None:
    """Convert a reading-order ``[x, y, w, h]`` box to integer ``[x1, y1, x2, y2]`` for cropping."""
    cleaned = _clean_bbox_xywh(bbox)
    if cleaned is None:
        return None
    x, y, width, height = cleaned
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + width))
    y2 = int(round(y + height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _write_tight_crop(source_image: Path, crop_path: Path, bbox: list[int]) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # noqa: BLE001 - optional dependency.
        raise DataValidationError("recognizer-eval crop extraction requires Pillow. Install ocr-tech[eval].") from exc
    with Image.open(source_image) as image:
        image_width, image_height = image.size
        x1, y1, x2, y2 = bbox
        clamped = (max(0, x1), max(0, y1), min(image_width, x2), min(image_height, y2))
        if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
            raise DataValidationError(f"crop bbox outside image bounds for {source_image}: {bbox}")
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop(clamped).save(crop_path)


def prepare_gorkhapatra_language_page_recognizer_eval(
    manifest_path: str | Path,
    references_dir: str | Path,
    output_dir: str | Path,
    *,
    candidate_languages: list[str] | None = None,
    split: str = "eval",
    dataset_name: str = "gorkhapatra-language-pages-real",
    min_text_length: int = 1,
    allow_draft: bool = False,
) -> GorkhapatraLanguagePageRecognizerEvalSummary:
    """Turn verified Gorkhapatra references into a line-level PaddleOCR recognizer eval manifest.

    Each reading-order line becomes a tight line crop + a manifest row (image -> text), so the
    real held-out page plugs into ``audit-recognizer-corpus`` / benchmark scoring. Only
    ``claim_evidence_eligible`` references contribute claim-grade rows; pass ``allow_draft`` to
    smoke-test against draft references (rows are then marked non-claim-eligible).
    """
    if min_text_length < 1:
        raise DataValidationError("min_text_length must be at least 1")
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    refs_dir = Path(references_dir)
    if not refs_dir.is_dir():
        raise DataValidationError(f"references directory does not exist: {refs_dir}")
    selected_languages = _normalized_language_filter(candidate_languages)
    entries = {entry.sample_id: entry for entry in load_manifest(source_manifest)}
    out = Path(output_dir)
    images_dir = out / "images"
    manifests_dir = out / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    eval_entries: list[ManifestEntry] = []
    warnings: list[str] = []
    language_counts: dict[str, int] = {}
    sample_ids: set[str] = set()
    line_total = 0
    crop_total = 0
    skipped_total = 0
    claim_eligible_total = 0

    for reference_path in sorted(refs_dir.glob("*.ref.json")):
        sample_id = reference_path.name[: -len(".ref.json")]
        entry = entries.get(sample_id)
        if entry is None:
            warnings.append(f"{sample_id}: reference present but sample not in manifest")
            continue
        try:
            reference = _load_json_object(reference_path, "verified reference")
        except DataValidationError as exc:
            warnings.append(f"{sample_id}: {exc}")
            continue
        ref_meta = reference.get("metadata") if isinstance(reference.get("metadata"), dict) else {}
        language = (
            _clean_optional_str(ref_meta.get("candidate_language"))
            or _clean_optional_str(ref_meta.get("language"))
            or _clean_optional_str((entry.metadata or {}).get("candidate_language"))
        )
        if selected_languages and language not in selected_languages:
            continue
        claim_eligible = bool(ref_meta.get("claim_evidence_eligible"))
        if not claim_eligible and not allow_draft:
            warnings.append(f"{sample_id}: reference is not claim_evidence_eligible (use allow_draft to smoke-test)")
            continue
        reading_order = reference.get("reading_order")
        if not isinstance(reading_order, list):
            warnings.append(f"{sample_id}: reference reading_order must be a list")
            continue
        image_path = _resolve_manifest_entry_path(entry.image_path, source_manifest=source_manifest)
        if not image_path.is_file():
            warnings.append(f"{sample_id}: image_path does not exist: {image_path}")
            continue
        script = _clean_optional_str(ref_meta.get("script")) or _clean_optional_str((entry.metadata or {}).get("script"))
        sample_ids.add(sample_id)
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        for order_index, item in enumerate(reading_order, start=1):
            if not isinstance(item, dict):
                continue
            line_total += 1
            text = _clean_optional_str(item.get("text"))
            if not text or len(text) < min_text_length:
                skipped_total += 1
                continue
            bbox = _reading_order_bbox_to_pixels(item.get("bbox"))
            if bbox is None:
                skipped_total += 1
                continue
            line_id = _clean_optional_str(item.get("id")) or f"line-{order_index:04d}"
            crop_path = images_dir / f"{sample_id}-{_safe_filename(line_id)}.png"
            _write_tight_crop(image_path, crop_path, bbox)
            crop_total += 1
            if claim_eligible:
                claim_eligible_total += 1
            crop_sha = sha256_file(crop_path)
            text_sha = sha256_text(text)
            slices = sorted({*_verified_gorkhapatra_slices(language), "recognizer_eval", "line_crop"})
            eval_entries.append(
                ManifestEntry(
                    sample_id=f"{sample_id}::line::{line_id}",
                    dataset=dataset_name,
                    split=split,
                    image_path=str(crop_path),
                    text=text,
                    sha256=crop_sha,
                    metadata={
                        "slices": slices,
                        "source_sample_id": sample_id,
                        "source_manifest": str(source_manifest),
                        "source_image_path": str(image_path),
                        "source_reference_path": str(reference_path),
                        "candidate_language": language,
                        "language": language,
                        "script": script,
                        "line_id": line_id,
                        "order_index": order_index,
                        "bbox": bbox,
                        "text_sha256": text_sha,
                        "sample_sha256": sha256_text(f"{crop_sha}\n{text_sha}"),
                        "claim_evidence_eligible": claim_eligible,
                        "real_evaluation": claim_eligible,
                        "source": "gorkhapatra_verified_reference" if claim_eligible else "gorkhapatra_draft_reference",
                    },
                )
            )

    if not eval_entries:
        warnings.append("no recognizer-eval crops written")
    manifest_out = manifests_dir / "recognizer-eval.jsonl"
    if eval_entries:
        write_manifest(eval_entries, manifest_out)
    summary = GorkhapatraLanguagePageRecognizerEvalSummary(
        output_dir=str(out),
        manifest_path=str(manifest_out),
        summary_json_path=str(out / "gorkhapatra-language-page-recognizer-eval.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-recognizer-eval.md"),
        source_manifest=str(source_manifest),
        references_dir=str(refs_dir),
        sample_count=len(sample_ids),
        line_count=line_total,
        crop_count=crop_total,
        skipped_count=skipped_total,
        claim_eligible_crop_count=claim_eligible_total,
        language_counts=dict(sorted(language_counts.items())),
        warnings=warnings,
    )
    Path(summary.summary_json_path).write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_gorkhapatra_language_page_recognizer_eval_markdown(summary)
    return summary


def _write_gorkhapatra_language_page_recognizer_eval_markdown(summary: GorkhapatraLanguagePageRecognizerEvalSummary) -> None:
    lines = [
        "# Gorkhapatra Language-Page Recognizer Eval",
        "",
        f"- passed: {summary.passed}",
        f"- source manifest: {summary.source_manifest}",
        f"- references dir: {summary.references_dir}",
        f"- recognizer eval manifest: {summary.manifest_path}",
        f"- samples: {summary.sample_count}",
        f"- reading-order lines: {summary.line_count}",
        f"- line crops: {summary.crop_count}",
        f"- claim-eligible crops: {summary.claim_eligible_crop_count}",
        f"- skipped lines: {summary.skipped_count}",
        "",
        "## Language counts",
        "",
    ]
    for language, count in summary.language_counts.items():
        lines.append(f"- {language}: {count}")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    Path(summary.summary_md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_gorkhapatra_language_page_ocr_sidecars(
    manifest_path: str | Path,
    review_csv_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_languages: list[str] | None = None,
    engine: str = "paddleocr",
    tesseract_language: str = "nep+eng",
    overwrite: bool = False,
    in_place: bool = False,
) -> GorkhapatraLanguagePageOcrSidecarSummary:
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    review_csv = Path(review_csv_path)
    if not review_csv.is_file():
        raise DataValidationError(f"Gorkhapatra language-page review CSV does not exist: {review_csv}")
    selected_languages = _normalized_language_filter(candidate_languages)
    entries = {entry.sample_id: entry for entry in load_manifest(source_manifest)}
    review_rows = _read_language_page_review_rows(review_csv)

    try:
        from .engines import create_engine
    except Exception as exc:  # noqa: BLE001 - optional engine dependencies.
        raise DataValidationError(f"could not import OCR engine adapters: {exc}") from exc
    engine_kwargs: dict[str, Any] = {}
    if engine in {"tesseract", "tess"}:
        engine_kwargs["language"] = tesseract_language
    try:
        ocr_engine = create_engine(engine, **engine_kwargs)
    except Exception as exc:  # noqa: BLE001 - engine availability is runtime-specific.
        raise DataValidationError(f"could not initialize OCR engine {engine}: {exc}") from exc

    out = Path(output_dir)
    sidecars_dir = out / "sidecars"
    out.mkdir(parents=True, exist_ok=True)
    if not in_place:
        sidecars_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    language_counts: dict[str, int] = {}
    sidecar_count = 0
    failed_count = 0
    skipped_count = 0

    for row in review_rows:
        if _clean_optional_str(row.get("review_decision")) != "accept_language_page":
            skipped_count += 1
            continue
        sample_id = _clean_optional_str(row.get("sample_id"))
        if not sample_id:
            warnings.append("accepted review row missing sample_id")
            failed_count += 1
            continue
        entry = entries.get(sample_id)
        if entry is None:
            warnings.append(f"{sample_id}: accepted review row not present in source manifest")
            failed_count += 1
            continue
        language = _entry_or_review_candidate_language(entry, row)
        if selected_languages and language not in selected_languages:
            skipped_count += 1
            continue
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        image_path = _resolve_manifest_entry_path(entry.image_path, source_manifest=source_manifest)
        sidecar_path = image_path.with_name(image_path.name + ".ocr.json") if in_place else sidecars_dir / f"{sample_id}.ocr.json"
        if sidecar_path.exists() and not overwrite:
            warnings.append(f"{sample_id}: OCR sidecar exists and --overwrite was not set: {sidecar_path}")
            skipped_count += 1
            continue
        status = "sidecar"
        error = ""
        line_count = 0
        confidence: float | None = None
        try:
            output = ocr_engine.recognize(image_path)
            payload = _engine_output_sidecar_payload(output, engine_name=engine, image_path=image_path)
            line_count = sum(len(page.get("lines", [])) for page in payload["pages"])
            confidence = _sidecar_average_confidence(payload)
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            sidecar_count += 1
        except Exception as exc:  # noqa: BLE001 - row-level OCR failures should be auditable.
            status = "failed"
            error = str(exc)
            failed_count += 1
        rows.append(
            {
                "sample_id": sample_id,
                "candidate_language": language,
                "status": status,
                "error": error,
                "image_path": str(image_path),
                "sidecar_path": str(sidecar_path),
                "ocr_line_count": line_count,
                "ocr_average_confidence": confidence,
            }
        )

    if not rows:
        warnings.append("no OCR sidecar rows written")
    summary = GorkhapatraLanguagePageOcrSidecarSummary(
        output_dir=str(out),
        summary_json_path=str(out / "gorkhapatra-language-page-ocr-sidecars.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-ocr-sidecars.md"),
        index_csv_path=str(out / "gorkhapatra-language-page-ocr-sidecars.csv"),
        source_manifest=str(source_manifest),
        review_csv_path=str(review_csv),
        engine=engine,
        in_place=in_place,
        sample_count=len(rows),
        sidecar_count=sidecar_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        language_counts=dict(sorted(language_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_ocr_sidecars_summary(summary, rows)
    return summary


def prepare_gorkhapatra_language_page_verification_bundle(
    manifest_path: str | Path,
    review_csv_path: str | Path,
    assisted_references_dir: str | Path,
    output_dir: str | Path,
    *,
    candidate_languages: list[str] | None = None,
    overwrite: bool = False,
) -> GorkhapatraLanguagePageVerificationBundleSummary:
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    review_csv = Path(review_csv_path)
    if not review_csv.is_file():
        raise DataValidationError(f"Gorkhapatra language-page review CSV does not exist: {review_csv}")
    assisted_dir = Path(assisted_references_dir)
    references_dir = assisted_dir / "references"
    if not references_dir.is_dir():
        raise DataValidationError(f"assisted references directory does not exist: {references_dir}")
    selected_languages = _normalized_language_filter(candidate_languages)
    entries = {entry.sample_id: entry for entry in load_manifest(source_manifest)}
    review_rows = _read_language_page_review_rows(review_csv)
    out = Path(output_dir)
    crops_dir = out / "line-crops"
    out.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    language_counts: dict[str, int] = {}
    sample_ids: set[str] = set()
    skipped_count = 0
    crop_count = 0
    missing_bbox_count = 0
    for row in review_rows:
        if _clean_optional_str(row.get("review_decision")) != "accept_language_page":
            skipped_count += 1
            continue
        sample_id = _clean_optional_str(row.get("sample_id"))
        if not sample_id:
            warnings.append("accepted review row missing sample_id")
            continue
        entry = entries.get(sample_id)
        if entry is None:
            warnings.append(f"{sample_id}: accepted review row not present in source manifest")
            continue
        language = _entry_or_review_candidate_language(entry, row)
        if selected_languages and language not in selected_languages:
            skipped_count += 1
            continue
        reference_path = references_dir / f"{sample_id}.ref.json"
        if not reference_path.is_file():
            warnings.append(f"{sample_id}: assisted reference does not exist: {reference_path}")
            continue
        try:
            reference = _load_json_object(reference_path, "assisted reference")
        except DataValidationError as exc:
            warnings.append(f"{sample_id}: {exc}")
            continue
        reading_order = reference.get("reading_order")
        if not isinstance(reading_order, list):
            warnings.append(f"{sample_id}: assisted reference reading_order must be a list")
            continue
        image_path = _resolve_manifest_entry_path(entry.image_path, source_manifest=source_manifest)
        if not image_path.is_file():
            warnings.append(f"{sample_id}: image_path does not exist: {image_path}")
            continue
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        sample_ids.add(sample_id)
        for order_index, item in enumerate(reading_order, start=1):
            if not isinstance(item, dict):
                warnings.append(f"{sample_id}: reading_order item {order_index} is not an object")
                continue
            text = _clean_optional_str(item.get("text"))
            if not text:
                continue
            line_id = _clean_optional_str(item.get("id")) or f"line-{order_index:04d}"
            bbox = _clean_bbox_xywh(item.get("bbox"))
            crop_path = ""
            crop_note = ""
            if bbox is None:
                missing_bbox_count += 1
            else:
                crop_target = crops_dir / sample_id / f"{_safe_filename(line_id)}.png"
                crop_note = _xywh_crop_note(image_path, bbox)
                if crop_target.exists() and not overwrite:
                    crop_path = str(crop_target)
                else:
                    _write_xywh_crop(image_path, crop_target, bbox)
                crop_path = str(crop_target)
                crop_count += 1
            item_note = _clean_optional_str(item.get("review_note")) or ""
            combined_note = "; ".join(part for part in (item_note, crop_note) if part)
            rows.append(
                {
                    "sample_id": sample_id,
                    "candidate_language": language,
                    "line_id": line_id,
                    "order_index": order_index,
                    "page_index": item.get("page_index", 0),
                    "ocr_text": text,
                    "ocr_confidence": item.get("confidence"),
                    "bbox": json.dumps(bbox, ensure_ascii=False) if bbox is not None else "",
                    "crop_path": crop_path,
                    "review_text": "",
                    "review_status": "pending",
                    "notes": combined_note,
                }
            )
    if not rows:
        warnings.append("no verification rows written")
    summary = GorkhapatraLanguagePageVerificationBundleSummary(
        output_dir=str(out),
        summary_json_path=str(out / "gorkhapatra-language-page-verification-bundle.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-verification-bundle.md"),
        index_csv_path=str(out / "gorkhapatra-language-page-verification-lines.csv"),
        review_html_path=str(out / "gorkhapatra-language-page-verification-review.html"),
        source_manifest=str(source_manifest),
        review_csv_path=str(review_csv),
        assisted_references_dir=str(assisted_dir),
        sample_count=len(sample_ids),
        line_count=len(rows),
        crop_count=crop_count,
        missing_bbox_count=missing_bbox_count,
        skipped_count=skipped_count,
        language_counts=dict(sorted(language_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_verification_bundle_summary(summary, rows)
    return summary


def apply_gorkhapatra_language_page_verification_bundle(
    manifest_path: str | Path,
    review_csv_path: str | Path,
    verification_csv_path: str | Path,
    assisted_references_dir: str | Path,
    output_dir: str | Path,
    *,
    candidate_languages: list[str] | None = None,
    reviewer: str | None = None,
    reviewed_at: str | None = None,
    tables_reviewed: bool = False,
    figures_reviewed: bool = False,
    captions_reviewed: bool = False,
    required_codepoint_ranges: list[str] | None = None,
    overwrite: bool = False,
) -> GorkhapatraLanguagePageVerificationApplySummary:
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    review_csv = Path(review_csv_path)
    if not review_csv.is_file():
        raise DataValidationError(f"Gorkhapatra language-page review CSV does not exist: {review_csv}")
    verification_csv = Path(verification_csv_path)
    if not verification_csv.is_file():
        raise DataValidationError(f"Gorkhapatra verification CSV does not exist: {verification_csv}")
    assisted_dir = Path(assisted_references_dir)
    references_dir = assisted_dir / "references"
    if not references_dir.is_dir():
        raise DataValidationError(f"assisted references directory does not exist: {references_dir}")
    structural_missing = [
        label
        for label, reviewed in (
            ("tables", tables_reviewed),
            ("figures", figures_reviewed),
            ("captions", captions_reviewed),
        )
        if not reviewed
    ]

    entries = {entry.sample_id: entry for entry in load_manifest(source_manifest)}
    review_rows = _read_language_page_review_rows(review_csv)
    verification_rows = _read_language_page_verification_rows(verification_csv)
    selected_languages = _normalized_language_filter(candidate_languages)
    required_ranges = _parse_codepoint_ranges(required_codepoint_ranges)
    verification_by_sample = _group_verification_rows_by_sample(verification_rows)
    out = Path(output_dir)
    verified_refs_dir = out / "references"
    out.mkdir(parents=True, exist_ok=True)
    verified_refs_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    updated_review_rows: list[dict[str, str]] = []
    verified_sample_ids: set[str] = set()
    blocked_sample_ids: set[str] = set()
    language_counts: dict[str, int] = {}
    reviewed_line_count = 0
    dropped_line_count = 0

    for review_row in review_rows:
        updated_row = dict(review_row)
        sample_id = _clean_optional_str(review_row.get("sample_id"))
        decision = _clean_optional_str(review_row.get("review_decision"))
        if decision != "accept_language_page":
            updated_review_rows.append(updated_row)
            continue
        if not sample_id:
            warnings.append("accepted review row missing sample_id")
            updated_review_rows.append(updated_row)
            continue
        entry = entries.get(sample_id)
        if entry is None:
            warnings.append(f"{sample_id}: accepted review row not present in source manifest")
            updated_review_rows.append(updated_row)
            continue
        language = _entry_or_review_candidate_language(entry, review_row)
        if selected_languages and language not in selected_languages:
            updated_review_rows.append(updated_row)
            continue
        if structural_missing:
            warnings.append(f"{sample_id}: structural review declarations missing: {', '.join(structural_missing)}")
            blocked_sample_ids.add(sample_id)
            updated_review_rows.append(updated_row)
            continue
        rows = verification_by_sample.get(sample_id, [])
        if not rows:
            warnings.append(f"{sample_id}: no verification rows found")
            blocked_sample_ids.add(sample_id)
            updated_review_rows.append(updated_row)
            continue
        reference_path = references_dir / f"{sample_id}.ref.json"
        try:
            assisted_reference = _load_json_object(reference_path, "assisted reference")
            reference, line_count, drop_count = _verified_reference_from_line_reviews(
                sample_id=sample_id,
                language=language,
                assisted_reference=assisted_reference,
                assisted_reference_path=reference_path,
                verification_rows=rows,
                verification_csv=verification_csv,
                reviewer=reviewer,
                reviewed_at=reviewed_at,
                tables_reviewed=tables_reviewed,
                figures_reviewed=figures_reviewed,
                captions_reviewed=captions_reviewed,
                required_codepoint_ranges=required_ranges,
            )
        except DataValidationError as exc:
            warnings.append(f"{sample_id}: {exc}")
            blocked_sample_ids.add(sample_id)
            updated_review_rows.append(updated_row)
            continue
        target = verified_refs_dir / f"{sample_id}.ref.json"
        if target.exists() and not overwrite:
            warnings.append(f"{sample_id}: verified reference already exists and --overwrite was not set: {target}")
            blocked_sample_ids.add(sample_id)
            updated_review_rows.append(updated_row)
            continue
        target.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        updated_row["reference_path"] = str(Path("references") / target.name)
        verified_sample_ids.add(sample_id)
        reviewed_line_count += line_count
        dropped_line_count += drop_count
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        updated_review_rows.append(updated_row)

    updated_review_csv = out / "gorkhapatra-language-page-review-with-verified-references.csv"
    _write_updated_language_page_review_csv(review_csv, updated_review_csv, updated_review_rows)
    summary = GorkhapatraLanguagePageVerificationApplySummary(
        output_dir=str(out),
        summary_json_path=str(out / "gorkhapatra-language-page-verification-apply.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-verification-apply.md"),
        updated_review_csv_path=str(updated_review_csv),
        references_dir=str(verified_refs_dir),
        source_manifest=str(source_manifest),
        review_csv_path=str(review_csv),
        verification_csv_path=str(verification_csv),
        assisted_references_dir=str(assisted_dir),
        sample_count=len(verified_sample_ids | blocked_sample_ids),
        verified_reference_count=len(verified_sample_ids),
        blocked_count=len(blocked_sample_ids),
        reviewed_line_count=reviewed_line_count,
        dropped_line_count=dropped_line_count,
        language_counts=dict(sorted(language_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_verification_apply_summary(summary)
    return summary


def audit_gorkhapatra_language_page_verification_csv(
    verification_csv_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_languages: list[str] | None = None,
    required_codepoint_ranges: list[str] | None = None,
) -> GorkhapatraLanguagePageVerificationCsvAuditSummary:
    verification_csv = Path(verification_csv_path)
    if not verification_csv.is_file():
        raise DataValidationError(f"Gorkhapatra verification CSV does not exist: {verification_csv}")
    selected_languages = _normalized_language_filter(candidate_languages)
    required_ranges = _parse_codepoint_ranges(required_codepoint_ranges)
    rows = [
        row
        for row in _read_language_page_verification_rows(verification_csv)
        if not selected_languages or _clean_optional_str(row.get("candidate_language")) in selected_languages
    ]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    warnings: list[str] = []
    status_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    sample_ids: set[str] = set()
    sample_line_ids: dict[str, set[str]] = {}
    sample_order_indices: dict[str, set[int]] = {}
    ready_line_count = 0
    dropped_line_count = 0
    blocked_line_count = 0
    for row_index, row in enumerate(rows, start=2):
        sample_id = _clean_optional_str(row.get("sample_id"))
        line_id = _clean_optional_str(row.get("line_id"))
        status = _clean_optional_str(row.get("review_status")) or "<empty>"
        status_counts[status] = status_counts.get(status, 0) + 1
        language = _clean_optional_str(row.get("candidate_language"))
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        if not sample_id:
            blocked_line_count += 1
            issues.append(f"row {row_index}: missing sample_id")
            continue
        sample_ids.add(sample_id)
        if not line_id:
            blocked_line_count += 1
            issues.append(f"{sample_id}: row {row_index} missing line_id")
            continue
        seen_line_ids = sample_line_ids.setdefault(sample_id, set())
        if line_id in seen_line_ids:
            blocked_line_count += 1
            issues.append(f"{sample_id}: duplicate line_id {line_id}")
        seen_line_ids.add(line_id)
        order_index = _verification_order_index(row)
        if order_index == 10**9:
            warnings.append(f"{sample_id}:{line_id}: missing or invalid order_index")
        else:
            seen_order_indices = sample_order_indices.setdefault(sample_id, set())
            if order_index in seen_order_indices:
                blocked_line_count += 1
                issues.append(f"{sample_id}: duplicate order_index {order_index}")
            seen_order_indices.add(order_index)
        if status == "accept":
            text = _clean_optional_str(row.get("ocr_text"))
            if not text:
                blocked_line_count += 1
                issues.append(f"{sample_id}:{line_id}: accept row has empty ocr_text")
            elif required_ranges and not _text_contains_codepoint_in_ranges(text, required_ranges):
                blocked_line_count += 1
                issues.append(f"{sample_id}:{line_id}: accepted text has no character in required range {_format_codepoint_ranges(required_ranges)}")
            else:
                ready_line_count += 1
            continue
        if status == "corrected":
            text = _clean_optional_str(row.get("review_text"))
            if not text:
                blocked_line_count += 1
                issues.append(f"{sample_id}:{line_id}: corrected row has empty review_text")
            elif required_ranges and not _text_contains_codepoint_in_ranges(text, required_ranges):
                blocked_line_count += 1
                issues.append(f"{sample_id}:{line_id}: corrected text has no character in required range {_format_codepoint_ranges(required_ranges)}")
            else:
                ready_line_count += 1
            continue
        if status == "drop":
            dropped_line_count += 1
            continue
        blocked_line_count += 1
        if status in {"pending", "needs_context", "bad_segmentation", "needs_resegmentation", "<empty>"}:
            issues.append(f"{sample_id}:{line_id}: review_status is {status}")
        else:
            issues.append(f"{sample_id}:{line_id}: unsupported review_status {status}")
    if not rows:
        issues.append("verification CSV has no rows after filtering")
    summary = GorkhapatraLanguagePageVerificationCsvAuditSummary(
        verification_csv_path=str(verification_csv),
        output_dir=str(out),
        summary_json_path=str(out / "gorkhapatra-language-page-verification-csv-audit.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-verification-csv-audit.md"),
        sample_count=len(sample_ids),
        line_count=len(rows),
        ready_line_count=ready_line_count,
        dropped_line_count=dropped_line_count,
        blocked_line_count=blocked_line_count,
        status_counts=dict(sorted(status_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        issues=issues,
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_verification_csv_audit(summary)
    return summary


def split_gorkhapatra_language_page_verification_csv(
    verification_csv_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 50,
    candidate_languages: list[str] | None = None,
) -> GorkhapatraLanguagePageVerificationSplitSummary:
    if batch_size < 1:
        raise DataValidationError("batch_size must be at least 1")
    verification_csv = Path(verification_csv_path)
    if not verification_csv.is_file():
        raise DataValidationError(f"Gorkhapatra verification CSV does not exist: {verification_csv}")
    rows, fieldnames = _read_language_page_verification_rows_with_fieldnames(verification_csv)
    selected_languages = _normalized_language_filter(candidate_languages)
    if selected_languages:
        rows = [row for row in rows if _clean_optional_str(row.get("candidate_language")) in selected_languages]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    status_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    sample_ids: set[str] = set()
    for row in rows:
        sample_id = _clean_optional_str(row.get("sample_id"))
        if sample_id:
            sample_ids.add(sample_id)
        status = _clean_optional_str(row.get("review_status")) or "<empty>"
        status_counts[status] = status_counts.get(status, 0) + 1
        language = _clean_optional_str(row.get("candidate_language"))
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
    if not rows:
        warnings.append("no verification rows after filtering")

    batch_rows: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(rows), batch_size), start=1):
        chunk = rows[start : start + batch_size]
        batch_id = f"batch-{batch_index:04d}"
        batch_dir = out / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        csv_path = batch_dir / "verification-lines.csv"
        html_path = batch_dir / "verification-review.html"
        _write_verification_csv_rows(csv_path, chunk, fieldnames)
        _write_verification_rows_review_html(
            title=f"Gorkhapatra Verification {batch_id}",
            csv_path=csv_path,
            rows=chunk,
            html_path=html_path,
        )
        batch_status_counts: dict[str, int] = {}
        for row in chunk:
            status = _clean_optional_str(row.get("review_status")) or "<empty>"
            batch_status_counts[status] = batch_status_counts.get(status, 0) + 1
        batch_rows.append(
            {
                "batch_id": batch_id,
                "line_count": len(chunk),
                "start_order_index": chunk[0].get("order_index") if chunk else "",
                "end_order_index": chunk[-1].get("order_index") if chunk else "",
                "csv_path": str(csv_path),
                "review_html_path": str(html_path),
                "status_counts": json.dumps(dict(sorted(batch_status_counts.items())), ensure_ascii=False),
            }
        )
    summary = GorkhapatraLanguagePageVerificationSplitSummary(
        verification_csv_path=str(verification_csv),
        output_dir=str(out),
        summary_json_path=str(out / "gorkhapatra-language-page-verification-split.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-verification-split.md"),
        index_csv_path=str(out / "gorkhapatra-language-page-verification-split.csv"),
        batch_size=batch_size,
        sample_count=len(sample_ids),
        line_count=len(rows),
        batch_count=len(batch_rows),
        status_counts=dict(sorted(status_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_verification_split_summary(summary, batch_rows)
    return summary


def merge_gorkhapatra_language_page_verification_batches(
    source_verification_csv_path: str | Path,
    batches_dir: str | Path,
    output_dir: str | Path,
) -> GorkhapatraLanguagePageVerificationMergeSummary:
    source_csv = Path(source_verification_csv_path)
    if not source_csv.is_file():
        raise DataValidationError(f"Gorkhapatra source verification CSV does not exist: {source_csv}")
    batch_root = Path(batches_dir)
    if not batch_root.is_dir():
        raise DataValidationError(f"Gorkhapatra verification batches directory does not exist: {batch_root}")
    source_rows, source_fieldnames = _read_language_page_verification_rows_with_fieldnames(source_csv)
    source_order: list[tuple[str, str]] = []
    source_keys: set[tuple[str, str]] = set()
    issues: list[str] = []
    warnings: list[str] = []
    for row_index, row in enumerate(source_rows, start=2):
        key = _verification_row_key(row)
        if key is None:
            issues.append(f"source row {row_index}: missing sample_id or line_id")
            continue
        if key in source_keys:
            issues.append(f"source row {row_index}: duplicate source key {_format_verification_key(key)}")
            continue
        source_keys.add(key)
        source_order.append(key)

    batch_files = sorted(batch_root.glob("batch-*/verification-lines.csv"))
    if not batch_files:
        issues.append(f"no batch verification CSVs found under {batch_root}/batch-*/verification-lines.csv")
    merged_by_key: dict[tuple[str, str], dict[str, str]] = {}
    merged_fieldnames = list(source_fieldnames)
    for batch_file in batch_files:
        batch_rows, batch_fieldnames = _read_language_page_verification_rows_with_fieldnames(batch_file)
        for field_name in batch_fieldnames:
            if field_name not in merged_fieldnames:
                merged_fieldnames.append(field_name)
        for row_index, row in enumerate(batch_rows, start=2):
            key = _verification_row_key(row)
            if key is None:
                issues.append(f"{batch_file}: row {row_index}: missing sample_id or line_id")
                continue
            if key not in source_keys:
                issues.append(f"{batch_file}: row {row_index}: unknown line {_format_verification_key(key)}")
                continue
            if key in merged_by_key:
                issues.append(f"{batch_file}: row {row_index}: duplicate merged line {_format_verification_key(key)}")
                continue
            merged_by_key[key] = row
    missing = [key for key in source_order if key not in merged_by_key]
    if missing:
        issues.extend(f"missing reviewed line {_format_verification_key(key)}" for key in missing[:200])
        if len(missing) > 200:
            warnings.append(f"{len(missing) - 200} additional missing reviewed lines omitted from issue list")
    merged_rows = [merged_by_key[key] for key in source_order if key in merged_by_key]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged_csv = out / "gorkhapatra-language-page-verification-lines-merged.csv"
    _write_verification_csv_rows(merged_csv, merged_rows, merged_fieldnames)
    summary = GorkhapatraLanguagePageVerificationMergeSummary(
        source_verification_csv_path=str(source_csv),
        batches_dir=str(batch_root),
        output_dir=str(out),
        merged_csv_path=str(merged_csv),
        summary_json_path=str(out / "gorkhapatra-language-page-verification-merge.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-verification-merge.md"),
        source_line_count=len(source_rows),
        merged_line_count=len(merged_rows),
        batch_file_count=len(batch_files),
        issues=issues,
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_verification_merge_summary(summary)
    return summary


def assign_gorkhapatra_language_page_verification_batches(
    split_index_csv_path: str | Path,
    output_dir: str | Path,
    *,
    reviewers: list[str] | None = None,
    due_date: str | None = None,
) -> GorkhapatraLanguagePageVerificationAssignmentSummary:
    split_index = Path(split_index_csv_path)
    if not split_index.is_file():
        raise DataValidationError(f"Gorkhapatra verification split index CSV does not exist: {split_index}")
    with split_index.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DataValidationError(f"Gorkhapatra verification split index CSV has no header: {split_index}")
        required = {"batch_id", "line_count", "csv_path", "review_html_path"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise DataValidationError(f"Gorkhapatra verification split index missing fields: {', '.join(missing)}")
        batches = [dict(row) for row in reader]
    reviewer_list = [
        reviewer
        for reviewer in (_clean_optional_str(value) for value in (reviewers or []))
        if reviewer
    ]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assignment_rows: list[dict[str, Any]] = []
    reviewer_counts: dict[str, int] = {}
    for index, batch in enumerate(batches):
        reviewer = reviewer_list[index % len(reviewer_list)] if reviewer_list else ""
        if reviewer:
            reviewer_counts[reviewer] = reviewer_counts.get(reviewer, 0) + 1
        assignment_rows.append(
            {
                "batch_id": batch.get("batch_id", ""),
                "reviewer": reviewer,
                "assignment_status": "assigned" if reviewer else "unassigned",
                "due_date": due_date or "",
                "line_count": batch.get("line_count", ""),
                "start_order_index": batch.get("start_order_index", ""),
                "end_order_index": batch.get("end_order_index", ""),
                "csv_path": batch.get("csv_path", ""),
                "review_html_path": batch.get("review_html_path", ""),
                "completed_at": "",
                "notes": "",
            }
        )
    warnings: list[str] = []
    if not batches:
        warnings.append("split index has no batch rows")
    if not reviewer_list:
        warnings.append("no reviewers supplied; assignment rows are unassigned")
    assignment_csv = out / "gorkhapatra-language-page-verification-assignments.csv"
    summary = GorkhapatraLanguagePageVerificationAssignmentSummary(
        split_index_csv_path=str(split_index),
        output_dir=str(out),
        assignment_csv_path=str(assignment_csv),
        summary_json_path=str(out / "gorkhapatra-language-page-verification-assignments.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-verification-assignments.md"),
        batch_count=len(batches),
        assigned_count=sum(1 for row in assignment_rows if row["assignment_status"] == "assigned"),
        reviewer_counts=dict(sorted(reviewer_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_verification_assignment_summary(summary, assignment_rows)
    return summary


def _reviewer_bundle_reference_source(
    sample_id: str,
    *,
    row: dict[str, str],
    metadata: dict[str, Any],
    source_manifest: Path,
    review_csv: Path,
    template_dir: Path | None,
) -> Path | None:
    if template_dir is not None:
        candidate = template_dir / "references" / f"{sample_id}.ref.json"
        if candidate.is_file():
            return candidate
        candidate = template_dir / f"{sample_id}.ref.json"
        if candidate.is_file():
            return candidate
    reference_path_value = _clean_optional_str(row.get("reference_path")) or _clean_optional_str(metadata.get("reference_path"))
    if not reference_path_value:
        return None
    reference_path = _resolve_review_reference_path(reference_path_value, source_manifest=source_manifest, review_csv=review_csv)
    return reference_path if reference_path.is_file() else None


def audit_gorkhapatra_language_page_review(
    manifest_path: str | Path,
    review_csv_path: str | Path,
    output_dir: str | Path,
    *,
    require_verified_references: bool = False,
    candidate_languages: list[str] | None = None,
) -> GorkhapatraLanguagePageReviewAuditSummary:
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    review_csv = Path(review_csv_path)
    if not review_csv.is_file():
        raise DataValidationError(f"Gorkhapatra language-page review CSV does not exist: {review_csv}")
    entries = load_manifest(source_manifest)
    review_rows = _read_language_page_review_rows(review_csv)
    rows_by_sample_id = {_clean_optional_str(row.get("sample_id")): row for row in review_rows if _clean_optional_str(row.get("sample_id"))}
    selected_languages = _normalized_language_filter(candidate_languages)
    if selected_languages:
        entries = [
            entry
            for entry in entries
            if _entry_or_review_candidate_language(entry, rows_by_sample_id.get(entry.sample_id)) in selected_languages
        ]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    issues: list[str] = []
    warnings: list[str] = []
    groups: dict[str, list[ManifestEntry]] = {}
    accepted_count = 0
    rejected_count = 0
    unresolved_count = 0
    missing_reference_count = 0
    verified_reference_count = 0

    for entry in entries:
        metadata = entry.metadata or {}
        if metadata.get("source_kind") != "language_page_candidate":
            warnings.append(f"{entry.sample_id}: source_kind is not language_page_candidate: {metadata.get('source_kind')}")
        group_key = _language_page_review_group_key(entry)
        groups.setdefault(group_key, []).append(entry)
        row = rows_by_sample_id.get(entry.sample_id)
        if row is None:
            issues.append(f"{entry.sample_id}: missing review row")
            unresolved_count += 1
            continue
        decision = _clean_optional_str(row.get("review_decision"))
        if decision == "accept_language_page":
            accepted_count += 1
            reference_path_value = _clean_optional_str(row.get("reference_path")) or _clean_optional_str(metadata.get("reference_path"))
            if not reference_path_value:
                issues.append(f"{entry.sample_id}: accepted row missing reference_path")
                missing_reference_count += 1
                continue
            reference_path = _resolve_review_reference_path(reference_path_value, source_manifest=source_manifest, review_csv=review_csv)
            if not reference_path.is_file():
                issues.append(f"{entry.sample_id}: accepted row reference_path does not exist: {reference_path}")
                missing_reference_count += 1
                continue
            if require_verified_references:
                try:
                    _load_verified_language_page_reference(reference_path, sample_id=entry.sample_id)
                    verified_reference_count += 1
                except DataValidationError as exc:
                    issues.append(f"{entry.sample_id}: {exc}")
            continue
        if decision in {"reject_wrong_language", "reject_not_language_page", "duplicate_candidate"}:
            rejected_count += 1
            continue
        unresolved_count += 1
        issues.append(f"{entry.sample_id}: unresolved review_decision {decision or '<empty>'}")

    duplicate_accept_group_count = 0
    for group_key, group_entries in sorted(groups.items()):
        accepted_entries = [
            entry
            for entry in group_entries
            if _clean_optional_str(rows_by_sample_id.get(entry.sample_id, {}).get("review_decision")) == "accept_language_page"
        ]
        if len(accepted_entries) == 0:
            issues.append(f"{group_key}: no accepted language page candidate")
        elif len(accepted_entries) > 1:
            duplicate_accept_group_count += 1
            accepted_ids = ", ".join(entry.sample_id for entry in accepted_entries)
            issues.append(f"{group_key}: multiple accepted language page candidates: {accepted_ids}")

    entry_sample_ids = {entry.sample_id for entry in entries}
    extra_rows = sorted(
        sample_id
        for sample_id, row in rows_by_sample_id.items()
        if sample_id not in entry_sample_ids
        and (not selected_languages or _clean_optional_str(row.get("candidate_language")) in selected_languages)
    )
    for sample_id in extra_rows:
        warnings.append(f"{sample_id}: review row not present in source manifest")

    summary = GorkhapatraLanguagePageReviewAuditSummary(
        source_manifest=str(source_manifest),
        review_csv_path=str(review_csv),
        summary_json_path=str(out / "gorkhapatra-language-page-review-audit.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-review-audit.md"),
        sample_count=len(entries),
        group_count=len(groups),
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        unresolved_count=unresolved_count,
        duplicate_accept_group_count=duplicate_accept_group_count,
        missing_reference_count=missing_reference_count,
        verified_reference_count=verified_reference_count,
        require_verified_references=require_verified_references,
        issues=issues,
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_review_audit(summary)
    return summary


def finalize_gorkhapatra_language_page_review(
    manifest_path: str | Path,
    review_csv_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "eval",
    dataset_name: str = "gorkhapatra-language-pages-verified",
    candidate_languages: list[str] | None = None,
) -> GorkhapatraLanguagePageFinalizeSummary:
    source_manifest = Path(manifest_path)
    if not source_manifest.is_file():
        raise DataValidationError(f"Gorkhapatra language-page pack manifest does not exist: {source_manifest}")
    review_csv = Path(review_csv_path)
    if not review_csv.is_file():
        raise DataValidationError(f"Gorkhapatra language-page review CSV does not exist: {review_csv}")
    entries = {entry.sample_id: entry for entry in load_manifest(source_manifest)}
    review_rows = _read_language_page_review_rows(review_csv)
    selected_languages = _normalized_language_filter(candidate_languages)
    out = Path(output_dir)
    manifests_dir = out / "manifests"
    refs_dir = out / "references"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    finalized: list[ManifestEntry] = []
    warnings: list[str] = []
    accepted_count = 0
    skipped_count = 0
    language_counts: dict[str, int] = {}
    for row in review_rows:
        sample_id = _clean_optional_str(row.get("sample_id"))
        decision = _clean_optional_str(row.get("review_decision"))
        if decision != "accept_language_page":
            skipped_count += 1
            continue
        accepted_count += 1
        if not sample_id:
            warnings.append("accepted review row missing sample_id")
            continue
        entry = entries.get(sample_id)
        if entry is None:
            warnings.append(f"{sample_id}: accepted review row not present in source manifest")
            continue
        if selected_languages and _entry_or_review_candidate_language(entry, row) not in selected_languages:
            skipped_count += 1
            accepted_count -= 1
            continue
        metadata = dict(entry.metadata or {})
        reference_path_value = _clean_optional_str(row.get("reference_path")) or _clean_optional_str(metadata.get("reference_path"))
        if not reference_path_value:
            warnings.append(f"{sample_id}: accepted review row missing reference_path")
            continue
        reference_path = _resolve_review_reference_path(reference_path_value, source_manifest=source_manifest, review_csv=review_csv)
        try:
            reference = _load_verified_language_page_reference(reference_path, sample_id=sample_id)
        except DataValidationError as exc:
            warnings.append(f"{sample_id}: {exc}")
            continue
        image_path = _resolve_manifest_entry_path(entry.image_path, source_manifest=source_manifest)
        if not image_path.is_file():
            warnings.append(f"{sample_id}: image_path does not exist: {image_path}")
            continue
        reference_copy = refs_dir / f"{sample_id}.ref.json"
        shutil.copy2(reference_path, reference_copy)
        text = str(reference.get("text") or "")
        image_hash = sha256_file(image_path)
        reference_hash = sha256_file(reference_copy)
        language = _clean_optional_str(row.get("candidate_language")) or _clean_optional_str(metadata.get("candidate_language")) or _clean_optional_str(metadata.get("language"))
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        reference_metadata = reference.get("metadata") if isinstance(reference.get("metadata"), dict) else {}
        new_metadata = dict(metadata)
        new_metadata.update(
            {
                "source_kind": "gorkhapatra_language_page_verified",
                "document_type": "language_page",
                "reference_path": str(reference_copy),
                "reference_sha256": reference_hash,
                "reference_status": "verified",
                "claim_evidence_eligible": True,
                "page_disambiguation_status": "reviewed_accepted",
                "review_decision": decision,
                "reviewer": row.get("reviewer"),
                "reviewed_at": row.get("reviewed_at"),
                "language": language,
                "script": _clean_optional_str(reference_metadata.get("script")) or metadata.get("script") or "unknown",
                "text_sha256": sha256_text(text),
                "sample_sha256": sha256_text(f"{image_hash}\n{sha256_text(text)}\n{reference_hash}\n"),
                "slices": _verified_gorkhapatra_slices(language),
            }
        )
        finalized.append(
            ManifestEntry(
                sample_id=entry.sample_id,
                dataset=dataset_name,
                split=split,
                image_path=entry.image_path,
                text=text,
                sha256=image_hash,
                metadata=new_metadata,
            )
        )
    if accepted_count == 0:
        warnings.append("no accepted language page review rows found")
    manifest_out = manifests_dir / "gorkhapatra-language-page-verified.jsonl"
    write_manifest(finalized, manifest_out)
    summary = GorkhapatraLanguagePageFinalizeSummary(
        manifest_path=str(manifest_out),
        summary_json_path=str(out / "gorkhapatra-language-page-finalize.json"),
        summary_md_path=str(out / "gorkhapatra-language-page-finalize.md"),
        source_manifest=str(source_manifest),
        review_csv_path=str(review_csv),
        sample_count=len(finalized),
        accepted_review_count=accepted_count,
        skipped_review_count=skipped_count,
        language_counts=dict(sorted(language_counts.items())),
        warnings=warnings,
    )
    _write_gorkhapatra_language_page_finalize_summary(summary)
    return summary


def _network_retry_delay_seconds(base_delay: float) -> float:
    return base_delay + random.uniform(0.0, min(_NETWORK_RETRY_JITTER_SECONDS, base_delay * 0.1))


def _run_network_fetch_with_retries(description: str, fetch: Callable[[], _T]) -> _T:
    attempts = len(_NETWORK_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(1, attempts + 1):
        try:
            return fetch()
        except _TRANSIENT_NETWORK_EXCEPTIONS as exc:
            if attempt == attempts:
                LOGGER.error(
                    "network fetch failed after %d attempts for %s: %s: %s",
                    attempts,
                    description,
                    type(exc).__name__,
                    exc,
                )
                raise
            delay = _network_retry_delay_seconds(_NETWORK_RETRY_DELAYS_SECONDS[attempt - 1])
            LOGGER.warning(
                "network fetch failed for %s on attempt %d/%d: %s: %s; retrying in %.2fs",
                description,
                attempt,
                attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable network retry loop")


def _read_html_inputs(
    urls: list[str],
    paths: list[str | Path],
    *,
    timeout_seconds: float,
    output_dir: Path,
    prefix: str,
) -> list[tuple[str, str]]:
    inputs: list[tuple[str, str]] = []
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index, url in enumerate(urls, start=1):
        text = _fetch_text(url, timeout_seconds=timeout_seconds)
        raw_path = raw_dir / f"{prefix}-url-{index}.html"
        raw_path.write_text(text, encoding="utf-8")
        inputs.append((url, text))
    for path in paths:
        html_path = Path(path)
        if not html_path.is_file():
            raise DataValidationError(f"{prefix} HTML path does not exist: {html_path}")
        inputs.append((str(html_path), html_path.read_text(encoding="utf-8")))
    return inputs


def _fetch_text(url: str, *, timeout_seconds: float) -> str:
    def fetch() -> bytes:
        request = Request(url, headers={"User-Agent": _SOURCE_AUDIT_USER_AGENT})
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read()

    data = _run_network_fetch_with_retries(f"HTML {url}", fetch)
    return data.decode("utf-8", errors="replace")


def _download_url(url: str, path: Path, *, timeout_seconds: float) -> DownloadedArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    def fetch() -> bytes:
        request = Request(url, headers={"User-Agent": _SOURCE_AUDIT_USER_AGENT})
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read()

    path.write_bytes(_run_network_fetch_with_retries(f"download {url}", fetch))
    return DownloadedArtifact(url=url, path=str(path), sha256=sha256_file(path), size_bytes=path.stat().st_size, kind=_artifact_kind(path))


def _download_or_reuse_epaper_pdf(url: str, path: Path, *, timeout_seconds: float) -> DownloadedArtifact:
    artifact = _reuse_existing_epaper_pdf(url, path)
    if artifact is not None:
        return artifact
    artifact = _download_url(url, path, timeout_seconds=timeout_seconds)
    artifact.pdf_audit = _audit_pdf_with_pymupdf(path)
    return artifact


def _reuse_existing_epaper_pdf(url: str, path: Path) -> DownloadedArtifact | None:
    if not path.exists():
        return None
    if not path.is_file():
        LOGGER.warning("cached epaper PDF path is not a file; downloading fresh: %s", path)
        return None
    if path.stat().st_size <= 0:
        LOGGER.warning("discarding empty cached epaper PDF before download: %s", path)
        _delete_cached_epaper_pdf(path)
        return None
    try:
        pdf_audit = _audit_pdf_with_pymupdf(path)
    except Exception as exc:  # noqa: BLE001 - corrupt cached PDFs are replaced before retrying the item.
        LOGGER.warning("discarding unreadable cached epaper PDF before download: %s: %s: %s", path, type(exc).__name__, exc)
        _delete_cached_epaper_pdf(path)
        return None
    return DownloadedArtifact(
        url=url,
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        kind=_artifact_kind(path),
        pdf_audit=pdf_audit,
        reused_from_disk=True,
    )


def _delete_cached_epaper_pdf(path: Path) -> None:
    try:
        path.unlink()
    except OSError as exc:
        raise DataValidationError(f"failed to delete cached epaper PDF {path}: {type(exc).__name__}: {exc}") from exc


def _download_article_images(
    articles: list[GorkhapatraArticleAsset],
    output_dir: Path,
    *,
    max_count: int,
    timeout_seconds: float,
    warnings: list[str],
) -> None:
    count = 0
    for index, article in enumerate(articles, start=1):
        if max_count and count >= max_count:
            return
        if not article.image_url:
            continue
        suffix = _url_suffix(article.image_url, default=".jpg")
        target = output_dir / "article-images" / f"article-{index:04d}{suffix}"
        try:
            article.downloaded_image = _download_url(article.image_url, target, timeout_seconds=timeout_seconds)
            count += 1
        except Exception as exc:  # noqa: BLE001 - wrapped as provenance warning, not ignored.
            warnings.append(f"failed to download article image {article.image_url}: {type(exc).__name__}: {exc}")


def _download_epaper_pdfs(
    epapers: list[GorkhapatraEpaperAsset],
    output_dir: Path,
    *,
    max_count: int,
    timeout_seconds: float,
    warnings: list[str],
) -> None:
    count = 0
    for index, epaper in enumerate(epapers, start=1):
        if max_count and count >= max_count:
            return
        target = output_dir / "epapers" / f"epaper-{index:04d}.pdf"
        try:
            artifact = _download_url(epaper.direct_pdf_url, target, timeout_seconds=timeout_seconds)
            artifact.pdf_audit = _audit_pdf_with_pymupdf(target)
            epaper.downloaded_pdf = artifact
            count += 1
        except Exception as exc:  # noqa: BLE001 - wrapped as provenance warning, not ignored.
            warnings.append(f"failed to download epaper PDF {epaper.direct_pdf_url}: {type(exc).__name__}: {exc}")


def _audit_pdf_with_pymupdf(path: Path) -> dict[str, Any]:
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    with fitz.open(path) as document:
        pages: list[dict[str, Any]] = []
        total_text_chars = 0
        for index, page in enumerate(document):
            text = page.get_text("text") or ""
            total_text_chars += len(text)
            if index < 5:
                pages.append(
                    {
                        "page_index": index,
                        "text_chars": len(text),
                        "text_preview": text[:300],
                        "block_count": len(page.get_text("blocks") or []),
                        "image_count": len(page.get_images(full=True) or []),
                    }
                )
        return {
            "available": True,
            "page_count": document.page_count,
            "total_text_chars": total_text_chars,
            "first_pages": pages,
            "text_layer_warning": _pdf_text_layer_warning(pages),
        }


def _pdf_text_layer_warning(pages: list[dict[str, Any]]) -> str | None:
    preview = "\n".join(str(page.get("text_preview") or "") for page in pages)
    if not preview.strip():
        return "no_text_extracted"
    devanagari = sum(1 for char in preview if "\u0900" <= char <= "\u097f")
    ascii_like = sum(1 for char in preview if char.isascii() and char.isalnum())
    if ascii_like > devanagari * 5 and devanagari < 20:
        return "extracted_text_may_be_legacy_encoded_or_mojibake"
    return None


def _find_language_page_hits(pdf_path: Path, output_dir: Path, *, render_pages: bool) -> list[GorkhapatraLanguagePageHit]:
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        raise DataValidationError(f"PDF page marker search requires PyMuPDF: {type(exc).__name__}: {exc}") from exc
    hits: list[GorkhapatraLanguagePageHit] = []
    if render_pages:
        output_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text") or ""
            marker = _language_page_marker(text)
            if marker is None:
                continue
            rendered_path: str | None = None
            if render_pages:
                target = output_dir / f"{pdf_path.stem}-page-{page_index + 1:04d}.png"
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                pixmap.save(target)
                rendered_path = str(target)
            hits.append(
                GorkhapatraLanguagePageHit(
                    page_index=page_index,
                    marker=marker,
                    text_chars=len(text),
                    text_preview=" ".join(text.split())[:500],
                    rendered_page_path=rendered_path,
                )
            )
    return hits


def _language_page_marker(text: str) -> str | None:
    normalized = " ".join(text.split()).lower()
    for marker in LANGUAGE_PAGE_MARKERS:
        if marker.lower() in normalized:
            return marker
    if "भाषा" in normalized and "पृष्ठ" in normalized:
        return "भाषा+पृष्ठ"
    return None


def _write_gorkhapatra_audit(audit: GorkhapatraSourceAudit, output_dir: Path) -> None:
    json_path = output_dir / "gorkhapatra-source-audit.json"
    md_path = output_dir / "gorkhapatra-source-audit.md"
    audit.summary_json_path = str(json_path)
    audit.summary_md_path = str(md_path)
    json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gorkhapatra Source Audit",
        "",
        f"- article assets: {audit.article_count}",
        f"- epaper assets: {audit.epaper_count}",
        f"- downloaded artifacts: {audit.downloaded_count}",
        f"- warnings: {len(audit.warnings)}",
        "",
        "## Languages",
        "",
    ]
    if audit.language_counts:
        lines.extend(f"- {language}: {count}" for language, count in audit.language_counts.items())
    else:
        lines.append("- none detected")
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    lines.extend(["", "## Sample Article Assets", ""])
    for article in audit.articles[:20]:
        lines.append(f"- {article.language or 'unknown'}: {article.title} ({article.article_url or 'no article URL'})")
    lines.extend(["", "## Sample Epaper Assets", ""])
    for epaper in audit.epapers[:20]:
        lines.append(f"- {epaper.date_label or 'unknown date'}: {epaper.direct_pdf_url}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_audit(audit: GorkhapatraLanguagePageAudit, output_dir: Path) -> None:
    json_path = output_dir / "gorkhapatra-language-pages.json"
    md_path = output_dir / "gorkhapatra-language-pages.md"
    audit.summary_json_path = str(json_path)
    audit.summary_md_path = str(md_path)
    json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gorkhapatra Language Page Alignment",
        "",
        f"- publication items: {audit.publication_item_count}",
        f"- aligned to epaper date: {audit.aligned_count}",
        f"- language page hits: {audit.language_page_hit_count}",
        f"- warnings: {len(audit.warnings)}",
        "",
        "## Status",
        "",
    ]
    lines.extend(f"- `{status}`: {count}" for status, count in audit.status_counts.items()) if audit.status_counts else lines.append("- none")
    lines.extend(["", "## Languages", ""])
    lines.extend(f"- {language}: {count}" for language, count in audit.language_counts.items()) if audit.language_counts else lines.append("- none")
    lines.extend(["", "## Alignments", ""])
    for alignment in audit.alignments[:50]:
        item = alignment.publication_item
        page_list = ", ".join(str(hit.page_index + 1) for hit in alignment.page_hits) or "none"
        lines.append(
            f"- {item.date_label or 'unknown date'} / {item.language or 'unknown'}: "
            f"`{alignment.alignment_status}` pages={page_list} title={item.title}"
        )
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _direct_epaper_pdf_url(viewer_url: str) -> str:
    parsed = urlparse(viewer_url)
    values = parse_qs(parsed.query).get("file")
    if not values:
        raise DataValidationError(f"epaper viewer URL does not include file parameter: {viewer_url}")
    return urljoin(GORKHAPATRA_EPAPER_BASE, unquote(values[0]))


def _extract_language_label(title: str) -> str | None:
    match = re.search(r"प्रकाशित\s+(.+?)\s+भाषा", title)
    if not match:
        return None
    return " ".join(match.group(1).split())


def _nepali_date_key(text: str) -> str | None:
    normalized = _normalize_nepali_digits(" ".join(text.split()))
    patterns = (
        r"(?P<day>\d{1,2})\s+(?P<month>[^\s,]+)\s+(?P<year>\d{4})",
        r"(?P<year>\d{4})\s+(?P<month>[^\s,]+)\s+(?P<day>\d{1,2})",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        month = NEPALI_MONTHS.get(match.group("month"))
        if month is None:
            continue
        return f"{int(match.group('year')):04d}-{month:02d}-{int(match.group('day')):02d}"
    return None


def _normalize_nepali_digits(text: str) -> str:
    table = str.maketrans("०१२३४५६७८९", "0123456789")
    return text.translate(table)


def _clean_date_label(text: str) -> str:
    return " ".join(text.split()).strip()


def _dedupe_publication_items(items: list[NayaNepalPublicationItem]) -> list[NayaNepalPublicationItem]:
    seen: set[tuple[str | None, str | None, str, str | None]] = set()
    unique: list[NayaNepalPublicationItem] = []
    for item in items:
        key = (item.article_url, item.image_url, item.title, item.date_key)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _language_page_alignment(
    item: NayaNepalPublicationItem,
    epaper: GorkhapatraEpaperAsset | None,
    pdf_path: str | None,
    pdf_sha256: str | None,
    hits: list[GorkhapatraLanguagePageHit],
    status: str,
    warnings: list[str],
) -> GorkhapatraLanguagePageAlignment:
    return GorkhapatraLanguagePageAlignment(
        publication_item=item,
        epaper=epaper,
        pdf_path=pdf_path,
        pdf_sha256=pdf_sha256,
        page_hits=hits,
        alignment_status=status,
        claim_evidence_eligible=False,
        warnings=warnings,
    )


def _alignment_source_url(alignment: dict[str, Any]) -> str | None:
    epaper = alignment.get("epaper")
    if isinstance(epaper, dict):
        return _clean_optional_str(epaper.get("direct_pdf_url")) or _clean_optional_str(epaper.get("viewer_url"))
    publication_item = alignment.get("publication_item")
    if isinstance(publication_item, dict):
        return _clean_optional_str(publication_item.get("article_url")) or _clean_optional_str(publication_item.get("image_url"))
    return None


def _gorkhapatra_language_page_audit(
    *,
    publication_sources: list[str],
    epaper_sources: list[str],
    publication_items: list[NayaNepalPublicationItem],
    alignments: list[GorkhapatraLanguagePageAlignment],
    warnings: list[str],
) -> GorkhapatraLanguagePageAudit:
    status_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    for alignment in alignments:
        status_counts[alignment.alignment_status] = status_counts.get(alignment.alignment_status, 0) + 1
        language = alignment.publication_item.language
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
    return GorkhapatraLanguagePageAudit(
        publication_sources=publication_sources,
        epaper_sources=epaper_sources,
        publication_item_count=len(publication_items),
        aligned_count=sum(1 for alignment in alignments if alignment.epaper is not None),
        language_page_hit_count=sum(len(alignment.page_hits) for alignment in alignments),
        status_counts=dict(sorted(status_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        alignments=alignments,
        warnings=warnings,
    )


def _gorkhapatra_language_page_pack_summary(
    manifest_path: Path,
    entries: list[ManifestEntry],
    *,
    copied_page_count: int,
    warnings: list[str],
) -> GorkhapatraLanguagePagePackSummary:
    language_counts: dict[str, int] = {}
    reference_status_counts: dict[str, int] = {}
    for entry in entries:
        language = entry.metadata.get("language")
        if isinstance(language, str) and language:
            language_counts[language] = language_counts.get(language, 0) + 1
        status = str(entry.metadata.get("reference_status") or "unknown")
        reference_status_counts[status] = reference_status_counts.get(status, 0) + 1
    return GorkhapatraLanguagePagePackSummary(
        manifest_path=str(manifest_path),
        sample_count=len(entries),
        copied_page_count=copied_page_count,
        language_counts=dict(sorted(language_counts.items())),
        reference_status_counts=dict(sorted(reference_status_counts.items())),
        warnings=warnings,
    )


def _dedupe_articles(articles: list[GorkhapatraArticleAsset]) -> list[GorkhapatraArticleAsset]:
    seen: set[tuple[str | None, str | None, str]] = set()
    unique: list[GorkhapatraArticleAsset] = []
    for article in articles:
        key = (article.article_url, article.image_url, article.title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
    return unique


def _dedupe_epapers(epapers: list[GorkhapatraEpaperAsset]) -> list[GorkhapatraEpaperAsset]:
    seen: set[str] = set()
    unique: list[GorkhapatraEpaperAsset] = []
    for epaper in epapers:
        if epaper.direct_pdf_url in seen:
            continue
        seen.add(epaper.direct_pdf_url)
        unique.append(epaper)
    return unique


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return "image"
    return "file"


def _url_suffix(url: str, *, default: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".pdf"}:
        return suffix
    return default


def _resolve_audit_artifact_path(path_value: str, audit_json_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    audit_dir = audit_json_path.parent
    candidate = audit_dir / path
    if candidate.exists():
        return candidate
    return path


def _write_pending_reference(
    path: Path,
    *,
    sample_id: str,
    source_url: str | None,
    language: str | None,
    source_kind: str,
) -> None:
    payload = {
        "text": "",
        "reading_order": [],
        "tables": [],
        "figures": [],
        "metadata": {
            "sample_id": sample_id,
            "source_url": source_url,
            "language": language,
            "source_kind": source_kind,
            "reference_status": "pending_manual_label",
            "claim_evidence_eligible": False,
            "notes": "Real Gorkhapatra source input. Text/layout reference must be manually verified before claim use.",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_language_page_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DataValidationError(f"Gorkhapatra language-page review CSV has no header: {path}")
        required = {"sample_id", "review_decision", "reference_path"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise DataValidationError(f"Gorkhapatra language-page review CSV missing fields: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _read_language_page_verification_rows(path: Path) -> list[dict[str, str]]:
    rows, _fieldnames = _read_language_page_verification_rows_with_fieldnames(path)
    return rows


def _read_language_page_verification_rows_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DataValidationError(f"Gorkhapatra verification CSV has no header: {path}")
        required = {"sample_id", "line_id", "order_index", "ocr_text", "review_text", "review_status"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise DataValidationError(f"Gorkhapatra verification CSV missing fields: {', '.join(missing)}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_verification_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _verification_row_key(row: dict[str, Any]) -> tuple[str, str] | None:
    sample_id = _clean_optional_str(row.get("sample_id"))
    line_id = _clean_optional_str(row.get("line_id"))
    if not sample_id or not line_id:
        return None
    return sample_id, line_id


def _format_verification_key(key: tuple[str, str]) -> str:
    return f"{key[0]}:{key[1]}"


def _group_verification_rows_by_sample(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        sample_id = _clean_optional_str(row.get("sample_id"))
        if not sample_id:
            continue
        grouped.setdefault(sample_id, []).append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: _verification_order_index(row))
    return grouped


def _verification_order_index(row: dict[str, str]) -> int:
    value = _clean_optional_str(row.get("order_index"))
    if value is None:
        return 10**9
    try:
        return int(value)
    except ValueError:
        return 10**9


def _parse_codepoint_ranges(values: list[str] | None) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for value in values or []:
        cleaned = _clean_optional_str(value)
        if not cleaned:
            continue
        if ".." in cleaned:
            start_text, end_text = cleaned.split("..", 1)
        else:
            start_text = end_text = cleaned
        try:
            start = _parse_codepoint_value(start_text)
            end = _parse_codepoint_value(end_text)
        except ValueError as exc:
            raise DataValidationError(f"invalid codepoint range {cleaned!r}; expected U+1900..U+194F") from exc
        if end < start:
            raise DataValidationError(f"invalid codepoint range {cleaned!r}; end before start")
        ranges.append((start, end))
    return ranges


def _parse_codepoint_value(value: str) -> int:
    text = value.strip()
    if text.upper().startswith("U+"):
        text = text[2:]
    return int(text, 16)


def _text_contains_codepoint_in_ranges(text: str, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= ord(char) <= end for char in text for start, end in ranges)


def _format_codepoint_ranges(ranges: list[tuple[int, int]]) -> str:
    return ",".join(f"U+{start:04X}..U+{end:04X}" if start != end else f"U+{start:04X}" for start, end in ranges)


def _verified_reference_from_line_reviews(
    *,
    sample_id: str,
    language: str | None,
    assisted_reference: dict[str, Any],
    assisted_reference_path: Path,
    verification_rows: list[dict[str, str]],
    verification_csv: Path,
    reviewer: str | None,
    reviewed_at: str | None,
    tables_reviewed: bool,
    figures_reviewed: bool,
    captions_reviewed: bool,
    required_codepoint_ranges: list[tuple[int, int]],
) -> tuple[dict[str, Any], int, int]:
    draft_order = assisted_reference.get("reading_order")
    if not isinstance(draft_order, list):
        raise DataValidationError(f"assisted reference reading_order must be a list: {assisted_reference_path}")
    draft_by_line_id = {
        line_id: item
        for item in draft_order
        if isinstance(item, dict)
        for line_id in [_clean_optional_str(item.get("id"))]
        if line_id
    }
    reading_order: list[dict[str, Any]] = []
    blockers: list[str] = []
    dropped_count = 0
    for row in verification_rows:
        line_id = _clean_optional_str(row.get("line_id"))
        if not line_id:
            blockers.append("line row missing line_id")
            continue
        status = _clean_optional_str(row.get("review_status"))
        if status in {None, "", "pending", "needs_context", "bad_segmentation", "needs_resegmentation"}:
            blockers.append(f"{line_id}: review_status is {status or '<empty>'}")
            continue
        if status == "drop":
            dropped_count += 1
            continue
        if status == "accept":
            text = _clean_optional_str(row.get("ocr_text"))
        elif status == "corrected":
            text = _clean_optional_str(row.get("review_text"))
        else:
            blockers.append(f"{line_id}: unsupported review_status {status}")
            continue
        if not text:
            blockers.append(f"{line_id}: reviewed text is empty")
            continue
        if required_codepoint_ranges and not _text_contains_codepoint_in_ranges(text, required_codepoint_ranges):
            blockers.append(f"{line_id}: reviewed text has no character in required range {_format_codepoint_ranges(required_codepoint_ranges)}")
            continue
        item = dict(draft_by_line_id.get(line_id) or {})
        item["id"] = line_id
        item["text"] = text
        item["review_status"] = status
        if _clean_optional_str(row.get("bbox")):
            item["bbox"] = _parse_json_field(row["bbox"], field_name=f"{line_id}.bbox")
        if _clean_optional_str(row.get("page_index")) is not None:
            item["page_index"] = _parse_int_field(row.get("page_index"), field_name=f"{line_id}.page_index")
        if _clean_optional_str(row.get("ocr_confidence")) is not None:
            item["ocr_confidence"] = _parse_float_field(row.get("ocr_confidence"), field_name=f"{line_id}.ocr_confidence")
        reading_order.append(item)
    if blockers:
        raise DataValidationError("verification rows are not complete: " + "; ".join(blockers[:20]))
    if not reading_order:
        raise DataValidationError("verification rows produced no kept text lines")
    text = "\n".join(str(item["text"]) for item in reading_order)
    metadata = dict(assisted_reference.get("metadata") if isinstance(assisted_reference.get("metadata"), dict) else {})
    metadata.update(
        {
            "sample_id": sample_id,
            "candidate_language": language,
            "language": language,
            "reference_status": "verified",
            "claim_evidence_eligible": True,
            "machine_generated": False,
            "requires_human_review": False,
            "text_reviewed": True,
            "reading_order_reviewed": True,
            "tables_reviewed": tables_reviewed,
            "figures_reviewed": figures_reviewed,
            "captions_reviewed": captions_reviewed,
            "structural_review_status": "verified",
            "verification_source_csv": str(verification_csv),
            "assisted_reference_path": str(assisted_reference_path),
            "verified_line_count": len(reading_order),
            "dropped_line_count": dropped_count,
            "text_sha256": sha256_text(text),
        }
    )
    if required_codepoint_ranges:
        metadata["required_codepoint_ranges"] = _format_codepoint_ranges(required_codepoint_ranges)
    if reviewer:
        metadata["reviewer"] = reviewer
    if reviewed_at:
        metadata["reviewed_at"] = reviewed_at
    reference = {
        "text": text,
        "reading_order": reading_order,
        "tables": assisted_reference.get("tables") if isinstance(assisted_reference.get("tables"), list) else [],
        "figures": assisted_reference.get("figures") if isinstance(assisted_reference.get("figures"), list) else [],
        "metadata": metadata,
    }
    _load_verified_language_page_reference_payload(reference, sample_id=sample_id, label=str(assisted_reference_path))
    return reference, len(reading_order), dropped_count


def _parse_json_field(value: str, *, field_name: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"{field_name} is invalid JSON: {exc}") from exc


def _parse_int_field(value: Any, *, field_name: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{field_name} must be an integer") from exc


def _parse_float_field(value: Any, *, field_name: str) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{field_name} must be numeric") from exc


def _write_updated_language_page_review_csv(source_csv: Path, target_csv: Path, rows: list[dict[str, str]]) -> None:
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
    if "reference_path" not in fieldnames:
        fieldnames.append("reference_path")
    for row in rows:
        for field_name in row:
            if field_name not in fieldnames:
                fieldnames.append(field_name)
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    with target_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _normalized_language_filter(candidate_languages: list[str] | None) -> set[str]:
    return {
        value
        for value in (_clean_optional_str(language) for language in (candidate_languages or []))
        if value
    }


def _entry_or_review_candidate_language(entry: ManifestEntry, row: dict[str, str] | None) -> str | None:
    row_language = _clean_optional_str((row or {}).get("candidate_language"))
    if row_language:
        return row_language
    metadata = entry.metadata or {}
    return _clean_optional_str(metadata.get("candidate_language")) or _clean_optional_str(metadata.get("language"))


def _target_font_needles_for_language(language: str | None, requested_target_fonts: list[str]) -> list[str]:
    needles = list(requested_target_fonts)
    normalized = _clean_optional_str(language)
    if normalized in {"लिम्बू", "लिम्बु", "limbu", "Limbu"}:
        needles.extend(["namdhinggo", "sirijonga", "srijanga", "limbu"])
    return sorted({needle.lower() for needle in needles if needle})


def _extract_pdf_text_spans(page: Any, *, sample_target_fonts: list[str]) -> list[dict[str, Any]]:
    raw = page.get_text("rawdict")
    spans: list[dict[str, Any]] = []
    span_index = 0
    for block_index, block in enumerate(raw.get("blocks", [])):
        for line_index, line in enumerate(block.get("lines", [])):
            for raw_span in line.get("spans", []):
                text = "".join(char.get("c", "") for char in raw_span.get("chars", []))
                if not text:
                    continue
                font = str(raw_span.get("font") or "")
                font_lower = font.lower()
                span_index += 1
                spans.append(
                    {
                        "span_index": span_index,
                        "block_index": block_index,
                        "line_index": line_index,
                        "font": font,
                        "size": raw_span.get("size"),
                        "bbox": [round(float(value), 3) for value in raw_span.get("bbox", [])],
                        "raw_text": text,
                        "codepoints": [f"U+{ord(char):04X}" for char in text],
                        "is_target_font": any(needle in font_lower for needle in sample_target_fonts),
                    }
                )
    return spans


def _extract_pdf_page_fonts(document: Any, page: Any, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fonts: list[dict[str, Any]] = []
    for font in page.get_fonts(full=True):
        xref = int(font[0])
        extension = str(font[1] or "bin")
        font_type = str(font[2] or "")
        name = str(font[3] or f"font-{xref}")
        encoding = str(font[5] or "") if len(font) > 5 else ""
        saved_path = ""
        sha256 = ""
        size_bytes = 0
        try:
            extracted = document.extract_font(xref)
            font_bytes = extracted[3] if isinstance(extracted, tuple) else extracted.get("content", b"")
            extracted_ext = extracted[1] if isinstance(extracted, tuple) else extracted.get("ext", extension)
            if font_bytes:
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or f"font-{xref}"
                target = output_dir / f"{xref}-{safe_name}.{extracted_ext or extension}"
                target.write_bytes(font_bytes)
                saved_path = str(target)
                sha256 = hashlib.sha256(font_bytes).hexdigest()
                size_bytes = len(font_bytes)
        except Exception:  # noqa: BLE001 - some PDF font references are not extractable.
            saved_path = ""
        fonts.append(
            {
                "xref": xref,
                "name": name,
                "extension": extension,
                "type": font_type,
                "encoding": encoding,
                "saved_path": saved_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    return fonts


def _pdf_spans_plain_text(spans: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    current_line: tuple[int, int] | None = None
    current_parts: list[str] = []
    for span in spans:
        line_key = (int(span["block_index"]), int(span["line_index"]))
        if current_line is not None and line_key != current_line:
            lines.append("".join(current_parts).rstrip())
            current_parts = []
        current_line = line_key
        current_parts.append(str(span["raw_text"]))
    if current_parts:
        lines.append("".join(current_parts).rstrip())
    return "\n".join(line for line in lines if line)


def _language_page_review_group_key(entry: ManifestEntry) -> str:
    metadata = entry.metadata or {}
    date_key = _clean_optional_str(metadata.get("date_key")) or "unknown-date"
    language = _clean_optional_str(metadata.get("candidate_language")) or _clean_optional_str(metadata.get("language")) or "unknown-language"
    publication = (
        _clean_optional_str(metadata.get("article_url"))
        or _clean_optional_str(metadata.get("source_image_url"))
        or _clean_optional_str(metadata.get("source_url"))
        or "unknown-publication"
    )
    return f"date={date_key}|language={language}|publication={publication}"


def _resolve_review_reference_path(path_value: str, *, source_manifest: Path, review_csv: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    for base in (review_csv.parent, source_manifest.parent, Path.cwd()):
        candidate = base / path
        if candidate.exists():
            return candidate
    return review_csv.parent / path


def _resolve_manifest_entry_path(path_value: str, *, source_manifest: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = source_manifest.parent / path
    if candidate.exists():
        return candidate
    return path


def _load_verified_language_page_reference(path: Path, *, sample_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"verified reference JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid verified reference JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"verified reference JSON must be an object: {path}")
    _load_verified_language_page_reference_payload(payload, sample_id=sample_id, label=str(path))
    return payload


def _load_verified_language_page_reference_payload(payload: dict[str, Any], *, sample_id: str, label: str) -> None:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise DataValidationError(f"verified reference JSON has empty text: {label}")
    for field_name in ("reading_order", "tables", "figures"):
        if field_name not in payload:
            raise DataValidationError(f"verified reference JSON missing {field_name}: {label}")
        if not isinstance(payload[field_name], list):
            raise DataValidationError(f"verified reference JSON {field_name} must be a list: {label}")
    for index, item in enumerate(payload["reading_order"], start=1):
        if not isinstance(item, dict):
            raise DataValidationError(f"verified reference reading_order item {index} must be an object: {label}")
        if not _clean_optional_str(item.get("id")):
            raise DataValidationError(f"verified reference reading_order item {index} missing id: {label}")
        if not _clean_optional_str(item.get("text")):
            raise DataValidationError(f"verified reference reading_order item {index} missing text: {label}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise DataValidationError(f"verified reference JSON missing metadata object: {label}")
    if metadata.get("sample_id") not in {sample_id, None, ""}:
        raise DataValidationError(f"verified reference sample_id mismatch: {metadata.get('sample_id')} != {sample_id}")
    if metadata.get("reference_status") != "verified":
        raise DataValidationError(f"verified reference metadata.reference_status must be verified: {label}")
    if metadata.get("claim_evidence_eligible") is not True:
        raise DataValidationError(f"verified reference metadata.claim_evidence_eligible must be true: {label}")


def _language_page_reference_template(
    entry: ManifestEntry,
    *,
    metadata: dict[str, Any],
    review_row: dict[str, Any],
    language: str | None,
) -> dict[str, Any]:
    return {
        "text": "",
        "reading_order": [
            {
                "id": "block-0001",
                "type": "paragraph",
                "text": "",
                "bbox": None,
            }
        ],
        "tables": [],
        "figures": [],
        "metadata": {
            "sample_id": entry.sample_id,
            "source_kind": "gorkhapatra_language_page_reference_template",
            "source_url": metadata.get("source_url"),
            "image_path": entry.image_path,
            "candidate_language": language,
            "language": language,
            "script": metadata.get("script") or "unknown",
            "date_key": metadata.get("date_key"),
            "pdf_page_number": metadata.get("pdf_page_number"),
            "language_page_marker": metadata.get("language_page_marker"),
            "review_decision": review_row.get("review_decision"),
            "reviewer": review_row.get("reviewer"),
            "reviewed_at": review_row.get("reviewed_at"),
            "reference_status": "draft",
            "claim_evidence_eligible": False,
            "required_for_finalization": [
                "fill non-empty text",
                "fill reading_order items with id and text",
                "keep tables and figures as arrays, empty if absent",
                "set metadata.reference_status=verified",
                "set metadata.claim_evidence_eligible=true",
            ],
        },
    }


def _assisted_language_page_reference(
    entry: ManifestEntry,
    *,
    row: dict[str, Any],
    language: str | None,
    engine_name: str,
    engine_output: Any,
) -> dict[str, Any]:
    metadata = dict(entry.metadata or {})
    reading_order: list[dict[str, Any]] = []
    text_lines: list[str] = []
    confidences: list[float] = []
    for page in getattr(engine_output, "pages", []) or []:
        page_index = int(getattr(page, "page_index", 0) or 0)
        for line_index, line in enumerate(getattr(page, "text_lines", []) or []):
            text = _clean_optional_str(getattr(line, "text", ""))
            if not text:
                continue
            confidence = getattr(line, "confidence", None)
            if isinstance(confidence, int | float) and not isinstance(confidence, bool):
                confidences.append(float(confidence))
            line_id = _clean_optional_str(getattr(line, "line_id", None)) or f"p{page_index}-l{line_index:04d}"
            bbox = getattr(line, "bbox", None)
            bbox_value = bbox.to_list() if hasattr(bbox, "to_list") else None
            reading_order.append(
                {
                    "id": line_id,
                    "type": "ocr_line",
                    "text": text,
                    "bbox": bbox_value,
                    "confidence": confidence,
                    "page_index": page_index,
                    "needs_review": True,
                }
            )
            text_lines.append(text)
    average_confidence = sum(confidences) / len(confidences) if confidences else None
    return {
        "text": "\n".join(text_lines),
        "reading_order": reading_order,
        "tables": [],
        "figures": [],
        "metadata": {
            "sample_id": entry.sample_id,
            "source_kind": "gorkhapatra_language_page_assisted_reference_draft",
            "source_url": metadata.get("source_url"),
            "image_path": entry.image_path,
            "candidate_language": language,
            "language": language,
            "script": metadata.get("script") or "unknown",
            "date_key": metadata.get("date_key"),
            "pdf_page_number": metadata.get("pdf_page_number"),
            "language_page_marker": metadata.get("language_page_marker"),
            "review_decision": row.get("review_decision"),
            "reviewer": row.get("reviewer"),
            "reviewed_at": row.get("reviewed_at"),
            "reference_status": "draft",
            "claim_evidence_eligible": False,
            "machine_generated": True,
            "requires_human_review": True,
            "assisted_ocr_engine": engine_name,
            "ocr_line_count": len(reading_order),
            "ocr_average_confidence": average_confidence,
            "engine_metadata": dict(getattr(engine_output, "metadata", {}) or {}),
            "required_for_finalization": [
                "verify or replace machine OCR text",
                "correct reading_order text and order",
                "add tables and figures where present",
                "set metadata.reference_status=verified only after human review",
                "set metadata.claim_evidence_eligible=true only after human review",
            ],
        },
    }


def _engine_output_sidecar_payload(engine_output: Any, *, engine_name: str, image_path: Path) -> dict[str, Any]:
    return {
        "pages": [
            {
                "page_index": getattr(page, "page_index", 0),
                "width": getattr(page, "width", None),
                "height": getattr(page, "height", None),
                "lines": [line.to_dict() for line in getattr(page, "text_lines", []) or []],
                "metadata": dict(getattr(page, "metadata", {}) or {}),
            }
            for page in getattr(engine_output, "pages", []) or []
        ],
        "tables": [table.to_dict() for table in getattr(engine_output, "tables", []) or []],
        "figures": [figure.to_dict() for figure in getattr(engine_output, "figures", []) or []],
        "metadata": {
            **dict(getattr(engine_output, "metadata", {}) or {}),
            "sidecar_source_engine": engine_name,
            "source_image": str(image_path),
        },
    }


def _sidecar_average_confidence(payload: dict[str, Any]) -> float | None:
    values: list[float] = []
    for page in payload.get("pages") or []:
        if not isinstance(page, dict):
            continue
        for line in page.get("lines") or []:
            if not isinstance(line, dict):
                continue
            confidence = line.get("confidence")
            if isinstance(confidence, int | float) and not isinstance(confidence, bool):
                values.append(float(confidence))
    return sum(values) / len(values) if values else None


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} must be a JSON object: {path}")
    return payload


def _clean_bbox_xywh(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        candidate = [value.get("x"), value.get("y"), value.get("w"), value.get("h")]
    elif isinstance(value, list | tuple) and len(value) == 4:
        candidate = list(value)
    else:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in candidate):
        return None
    x, y, width, height = [float(item) for item in candidate]
    if width <= 0 or height <= 0:
        return None
    return [x, y, width, height]


def _xywh_crop_note(source_image: Path, bbox: list[float]) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DataValidationError("verification bundle crop extraction requires Pillow. Install ocr-tech[eval].") from exc
    with Image.open(source_image) as image:
        image_width, image_height = image.size
    x, y, width, height = bbox
    touches_edge = x <= 1 or y <= 1 or x + width >= image_width - 1 or y + height >= image_height - 1
    if touches_edge:
        return "OCR bbox touches page edge; verify against context crop before accepting"
    return ""


def _write_xywh_crop(source_image: Path, crop_path: Path, bbox: list[float]) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DataValidationError("verification bundle crop extraction requires Pillow. Install ocr-tech[eval].") from exc
    with Image.open(source_image) as image:
        image_width, image_height = image.size
        x, y, width, height = bbox
        pad_x = max(120, int(width * 0.35))
        pad_y = max(90, int(height * 0.85))
        left = max(0, int(x - pad_x))
        top = max(0, int(y - pad_y))
        right = min(image_width, int(x + width + pad_x))
        bottom = min(image_height, int(y + height + pad_y))
        if right <= left or bottom <= top:
            raise DataValidationError(f"crop bbox outside image bounds for {source_image}: {bbox}")
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop((left, top, right, bottom)).save(crop_path)


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "line"


def _render_pdf_pages(source_pdf: Path, inputs_dir: Path, *, epaper_index: int, max_pages: int) -> list[tuple[int, Path]]:
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        raise DataValidationError(f"PDF page rendering requires PyMuPDF: {type(exc).__name__}: {exc}") from exc
    rendered: list[tuple[int, Path]] = []
    with fitz.open(source_pdf) as document:
        page_count = min(document.page_count, max_pages)
        for page_index in range(page_count):
            page = document[page_index]
            target = inputs_dir / f"gorkhapatra-epaper-{epaper_index:04d}-page-{page_index + 1:04d}.png"
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            pixmap.save(target)
            rendered.append((page_index, target))
    return rendered


def _gorkhapatra_slices(language: str | None, source_kind: str) -> list[str]:
    slices = {"gorkhapatra", "naya_nepal", "real_publication", "image"}
    if source_kind == "article_image":
        slices.add("article_image")
    if source_kind == "epaper_page":
        slices.update({"epaper", "page"})
    if source_kind == "language_page_candidate":
        slices.update({"epaper", "page", "language_page_candidate", "pending_reference"})
    if language:
        slices.add(f"language:{language}")
    return sorted(slices)


def _verified_gorkhapatra_slices(language: str | None) -> list[str]:
    slices = {"gorkhapatra", "naya_nepal", "real", "real_publication", "image", "epaper", "page", "language_page", "hard_eval"}
    if language:
        slices.add(f"language:{language}")
    return sorted(slices)


def _gorkhapatra_pack_summary(
    manifest_path: Path,
    entries: list[ManifestEntry],
    *,
    article_image_count: int,
    epaper_page_count: int,
    warnings: list[str],
) -> GorkhapatraPackSummary:
    language_counts: dict[str, int] = {}
    reference_status_counts: dict[str, int] = {}
    for entry in entries:
        language = entry.metadata.get("language")
        if isinstance(language, str) and language:
            language_counts[language] = language_counts.get(language, 0) + 1
        status = str(entry.metadata.get("reference_status") or "unknown")
        reference_status_counts[status] = reference_status_counts.get(status, 0) + 1
    return GorkhapatraPackSummary(
        manifest_path=str(manifest_path),
        sample_count=len(entries),
        article_image_count=article_image_count,
        epaper_page_count=epaper_page_count,
        language_counts=dict(sorted(language_counts.items())),
        reference_status_counts=dict(sorted(reference_status_counts.items())),
        warnings=warnings,
    )


def _write_gorkhapatra_pack_summary(summary: GorkhapatraPackSummary, output_dir: Path) -> None:
    json_path = output_dir / "gorkhapatra-pack.json"
    md_path = output_dir / "gorkhapatra-pack.md"
    summary.summary_json_path = str(json_path)
    summary.summary_md_path = str(md_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gorkhapatra Validation Pack",
        "",
        f"- manifest: `{summary.manifest_path}`",
        f"- samples: {summary.sample_count}",
        f"- article images: {summary.article_image_count}",
        f"- epaper pages: {summary.epaper_page_count}",
        "",
        "## Reference Status",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.reference_status_counts.items()) if summary.reference_status_counts else lines.append("- none")
    lines.extend(["", "## Languages", ""])
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary.warnings) if summary.warnings else lines.append("- none")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_pack_summary(summary: GorkhapatraLanguagePagePackSummary, output_dir: Path) -> None:
    json_path = output_dir / "gorkhapatra-language-page-pack.json"
    md_path = output_dir / "gorkhapatra-language-page-pack.md"
    summary.summary_json_path = str(json_path)
    summary.summary_md_path = str(md_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gorkhapatra Language Page Candidate Pack",
        "",
        f"- manifest: `{summary.manifest_path}`",
        f"- samples: {summary.sample_count}",
        f"- copied page candidates: {summary.copied_page_count}",
        "",
        "## Reference Status",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.reference_status_counts.items()) if summary.reference_status_counts else lines.append("- none")
    lines.extend(["", "## Languages", ""])
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary.warnings) if summary.warnings else lines.append("- none")
    lines.extend(
        [
            "",
            "## Claim Status",
            "",
            "All rows are `claim_evidence_eligible=false` and `pending_manual_label` until page-language disambiguation and reference transcription are complete.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_review(
    summary: GorkhapatraLanguagePageReviewSummary,
    rows: list[dict[str, Any]],
) -> None:
    json_path = Path(summary.review_json_path)
    csv_path = Path(summary.review_csv_path)
    md_path = Path(summary.review_md_path)
    payload = {
        "summary": summary.to_dict(),
        "rows": rows,
        "review_decision_values": [
            "accept_language_page",
            "reject_wrong_language",
            "reject_not_language_page",
            "duplicate_candidate",
            "needs_additional_review",
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "sample_id",
        "candidate_language",
        "date_label",
        "date_key",
        "pdf_page_index",
        "pdf_page_number",
        "image_path",
        "reference_path",
        "article_url",
        "source_image_url",
        "source_url",
        "pdf_path",
        "pdf_sha256",
        "language_page_marker",
        "alignment_status",
        "page_disambiguation_status",
        "reference_status",
        "claim_evidence_eligible",
        "review_decision",
        "reviewer",
        "reviewed_at",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    lines = [
        "# Gorkhapatra Language Page Review",
        "",
        f"- manifest: `{summary.manifest_path}`",
        f"- samples: {summary.sample_count}",
        f"- pending review: {summary.pending_review_count}",
        f"- claim eligible: {summary.claim_evidence_eligible_count}",
        f"- CSV: `{summary.review_csv_path}`",
        "",
        "## Review Rule",
        "",
        "Rows remain non-claim-ready until a reviewer confirms the language-page match and creates a verified reference transcript/layout.",
        "",
        "## Languages",
        "",
    ]
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    lines.extend(["", "## Same Page Candidate Counts", ""])
    lines.extend(f"- `{key}`: {count}" for key, count in summary.page_candidate_counts.items()) if summary.page_candidate_counts else lines.append("- none")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    lines.extend(["", "## Rows", ""])
    for row in rows[:100]:
        lines.append(
            f"- `{row['sample_id']}` {row.get('candidate_language') or 'unknown'} "
            f"date={row.get('date_key') or 'unknown'} page={row.get('pdf_page_number') or 'unknown'} "
            f"status={row.get('page_disambiguation_status') or 'unknown'}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_reference_templates_summary(
    summary: GorkhapatraLanguagePageReferenceTemplateSummary,
    rows: list[dict[str, Any]],
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    csv_path = Path(summary.index_csv_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "sample_id",
        "candidate_language",
        "review_decision",
        "image_path",
        "template_path",
        "source_reference_path",
        "pdf_page_number",
        "date_key",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    lines = [
        "# Gorkhapatra Language Page Reference Templates",
        "",
        f"- passed: {summary.passed}",
        f"- source manifest: `{summary.source_manifest}`",
        f"- review CSV: `{summary.review_csv_path or 'not supplied'}`",
        f"- index CSV: `{summary.index_csv_path}`",
        f"- templates: {summary.sample_count}",
        f"- skipped: {summary.skipped_count}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Review Contract",
        "",
        "Templates are drafts. A reviewer must fill text and reading order, keep tables/figures as arrays, then set `metadata.reference_status=verified` and `metadata.claim_evidence_eligible=true` before finalization.",
        "",
        "## Languages",
        "",
    ]
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_reviewer_bundle_summary(
    summary: GorkhapatraLanguagePageReviewerBundleSummary,
    rows: list[dict[str, Any]],
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    csv_path = Path(summary.index_csv_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "sample_id",
        "candidate_language",
        "image_path",
        "reference_path",
        "source_image_path",
        "source_reference_path",
        "article_url",
        "source_url",
        "date_key",
        "pdf_page_number",
        "reference_status",
        "claim_evidence_eligible",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    lines = [
        "# Gorkhapatra Language Page Reviewer Bundle",
        "",
        f"- passed: {summary.passed}",
        f"- source manifest: `{summary.source_manifest}`",
        f"- review CSV: `{summary.review_csv_path}`",
        f"- reference template dir: `{summary.reference_template_dir or 'not supplied'}`",
        f"- index CSV: `{summary.index_csv_path}`",
        f"- bundled samples: {summary.sample_count}",
        f"- skipped review rows: {summary.skipped_count}",
        f"- copied images: {summary.copied_image_count}",
        f"- copied references: {summary.copied_reference_count}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Labeling Contract",
        "",
        "This bundle is for manual reference creation only. It is not claim-ready.",
        "For each accepted row, fill the matching `.ref.json` with full page text, reading order, and any tables or figures. Keep `tables` and `figures` as arrays even when empty. Only after review is complete may `metadata.reference_status` be changed to `verified` and `metadata.claim_evidence_eligible` to `true`.",
        "",
        "## Languages",
        "",
    ]
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    lines.extend(["", "## Rows", ""])
    for row in rows:
        lines.append(
            f"- `{row['sample_id']}` {row.get('candidate_language') or 'unknown'} "
            f"page={row.get('pdf_page_number') or 'unknown'} image=`{row.get('image_path')}` reference=`{row.get('reference_path')}`"
        )
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_transcription_work_order(
    summary: GorkhapatraLanguagePageTranscriptionWorkOrderSummary,
    rows: list[dict[str, Any]],
    *,
    skipped_count: int,
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    csv_path = Path(summary.index_csv_path)
    html_path = Path(summary.transcription_html_path)
    payload = {**summary.to_dict(), "skipped_count": skipped_count, "rows": rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "sample_id",
        "candidate_language",
        "status",
        "blocker",
        "image_path",
        "image_exists",
        "reference_path",
        "reference_exists",
        "date_key",
        "date_label",
        "pdf_page_number",
        "article_url",
        "source_image_url",
        "source_url",
        "required_fields",
        "next_command",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    _write_gorkhapatra_language_page_transcription_dashboard(summary, rows, html_path)
    lines = [
        "# Gorkhapatra Language Page Transcription Work Order",
        "",
        f"- passed: {summary.passed}",
        f"- source manifest: `{summary.source_manifest}`",
        f"- review CSV: `{summary.review_csv_path}`",
        f"- index CSV: `{summary.index_csv_path}`",
        f"- transcription dashboard: `{summary.transcription_html_path}`",
        f"- samples: {summary.sample_count}",
        f"- blocked: {summary.blocked_count}",
        f"- verified: {summary.verified_count}",
        f"- skipped review rows: {skipped_count}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Contract",
        "",
        "Fill each blocked reference JSON with non-empty `text`, structured `reading_order` items with `id` and `text`, `tables` and `figures` arrays, then set `metadata.reference_status=verified` and `metadata.claim_evidence_eligible=true`.",
        "",
        "## Languages",
        "",
    ]
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    lines.extend(["", "## Rows", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.get('sample_id')}",
                "",
                f"- language: `{row.get('candidate_language') or 'unknown'}`",
                f"- status: `{row.get('status')}`",
                f"- blocker: `{row.get('blocker') or 'none'}`",
                f"- image: `{row.get('image_path')}`",
                f"- reference: `{row.get('reference_path')}`",
                f"- page: `{row.get('pdf_page_number') or 'unknown'}`",
                f"- date: `{row.get('date_key') or row.get('date_label') or 'unknown'}`",
                f"- source: `{row.get('source_url') or 'unknown'}`",
                f"- next audit: `{row.get('next_command')}`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_transcription_dashboard(
    summary: GorkhapatraLanguagePageTranscriptionWorkOrderSummary,
    rows: list[dict[str, Any]],
    html_path: Path,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    rows_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    body = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Gorkhapatra Language Page Transcription</title>",
        "<style>",
        ":root{--bg:#f4f1ea;--panel:#fffdf8;--ink:#171717;--muted:#5f6368;--line:#d8d1c4;--accent:#0f766e;--bad:#b42318}",
        "*{box-sizing:border-box}",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:var(--bg);color:var(--ink)}",
        ".topbar{position:sticky;top:0;z-index:4;background:rgba(255,253,248,.96);border-bottom:1px solid var(--line);padding:14px 18px;display:grid;grid-template-columns:1fr auto;gap:16px;align-items:center}",
        "h1{font-size:20px;margin:0 0 6px}.meta{font-size:13px;color:var(--muted)}",
        ".actions{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}",
        "button{border:1px solid #b8b1a5;background:#fff;color:#111;border-radius:6px;padding:8px 10px;font-size:13px;cursor:pointer}",
        "button.primary{background:var(--accent);border-color:var(--accent);color:white}",
        "main{padding:18px;max-width:1700px;margin:0 auto}",
        ".sample{display:grid;grid-template-columns:minmax(420px,1fr) minmax(360px,520px);gap:16px;background:var(--panel);border:1px solid var(--line);border-radius:8px;margin-bottom:18px;padding:14px}",
        ".page{max-height:calc(100vh - 170px);overflow:auto;border:1px solid var(--line);background:white}.page img{display:block;max-width:100%;height:auto}",
        "textarea{width:100%;min-height:420px;border:1px solid var(--line);border-radius:6px;padding:10px;font-size:22px;line-height:1.55;resize:vertical;background:white}",
        "select,input{width:100%;border:1px solid var(--line);border-radius:6px;padding:8px;margin-top:6px}.field{margin-top:10px}.issue{font-size:12px;color:var(--bad);min-height:18px;margin-top:8px}",
        "@media(max-width:980px){.topbar{grid-template-columns:1fr}.sample{grid-template-columns:1fr}.page{max-height:none}}",
        "</style>",
        "</head>",
        "<body>",
        '<div class="topbar"><div>',
        "<h1>Gorkhapatra Language Page Transcription</h1>",
        f'<div class="meta">Work order: <code>{html_escape(summary.index_csv_path)}</code><br>Type verified Sirijonga Unicode text from the page. Ignore bad OCR candidates.</div>',
        '</div><div class="actions"><button type="button" id="export-json" class="primary">Export transcription JSON</button><button type="button" id="export-refs">Export reference JSON</button></div></div>',
        "<main>",
    ]
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        image_path = _clean_optional_str(row.get("image_path")) or ""
        image_src = _relative_html_path(image_path, html_path.parent) if image_path else ""
        body.extend(
            [
                f'<section class="sample" data-sample-id="{html_escape(sample_id)}" data-language="{html_escape(str(row.get("candidate_language") or ""))}" data-image-path="{html_escape(image_path)}">',
                '<div class="page">',
                f'<img src="{html_escape(image_src)}" alt="source page {html_escape(sample_id)}">' if image_src else '<div class="meta">missing image</div>',
                "</div><div>",
                f"<h2>{html_escape(sample_id)}</h2>",
                f'<div class="meta">language: {html_escape(str(row.get("candidate_language") or ""))}<br>status: {html_escape(str(row.get("status") or ""))}<br>blocker: {html_escape(str(row.get("blocker") or ""))}</div>',
                '<div class="field"><label class="meta">Transcription status</label><select data-field="transcription_status"><option value="pending">pending</option><option value="transcribed">transcribed</option><option value="needs_review">needs_review</option><option value="unreadable">unreadable</option></select></div>',
                '<div class="field"><label class="meta">Verified Sirijonga text in reading order</label><textarea data-field="text" placeholder="Type the page/region text here in Limbu Unicode U+1900..U+194F"></textarea></div>',
                '<div class="field"><label class="meta">Notes</label><input data-field="notes" placeholder="columns, omitted headers, uncertainty, etc."></div>',
                '<div class="issue" data-issue></div>',
                "</div></section>",
            ]
        )
    body.extend(
        [
            "</main>",
            f'<script type="application/json" id="work-order-rows">{rows_json}</script>',
            "<script>",
            "const sourceRows = JSON.parse(document.getElementById('work-order-rows').textContent);",
            "const byId = new Map(sourceRows.map(row => [row.sample_id || '', row]));",
            "function collect(){return [...document.querySelectorAll('.sample')].map(el => {const source = byId.get(el.dataset.sampleId) || {}; return {sample_id: el.dataset.sampleId, candidate_language: el.dataset.language, image_path: el.dataset.imagePath, transcription_status: el.querySelector('[data-field=transcription_status]').value, text: el.querySelector('[data-field=text]').value, notes: el.querySelector('[data-field=notes]').value, source};});}",
            "function validate(){document.querySelectorAll('.sample').forEach(el => {const status = el.querySelector('[data-field=transcription_status]').value; const text = el.querySelector('[data-field=text]').value.trim(); const issue = el.querySelector('[data-issue]'); let message = ''; if(status === 'transcribed' && !/[\\u1900-\\u194F]/u.test(text)) message = 'transcribed Limbu page needs Sirijonga Unicode U+1900..U+194F'; if(status === 'pending') message = 'pending blocks verification'; issue.textContent = message;});}",
            "document.addEventListener('input', validate); document.addEventListener('change', validate);",
            "function download(name, payload){const blob = new Blob([payload], {type:'application/json;charset=utf-8'}); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = name; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url);}",
            "document.getElementById('export-json').addEventListener('click', () => download('gorkhapatra-language-page-transcriptions.json', JSON.stringify({transcriptions: collect()}, null, 2) + '\\n'));",
            "document.getElementById('export-refs').addEventListener('click', () => {const refs = collect().filter(row => row.transcription_status === 'transcribed').map(row => ({sample_id: row.sample_id, reference: {text: row.text, reading_order: row.text.split(/\\n+/).filter(Boolean).map((line, index) => ({id: `manual-l${index + 1}`, text: line})), tables: [], figures: [], metadata: {sample_id: row.sample_id, candidate_language: row.candidate_language, language: row.candidate_language, source_kind: 'gorkhapatra_language_page_manual_transcription', image_path: row.image_path, reference_status: 'draft', claim_evidence_eligible: false, machine_generated: false, requires_human_review: true, transcription_status: row.transcription_status, notes: row.notes}}})); download('gorkhapatra-language-page-reference-drafts.json', JSON.stringify({references: refs}, null, 2) + '\\n');});",
            "validate();",
            "</script>",
            "</body>",
            "</html>",
        ]
    )
    html_path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_pdf_text_summary(
    summary: GorkhapatraLanguagePagePdfTextSummary,
    index_rows: list[dict[str, Any]],
    span_rows: list[dict[str, Any]],
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    csv_path = Path(summary.index_csv_path)
    spans_csv_path = Path(summary.spans_csv_path)
    json_path.write_text(
        json.dumps({"summary": summary.to_dict(), "rows": index_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    index_fieldnames = [
        "sample_id",
        "candidate_language",
        "date_key",
        "pdf_path",
        "pdf_page_number",
        "span_count",
        "target_span_count",
        "font_count",
        "sample_json_path",
        "sample_text_path",
        "raw_text_preview",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_fieldnames)
        writer.writeheader()
        for row in index_rows:
            writer.writerow({field: row.get(field, "") for field in index_fieldnames})
    span_fieldnames = [
        "sample_id",
        "candidate_language",
        "date_key",
        "pdf_path",
        "pdf_page_index",
        "pdf_page_number",
        "span_index",
        "block_index",
        "line_index",
        "font",
        "size",
        "bbox",
        "raw_text",
        "codepoints",
        "is_target_font",
    ]
    with spans_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=span_fieldnames)
        writer.writeheader()
        for row in span_rows:
            writer.writerow({field: row.get(field, "") for field in span_fieldnames})
    lines = [
        "# Gorkhapatra Language Page PDF Text Extraction",
        "",
        f"- passed: {summary.passed}",
        f"- source manifest: `{summary.source_manifest}`",
        f"- selected samples: {summary.sample_count}",
        f"- extracted samples: {summary.extracted_sample_count}",
        f"- spans: {summary.span_count}",
        f"- target-font spans: {summary.target_span_count}",
        f"- unique fonts: {summary.font_count}",
        f"- embedded font files extracted: {summary.embedded_font_count}",
        f"- index CSV: `{summary.index_csv_path}`",
        f"- span CSV: `{summary.spans_csv_path}`",
        "",
        "## Contract",
        "",
        "This is the primary extraction step for Gorkhapatra epaper language pages. It uses the PDF text layer and embedded font metadata from the date-matched newspaper PDF before OCR or manual transcription.",
        "",
        "Raw text is preserved exactly as exposed by PyMuPDF. Legacy-font pages, including Limbu pages backed by Namdhinggo-style fonts, may require a font/encoding conversion step before the text is valid Unicode reference text.",
        "",
        "## Languages",
        "",
    ]
    lines.extend(f"- `{language}`: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    lines.extend(["", "## Fonts", ""])
    if summary.font_counts:
        lines.extend(f"- `{font}`: {count}" for font, count in sorted(summary.font_counts.items(), key=lambda item: (-item[1], item[0]))[:30])
    else:
        lines.append("- none")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    lines.extend(["", "## Sample Outputs", ""])
    lines.extend(f"- `{row['sample_id']}`: `{row['sample_json_path']}`" for row in index_rows[:20]) if index_rows else lines.append("- none")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_assisted_references_summary(
    summary: GorkhapatraLanguagePageAssistedReferenceSummary,
    rows: list[dict[str, Any]],
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    csv_path = Path(summary.index_csv_path)
    payload = {**summary.to_dict(), "rows": rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "sample_id",
        "candidate_language",
        "status",
        "error",
        "image_path",
        "reference_path",
        "ocr_line_count",
        "ocr_average_confidence",
        "reference_status",
        "claim_evidence_eligible",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    lines = [
        "# Gorkhapatra Language Page Assisted References",
        "",
        f"- passed: {summary.passed}",
        f"- source manifest: `{summary.source_manifest}`",
        f"- review CSV: `{summary.review_csv_path}`",
        f"- engine: `{summary.engine}`",
        f"- samples: {summary.sample_count}",
        f"- assisted: {summary.assisted_count}",
        f"- failed: {summary.failed_count}",
        f"- skipped: {summary.skipped_count}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Contract",
        "",
        "These references are machine-assisted drafts only. They must keep `metadata.reference_status=draft` and `metadata.claim_evidence_eligible=false` until a human verifies text, reading order, tables, figures, and metadata.",
        "",
        "## Languages",
        "",
    ]
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    lines.extend(["", "## Rows", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.get('sample_id')}",
                "",
                f"- language: `{row.get('candidate_language') or 'unknown'}`",
                f"- status: `{row.get('status')}`",
                f"- OCR lines: `{row.get('ocr_line_count')}`",
                f"- OCR confidence: `{row.get('ocr_average_confidence')}`",
                f"- reference: `{row.get('reference_path')}`",
                f"- error: `{row.get('error') or 'none'}`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_ocr_sidecars_summary(
    summary: GorkhapatraLanguagePageOcrSidecarSummary,
    rows: list[dict[str, Any]],
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    csv_path = Path(summary.index_csv_path)
    payload = {**summary.to_dict(), "rows": rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "sample_id",
        "candidate_language",
        "status",
        "error",
        "image_path",
        "sidecar_path",
        "ocr_line_count",
        "ocr_average_confidence",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    lines = [
        "# Gorkhapatra Language Page OCR Sidecars",
        "",
        f"- passed: {summary.passed}",
        f"- source manifest: `{summary.source_manifest}`",
        f"- review CSV: `{summary.review_csv_path}`",
        f"- engine: `{summary.engine}`",
        f"- in place: {summary.in_place}",
        f"- samples: {summary.sample_count}",
        f"- sidecars: {summary.sidecar_count}",
        f"- failed: {summary.failed_count}",
        f"- skipped: {summary.skipped_count}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Contract",
        "",
        "These OCR sidecars are machine output used to accelerate draft reference creation. They are not references and are not claim evidence.",
        "",
        "## Languages",
        "",
    ]
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    lines.extend(["", "## Rows", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.get('sample_id')}",
                "",
                f"- language: `{row.get('candidate_language') or 'unknown'}`",
                f"- status: `{row.get('status')}`",
                f"- OCR lines: `{row.get('ocr_line_count')}`",
                f"- OCR confidence: `{row.get('ocr_average_confidence')}`",
                f"- sidecar: `{row.get('sidecar_path')}`",
                f"- error: `{row.get('error') or 'none'}`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_verification_bundle_summary(
    summary: GorkhapatraLanguagePageVerificationBundleSummary,
    rows: list[dict[str, Any]],
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    csv_path = Path(summary.index_csv_path)
    html_path = Path(summary.review_html_path)
    payload = {**summary.to_dict(), "rows": rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "sample_id",
        "candidate_language",
        "line_id",
        "order_index",
        "page_index",
        "ocr_text",
        "ocr_confidence",
        "bbox",
        "crop_path",
        "review_text",
        "review_status",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    _write_gorkhapatra_language_page_verification_review_html(summary, rows, html_path)
    lines = [
        "# Gorkhapatra Language Page Verification Bundle",
        "",
        f"- passed: {summary.passed}",
        f"- source manifest: `{summary.source_manifest}`",
        f"- review CSV: `{summary.review_csv_path}`",
        f"- assisted references: `{summary.assisted_references_dir}`",
        f"- samples: {summary.sample_count}",
        f"- lines: {summary.line_count}",
        f"- crops: {summary.crop_count}",
        f"- missing bboxes: {summary.missing_bbox_count}",
        f"- skipped review rows: {summary.skipped_count}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Contract",
        "",
        "This bundle is for human verification only. Fill `review_text`, set `review_status`, and use the result to update the draft `.ref.json`; the bundle itself is not claim evidence.",
        "",
        "## Languages",
        "",
    ]
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    lines.extend(["", "## Review Sheet", ""])
    lines.append(f"- CSV: `{summary.index_csv_path}`")
    lines.append(f"- HTML: `{summary.review_html_path}`")
    lines.append("- Status values: `accept`, `corrected`, `drop`, `needs_context`, `bad_segmentation`, `needs_resegmentation`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_verification_review_html(
    summary: GorkhapatraLanguagePageVerificationBundleSummary,
    rows: list[dict[str, Any]],
    html_path: Path,
) -> None:
    _write_verification_rows_review_html(
        title="Gorkhapatra Language Page Verification",
        csv_path=Path(summary.index_csv_path),
        rows=rows,
        html_path=html_path,
        meta=[
            f"Samples: {summary.sample_count}",
            f"Lines: {summary.line_count}",
            f"Crops: {summary.crop_count}",
            f"Missing bboxes: {summary.missing_bbox_count}",
        ],
    )


def _write_verification_rows_review_html(
    *,
    title: str,
    csv_path: Path,
    rows: list[dict[str, Any]],
    html_path: Path,
    meta: list[str] | None = None,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sample_id") or "unknown"), []).append(row)
    meta_text = " | ".join(meta or [f"Lines: {len(rows)}"])
    rows_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    body: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html_escape(title)}</title>",
        "<style>",
        ":root{color-scheme:light;--bg:#f4f1ea;--panel:#fffdf8;--ink:#171717;--muted:#5f6368;--line:#d8d1c4;--accent:#0f766e;--bad:#b42318;--warn:#a15c07}",
        "*{box-sizing:border-box}",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:var(--bg);color:var(--ink)}",
        ".topbar{position:sticky;top:0;z-index:4;background:rgba(255,253,248,.96);border-bottom:1px solid var(--line);padding:14px 18px;display:grid;grid-template-columns:1fr auto;gap:16px;align-items:center}",
        "h1{font-size:20px;margin:0 0 6px}",
        "h2{font-size:17px;margin:26px 0 10px}",
        ".meta{font-size:13px;color:var(--muted)}",
        ".actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end}",
        "button{border:1px solid #b8b1a5;background:#fff;color:#111;border-radius:6px;padding:8px 10px;font-size:13px;cursor:pointer}",
        "button.primary{background:var(--accent);border-color:var(--accent);color:white}",
        "button:disabled{opacity:.45;cursor:not-allowed}",
        ".stats{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}",
        ".pill{font-size:12px;border:1px solid var(--line);background:#fff;border-radius:999px;padding:4px 8px}",
        "main{padding:18px;max-width:1500px;margin:0 auto}",
        ".filters{display:flex;gap:10px;flex-wrap:wrap;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:14px}",
        "select,input[type=search]{border:1px solid var(--line);border-radius:6px;background:white;padding:8px;font-size:14px}",
        ".row{display:grid;grid-template-columns:92px minmax(260px,440px) minmax(280px,1fr) 240px;gap:12px;align-items:start;background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:9px 0;padding:10px}",
        ".row[data-status=accept]{border-left:5px solid var(--accent)}",
        ".row[data-status=corrected]{border-left:5px solid #2563eb}",
        ".row[data-status=drop]{border-left:5px solid var(--bad);opacity:.78}",
        ".row[data-status=needs_context]{border-left:5px solid var(--warn)}",
        ".row[data-status=bad_segmentation],.row[data-status=needs_resegmentation]{border-left:5px solid #7c3aed}",
        ".crop{max-width:100%;height:auto;border:1px solid #ddd;background:white}",
        ".mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#374151;overflow-wrap:anywhere}",
        ".text{font-size:18px;line-height:1.45;white-space:pre-wrap;border:1px solid var(--line);background:white;border-radius:6px;padding:8px;min-height:44px}",
        ".review textarea{width:100%;min-height:84px;border:1px solid var(--line);border-radius:6px;padding:8px;font-size:17px;line-height:1.45;resize:vertical}",
        ".review select,.review input{width:100%;margin-top:7px}",
        ".hint{font-size:12px;color:var(--muted);margin-top:6px}",
        ".quick{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}",
        ".issue{color:var(--bad);font-size:12px;margin-top:6px;min-height:16px}",
        ".hidden{display:none}",
        "@media(max-width:980px){.topbar{grid-template-columns:1fr}.actions{justify-content:flex-start}.row{grid-template-columns:1fr}.crop{width:100%}}",
        "</style>",
        "</head>",
        "<body>",
        '<div class="topbar">',
        "<div>",
        f"<h1>{html_escape(title)}</h1>",
        '<div class="meta">',
        f"CSV contract: <code>{html_escape(str(csv_path))}</code><br>",
        html_escape(meta_text),
        "</div>",
        '<div class="stats" id="stats"></div>',
        "</div>",
        '<div class="actions">',
        '<button type="button" id="mark-visible-accept">Accept visible OCR</button>',
        '<button type="button" id="export-csv" class="primary">Export reviewed CSV</button>',
        "</div>",
        "</div>",
        "<main>",
        '<div class="filters">',
        '<label class="meta">Status <select id="status-filter"><option value="">all</option><option value="pending">pending</option><option value="accept">accept</option><option value="corrected">corrected</option><option value="drop">drop</option><option value="needs_context">needs_context</option><option value="bad_segmentation">bad_segmentation</option><option value="needs_resegmentation">needs_resegmentation</option></select></label>',
        '<label class="meta">Search <input type="search" id="search-filter" placeholder="line id or OCR text"></label>',
        '<span class="hint">Review every crop. Use corrected only for one clear line. Use bad_segmentation or needs_resegmentation when a crop contains multiple lines, photos, or partial columns. Drop non-text/noise. Limbu/Sirijonga kept rows must contain U+1900..U+194F text.</span>',
        "</div>",
    ]
    for sample_id, sample_rows in sorted(grouped.items()):
        language = next((_clean_optional_str(row.get("candidate_language")) for row in sample_rows if _clean_optional_str(row.get("candidate_language"))), None)
        body.append(f"<h2>{html_escape(sample_id)}{f' · {html_escape(language)}' if language else ''}</h2>")
        for row in sample_rows:
            crop_path = _clean_optional_str(row.get("crop_path"))
            crop_html = '<div class="hint">missing crop</div>'
            if crop_path:
                crop_src = _relative_html_path(crop_path, html_path.parent)
                crop_html = f'<img class="crop" src="{html_escape(crop_src)}" alt="line crop {html_escape(str(row.get("line_id") or ""))}">'
            row_id = f"{row.get('sample_id') or ''}:{row.get('line_id') or ''}"
            review_status = _clean_optional_str(row.get("review_status")) or "pending"
            body.extend(
                [
                    f'<section class="row" data-row-id="{html_escape(row_id)}" data-status="{html_escape(review_status)}" data-search="{html_escape((str(row.get("line_id") or "") + " " + str(row.get("ocr_text") or "")).lower())}">',
                    '<div class="mono">',
                    f"order {html_escape(str(row.get('order_index') or ''))}<br>",
                    f"{html_escape(str(row.get('line_id') or ''))}<br>",
                    f"conf {html_escape(str(row.get('ocr_confidence') or ''))}",
                    "</div>",
                    f"<div>{crop_html}</div>",
                    "<div>",
                    f'<div class="text">{html_escape(str(row.get("ocr_text") or ""))}</div>',
                    '<div class="hint">OCR candidate</div>',
                    "</div>",
                    '<div class="review">',
                    '<label class="meta">Review text</label>',
                    f'<textarea data-field="review_text">{html_escape(str(row.get("review_text") or ""))}</textarea>',
                    '<label class="meta">Status</label>',
                    '<select data-field="review_status">',
                    "".join(
                        f'<option value="{html_escape(status)}"{" selected" if status == review_status else ""}>{html_escape(status)}</option>'
                        for status in ["pending", "accept", "corrected", "drop", "needs_context", "bad_segmentation", "needs_resegmentation"]
                    ),
                    "</select>",
                    '<label class="meta">Notes</label>',
                    f'<input data-field="notes" value="{html_escape(str(row.get("notes") or ""))}">',
                    '<div class="quick">',
                    '<button type="button" data-action="accept">accept</button>',
                    '<button type="button" data-action="copy-correct">copy to edit</button>',
                    '<button type="button" data-action="drop">drop</button>',
                    '<button type="button" data-action="bad-segmentation">bad segmentation</button>',
                    "</div>",
                    '<div class="issue" data-issue></div>',
                    "</div>",
                    "</section>",
                ]
            )
    body.extend(
        [
            "</main>",
            f'<script type="application/json" id="verification-rows">{rows_json}</script>',
            "<script>",
            "const originalRows = JSON.parse(document.getElementById('verification-rows').textContent);",
            "const rowsById = new Map(originalRows.map(row => [`${row.sample_id || ''}:${row.line_id || ''}`, row]));",
            "const csvFields = ['sample_id','candidate_language','line_id','order_index','page_index','ocr_text','ocr_confidence','bbox','crop_path','review_text','review_status','notes'];",
            "const statusFilter = document.getElementById('status-filter');",
            "const searchFilter = document.getElementById('search-filter');",
            "function csvCell(value){const text = value == null ? '' : String(value); return /[\",\\n\\r]/.test(text) ? '\"' + text.replaceAll('\"','\"\"') + '\"' : text;}",
            "function currentRows(){return [...document.querySelectorAll('.row')].map(el => {const base = {...rowsById.get(el.dataset.rowId)}; base.review_text = el.querySelector('[data-field=review_text]').value; base.review_status = el.querySelector('[data-field=review_status]').value; base.notes = el.querySelector('[data-field=notes]').value; return base;});}",
            "function validateRow(el){const base = rowsById.get(el.dataset.rowId) || {}; const status = el.querySelector('[data-field=review_status]').value; const reviewText = el.querySelector('[data-field=review_text]').value.trim(); const keptText = status === 'corrected' ? reviewText : (base.ocr_text || ''); const issue = el.querySelector('[data-issue]'); let message = ''; if(status === 'pending') message = 'pending blocks verification'; if(status === 'corrected' && !reviewText) message = 'corrected needs review text'; if(['needs_context','bad_segmentation','needs_resegmentation'].includes(status)) message = `${status} blocks verification`; if(!message && (base.candidate_language || '') === 'लिम्बू' && ['accept','corrected'].includes(status) && !/[\\u1900-\\u194F]/u.test(keptText)) message = 'kept Limbu row needs Sirijonga Unicode U+1900..U+194F'; issue.textContent = message; el.dataset.status = status; return !message;}",
            "function refresh(){let visible = 0; const counts = {}; document.querySelectorAll('.row').forEach(el => {validateRow(el); const status = el.dataset.status; counts[status] = (counts[status] || 0) + 1; const statusOk = !statusFilter.value || status === statusFilter.value; const searchOk = !searchFilter.value || el.dataset.search.includes(searchFilter.value.toLowerCase()); const show = statusOk && searchOk; el.classList.toggle('hidden', !show); if(show) visible += 1;}); document.getElementById('stats').innerHTML = Object.entries(counts).sort().map(([k,v]) => `<span class=\"pill\">${k}: ${v}</span>`).join('') + `<span class=\"pill\">visible: ${visible}</span>`;}",
            "document.addEventListener('input', event => {if(event.target.matches('[data-field],#status-filter,#search-filter')) refresh();});",
            "document.addEventListener('change', event => {if(event.target.matches('[data-field],#status-filter,#search-filter')) refresh();});",
            "document.addEventListener('click', event => {const action = event.target.dataset.action; if(!action) return; const rowEl = event.target.closest('.row'); const base = rowsById.get(rowEl.dataset.rowId) || {}; const status = rowEl.querySelector('[data-field=review_status]'); const text = rowEl.querySelector('[data-field=review_text]'); const notes = rowEl.querySelector('[data-field=notes]'); if(action === 'accept'){status.value = 'accept'; text.value = '';} if(action === 'copy-correct'){status.value = 'corrected'; text.value = base.ocr_text || '';} if(action === 'drop'){status.value = 'drop'; text.value = '';} if(action === 'bad-segmentation'){status.value = 'bad_segmentation'; text.value = ''; if(!notes.value) notes.value = 'crop contains multiple lines/regions; needs resegmentation';} refresh();});",
            "document.getElementById('mark-visible-accept').addEventListener('click', () => {document.querySelectorAll('.row:not(.hidden)').forEach(el => {el.querySelector('[data-field=review_status]').value = 'accept'; el.querySelector('[data-field=review_text]').value = '';}); refresh();});",
            "document.getElementById('export-csv').addEventListener('click', () => {const rows = currentRows(); const csv = [csvFields.join(','), ...rows.map(row => csvFields.map(field => csvCell(row[field])).join(','))].join('\\n') + '\\n'; const blob = new Blob([csv], {type:'text/csv;charset=utf-8'}); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = 'gorkhapatra-language-page-verification-lines-reviewed.csv'; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url);});",
            "refresh();",
            "</script>",
            "</body>",
            "</html>",
        ]
    )
    html_path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _relative_html_path(path_value: str, base_dir: Path) -> str:
    path = Path(path_value)
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path)


def _write_gorkhapatra_language_page_verification_apply_summary(
    summary: GorkhapatraLanguagePageVerificationApplySummary,
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gorkhapatra Language Page Verification Apply",
        "",
        f"- passed: {summary.passed}",
        f"- source manifest: `{summary.source_manifest}`",
        f"- review CSV: `{summary.review_csv_path}`",
        f"- verification CSV: `{summary.verification_csv_path}`",
        f"- assisted references: `{summary.assisted_references_dir}`",
        f"- updated review CSV: `{summary.updated_review_csv_path}`",
        f"- references: `{summary.references_dir}`",
        f"- samples: {summary.sample_count}",
        f"- verified references: {summary.verified_reference_count}",
        f"- blocked: {summary.blocked_count}",
        f"- reviewed lines: {summary.reviewed_line_count}",
        f"- dropped lines: {summary.dropped_line_count}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Contract",
        "",
        "This step may write claim-eligible `.ref.json` files only when every kept line has an explicit `accept` or `corrected` review status and every corrected line has non-empty `review_text`. `drop` lines are omitted. `pending`, `needs_context`, empty, or unknown statuses block the sample.",
        "",
        "Run `audit-gorkhapatra-language-page-review --require-verified-references` and then `finalize-gorkhapatra-language-page-review` with the updated review CSV before using these references as evaluation evidence.",
        "",
        "## Languages",
        "",
    ]
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_verification_csv_audit(
    summary: GorkhapatraLanguagePageVerificationCsvAuditSummary,
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gorkhapatra Language Page Verification CSV Audit",
        "",
        f"- passed: {summary.passed}",
        f"- verification CSV: `{summary.verification_csv_path}`",
        f"- samples: {summary.sample_count}",
        f"- lines: {summary.line_count}",
        f"- ready lines: {summary.ready_line_count}",
        f"- dropped lines: {summary.dropped_line_count}",
        f"- blocked lines: {summary.blocked_line_count}",
        f"- issues: {len(summary.issues)}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Status Counts",
        "",
    ]
    lines.extend(f"- `{status}`: {count}" for status, count in summary.status_counts.items()) if summary.status_counts else lines.append("- none")
    lines.extend(["", "## Languages", ""])
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    if summary.issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in summary.issues[:200])
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings[:200])
    lines.extend(
        [
            "",
            "## Apply Readiness",
            "",
            "This audit only checks the line-review CSV. `apply-gorkhapatra-language-page-verification-bundle` still requires explicit table, figure, and caption review declarations before writing claim-eligible references.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_verification_split_summary(
    summary: GorkhapatraLanguagePageVerificationSplitSummary,
    batch_rows: list[dict[str, Any]],
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    csv_path = Path(summary.index_csv_path)
    json_path.write_text(json.dumps({**summary.to_dict(), "batches": batch_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = ["batch_id", "line_count", "start_order_index", "end_order_index", "csv_path", "review_html_path", "status_counts"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in batch_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    lines = [
        "# Gorkhapatra Language Page Verification Split",
        "",
        f"- passed: {summary.passed}",
        f"- source CSV: `{summary.verification_csv_path}`",
        f"- output: `{summary.output_dir}`",
        f"- batch size: {summary.batch_size}",
        f"- samples: {summary.sample_count}",
        f"- lines: {summary.line_count}",
        f"- batches: {summary.batch_count}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Status Counts",
        "",
    ]
    lines.extend(f"- `{status}`: {count}" for status, count in summary.status_counts.items()) if summary.status_counts else lines.append("- none")
    lines.extend(["", "## Batches", ""])
    for row in batch_rows:
        lines.extend(
            [
                f"### {row.get('batch_id')}",
                "",
                f"- lines: {row.get('line_count')}",
                f"- order range: `{row.get('start_order_index')}` to `{row.get('end_order_index')}`",
                f"- CSV: `{row.get('csv_path')}`",
                f"- HTML: `{row.get('review_html_path')}`",
                f"- status counts: `{row.get('status_counts')}`",
                "",
            ]
        )
    if summary.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_verification_merge_summary(
    summary: GorkhapatraLanguagePageVerificationMergeSummary,
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gorkhapatra Language Page Verification Merge",
        "",
        f"- passed: {summary.passed}",
        f"- source CSV: `{summary.source_verification_csv_path}`",
        f"- batches: `{summary.batches_dir}`",
        f"- merged CSV: `{summary.merged_csv_path}`",
        f"- source lines: {summary.source_line_count}",
        f"- merged lines: {summary.merged_line_count}",
        f"- batch files: {summary.batch_file_count}",
        f"- issues: {len(summary.issues)}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Contract",
        "",
        "The merge succeeds only when every `(sample_id, line_id)` from the source verification CSV appears exactly once across batch CSVs and no unknown lines are introduced.",
    ]
    if summary.issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in summary.issues[:200])
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings[:200])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_verification_assignment_summary(
    summary: GorkhapatraLanguagePageVerificationAssignmentSummary,
    assignment_rows: list[dict[str, Any]],
) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    csv_path = Path(summary.assignment_csv_path)
    json_path.write_text(json.dumps({**summary.to_dict(), "assignments": assignment_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "batch_id",
        "reviewer",
        "assignment_status",
        "due_date",
        "line_count",
        "start_order_index",
        "end_order_index",
        "csv_path",
        "review_html_path",
        "completed_at",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in assignment_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    lines = [
        "# Gorkhapatra Language Page Verification Assignments",
        "",
        f"- passed: {summary.passed}",
        f"- split index: `{summary.split_index_csv_path}`",
        f"- assignments CSV: `{summary.assignment_csv_path}`",
        f"- batches: {summary.batch_count}",
        f"- assigned: {summary.assigned_count}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Reviewer Counts",
        "",
    ]
    lines.extend(f"- `{reviewer}`: {count}" for reviewer, count in summary.reviewer_counts.items()) if summary.reviewer_counts else lines.append("- none")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    lines.extend(["", "## Assignments", ""])
    for row in assignment_rows:
        lines.extend(
            [
                f"### {row.get('batch_id')}",
                "",
                f"- reviewer: `{row.get('reviewer') or 'unassigned'}`",
                f"- status: `{row.get('assignment_status')}`",
                f"- due: `{row.get('due_date') or 'unset'}`",
                f"- lines: {row.get('line_count')}",
                f"- order range: `{row.get('start_order_index')}` to `{row.get('end_order_index')}`",
                f"- CSV: `{row.get('csv_path')}`",
                f"- HTML: `{row.get('review_html_path')}`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_review_audit(summary: GorkhapatraLanguagePageReviewAuditSummary) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gorkhapatra Language Page Review Audit",
        "",
        f"- passed: {summary.passed}",
        f"- source manifest: `{summary.source_manifest}`",
        f"- review CSV: `{summary.review_csv_path}`",
        f"- samples: {summary.sample_count}",
        f"- publication/language/date groups: {summary.group_count}",
        f"- accepted rows: {summary.accepted_count}",
        f"- rejected rows: {summary.rejected_count}",
        f"- unresolved rows: {summary.unresolved_count}",
        f"- duplicate accepted groups: {summary.duplicate_accept_group_count}",
        f"- missing references: {summary.missing_reference_count}",
        f"- verified references: {summary.verified_reference_count}",
        f"- require verified references: {summary.require_verified_references}",
        "",
        "## Contract",
        "",
        "Each publication/language/date group must have exactly one `accept_language_page` row. Other candidates in the same group must be rejected or marked duplicate. Accepted rows must point to an existing reference file; with verified-reference mode, that JSON must already satisfy finalization rules.",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {issue}" for issue in summary.issues) if summary.issues else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary.warnings) if summary.warnings else lines.append("- none")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gorkhapatra_language_page_finalize_summary(summary: GorkhapatraLanguagePageFinalizeSummary) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gorkhapatra Language Page Verified Eval",
        "",
        f"- passed: {summary.passed}",
        f"- manifest: `{summary.manifest_path}`",
        f"- source manifest: `{summary.source_manifest}`",
        f"- review CSV: `{summary.review_csv_path}`",
        f"- samples: {summary.sample_count}",
        f"- accepted review rows: {summary.accepted_review_count}",
        f"- skipped review rows: {summary.skipped_review_count}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Languages",
        "",
    ]
    lines.extend(f"- {language}: {count}" for language, count in summary.language_counts.items()) if summary.language_counts else lines.append("- none")
    lines.extend(["", "## Claim Status", ""])
    lines.append("Promoted rows are `claim_evidence_eligible=true`, `reference_status=verified`, `real`, and `hard_eval`.")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clean_optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
