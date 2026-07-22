"""Command-line interface for ocrtech."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .ablation import audit_augmentation_ablation
from .admission import assert_recognizer_family_not_stopped, decide_recognizer_admission, review_recognizer_failures
from .benchmark import derive_table_postprocess_benchmark, rescore_benchmark_report, run_benchmark, summarize_benchmark_report
from .calibration import calibrate_quality_router, calibrate_script_router, calibrate_tesseract_psm_ensemble
from .claim_review import review_claim
from .capture import audit_limbu_capture, parse_crop_box, parse_perspective_quad, prepare_limbu_capture
from .crops import extract_crop_manifest
from .datasets import audit_manifest, audit_recognizer_corpus, build_recognizer_corpus, filter_manifest, merge_manifests, rebalance_manifest, split_manifest
from .error_analysis import analyze_benchmark_errors, export_failure_manifest
from .errors import OcrTechError
from .evalpack import audit_eval_pack, create_eval_pack
from .experiments import run_experiment
from .font_assets import audit_font_asset_readiness, prepare_font_assets
from .language_registry import audit_font_renderability, audit_language_readiness, audit_language_registry, audit_paddle_dictionary, inventory_fonts
from .limbu_legacy import convert_gorkhapatra_limbu_pdf_text, convert_limbu_legacy_text
from .limbu_runtime import preflight_limbu_pipeline_runtime
from .manifest import inspect_hf_dataset, load_manifest, normalize_manifest_images, prepare_hf_dataset, prepare_local_dataset
from .models import audit_model_card, write_model_card
from .oracle import build_benchmark_oracle
from .paper import export_paper_benchmark_table
from .pipeline import (
    admit_limbu_post_correction_profile,
    apply_limbu_review_corrections,
    audit_limbu_correction_pair_pack,
    audit_limbu_output,
    audit_limbu_post_correction_profile,
    build_limbu_correction_pair_pack,
    derive_limbu_post_correction_profile,
    parse_document,
    parse_limbu_document,
    score_limbu_post_correction_profile,
)
from .preflight import run_claim_preflight
from .public_benchmarks import audit_public_benchmark_manifest, list_public_benchmarks, prepare_public_benchmark
from .references import audit_references, materialize_references
from .remote import DEFAULT_PYTHON_CANDIDATES, audit_remote_host, bootstrap_remote_host, sync_remote_workspace
from .source_audit import (
    DEFAULT_GORKHAPATRA_CATEGORY_URLS,
    DEFAULT_GORKHAPATRA_EPAPER_URL,
    DEFAULT_NAYA_NEPAL_PUBLICATION_URL,
    apply_gorkhapatra_language_page_verification_bundle,
    assign_gorkhapatra_language_page_verification_batches,
    audit_gorkhapatra_language_page_verification_csv,
    audit_gorkhapatra_language_pages,
    audit_gorkhapatra_source,
    extract_gorkhapatra_language_page_pdf_text,
    audit_gorkhapatra_language_page_review,
    finalize_gorkhapatra_language_page_review,
    merge_gorkhapatra_language_page_verification_batches,
    prepare_gorkhapatra_language_page_assisted_references,
    prepare_gorkhapatra_language_page_ocr_sidecars,
    prepare_gorkhapatra_language_page_pdf_native_references,
    prepare_gorkhapatra_language_page_recognizer_eval,
    prepare_gorkhapatra_language_page_reference_templates,
    prepare_gorkhapatra_language_page_review,
    prepare_gorkhapatra_language_page_reviewer_bundle,
    prepare_gorkhapatra_language_page_transcription_work_order,
    prepare_gorkhapatra_language_page_verification_bundle,
    prepare_gorkhapatra_language_page_pack,
    prepare_gorkhapatra_pack,
    split_gorkhapatra_language_page_verification_csv,
)
from .synthesis_resources import (
    audit_rendered_degradation_split,
    audit_rendered_synthesis_lines,
    audit_synthesis_text_manifest,
    audit_synthesis_text_promotion,
    audit_synthesis_resources,
    prepare_bible_brain_text,
    prepare_limbu_limdic_text,
    prepare_limbu_unicode_text,
    prepare_magar_text,
    prepare_tamang_text,
    prepare_toolkit_parallel_text,
    render_synthesis_text_split,
    render_synthesis_text_lines,
    split_synthesis_text_manifest,
)
from .table_analysis import analyze_table_cells
from .training import (
    PADDLE_DEVANAGARI_PPOCRV3_PRETRAINED,
    PADDLE_PPOCRV5_MOBILE_REC_PRETRAINED,
    bundle_recognizer_training,
    finalize_hf_recognizer_run,
    generate_correction_pairs,
    package_hf_text_corrector_model,
    package_hf_recognizer_model,
    package_paddleocr_model,
    write_corrector_recipe,
    write_hf_text_corrector_recipe,
    write_hf_recognizer_recipe,
    write_recognizer_export_recipe,
    write_recognizer_recipe,
)
from .validation import validate_claim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocrtech", description="Nepali + English OCR document parser")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="download/verify/convert datasets into manifests")
    prepare.add_argument("source", help="local CSV/JSONL/directory or Hugging Face dataset id")
    prepare.add_argument("--out", default="data/manifests", help="output manifest directory")
    prepare.add_argument("--dataset-name", help="manifest dataset name")
    prepare.add_argument("--split", default="train", help="dataset split")
    prepare.add_argument("--image-field", help="explicit image field")
    prepare.add_argument("--text-field", help="explicit text field")
    prepare.add_argument("--hf-subset", help="known HF dataset subset/language selector (for example: en)")
    prepare.add_argument("--limit", type=int, help="max rows to convert")
    prepare.add_argument("--hf", action="store_true", help="load source with Hugging Face datasets")
    prepare.add_argument("--skip-invalid", action="store_true", help="skip invalid rows and write a reject report")
    prepare.add_argument("--strict-chars", action="store_true", help="reject unsupported dictionary characters")
    prepare.add_argument("--slice", action="append", default=[], help="slice label to stamp into manifest metadata; repeatable")
    prepare.set_defaults(func=cmd_prepare_data)

    inspect_hf = subparsers.add_parser("inspect-hf-dataset", help="inspect a Hugging Face dataset schema before manifest conversion")
    inspect_hf.add_argument("dataset_id", help="Hugging Face dataset id")
    inspect_hf.add_argument("--out", default="outputs/hf-dataset-inspection")
    inspect_hf.add_argument("--split", default="train")
    inspect_hf.add_argument("--hf-subset", help="dataset subset/language selector")
    inspect_hf.add_argument("--image-field", help="explicit image field")
    inspect_hf.add_argument("--text-field", help="explicit text field")
    inspect_hf.add_argument("--limit", type=int, default=3)
    inspect_hf.set_defaults(func=cmd_inspect_hf_dataset)

    audit_manifest_parser = subparsers.add_parser("audit-manifest", help="audit a training or evaluation manifest")
    audit_manifest_parser.add_argument("manifest")
    audit_manifest_parser.add_argument("--out", default="outputs/manifest-audit")
    audit_manifest_parser.add_argument("--no-verify-hashes", action="store_true")
    audit_manifest_parser.add_argument("--require-slice", action="append", default=[])
    audit_manifest_parser.add_argument("--strict-chars", action="store_true")
    audit_manifest_parser.set_defaults(func=cmd_audit_manifest)

    split = subparsers.add_parser("split-manifest", help="create deterministic leakage-safe train/eval manifests")
    split.add_argument("manifest")
    split.add_argument("--out", default="data/splits")
    split.add_argument("--eval-ratio", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=13)
    split.add_argument("--stratify-by", choices=["slices", "dataset", "document_type", "none"], default="slices")
    split.add_argument("--group-by", choices=["sample_signature", "text_sha256", "image_sha256", "image_path", "sample_id"], default="sample_signature")
    split.set_defaults(func=cmd_split_manifest)

    rebalance = subparsers.add_parser("rebalance-manifest", help="upsample underrepresented slices in a manifest for training")
    rebalance.add_argument("manifest")
    rebalance.add_argument("--out", default="data/manifests/rebalanced.jsonl")
    rebalance.add_argument("--slice", action="append", default=[], help="slice to rebalance; repeatable")
    rebalance.add_argument("--target-count", type=int, help="target count per requested slice; defaults to the largest requested slice")
    rebalance.add_argument("--seed", type=int, default=13)
    rebalance.set_defaults(func=cmd_rebalance_manifest)

    corpus_audit = subparsers.add_parser("audit-recognizer-corpus", help="audit train/eval manifests before recognizer training")
    corpus_audit.add_argument("--train-manifest", required=True)
    corpus_audit.add_argument("--eval-manifest", required=True)
    corpus_audit.add_argument("--out", default="outputs/recognizer-corpus-audit")
    corpus_audit.add_argument("--min-train-samples", type=int, default=1000)
    corpus_audit.add_argument("--min-eval-samples", type=int, default=100)
    corpus_audit.add_argument("--min-train-english", type=int, default=200)
    corpus_audit.add_argument("--min-train-devanagari", type=int, default=200)
    corpus_audit.add_argument("--min-train-mixed", type=int, default=25)
    corpus_audit.add_argument("--min-eval-english", type=int, default=50)
    corpus_audit.add_argument("--min-eval-devanagari", type=int, default=50)
    corpus_audit.add_argument("--min-eval-mixed", type=int, default=10)
    corpus_audit.add_argument("--min-train-latin-only", type=int, default=50)
    corpus_audit.add_argument("--min-train-devanagari-only", type=int, default=50)
    corpus_audit.add_argument("--min-eval-latin-only", type=int, default=10)
    corpus_audit.add_argument("--min-eval-devanagari-only", type=int, default=10)
    corpus_audit.add_argument("--min-train-real", type=int, default=100)
    corpus_audit.add_argument("--min-train-synthetic", type=int, default=100)
    corpus_audit.add_argument("--require-eval-real", action="store_true", help="fail if eval is not held-out real data or contains synthetic rows")
    corpus_audit.set_defaults(func=cmd_audit_recognizer_corpus)

    corpus_build = subparsers.add_parser("build-recognizer-corpus", help="assemble and upsample a recognizer training manifest by script buckets")
    corpus_build.add_argument("manifests", nargs="+")
    corpus_build.add_argument("--out", required=True)
    corpus_build.add_argument("--target-latin-only", type=int, default=0)
    corpus_build.add_argument("--target-devanagari-only", type=int, default=0)
    corpus_build.add_argument("--target-mixed", type=int, default=0)
    corpus_build.add_argument("--target-real", type=int, default=0)
    corpus_build.add_argument("--target-synthetic", type=int, default=0)
    corpus_build.add_argument("--seed", type=int, default=13)
    corpus_build.set_defaults(func=cmd_build_recognizer_corpus)

    merge = subparsers.add_parser("merge-manifests", help="combine multiple manifests into one auditable corpus")
    merge.add_argument("manifests", nargs="+")
    merge.add_argument("--out", default="data/manifests/combined.jsonl")
    merge.set_defaults(func=cmd_merge_manifests)

    filter_parser = subparsers.add_parser("filter-manifest", help="select a reproducible subset of a manifest")
    filter_parser.add_argument("manifest")
    filter_parser.add_argument("--out", default="data/manifests/filtered.jsonl")
    filter_parser.add_argument("--sample-id", action="append", default=[], help="exact sample_id to include; repeatable")
    filter_parser.add_argument("--slice", action="append", default=[], help="required slice label; repeatable")
    filter_parser.add_argument("--document-type", action="append", default=[], help="required document_type metadata value; repeatable")
    filter_parser.add_argument("--degradation", action="append", default=[], help="required degradation metadata value; repeatable")
    filter_parser.add_argument("--limit", type=int)
    filter_parser.set_defaults(func=cmd_filter_manifest)

    refs = subparsers.add_parser("materialize-references", help="write benchmark reference files from a manifest and optionally rewrite manifest metadata")
    refs.add_argument("manifest")
    refs.add_argument("--out", default="data/references")
    refs.add_argument("--rewritten-manifest", help="optional manifest output path with metadata.reference_path populated")
    refs.set_defaults(func=cmd_materialize_references)

    refs_audit = subparsers.add_parser("audit-references", help="audit benchmark reference files using the same parser as scoring")
    refs_audit.add_argument("manifest")
    refs_audit.add_argument("--out", default="outputs/reference-audit")
    refs_audit.add_argument("--require-claim-ready", action="store_true")
    refs_audit.set_defaults(func=cmd_audit_references)

    normalize_images = subparsers.add_parser("normalize-manifest-images", help="copy/convert manifest images to OCR-supported suffixes")
    normalize_images.add_argument("manifest")
    normalize_images.add_argument("--out", default="data/normalized-images")
    normalize_images.add_argument("--image-dir-name", default="images")
    normalize_images.add_argument("--output-manifest-name", default="manifest-images-normalized.jsonl")
    normalize_images.set_defaults(func=cmd_normalize_manifest_images)

    create_eval = subparsers.add_parser("create-eval-pack", help="generate a held-out Nepali + English document eval pack")
    create_eval.add_argument("--out", default="data/eval-pack", help="eval-pack output directory")
    create_eval.add_argument("--count-per-template", type=int, default=1)
    create_eval.add_argument("--input-format", choices=["text", "image"], default="text")
    create_eval.add_argument("--degradation", action="append", help="degradation to generate; repeatable")
    create_eval.add_argument("--seed", type=int, default=13)
    create_eval.add_argument("--font-path", help="font path for image rendering")
    create_eval.add_argument("--template", action="append", default=[], help="template name to generate; repeatable")
    create_eval.add_argument("--variant-offset", type=int, default=0, help="starting copy index for generated template variants")
    create_eval.set_defaults(func=cmd_create_eval_pack)

    audit_eval = subparsers.add_parser("audit-eval-pack", help="audit eval-pack manifest integrity and claim readiness")
    audit_eval.add_argument("manifest", help="eval-pack manifest JSONL")
    audit_eval.add_argument("--out", default="outputs/eval-pack-audit")
    audit_eval.add_argument("--require-claim-ready", action="store_true")
    audit_eval.set_defaults(func=cmd_audit_eval_pack)

    crops = subparsers.add_parser("extract-crops", help="extract line/cell/caption recognizer crops from rendered eval-pack layout metadata")
    crops.add_argument("manifest", help="source eval-pack manifest JSONL")
    crops.add_argument("--out", default="data/crops", help="crop manifest output directory")
    crops.add_argument("--crop-type", action="append", default=[], choices=["line", "table-cell", "figure-caption"], help="crop type to extract; repeatable")
    crops.add_argument("--split", default="train", help="split label for emitted crop manifest rows")
    crops.add_argument("--dataset-name", default="ocrtech-crops")
    crops.add_argument("--min-text-length", type=int, default=1)
    crops.set_defaults(func=cmd_extract_crops)

    train_rec = subparsers.add_parser("train-recognizer", help="write a PaddleOCR recognizer training recipe")
    train_rec.add_argument("--train-manifest", required=True)
    train_rec.add_argument("--eval-manifest", required=True)
    train_rec.add_argument("--out", default="runs/recognizer")
    train_rec.add_argument("--backend", choices=["paddleocr", "hf-vision-encoder-decoder"], default="paddleocr")
    train_rec.add_argument("--base-model", default=PADDLE_PPOCRV5_MOBILE_REC_PRETRAINED)
    train_rec.add_argument("--dictionary")
    train_rec.add_argument("--max-target-length", type=int, default=128)
    train_rec.add_argument("--train-batch-size", type=int, default=8)
    train_rec.add_argument("--eval-batch-size", type=int, default=8)
    train_rec.add_argument("--learning-rate", type=float, default=5e-5)
    train_rec.add_argument("--epochs", type=float, default=3.0)
    train_rec.add_argument("--warmup-epoch", type=float, default=5.0)
    train_rec.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    train_rec.add_argument("--num-workers", type=int, default=2)
    train_rec.add_argument("--train-drop-last", action=argparse.BooleanOptionalAction, default=True)
    train_rec.add_argument("--main-indicator", choices=["acc", "norm_edit_dis"], default="acc")
    train_rec.add_argument("--eval-batch-step", type=int, default=200)
    train_rec.add_argument("--failure-review", help="recognizer-failure-review.json used to reject stopped model families")
    train_rec.add_argument("--run", action="store_true", help="run PaddleOCR training command after writing config")
    train_rec.set_defaults(func=cmd_train_recognizer)

    export_rec = subparsers.add_parser("export-recognizer", help="write or run a PaddleOCR recognizer export recipe")
    export_rec.add_argument("--training-config", required=True)
    export_rec.add_argument("--checkpoint", required=True)
    export_rec.add_argument("--out", default="runs/recognizer/export")
    export_rec.add_argument("--inference-dir")
    export_rec.add_argument("--paddleocr-dir", default="PaddleOCR")
    export_rec.add_argument(
        "--export-option",
        action="append",
        default=[],
        help="additional PaddleOCR -o override such as Global.character_dict_path=/abs/dict.txt; repeatable",
    )
    export_rec.add_argument("--run", action="store_true")
    export_rec.set_defaults(func=cmd_export_recognizer)

    package_rec = subparsers.add_parser("package-recognizer", help="package PaddleOCR inference artifacts into a candidate model card")
    package_rec.add_argument("--inference-dir", required=True)
    package_rec.add_argument("--dictionary", required=True)
    package_rec.add_argument("--out", default="runs/recognizer/package")
    package_rec.add_argument("--model-id", required=True)
    package_rec.add_argument("--base-model", default=PADDLE_DEVANAGARI_PPOCRV3_PRETRAINED)
    package_rec.add_argument("--paddle-lang", default="ne")
    package_rec.add_argument("--text-recognition-model-name", default="PP-OCRv5_mobile_rec")
    package_rec.add_argument("--recognition-mode", choices=["full_page", "line", "auto", "recognition_only"], default="full_page")
    package_rec.add_argument("--line-mode-max-height", type=int, default=256)
    package_rec.add_argument("--line-mode-min-aspect-ratio", type=float, default=3.0)
    package_rec.add_argument("--train-manifest")
    package_rec.add_argument("--eval-manifest")
    package_rec.add_argument("--training-summary")
    package_rec.add_argument("--metrics-report")
    package_rec.add_argument("--admission-validation-report")
    package_rec.add_argument("--source-archive")
    package_rec.add_argument("--source-checkpoint")
    package_rec.add_argument("--source-training-config")
    package_rec.add_argument("--export-recipe")
    package_rec.set_defaults(func=cmd_package_recognizer)

    bundle_rec = subparsers.add_parser("bundle-recognizer-training", help="package recognizer recipe, manifests, and images for transfer to a training host")
    bundle_rec.add_argument("--recipe-dir", required=True)
    bundle_rec.add_argument("--train-manifest", required=True)
    bundle_rec.add_argument("--eval-manifest", required=True)
    bundle_rec.add_argument("--out", default="runs/recognizer-training-bundle")
    bundle_rec.add_argument("--archive-name", default="recognizer-training-bundle.tar.gz")
    bundle_rec.add_argument("--base-dir", default=".")
    bundle_rec.add_argument("--failure-review", help="recognizer-failure-review.json used to reject stopped model families before bundling")
    bundle_rec.add_argument("--require-corpus-audit", action="store_true", help="fail unless train/eval manifests pass recognizer corpus coverage checks")
    bundle_rec.add_argument("--min-train-samples", type=int, default=1000)
    bundle_rec.add_argument("--min-eval-samples", type=int, default=100)
    bundle_rec.add_argument("--min-train-english", type=int, default=200)
    bundle_rec.add_argument("--min-train-devanagari", type=int, default=200)
    bundle_rec.add_argument("--min-train-mixed", type=int, default=25)
    bundle_rec.add_argument("--min-eval-english", type=int, default=50)
    bundle_rec.add_argument("--min-eval-devanagari", type=int, default=50)
    bundle_rec.add_argument("--min-eval-mixed", type=int, default=10)
    bundle_rec.add_argument("--min-train-latin-only", type=int, default=50)
    bundle_rec.add_argument("--min-train-devanagari-only", type=int, default=50)
    bundle_rec.add_argument("--min-eval-latin-only", type=int, default=10)
    bundle_rec.add_argument("--min-eval-devanagari-only", type=int, default=10)
    bundle_rec.add_argument("--min-train-real", type=int, default=100)
    bundle_rec.add_argument("--min-train-synthetic", type=int, default=100)
    bundle_rec.add_argument("--require-eval-real", action="store_true", help="with --require-corpus-audit, require held-out real non-synthetic eval rows")
    bundle_rec.set_defaults(func=cmd_bundle_recognizer_training)

    package_hf = subparsers.add_parser("package-hf-recognizer", help="package a Hugging Face OCR model directory into a candidate model card")
    package_hf.add_argument("--model-dir", required=True)
    package_hf.add_argument("--out", default="runs/hf-recognizer/package")
    package_hf.add_argument("--model-id", required=True)
    package_hf.add_argument("--base-model", required=True)
    package_hf.add_argument("--train-manifest")
    package_hf.add_argument("--eval-manifest")
    package_hf.add_argument("--training-summary")
    package_hf.add_argument("--metrics-report")
    package_hf.add_argument("--admission-validation-report")
    package_hf.add_argument("--max-new-tokens", type=int, default=128)
    package_hf.set_defaults(func=cmd_package_hf_recognizer)

    finalize_hf = subparsers.add_parser("finalize-hf-recognizer-run", help="package and audit a completed Hugging Face recognizer training run")
    finalize_hf.add_argument("--run-dir", required=True)
    finalize_hf.add_argument("--out")
    finalize_hf.add_argument("--model-id", required=True)
    finalize_hf.add_argument("--base-model", required=True)
    finalize_hf.add_argument("--train-manifest")
    finalize_hf.add_argument("--eval-manifest")
    finalize_hf.add_argument("--training-summary")
    finalize_hf.add_argument("--admission-validation-report")
    finalize_hf.add_argument("--max-new-tokens", type=int, default=128)
    finalize_hf.set_defaults(func=cmd_finalize_hf_recognizer_run)

    package_hf_corrector = subparsers.add_parser("package-hf-corrector", help="package a Hugging Face text corrector into a composite OCR candidate model card")
    package_hf_corrector.add_argument("--model-dir", required=True)
    package_hf_corrector.add_argument("--out", default="runs/hf-corrector/package")
    package_hf_corrector.add_argument("--model-id", required=True)
    package_hf_corrector.add_argument("--base-model", required=True)
    package_hf_corrector.add_argument("--base-engine", default="tesseract")
    package_hf_corrector.add_argument("--base-engine-kwarg", action="append", default=[], help="base OCR engine kwarg as key=value; repeatable")
    package_hf_corrector.add_argument("--train-manifest")
    package_hf_corrector.add_argument("--eval-manifest")
    package_hf_corrector.add_argument("--metrics-report")
    package_hf_corrector.add_argument("--max-new-tokens", type=int, default=256)
    package_hf_corrector.set_defaults(func=cmd_package_hf_corrector)

    pair_mining = subparsers.add_parser("generate-correction-pairs", help="run an OCR engine on a manifest and write noisy-clean correction pairs")
    pair_mining.add_argument("--manifest", required=True)
    pair_mining.add_argument("--out", required=True)
    pair_mining.add_argument("--engine", default="tesseract", choices=["tesseract", "paddleocr", "surya", "candidate", "ours", "sidecar"])
    pair_mining.add_argument("--model-config")
    pair_mining.add_argument("--limit", type=int)
    pair_mining.add_argument("--only-errors", action="store_true")
    pair_mining.set_defaults(func=cmd_generate_correction_pairs)

    train_corr = subparsers.add_parser("train-corrector", help="write an OCR correction training recipe")
    train_corr.add_argument("--out", default="runs/corrector")
    train_corr.add_argument("--clean-manifest", help="clean OCR manifest used to synthesize noisy pairs")
    train_corr.add_argument("--pairs-source", help="JSONL with noisy_text and clean_text fields")
    train_corr.add_argument("--hf-dataset", help="verified HF correction dataset id, for example cfilt/RoundTripOCR-nepali")
    train_corr.add_argument("--hf-split", default="train")
    train_corr.add_argument("--limit", type=int)
    train_corr.add_argument("--seed", type=int, default=13)
    train_corr.add_argument("--backend", choices=["recipe", "hf-seq2seq"], default="recipe")
    train_corr.add_argument("--base-model", default="google/byt5-small")
    train_corr.add_argument("--eval-ratio", type=float, default=0.1)
    train_corr.add_argument("--max-source-length", type=int, default=256)
    train_corr.add_argument("--max-target-length", type=int, default=256)
    train_corr.add_argument("--train-batch-size", type=int, default=16)
    train_corr.add_argument("--eval-batch-size", type=int, default=16)
    train_corr.add_argument("--learning-rate", type=float, default=5e-5)
    train_corr.add_argument("--epochs", type=float, default=3.0)
    train_corr.add_argument("--run", action="store_true")
    train_corr.set_defaults(func=cmd_train_corrector)

    model_card = subparsers.add_parser("create-model-card", help="write a candidate model artifact card")
    model_card.add_argument("--out", required=True)
    model_card.add_argument("--model-id", required=True)
    model_card.add_argument("--backend", choices=["paddleocr", "sidecar", "tesseract", "surya", "hf_vision_encoder_decoder", "text_correction_composite", "quality_select_composite", "quality_ranked_ensemble", "script_select_composite", "line_align_composite", "table_cell_refine_composite", "figure_caption_refine_composite"], default="paddleocr")
    model_card.add_argument("--base-model", default=PADDLE_DEVANAGARI_PPOCRV3_PRETRAINED)
    model_card.add_argument("--artifact", action="append", default=[], help="required model artifact path; repeatable")
    model_card.add_argument("--backend-kwarg", action="append", default=[], help="backend kwarg as key=value; repeatable")
    model_card.add_argument("--paddleocr-kwarg", action="append", default=[], help="deprecated alias for --backend-kwarg; repeatable")
    model_card.add_argument("--train-manifest")
    model_card.add_argument("--eval-manifest")
    model_card.add_argument("--metrics-report")
    model_card.add_argument("--admission-validation-report")
    model_card.add_argument("--fallback-model-card", help="fallback component model-card path to record in provenance")
    model_card.add_argument("--notes")
    model_card.set_defaults(func=cmd_create_model_card)

    model_audit = subparsers.add_parser("audit-model", help="audit a candidate model card and artifacts")
    model_audit.add_argument("model_config")
    model_audit.add_argument("--out", default="outputs/model-audit")
    model_audit.set_defaults(func=cmd_audit_model)

    parse = subparsers.add_parser("parse-doc", help="convert image/PDF/text sidecar to Markdown, JSON, tables, and figures")
    parse.add_argument("input")
    parse.add_argument("--out", default="outputs/document")
    parse.add_argument("--engine", default="auto", choices=["auto", "sidecar", "tesseract", "paddleocr", "surya", "candidate", "ours"])
    parse.add_argument("--model-config", help="candidate model card path when --engine candidate/ours")
    parse.add_argument("--fallback-engine", choices=["sidecar", "tesseract", "paddleocr", "surya"])
    parse.add_argument("--fallback-model-config", help="candidate model card path to use as a selective fallback engine")
    parse.add_argument("--low-confidence-threshold", type=float, default=0.80)
    parse.add_argument("--fallback-min-quality-score", type=float, help="trigger configured fallback when primary document quality score is below this value")
    parse.set_defaults(func=cmd_parse_doc)

    parse_limbu = subparsers.add_parser("parse-limbu-doc", help="parse a mixed Devanagari/Sirijonga Limbu document with review/audit artifacts")
    parse_limbu.add_argument("input")
    parse_limbu.add_argument("--out", default="outputs/limbu-document")
    parse_limbu.add_argument("--engine", default="auto", choices=["auto", "sidecar", "tesseract", "paddleocr", "surya", "candidate", "ours"])
    parse_limbu.add_argument("--model-config", help="candidate model card path when --engine candidate/ours")
    parse_limbu.add_argument("--fallback-engine", choices=["sidecar", "tesseract", "paddleocr", "surya"])
    parse_limbu.add_argument("--fallback-model-config", help="candidate model card path to use as a selective fallback engine")
    parse_limbu.add_argument("--low-confidence-threshold", type=float, default=0.80)
    parse_limbu.add_argument("--sirijonga-low-confidence-threshold", type=float, help="override review threshold for Sirijonga/Limbu-script lines")
    parse_limbu.add_argument("--devanagari-low-confidence-threshold", type=float, help="override review threshold for Devanagari-written Limbu lines")
    parse_limbu.add_argument("--mixed-script-low-confidence-threshold", type=float, help="override review threshold for mixed Sirijonga/Devanagari lines")
    parse_limbu.add_argument("--other-script-low-confidence-threshold", type=float, help="override review threshold for lines without enough Limbu or Devanagari script")
    parse_limbu.add_argument("--fallback-min-quality-score", type=float, help="trigger configured fallback when primary document quality score is below this value")
    parse_limbu.add_argument("--script-ratio-threshold", type=float, default=0.20, help="minimum line-level codepoint ratio used to classify Devanagari or Sirijonga")
    parse_limbu.add_argument("--capture-prep-metadata", help="limbu-capture-prep.json from prepare-limbu-capture to bind raw/prepared hashes into OCR output")
    parse_limbu.add_argument("--post-correction-profile", help="guarded Limbu JSON correction profile to apply before review/audit export")
    parse_limbu.add_argument("--line-detection-mode", default="engine", choices=["engine", "image-line"], help="use the OCR engine's native page detection or detect line crops from image pixels before recognition")
    parse_limbu.add_argument("--image-line-threshold", default="otsu", help="image-line threshold: 'otsu' or an integer 0..255")
    parse_limbu.add_argument("--image-line-bbox-source", default="ink", choices=["ink", "dilated"])
    parse_limbu.add_argument(
        "--image-line-reading-order",
        default="auto_layout",
        choices=[
            "top_to_bottom",
            "column_major",
            "wide_column_right_to_left",
            "wide_column_right_to_left_split_tall_last",
            "auto_layout",
            "auto_layout_split_tall_last",
        ],
    )
    parse_limbu.add_argument("--image-line-horizontal-kernel", type=int, default=23)
    parse_limbu.add_argument("--image-line-vertical-kernel", type=int, default=3)
    parse_limbu.add_argument("--image-line-dilation-iterations", type=int, default=1)
    parse_limbu.add_argument("--image-line-min-width", type=int, default=35)
    parse_limbu.add_argument("--image-line-min-height", type=int, default=10)
    parse_limbu.add_argument("--image-line-min-area", type=int, default=100)
    parse_limbu.add_argument("--image-line-max-height", type=int, default=140)
    parse_limbu.add_argument("--image-line-min-aspect-ratio", type=float, default=0.0)
    parse_limbu.add_argument("--image-line-max-aspect-ratio", type=float, default=0.0)
    parse_limbu.add_argument("--image-line-detector-padding", type=int, default=2)
    parse_limbu.add_argument("--image-line-crop-padding", type=int, default=12)
    parse_limbu.add_argument("--image-line-split-tall-components", action="store_true")
    parse_limbu.add_argument("--image-line-split-tall-row-min-ink", type=int, default=2)
    parse_limbu.add_argument("--image-line-split-tall-max-row-gap", type=int, default=4)
    parse_limbu.add_argument("--image-line-split-wide-components", action="store_true")
    parse_limbu.add_argument("--image-line-split-wide-col-min-ink", type=int, default=2)
    parse_limbu.add_argument("--image-line-split-wide-max-col-gap", type=int, default=24)
    parse_limbu.add_argument("--image-line-split-wide-min-width", type=int, default=600)
    parse_limbu.add_argument("--image-line-split-detected-row-components", action="store_true")
    parse_limbu.add_argument("--image-line-split-detected-row-col-min-ink", type=int, default=2)
    parse_limbu.add_argument("--image-line-split-detected-row-max-col-gap", type=int, default=24)
    parse_limbu.add_argument("--image-line-split-detected-row-min-width", type=int, default=600)
    parse_limbu.add_argument("--image-line-split-detected-row-min-segment-width", type=int, default=40)
    parse_limbu.add_argument("--image-line-split-detected-tall-components", action="store_true")
    parse_limbu.add_argument("--image-line-split-detected-tall-row-min-ink", type=int, default=20)
    parse_limbu.add_argument("--image-line-split-detected-tall-max-row-gap", type=int, default=4)
    parse_limbu.add_argument("--image-line-split-detected-tall-min-height", type=int, default=90)
    parse_limbu.add_argument("--image-line-split-detected-tall-min-segment-height", type=int, default=24)
    parse_limbu.add_argument("--image-line-merge-same-row-components", action="store_true")
    parse_limbu.add_argument("--image-line-merge-same-row-y-tolerance", type=float, default=8.0)
    parse_limbu.add_argument("--image-line-merge-same-row-max-gap", type=float, default=120.0)
    parse_limbu.add_argument("--image-line-merge-same-row-max-center-delta", type=float, default=300.0)
    parse_limbu.add_argument("--image-line-merge-same-row-max-width", type=float, default=420.0)
    parse_limbu.add_argument("--image-line-merge-same-row-auto-fragmented-top-to-bottom", action="store_true")
    parse_limbu.add_argument("--image-line-merge-same-row-auto-min-reduction-ratio", type=float, default=0.18)
    parse_limbu.add_argument("--image-line-merge-same-row-auto-min-reduction-count", type=int, default=8)
    parse_limbu.add_argument(
        "--image-line-rescue-detector-pass",
        action="append",
        default=[],
        help=(
            "extra image-line detector pass as comma-separated key=value overrides; "
            "repeatable, for example horizontal_kernel=15,vertical_kernel=1,min_width=20"
        ),
    )
    parse_limbu.add_argument("--image-line-merge-iou-threshold", type=float, default=0.80)
    parse_limbu.add_argument("--image-line-allow-empty-lines", action="store_true")
    parse_limbu.add_argument("--image-line-filter-drop-empty", action="store_true")
    parse_limbu.add_argument("--image-line-filter-min-confidence", type=float)
    parse_limbu.add_argument("--image-line-filter-require-script", default="any", choices=["any", "sirijonga", "devanagari", "limbu_or_devanagari"])
    parse_limbu.add_argument("--image-line-filter-min-width-ratio", type=float)
    parse_limbu.add_argument("--image-line-filter-max-width-ratio", type=float)
    parse_limbu.add_argument("--image-line-filter-min-height-ratio", type=float)
    parse_limbu.add_argument("--image-line-filter-max-height-ratio", type=float)
    parse_limbu.set_defaults(func=cmd_parse_limbu_doc)

    prep_limbu = subparsers.add_parser("prepare-limbu-capture", help="preserve and normalize an imperfect Limbu page capture before OCR")
    prep_limbu.add_argument("input")
    prep_limbu.add_argument("--out", default="outputs/limbu-capture-prep")
    prep_limbu.add_argument("--rotate-degrees", type=float, default=0.0, help="manual deskew rotation in degrees; positive is counter-clockwise")
    prep_limbu.add_argument("--crop-box", help="manual crop box as left,top,right,bottom after EXIF transpose")
    prep_limbu.add_argument(
        "--perspective-quad",
        help=(
            "manual page corners as top_left_x,top_left_y,top_right_x,top_right_y,"
            "bottom_right_x,bottom_right_y,bottom_left_x,bottom_left_y after EXIF transpose/crop"
        ),
    )
    prep_limbu.add_argument("--auto-detect-page", action="store_true", help="detect a four-corner page contour and perspective-rectify it")
    prep_limbu.add_argument("--auto-deskew", action="store_true", help="detect near-horizontal text skew and rotate the prepared image")
    prep_limbu.add_argument("--max-auto-deskew-degrees", type=float, default=8.0, help="maximum absolute angle automatic deskew is allowed to apply")
    prep_limbu.add_argument("--no-autocontrast", action="store_true")
    prep_limbu.add_argument("--no-grayscale", action="store_true")
    prep_limbu.set_defaults(func=cmd_prepare_limbu_capture)

    audit_limbu_capture_parser = subparsers.add_parser("audit-limbu-capture", help="audit a prepared Limbu capture bundle for stale/missing raw and prepared images")
    audit_limbu_capture_parser.add_argument("capture_metadata", help="limbu-capture-prep.json or its containing directory")
    audit_limbu_capture_parser.add_argument("--out", help="audit report directory; defaults to the capture metadata directory")
    audit_limbu_capture_parser.add_argument(
        "--require-metadata-path-self",
        action="store_true",
        help="fail unless metadata_path points to the audited limbu-capture-prep.json file",
    )
    audit_limbu_capture_parser.add_argument("--min-prepared-width", type=int)
    audit_limbu_capture_parser.add_argument("--min-prepared-height", type=int)
    audit_limbu_capture_parser.add_argument("--min-prepared-entropy", type=float)
    audit_limbu_capture_parser.add_argument("--min-prepared-luminance-stddev", type=float)
    audit_limbu_capture_parser.add_argument("--min-prepared-edge-stddev", type=float)
    audit_limbu_capture_parser.set_defaults(func=cmd_audit_limbu_capture)

    limbu_preflight = subparsers.add_parser("preflight-limbu-pipeline", help="check runtime readiness for the packaged Limbu OCR pipeline")
    limbu_preflight.add_argument("--model-config", default="outputs/limbu-first-router-v1/model-card.json")
    limbu_preflight.add_argument("--out", default="outputs/limbu-first-router-v1/runtime-preflight")
    limbu_preflight.add_argument("--required-script", default="limbu", help="required router script key, defaults to limbu")
    limbu_preflight.add_argument(
        "--required-tesseract-language",
        action="append",
        default=None,
        help="required Tesseract language code; repeatable, defaults to nep and eng",
    )
    limbu_preflight.add_argument(
        "--require-validated-components",
        action="store_true",
        help="fail if any recorded router component admission is experimental, pending, missing, or not validated/admitted",
    )
    limbu_preflight.add_argument(
        "--required-component-role",
        action="append",
        default=None,
        help="component admission role that must be present; repeatable, e.g. devanagari_primary and sirijonga_secondary",
    )
    limbu_preflight.set_defaults(func=cmd_preflight_limbu_pipeline)

    limbu_apply_reviews = subparsers.add_parser("apply-limbu-review-corrections", help="apply accepted Limbu review-queue corrections to a parsed document")
    limbu_apply_reviews.add_argument("--document", required=True, help="document.json emitted by parse-limbu-doc")
    limbu_apply_reviews.add_argument("--review-queue", required=True, help="limbu-review-queue.tsv with accepted corrected_text rows")
    limbu_apply_reviews.add_argument("--out", default="outputs/limbu-corrected-document")
    limbu_apply_reviews.add_argument(
        "--accepted-status",
        action="append",
        default=None,
        help="review_status value to apply; repeatable, defaults to accepted/approved/corrected/verified/done",
    )
    limbu_apply_reviews.set_defaults(func=cmd_apply_limbu_review_corrections)

    limbu_pair_pack = subparsers.add_parser(
        "build-limbu-correction-pair-pack",
        help="build a frozen train/heldout Limbu correction-pair pack from reviewed correction pairs",
    )
    limbu_pair_pack.add_argument(
        "correction_pairs",
        nargs="+",
        help="limbu-correction-pairs.jsonl emitted by apply-limbu-review-corrections",
    )
    limbu_pair_pack.add_argument("--out", default="outputs/limbu-correction-pair-pack")
    limbu_pair_pack.add_argument("--pack-id", required=True)
    limbu_pair_pack.add_argument("--heldout-fraction", type=float, default=0.20)
    limbu_pair_pack.add_argument("--min-heldout", type=int, default=1)
    limbu_pair_pack.set_defaults(func=cmd_build_limbu_correction_pair_pack)

    limbu_pair_pack_audit = subparsers.add_parser(
        "audit-limbu-correction-pair-pack",
        help="audit a frozen Limbu correction-pair pack for stale or tampered artifacts",
    )
    limbu_pair_pack_audit.add_argument("pack_dir")
    limbu_pair_pack_audit.add_argument("--out", help="audit report directory; defaults to the pack directory")
    limbu_pair_pack_audit.set_defaults(func=cmd_audit_limbu_correction_pair_pack)

    limbu_derive_post_correction = subparsers.add_parser(
        "derive-limbu-post-correction-profile",
        help="derive an experimental deterministic Limbu post-correction profile from reviewed correction pairs",
    )
    limbu_derive_post_correction.add_argument(
        "correction_pairs",
        nargs="+",
        help="limbu-correction-pairs.jsonl emitted by apply-limbu-review-corrections",
    )
    limbu_derive_post_correction.add_argument("--out", default="outputs/limbu-post-correction-profile-derived")
    limbu_derive_post_correction.add_argument("--profile-id", default="limbu-derived-post-correction-profile-v1")
    limbu_derive_post_correction.add_argument("--min-support", type=int, default=1)
    limbu_derive_post_correction.set_defaults(func=cmd_derive_limbu_post_correction_profile)

    limbu_score_post_correction = subparsers.add_parser(
        "score-limbu-post-correction-profile",
        help="score a Limbu post-correction profile on held-out correction pairs",
    )
    limbu_score_post_correction.add_argument("profile", help="limbu-post-correction-profile.json")
    limbu_score_post_correction.add_argument(
        "correction_pairs",
        nargs="+",
        help="held-out Limbu correction-pairs JSONL files",
    )
    limbu_score_post_correction.add_argument("--out", default="outputs/limbu-post-correction-profile-eval")
    limbu_score_post_correction.add_argument("--frozen-eval-pack", help="identifier/path for the frozen held-out correction-pair pack")
    limbu_score_post_correction.set_defaults(func=cmd_score_limbu_post_correction_profile)

    limbu_admit_post_correction = subparsers.add_parser(
        "admit-limbu-post-correction-profile",
        help="write a validated Limbu post-correction profile from a passing held-out eval JSON",
    )
    limbu_admit_post_correction.add_argument("profile", help="source limbu-post-correction-profile.json")
    limbu_admit_post_correction.add_argument("eval_run", help="limbu-post-correction-profile-eval.json from score-limbu-post-correction-profile")
    limbu_admit_post_correction.add_argument("--out", default="outputs/limbu-post-correction-profile-admission")
    limbu_admit_post_correction.add_argument("--output-filename", default="limbu-post-correction-profile-admitted.json")
    limbu_admit_post_correction.set_defaults(func=cmd_admit_limbu_post_correction_profile)

    limbu_audit_post_correction = subparsers.add_parser(
        "audit-limbu-post-correction-profile",
        help="audit a deterministic Limbu post-correction profile and its admission evidence",
    )
    limbu_audit_post_correction.add_argument("profile")
    limbu_audit_post_correction.add_argument("--out", default="outputs/limbu-post-correction-profile-audit")
    limbu_audit_post_correction.set_defaults(func=cmd_audit_limbu_post_correction_profile)

    limbu_audit_output = subparsers.add_parser("audit-limbu-output", help="audit a Limbu OCR or correction output directory for stale/missing artifacts")
    limbu_audit_output.add_argument("output_dir")
    limbu_audit_output.add_argument("--out", help="audit report directory; defaults to the output directory")
    limbu_audit_output.add_argument("--require-capture-prep", action="store_true", help="fail unless the output is bound to a passing prepare-limbu-capture audit")
    limbu_audit_output.add_argument("--min-capture-prepared-width", type=int)
    limbu_audit_output.add_argument("--min-capture-prepared-height", type=int)
    limbu_audit_output.add_argument("--min-capture-prepared-entropy", type=float)
    limbu_audit_output.add_argument("--min-capture-prepared-luminance-stddev", type=float)
    limbu_audit_output.add_argument("--min-capture-prepared-edge-stddev", type=float)
    limbu_audit_output.add_argument(
        "--require-capture-metadata-path-self",
        action="store_true",
        help="fail capture-prep replay unless metadata_path points to the replayed metadata file",
    )
    limbu_audit_output.add_argument("--require-no-pending-review", action="store_true", help="fail when OCR/review output still has unresolved review rows")
    limbu_audit_output.add_argument(
        "--require-reviewer-for-corrections",
        action="store_true",
        help="fail review-correction outputs when accepted correction rows have no reviewer",
    )
    limbu_audit_output.add_argument(
        "--require-no-dropped-image-lines",
        action="store_true",
        help="fail image-line OCR outputs when detected lines were filtered or dropped",
    )
    limbu_audit_output.add_argument("--min-line-count", type=int, help="fail unless the output document contains at least this many OCR text lines")
    limbu_audit_output.add_argument("--min-average-line-confidence", type=float, help="fail unless quality.metrics.average_line_confidence is at least this value")
    limbu_audit_output.add_argument("--min-quality-score", type=float, help="fail unless quality.quality_score is at least this value")
    limbu_audit_output.add_argument(
        "--require-script",
        action="append",
        choices=["limbu_sirijonga", "devanagari_limbu", "mixed_limbu_devanagari", "other"],
        default=[],
        help="fail unless at least one line of this Limbu script class is observed; repeatable",
    )
    limbu_audit_output.add_argument(
        "--require-script-count",
        action="append",
        default=[],
        metavar="SCRIPT=COUNT",
        help="fail unless the observed Limbu script class has at least COUNT lines; repeatable",
    )
    limbu_audit_output.set_defaults(func=cmd_audit_limbu_output)

    benchmark = subparsers.add_parser("benchmark", help="compare OCR baselines on the same files")
    benchmark.add_argument("inputs", nargs="*")
    benchmark.add_argument("--inputs-from-manifest", help="load benchmark input paths from a manifest image_path field")
    benchmark.add_argument("--out", default="outputs/benchmark")
    benchmark.add_argument("--baselines", default="ours,tesseract,stock-paddle,glm-ocr,paddleocr-vl")
    benchmark.add_argument("--references-dir")
    benchmark.add_argument("--eval-manifest")
    benchmark.add_argument("--candidate-model-config")
    benchmark.add_argument("--fallback-engine", choices=["sidecar", "tesseract", "paddleocr", "surya"])
    benchmark.add_argument("--fallback-model-config", help="candidate model card path to use as a selective fallback engine")
    benchmark.add_argument("--low-confidence-threshold", type=float, default=0.80)
    benchmark.add_argument("--fallback-min-quality-score", type=float, help="trigger configured candidate fallback when primary document quality score is below this value")
    benchmark.add_argument("--resume-existing", action="store_true", help="score existing document.json outputs and run only missing pages")
    benchmark.add_argument("--sample-timeout-seconds", type=int, help="mark one page as error if a baseline exceeds this runtime")
    benchmark.add_argument("--capture-gpu-metrics", action="store_true", help="sample nvidia-smi before and after each benchmark row when available")
    benchmark.set_defaults(func=cmd_benchmark)

    summarize = subparsers.add_parser("summarize-benchmark", help="summarize aggregate and paired benchmark metrics from report.json")
    summarize.add_argument("--benchmark-report", required=True)
    summarize.add_argument("--out", default="outputs/benchmark-summary")
    summarize.add_argument("--candidate", default="candidate")
    summarize.add_argument("--baseline", action="append", default=[], help="baseline to compare against candidate; repeatable")
    summarize.add_argument("--metric", action="append", default=[], help="metric to summarize in paired comparisons; repeatable")
    summarize.set_defaults(func=cmd_summarize_benchmark)

    rescore = subparsers.add_parser("rescore-benchmark", help="recompute benchmark metrics from existing document.json outputs")
    rescore.add_argument("--benchmark-report", required=True)
    rescore.add_argument("--out", default="outputs/benchmark-rescored")
    rescore.add_argument("--references-dir")
    rescore.add_argument("--eval-manifest")
    rescore.set_defaults(func=cmd_rescore_benchmark)

    derive_table = subparsers.add_parser(
        "derive-table-postprocess-benchmark",
        help="derive a table-postprocessed candidate from an existing baseline document output",
    )
    derive_table.add_argument("--benchmark-report", required=True)
    derive_table.add_argument("--out", default="outputs/benchmark-table-postprocess-derived")
    derive_table.add_argument("--source-baseline", required=True)
    derive_table.add_argument("--candidate-baseline", default="candidate")
    derive_table.add_argument("--references-dir")
    derive_table.add_argument("--eval-manifest")
    derive_table.set_defaults(func=cmd_derive_table_postprocess_benchmark)

    table_cells = subparsers.add_parser("analyze-table-cells", help="analyze cell-level table OCR errors from benchmark document outputs")
    table_cells.add_argument("--benchmark-report", required=True)
    table_cells.add_argument("--eval-manifest", required=True)
    table_cells.add_argument("--out", default="outputs/table-cell-analysis")
    table_cells.add_argument("--candidate", default="candidate")
    table_cells.add_argument("--baseline", action="append", default=[], help="baseline to compare against candidate; repeatable")
    table_cells.add_argument("--top-n", type=int, default=25)
    table_cells.set_defaults(func=cmd_analyze_table_cells)

    paper_table = subparsers.add_parser("paper-benchmark-table", help="merge benchmark summary artifacts into a paper-ready comparison table")
    paper_table.add_argument("--summary", action="append", default=[], required=True, help="path to summarize-benchmark summary.json; repeatable")
    paper_table.add_argument("--out", default="outputs/paper-table")
    paper_table.add_argument("--slice", default="all")
    paper_table.add_argument("--metric", action="append", default=[], help="aggregate metric column to include; repeatable")
    paper_table.add_argument("--system", action="append", default=[], help="system row to include and order explicitly; repeatable")
    paper_table.add_argument("--label", action="append", default=[], help="display label override as system=Label; repeatable")
    paper_table.add_argument("--source-preference", action="append", default=[], help="pin a system to one summary artifact as system=path; repeatable")
    paper_table.add_argument("--title", help="optional table title")
    paper_table.set_defaults(func=cmd_paper_benchmark_table)

    oracle = subparsers.add_parser("benchmark-oracle", help="compute per-sample oracle headroom from a benchmark report")
    oracle.add_argument("--benchmark-report", required=True)
    oracle.add_argument("--out", default="outputs/benchmark-oracle")
    oracle.add_argument("--metric", default="cer")
    oracle.add_argument("--direction", choices=["lower", "higher"])
    oracle.add_argument("--system", action="append", default=[], help="system to include in oracle selection; repeatable")
    oracle.add_argument("--oracle-name", default="oracle")
    oracle.set_defaults(func=cmd_benchmark_oracle)

    error_analysis = subparsers.add_parser("benchmark-error-analysis", help="rank hard samples and slice gaps from a benchmark report")
    error_analysis.add_argument("--benchmark-report", required=True)
    error_analysis.add_argument("--out", default="outputs/benchmark-error-analysis")
    error_analysis.add_argument("--system", required=True)
    error_analysis.add_argument("--baseline", action="append", default=[], help="baseline to compare against system; repeatable")
    error_analysis.add_argument("--metric", default="cer")
    error_analysis.add_argument("--direction", choices=["lower", "higher"])
    error_analysis.add_argument("--top-n", type=int, default=20)
    error_analysis.set_defaults(func=cmd_benchmark_error_analysis)

    failure_manifest = subparsers.add_parser("export-failure-manifest", help="write a manifest from ranked benchmark failure rows")
    failure_manifest.add_argument("--source-manifest", required=True)
    failure_manifest.add_argument("--top-failures", required=True)
    failure_manifest.add_argument("--out", required=True)
    failure_manifest.add_argument("--split", help="optional split override for selected entries")
    failure_manifest.set_defaults(func=cmd_export_failure_manifest)

    admission_decision = subparsers.add_parser("recognizer-admission-decision", help="summarize whether a recognizer admission validation report promotes or rejects a model branch")
    admission_decision.add_argument("--validation-report", required=True)
    admission_decision.add_argument("--out", default="outputs/recognizer-admission-decision")
    admission_decision.set_defaults(func=cmd_recognizer_admission_decision)

    failure_review = subparsers.add_parser("recognizer-failure-review", help="aggregate recognizer admission failures across runs and flag exhausted model families")
    failure_review.add_argument("--admission-decision", action="append", default=[], help="recognizer-admission-decision.json path; repeatable")
    failure_review.add_argument("--validation-report", action="append", default=[], help="validation.json path to convert into a review input; repeatable")
    failure_review.add_argument("--model-card", action="append", default=[], help="model-card.json path matched to inputs by order when provenance is unavailable; repeatable")
    failure_review.add_argument("--min-rejections-to-stop", type=int, default=3)
    failure_review.add_argument("--out", default="outputs/recognizer-failure-review")
    failure_review.set_defaults(func=cmd_recognizer_failure_review)

    calibrate = subparsers.add_parser("calibrate-quality-router", help="calibrate a quality-routed composite candidate from a calibration experiment")
    calibrate.add_argument("--experiment-report", required=True)
    calibrate.add_argument("--out", default="outputs/quality-router-calibration")
    calibrate.add_argument("--model-id", required=True)
    calibrate.add_argument("--base-model", default="tesseract-ocr")
    calibrate.add_argument("--primary-engine", required=True)
    calibrate.add_argument("--secondary-engine", required=True)
    calibrate.add_argument("--primary-engine-kwarg", action="append", default=[], help="primary engine kwarg as key=value; repeatable")
    calibrate.add_argument("--secondary-engine-kwarg", action="append", default=[], help="secondary engine kwarg as key=value; repeatable")
    calibrate.add_argument("--candidate-baseline", default="candidate")
    calibrate.add_argument("--primary-baseline", default="tesseract")
    calibrate.add_argument("--metric", default="cer")
    calibrate.add_argument("--direction", choices=["lower", "higher"])
    calibrate.add_argument("--threshold-start", type=float, default=-0.20)
    calibrate.add_argument("--threshold-end", type=float, default=0.20)
    calibrate.add_argument("--threshold-step", type=float, default=0.01)
    calibrate.add_argument("--notes")
    calibrate.set_defaults(func=cmd_calibrate_quality_router)

    calibrate_script = subparsers.add_parser("calibrate-script-router", help="calibrate a script-routed composite candidate from a calibration experiment")
    calibrate_script.add_argument("--experiment-report", required=True)
    calibrate_script.add_argument("--out", default="outputs/script-router-calibration")
    calibrate_script.add_argument("--model-id", required=True)
    calibrate_script.add_argument("--base-model", default="tesseract+surya")
    calibrate_script.add_argument("--primary-engine", required=True)
    calibrate_script.add_argument("--secondary-engine", required=True)
    calibrate_script.add_argument("--primary-engine-kwarg", action="append", default=[], help="primary engine kwarg as key=value; repeatable")
    calibrate_script.add_argument("--secondary-engine-kwarg", action="append", default=[], help="secondary engine kwarg as key=value; repeatable")
    calibrate_script.add_argument("--primary-baseline", default="tesseract")
    calibrate_script.add_argument("--secondary-baseline", default="surya")
    calibrate_script.add_argument("--metric", default="cer")
    calibrate_script.add_argument("--direction", choices=["lower", "higher"])
    calibrate_script.add_argument("--script", default="devanagari")
    calibrate_script.add_argument("--threshold-start", type=float, default=0.0)
    calibrate_script.add_argument("--threshold-end", type=float, default=0.6)
    calibrate_script.add_argument("--threshold-step", type=float, default=0.05)
    calibrate_script.add_argument("--routing-granularity", choices=["document", "page"], default="document")
    calibrate_script.add_argument("--secondary-structure-backfill", action="store_true")
    calibrate_script.add_argument("--notes")
    calibrate_script.set_defaults(func=cmd_calibrate_script_router)

    calibrate_ensemble = subparsers.add_parser("calibrate-tesseract-psm-ensemble", help="calibrate a quality-ranked Tesseract PSM ensemble from separate variant experiments")
    calibrate_ensemble.add_argument("--primary-experiment-report", required=True)
    calibrate_ensemble.add_argument("--variant-experiment", action="append", default=[], help="variant label and experiment report as label=path; repeatable")
    calibrate_ensemble.add_argument("--out", default="outputs/tesseract-psm-ensemble-calibration")
    calibrate_ensemble.add_argument("--model-id", required=True)
    calibrate_ensemble.add_argument("--metric", default="cer")
    calibrate_ensemble.add_argument("--direction", choices=["lower", "higher"])
    calibrate_ensemble.add_argument("--bias-start", type=float, default=-0.12)
    calibrate_ensemble.add_argument("--bias-end", type=float, default=0.12)
    calibrate_ensemble.add_argument("--bias-step", type=float, default=0.02)
    calibrate_ensemble.add_argument("--language", default="nep+eng")
    calibrate_ensemble.add_argument("--selection-margin", type=float, default=0.0)
    calibrate_ensemble.add_argument("--notes")
    calibrate_ensemble.set_defaults(func=cmd_calibrate_tesseract_psm_ensemble)

    experiment = subparsers.add_parser("run-experiment", help="run benchmark and validation from an eval manifest")
    experiment.add_argument("--eval-manifest", required=True)
    experiment.add_argument("--out", default="outputs/experiment")
    experiment.add_argument("--baselines", default="ours,tesseract,stock-paddle,glm-ocr,paddleocr-vl")
    experiment.add_argument("--references-dir")
    experiment.add_argument("--validation-config")
    experiment.add_argument("--candidate-model-config")
    experiment.add_argument("--fallback-engine", choices=["sidecar", "tesseract", "paddleocr", "surya"])
    experiment.add_argument("--fallback-model-config", help="candidate model card path to use as a selective fallback engine")
    experiment.add_argument("--low-confidence-threshold", type=float, default=0.80)
    experiment.add_argument("--fallback-min-quality-score", type=float, help="trigger configured candidate fallback when primary document quality score is below this value")
    experiment.add_argument("--train-manifest", action="append", default=[])
    experiment.add_argument("--gorkhapatra-review-audit", action="append", default=[], help="passed Gorkhapatra language-page review-audit JSON; repeatable")
    experiment.add_argument("--skip-validation", action="store_true")
    experiment.add_argument("--require-validated", action="store_true")
    experiment.add_argument("--preflight", action="store_true", help="run claim preflight before benchmark and stop on failure")
    experiment.add_argument("--allow-smoke-eval-pack", action="store_true", help="allow smoke eval packs during run-experiment preflight")
    experiment.add_argument("--require-trained-recognizer", action="store_true", help="require trained recognizer provenance during run-experiment preflight")
    experiment.add_argument("--require-model-admission", action="store_true", help="require passing recognizer admission during run-experiment preflight")
    experiment.set_defaults(func=cmd_run_experiment)

    preflight = subparsers.add_parser("preflight-claim", help="fail early if claim benchmark inputs, models, references, or engines are not ready")
    preflight.add_argument("--eval-manifest", required=True)
    preflight.add_argument("--out", default="outputs/preflight")
    preflight.add_argument("--baselines", default="candidate,tesseract,stock-paddle,glm-ocr,paddleocr-vl")
    preflight.add_argument("--references-dir")
    preflight.add_argument("--candidate-model-config")
    preflight.add_argument("--fallback-model-config", help="fallback model card path to audit before claim benchmark runs")
    preflight.add_argument("--validation-config")
    preflight.add_argument("--train-manifest", action="append", default=[])
    preflight.add_argument("--gorkhapatra-review-audit", action="append", default=[], help="passed Gorkhapatra language-page review-audit JSON; repeatable")
    preflight.add_argument("--allow-smoke-eval-pack", action="store_true")
    preflight.add_argument("--require-trained-recognizer", action="store_true")
    preflight.add_argument("--require-model-admission", action="store_true")
    preflight.set_defaults(func=cmd_preflight_claim)

    public_list = subparsers.add_parser("list-public-benchmarks", help="list known public benchmark registry entries")
    public_list.set_defaults(func=cmd_list_public_benchmarks)

    public_prepare = subparsers.add_parser("prepare-public-benchmark", help="convert a verified public benchmark into an auditable manifest")
    public_prepare.add_argument("benchmark", choices=["opendatalab/OmniDocBench"])
    public_prepare.add_argument("--out", required=True)
    public_prepare.add_argument("--limit", type=int)
    public_prepare.add_argument("--language")
    public_prepare.add_argument("--subset")
    public_prepare.add_argument("--data-source")
    public_prepare.add_argument("--require-tables", action="store_true")
    public_prepare.add_argument("--annotation-path", help="local OmniDocBench.json path for offline/schema-fixture conversion")
    public_prepare.add_argument("--local-image-root", help="local dataset root containing images/ for offline conversion")
    public_prepare.set_defaults(func=cmd_prepare_public_benchmark)

    public_audit = subparsers.add_parser("audit-public-benchmark", help="audit a public benchmark manifest for claim readiness")
    public_audit.add_argument("manifest")
    public_audit.add_argument("--out", default="outputs/public-benchmark-audit")
    public_audit.add_argument("--benchmark", help="override benchmark name when manifest metadata is ambiguous")
    public_audit.add_argument("--min-samples", type=int)
    public_audit.add_argument("--allow-missing-reference-paths", action="store_true")
    public_audit.set_defaults(func=cmd_audit_public_benchmark)

    gorkhapatra_audit = subparsers.add_parser("audit-gorkhapatra-source", help="audit Gorkhapatra/Naya Nepal article images and epaper PDFs")
    gorkhapatra_audit.add_argument("--out", default="outputs/gorkhapatra-source-audit")
    gorkhapatra_audit.add_argument("--category-url", action="append", default=[], help="Naya Nepal category URL; repeatable")
    gorkhapatra_audit.add_argument("--epaper-url", action="append", default=[], help="Gorkhapatra epaper list URL; repeatable")
    gorkhapatra_audit.add_argument("--category-html", action="append", default=[], help="local category HTML fixture/path; repeatable")
    gorkhapatra_audit.add_argument("--epaper-html", action="append", default=[], help="local epaper list HTML fixture/path; repeatable")
    gorkhapatra_audit.add_argument("--no-default-urls", action="store_true", help="only use explicitly supplied URLs/HTML files")
    gorkhapatra_audit.add_argument("--download-assets", action="store_true", help="download and hash discovered article images/PDFs")
    gorkhapatra_audit.add_argument("--max-article-images", type=int, default=0, help="max article images to download; 0 means no limit")
    gorkhapatra_audit.add_argument("--max-epaper-pdfs", type=int, default=0, help="max epaper PDFs to download; 0 means no limit")
    gorkhapatra_audit.add_argument("--timeout-seconds", type=float, default=30.0)
    gorkhapatra_audit.set_defaults(func=cmd_audit_gorkhapatra_source)

    gorkhapatra_pack = subparsers.add_parser("prepare-gorkhapatra-pack", help="turn a Gorkhapatra source audit into a pending-reference validation manifest")
    gorkhapatra_pack.add_argument("source_audit_json")
    gorkhapatra_pack.add_argument("--out", default="data/gorkhapatra-pack")
    gorkhapatra_pack.add_argument("--no-article-images", action="store_true")
    gorkhapatra_pack.add_argument("--no-epaper-pages", action="store_true")
    gorkhapatra_pack.add_argument("--max-article-images", type=int, default=0)
    gorkhapatra_pack.add_argument("--max-epaper-pdfs", type=int, default=0)
    gorkhapatra_pack.add_argument("--max-pages-per-pdf", type=int, default=2)
    gorkhapatra_pack.add_argument("--split", default="eval")
    gorkhapatra_pack.add_argument("--dataset-name", default="gorkhapatra-naya-nepal")
    gorkhapatra_pack.set_defaults(func=cmd_prepare_gorkhapatra_pack)

    language_pages = subparsers.add_parser("audit-gorkhapatra-language-pages", help="align Naya Nepal publication cards to same-date Gorkhapatra PDF language pages")
    language_pages.add_argument("--out", default="outputs/gorkhapatra-language-pages")
    language_pages.add_argument("--publication-url", action="append", default=[])
    language_pages.add_argument("--publication-pages", type=int, default=1, help="expand each publication URL across this many ?page=N archive pages")
    language_pages.add_argument("--epaper-url", action="append", default=[])
    language_pages.add_argument("--publication-html", action="append", default=[])
    language_pages.add_argument("--epaper-html", action="append", default=[])
    language_pages.add_argument("--no-default-urls", action="store_true")
    language_pages.add_argument("--download-pdfs", action="store_true")
    language_pages.add_argument("--render-pages", action="store_true")
    language_pages.add_argument("--max-publication-items", type=int, default=0)
    language_pages.add_argument("--max-epaper-pdfs", type=int, default=0)
    language_pages.add_argument("--timeout-seconds", type=float, default=30.0)
    language_pages.set_defaults(func=cmd_audit_gorkhapatra_language_pages)

    language_page_pack = subparsers.add_parser("prepare-gorkhapatra-language-page-pack", help="turn language-page alignment hits into a pending-reference manifest")
    language_page_pack.add_argument("language_page_audit_json")
    language_page_pack.add_argument("--out", default="data/gorkhapatra-language-page-pack")
    language_page_pack.add_argument("--split", default="eval")
    language_page_pack.add_argument("--dataset-name", default="gorkhapatra-language-pages")
    language_page_pack.add_argument("--max-samples", type=int, default=0)
    language_page_pack.set_defaults(func=cmd_prepare_gorkhapatra_language_page_pack)

    language_page_review = subparsers.add_parser("prepare-gorkhapatra-language-page-review", help="write a manual review sheet for language-page candidates")
    language_page_review.add_argument("language_page_pack_manifest")
    language_page_review.add_argument("--out", default="data/gorkhapatra-language-page-review")
    language_page_review.set_defaults(func=cmd_prepare_gorkhapatra_language_page_review)

    language_page_review_audit = subparsers.add_parser(
        "audit-gorkhapatra-language-page-review",
        help="audit Gorkhapatra language-page manual decisions before reference template or finalization use",
    )
    language_page_review_audit.add_argument("language_page_pack_manifest")
    language_page_review_audit.add_argument("review_csv")
    language_page_review_audit.add_argument("--out", default="outputs/gorkhapatra-language-page-review-audit")
    language_page_review_audit.add_argument("--require-verified-references", action="store_true")
    language_page_review_audit.add_argument("--candidate-language", action="append", default=[], help="limit audit to accepted/reviewed rows for this candidate language; repeatable")
    language_page_review_audit.set_defaults(func=cmd_audit_gorkhapatra_language_page_review)

    language_page_templates = subparsers.add_parser(
        "prepare-gorkhapatra-language-page-reference-templates",
        help="write structured reference JSON drafts for Gorkhapatra language-page review rows",
    )
    language_page_templates.add_argument("language_page_pack_manifest")
    language_page_templates.add_argument("--out", default="data/gorkhapatra-language-page-reference-templates")
    language_page_templates.add_argument("--review-csv")
    language_page_templates.add_argument("--accepted-only", action="store_true")
    language_page_templates.set_defaults(func=cmd_prepare_gorkhapatra_language_page_reference_templates)

    language_page_reviewer_bundle = subparsers.add_parser(
        "prepare-gorkhapatra-language-page-reviewer-bundle",
        help="bundle accepted language-page images and draft references for manual labeling",
    )
    language_page_reviewer_bundle.add_argument("language_page_pack_manifest")
    language_page_reviewer_bundle.add_argument("review_csv")
    language_page_reviewer_bundle.add_argument("--out", default="data/gorkhapatra-language-page-reviewer-bundle")
    language_page_reviewer_bundle.add_argument("--reference-template-dir")
    language_page_reviewer_bundle.set_defaults(func=cmd_prepare_gorkhapatra_language_page_reviewer_bundle)

    language_page_work_order = subparsers.add_parser(
        "prepare-gorkhapatra-language-page-transcription-work-order",
        help="write a transcription work order for accepted Gorkhapatra language-page references",
    )
    language_page_work_order.add_argument("language_page_pack_manifest")
    language_page_work_order.add_argument("review_csv")
    language_page_work_order.add_argument("--out", default="data/gorkhapatra-language-page-transcription-work-order")
    language_page_work_order.add_argument("--candidate-language", action="append", default=[], help="include only accepted rows for this candidate language; repeatable")
    language_page_work_order.set_defaults(func=cmd_prepare_gorkhapatra_language_page_transcription_work_order)

    language_page_pdf_text = subparsers.add_parser(
        "extract-gorkhapatra-language-page-pdf-text",
        help="extract PDF-native text spans, font metadata, and embedded fonts from Gorkhapatra language-page candidates",
    )
    language_page_pdf_text.add_argument("language_page_pack_manifest")
    language_page_pdf_text.add_argument("--out", default="data/gorkhapatra-language-page-pdf-text")
    language_page_pdf_text.add_argument("--candidate-language", action="append", default=[], help="include only manifest rows for this candidate language; repeatable")
    language_page_pdf_text.add_argument("--target-font", action="append", default=[], help="extra target font substring to mark as language/script spans; repeatable")
    language_page_pdf_text.set_defaults(func=cmd_extract_gorkhapatra_language_page_pdf_text)

    limbu_legacy = subparsers.add_parser("convert-limbu-legacy", help="convert Namdhinggo/Sirijonga legacy text to Unicode Limbu")
    limbu_legacy.add_argument("--text", help="legacy text string to convert")
    limbu_legacy.add_argument("--pdf-text-json", help="Gorkhapatra PDF text sample JSON to convert target-font spans")
    limbu_legacy.add_argument("--out", default="data/limbu-legacy-converted")
    limbu_legacy.add_argument("--map", default="data/mappings/limbu-legacy/Limbu.map", help="SIL Limbu legacy TECkit map file")
    limbu_legacy.set_defaults(func=cmd_convert_limbu_legacy)

    language_page_assisted_refs = subparsers.add_parser(
        "prepare-gorkhapatra-language-page-assisted-references",
        help="write non-claim-ready machine-assisted reference drafts for accepted Gorkhapatra language pages",
    )
    language_page_assisted_refs.add_argument("language_page_pack_manifest")
    language_page_assisted_refs.add_argument("review_csv")
    language_page_assisted_refs.add_argument("--out", default="data/gorkhapatra-language-page-assisted-references")
    language_page_assisted_refs.add_argument("--candidate-language", action="append", default=[], help="include only accepted rows for this candidate language; repeatable")
    language_page_assisted_refs.add_argument("--engine", default="sidecar", choices=["sidecar", "tesseract", "auto"], help="OCR engine used only for draft suggestions")
    language_page_assisted_refs.add_argument("--tesseract-language", default="nep+eng", help="Tesseract language code when --engine tesseract")
    language_page_assisted_refs.add_argument("--overwrite", action="store_true", help="replace existing assisted draft reference files")
    language_page_assisted_refs.set_defaults(func=cmd_prepare_gorkhapatra_language_page_assisted_references)

    language_page_pdf_native_refs = subparsers.add_parser(
        "prepare-gorkhapatra-language-page-pdf-native-references",
        help="draft references from PDF-native legacy->Unicode conversion (pre-filled text for human review, not OCR)",
    )
    language_page_pdf_native_refs.add_argument("language_page_pack_manifest")
    language_page_pdf_native_refs.add_argument("review_csv")
    language_page_pdf_native_refs.add_argument("pdf_text_dir", help="extract-gorkhapatra-language-page-pdf-text output dir (contains samples/<id>.pdf-text.json)")
    language_page_pdf_native_refs.add_argument("--out", default="data/gorkhapatra-language-page-pdf-native-references")
    language_page_pdf_native_refs.add_argument("--candidate-language", action="append", default=[], help="include only accepted rows for this candidate language; repeatable")
    language_page_pdf_native_refs.add_argument("--converter", default="limbu-namdhinggo", help="legacy converter id (e.g. limbu-namdhinggo)")
    language_page_pdf_native_refs.add_argument("--map", dest="map_path", default=None, help="override the legacy TECkit map path for the converter")
    language_page_pdf_native_refs.add_argument("--overwrite", action="store_true", help="replace existing pdf-native draft reference files")
    language_page_pdf_native_refs.set_defaults(func=cmd_prepare_gorkhapatra_language_page_pdf_native_references)

    language_page_recognizer_eval = subparsers.add_parser(
        "prepare-gorkhapatra-language-page-recognizer-eval",
        help="turn verified Gorkhapatra references into a line-level PaddleOCR recognizer eval manifest",
    )
    language_page_recognizer_eval.add_argument("language_page_pack_manifest")
    language_page_recognizer_eval.add_argument("references_dir", help="directory of verified <sample_id>.ref.json files")
    language_page_recognizer_eval.add_argument("--out", default="data/gorkhapatra-language-page-recognizer-eval")
    language_page_recognizer_eval.add_argument("--candidate-language", action="append", default=[], help="include only this candidate language; repeatable")
    language_page_recognizer_eval.add_argument("--split", default="eval", help="split label for emitted manifest rows")
    language_page_recognizer_eval.add_argument("--dataset-name", default="gorkhapatra-language-pages-real")
    language_page_recognizer_eval.add_argument("--min-text-length", type=int, default=1)
    language_page_recognizer_eval.add_argument("--allow-draft", action="store_true", help="smoke-test against non-claim-eligible draft references (rows marked non-claim-eligible)")
    language_page_recognizer_eval.set_defaults(func=cmd_prepare_gorkhapatra_language_page_recognizer_eval)

    language_page_ocr_sidecars = subparsers.add_parser(
        "prepare-gorkhapatra-language-page-ocr-sidecars",
        help="run OCR on accepted Gorkhapatra language pages and write sidecar JSON for draft references",
    )
    language_page_ocr_sidecars.add_argument("language_page_pack_manifest")
    language_page_ocr_sidecars.add_argument("review_csv")
    language_page_ocr_sidecars.add_argument("--out", default="data/gorkhapatra-language-page-ocr-sidecars")
    language_page_ocr_sidecars.add_argument("--candidate-language", action="append", default=[], help="include only accepted rows for this candidate language; repeatable")
    language_page_ocr_sidecars.add_argument("--engine", default="paddleocr", choices=["sidecar", "tesseract", "paddleocr", "auto"], help="OCR engine used to create sidecars")
    language_page_ocr_sidecars.add_argument("--tesseract-language", default="nep+eng", help="Tesseract language code when --engine tesseract")
    language_page_ocr_sidecars.add_argument("--overwrite", action="store_true", help="replace existing OCR sidecar files")
    language_page_ocr_sidecars.add_argument("--in-place", action="store_true", help="write sidecars beside each source image so SidecarEngine can discover them")
    language_page_ocr_sidecars.set_defaults(func=cmd_prepare_gorkhapatra_language_page_ocr_sidecars)

    language_page_verification_bundle = subparsers.add_parser(
        "prepare-gorkhapatra-language-page-verification-bundle",
        help="create line-crop review sheets from assisted Gorkhapatra reference drafts",
    )
    language_page_verification_bundle.add_argument("language_page_pack_manifest")
    language_page_verification_bundle.add_argument("review_csv")
    language_page_verification_bundle.add_argument("assisted_references_dir")
    language_page_verification_bundle.add_argument("--out", default="data/gorkhapatra-language-page-verification-bundle")
    language_page_verification_bundle.add_argument("--candidate-language", action="append", default=[], help="include only accepted rows for this candidate language; repeatable")
    language_page_verification_bundle.add_argument("--overwrite", action="store_true", help="replace existing crop images")
    language_page_verification_bundle.set_defaults(func=cmd_prepare_gorkhapatra_language_page_verification_bundle)

    language_page_verification_audit = subparsers.add_parser(
        "audit-gorkhapatra-language-page-verification-csv",
        help="dry-run audit a completed line-level verification CSV before applying it",
    )
    language_page_verification_audit.add_argument("verification_csv")
    language_page_verification_audit.add_argument("--out", default="outputs/gorkhapatra-language-page-verification-csv-audit")
    language_page_verification_audit.add_argument("--candidate-language", action="append", default=[], help="include only rows for this candidate language; repeatable")
    language_page_verification_audit.add_argument("--require-codepoint-range", action="append", default=[], help="require kept text to include at least one codepoint from this range, e.g. U+1900..U+194F; repeatable")
    language_page_verification_audit.set_defaults(func=cmd_audit_gorkhapatra_language_page_verification_csv)

    language_page_verification_split = subparsers.add_parser(
        "split-gorkhapatra-language-page-verification-csv",
        help="split a line-level verification CSV into reviewer batches with per-batch HTML contact sheets",
    )
    language_page_verification_split.add_argument("verification_csv")
    language_page_verification_split.add_argument("--out", default="data/gorkhapatra-language-page-verification-batches")
    language_page_verification_split.add_argument("--batch-size", type=int, default=50)
    language_page_verification_split.add_argument("--candidate-language", action="append", default=[], help="include only rows for this candidate language; repeatable")
    language_page_verification_split.set_defaults(func=cmd_split_gorkhapatra_language_page_verification_csv)

    language_page_verification_merge = subparsers.add_parser(
        "merge-gorkhapatra-language-page-verification-batches",
        help="merge reviewed verification batch CSVs and validate coverage against the source CSV",
    )
    language_page_verification_merge.add_argument("source_verification_csv")
    language_page_verification_merge.add_argument("batches_dir")
    language_page_verification_merge.add_argument("--out", default="data/gorkhapatra-language-page-verification-merged")
    language_page_verification_merge.set_defaults(func=cmd_merge_gorkhapatra_language_page_verification_batches)

    language_page_verification_assign = subparsers.add_parser(
        "assign-gorkhapatra-language-page-verification-batches",
        help="create a reviewer assignment ledger from a verification batch split index",
    )
    language_page_verification_assign.add_argument("split_index_csv")
    language_page_verification_assign.add_argument("--out", default="data/gorkhapatra-language-page-verification-assignments")
    language_page_verification_assign.add_argument("--reviewer", action="append", default=[], help="reviewer id/name; repeatable, assigned round-robin")
    language_page_verification_assign.add_argument("--due-date", help="optional due date recorded in the assignment ledger")
    language_page_verification_assign.set_defaults(func=cmd_assign_gorkhapatra_language_page_verification_batches)

    language_page_verification_apply = subparsers.add_parser(
        "apply-gorkhapatra-language-page-verification-bundle",
        help="convert completed line-level review CSVs into verified reference JSON and an updated review CSV",
    )
    language_page_verification_apply.add_argument("language_page_pack_manifest")
    language_page_verification_apply.add_argument("review_csv")
    language_page_verification_apply.add_argument("verification_csv")
    language_page_verification_apply.add_argument("assisted_references_dir")
    language_page_verification_apply.add_argument("--out", default="data/gorkhapatra-language-page-verified-references")
    language_page_verification_apply.add_argument("--candidate-language", action="append", default=[], help="include only accepted rows for this candidate language; repeatable")
    language_page_verification_apply.add_argument("--reviewer", help="human reviewer identifier to write into verified reference metadata")
    language_page_verification_apply.add_argument("--reviewed-at", help="review completion timestamp/date to write into verified reference metadata")
    language_page_verification_apply.add_argument("--tables-reviewed", action="store_true", help="declare that table presence/absence and table content were checked by a human")
    language_page_verification_apply.add_argument("--figures-reviewed", action="store_true", help="declare that figure presence/absence and figure metadata were checked by a human")
    language_page_verification_apply.add_argument("--captions-reviewed", action="store_true", help="declare that captions were checked by a human")
    language_page_verification_apply.add_argument("--require-codepoint-range", action="append", default=[], help="require kept text to include at least one codepoint from this range, e.g. U+1900..U+194F; repeatable")
    language_page_verification_apply.add_argument("--overwrite", action="store_true", help="replace existing verified reference JSON files")
    language_page_verification_apply.set_defaults(func=cmd_apply_gorkhapatra_language_page_verification_bundle)

    language_page_finalize = subparsers.add_parser(
        "finalize-gorkhapatra-language-page-review",
        help="promote accepted reviewed language-page candidates with verified references into claim-ready eval",
    )
    language_page_finalize.add_argument("language_page_pack_manifest")
    language_page_finalize.add_argument("review_csv")
    language_page_finalize.add_argument("--out", default="data/gorkhapatra-language-page-verified")
    language_page_finalize.add_argument("--split", default="eval")
    language_page_finalize.add_argument("--dataset-name", default="gorkhapatra-language-pages-verified")
    language_page_finalize.add_argument("--candidate-language", action="append", default=[], help="finalize only accepted rows for this candidate language; repeatable")
    language_page_finalize.set_defaults(func=cmd_finalize_gorkhapatra_language_page_review)

    language_registry = subparsers.add_parser("audit-language-registry", help="audit a multilingual OCR language registry")
    language_registry.add_argument("registry")
    language_registry.add_argument("--out", default="outputs/language-registry-audit")
    language_registry.add_argument("--min-nepal-languages", type=int, default=10)
    language_registry.add_argument("--allow-non-limbu-first", action="store_true")
    language_registry.add_argument("--no-verify-local-paths", action="store_true")
    language_registry.set_defaults(func=cmd_audit_language_registry)

    language_readiness = subparsers.add_parser(
        "audit-language-readiness",
        help="audit target-language readiness across registry, synthesis, and validation evidence",
    )
    language_readiness.add_argument("registry")
    language_readiness.add_argument("--out", default="outputs/language-readiness-audit")
    language_readiness.add_argument(
        "--synthesis-text-audit",
        action="append",
        default=[],
        help="passed synthesis-text-manifest-audit.json evidence; repeatable",
    )
    language_readiness.add_argument("--min-target-languages", type=int, default=10)
    language_readiness.add_argument("--min-synthesis-ready-languages", type=int, default=1)
    language_readiness.add_argument("--no-require-limbu-synthesis", action="store_true")
    language_readiness.set_defaults(func=cmd_audit_language_readiness)

    font_audit = subparsers.add_parser("audit-font-renderability", help="audit registry font coverage and optional rendered samples")
    font_audit.add_argument("registry")
    font_audit.add_argument("--out", default="outputs/font-renderability-audit")
    font_audit.add_argument("--all-scripts", action="store_true", help="audit secondary/future scripts in addition to primary scripts")
    font_audit.add_argument("--min-coverage-ratio", type=float, default=0.95)
    font_audit.add_argument("--render-samples", action="store_true", help="write sample PNGs when Pillow is installed")
    font_audit.set_defaults(func=cmd_audit_font_renderability)

    font_inventory = subparsers.add_parser(
        "inventory-fonts",
        help="scan font directories and ZIP archives, hashing files and matching cmap coverage to registry scripts",
    )
    font_inventory.add_argument("registry")
    font_inventory.add_argument("--root", action="append", required=True, help="font directory, font file, or ZIP archive to scan; repeatable")
    font_inventory.add_argument("--out", default="outputs/font-inventory")
    font_inventory.add_argument("--min-script-coverage-ratio", type=float, default=0.95)
    font_inventory.set_defaults(func=cmd_inventory_fonts)

    font_assets = subparsers.add_parser(
        "prepare-font-assets",
        help="download/reuse font assets from a manifest and verify expected SHA-256 values",
    )
    font_assets.add_argument("manifest")
    font_assets.add_argument("--asset-root", default=".")
    font_assets.add_argument("--out", default="outputs/font-assets-preparation")
    font_assets.add_argument("--force", action="store_true", help="redownload files whose existing SHA-256 does not match")
    font_assets.add_argument("--timeout-seconds", type=float, default=60)
    font_assets.set_defaults(func=cmd_prepare_font_assets)

    font_asset_readiness = subparsers.add_parser(
        "audit-font-asset-readiness",
        help="gate font assets against preparation, inventory, and renderability evidence",
    )
    font_asset_readiness.add_argument("manifest")
    font_asset_readiness.add_argument("preparation_report")
    font_asset_readiness.add_argument("inventory_report")
    font_asset_readiness.add_argument("renderability_report")
    font_asset_readiness.add_argument("--asset-root", default=".")
    font_asset_readiness.add_argument("--out", default="outputs/font-asset-readiness")
    font_asset_readiness.set_defaults(func=cmd_audit_font_asset_readiness)

    dictionary_audit = subparsers.add_parser(
        "audit-paddle-dictionary",
        help="audit a PaddleOCR character dictionary against Unicode ranges and label files",
    )
    dictionary_audit.add_argument("dictionary")
    dictionary_audit.add_argument("--out", default="outputs/paddle-dictionary-audit")
    dictionary_audit.add_argument("--required-range", action="append", default=[], help="Unicode range such as U+1900..U+194F; repeatable")
    dictionary_audit.add_argument("--label-file", action="append", default=[], help="PaddleOCR label file to check for character coverage; repeatable")
    dictionary_audit.add_argument("--no-allow-space-char", action="store_true", help="require a literal space row instead of relying on use_space_char=true")
    dictionary_audit.set_defaults(func=cmd_audit_paddle_dictionary)

    synthesis_resources = subparsers.add_parser(
        "audit-synthesis-resources",
        help="audit real-text resource directories before synthetic OCR generation",
    )
    synthesis_resources.add_argument("--root", action="append", required=True, help="resource directory to audit; repeatable")
    synthesis_resources.add_argument(
        "--label",
        action="append",
        default=[],
        help="label for each --root; repeatable and must match root count when used",
    )
    synthesis_resources.add_argument("--out", default="outputs/synthesis-resource-audit")
    synthesis_resources.add_argument("--max-files-per-root", type=int, default=0)
    synthesis_resources.add_argument("--max-text-samples-per-root", type=int, default=20)
    synthesis_resources.add_argument("--sample-bytes", type=int, default=64000)
    synthesis_resources.set_defaults(func=cmd_audit_synthesis_resources)

    limbu_limdic = subparsers.add_parser("prepare-limbu-limdic-text", help="convert verified LTK Limdic JSON/TSV sources into synthesis text JSONL")
    limbu_limdic.add_argument("--source", action="append", required=True, help="Limdic JSON/TSV source; repeatable")
    limbu_limdic.add_argument("--out", default="data/synthesis-text/limbu-limdic")
    limbu_limdic.add_argument("--split", default="train")
    limbu_limdic.add_argument("--dataset", default="limbu-limdic")
    limbu_limdic.add_argument("--min-text-chars", type=int, default=1)
    limbu_limdic.set_defaults(func=cmd_prepare_limbu_limdic_text)

    tamang_text = subparsers.add_parser("prepare-tamang-text", help="convert verified LTK Tamang CSV/TSV/JSON sources into synthesis text JSONL")
    tamang_text.add_argument("--source", action="append", required=True, help="Tamang NepTam CSV or dictionary TSV/JSON source; repeatable")
    tamang_text.add_argument("--out", default="data/synthesis-text/tamang")
    tamang_text.add_argument("--split", default="train")
    tamang_text.add_argument("--dataset", default="tamang-real-text")
    tamang_text.add_argument("--min-text-chars", type=int, default=1)
    tamang_text.add_argument("--limit-per-source", type=int, default=0)
    tamang_text.set_defaults(func=cmd_prepare_tamang_text)

    magar_text = subparsers.add_parser("prepare-magar-text", help="convert verified LTK Magar dictionary TSV/JSON sources into synthesis text JSONL")
    magar_text.add_argument("--source", action="append", required=True, help="Magar dictionary TSV/JSON source; repeatable")
    magar_text.add_argument("--out", default="data/synthesis-text/magar")
    magar_text.add_argument("--split", default="train")
    magar_text.add_argument("--dataset", default="magar-real-text")
    magar_text.add_argument("--min-text-chars", type=int, default=1)
    magar_text.add_argument("--limit-per-source", type=int, default=0)
    magar_text.set_defaults(func=cmd_prepare_magar_text)

    bible_brain = subparsers.add_parser("prepare-bible-brain-text", help="convert verified Bible Brain text manifests into synthesis text JSONL")
    bible_brain.add_argument("--manifest", action="append", required=True, help="Bible Brain metadata/text_manifest.jsonl path; repeatable")
    bible_brain.add_argument("--language", action="append", required=True, help="language id for each --manifest; repeatable and must match manifest count")
    bible_brain.add_argument("--out", default="data/synthesis-text/bible-brain")
    bible_brain.add_argument("--split", default="train")
    bible_brain.add_argument("--dataset", default="bible-brain-nepal")
    bible_brain.add_argument("--min-text-chars", type=int, default=1)
    bible_brain.add_argument("--limit-per-manifest", type=int, default=0)
    bible_brain.set_defaults(func=cmd_prepare_bible_brain_text)

    limbu_unicode = subparsers.add_parser("prepare-limbu-unicode-text", help="convert verified Limbu/Sirijonga Unicode text sources into synthesis text JSONL")
    limbu_unicode.add_argument("--source", action="append", required=True, help="Limbu Unicode .txt or tagged .tsv source; repeatable")
    limbu_unicode.add_argument("--out", default="data/synthesis-text/limbu-unicode")
    limbu_unicode.add_argument("--split", default="train")
    limbu_unicode.add_argument("--dataset", default="limbu-unicode-real-text")
    limbu_unicode.add_argument("--min-text-chars", type=int, default=1)
    limbu_unicode.add_argument("--min-limbu-chars", type=int, default=1)
    limbu_unicode.add_argument("--limit-per-source", type=int, default=0)
    limbu_unicode.set_defaults(func=cmd_prepare_limbu_unicode_text)

    toolkit_parallel = subparsers.add_parser(
        "prepare-toolkit-parallel-text",
        help="convert explicitly mapped toolkit JSONL/TSV/CSV text fields into synthesis text JSONL",
    )
    toolkit_parallel.add_argument("--source", action="append", required=True, help="toolkit JSONL/TSV/CSV source; repeatable")
    toolkit_parallel.add_argument(
        "--text-field",
        action="append",
        required=True,
        help="text field mapping in FIELD=language form; repeatable",
    )
    toolkit_parallel.add_argument("--out", default="data/synthesis-text/toolkit-parallel")
    toolkit_parallel.add_argument("--split", default="train")
    toolkit_parallel.add_argument("--dataset", default="nepal-toolkit-parallel-text")
    toolkit_parallel.add_argument("--min-text-chars", type=int, default=1)
    toolkit_parallel.add_argument("--limit-per-source", type=int, default=0)
    toolkit_parallel.add_argument("--row-id-field", default=None)
    toolkit_parallel.add_argument("--metadata-field", action="append", default=[])
    toolkit_parallel.add_argument("--license-status", default="pending_review")
    toolkit_parallel.add_argument("--split-policy", default=None)
    toolkit_parallel.set_defaults(func=cmd_prepare_toolkit_parallel_text)

    audit_synthesis_text = subparsers.add_parser(
        "audit-synthesis-text-manifest",
        help="audit synthesis-text JSONL manifests before rendering or training",
    )
    audit_synthesis_text.add_argument("manifest")
    audit_synthesis_text.add_argument("--out", default="outputs/synthesis-text-manifest-audit")
    audit_synthesis_text.add_argument("--require-language", action="append", default=[], help="language id that must appear; repeatable")
    audit_synthesis_text.add_argument("--require-script", action="append", default=[], help="script label that must appear; repeatable")
    audit_synthesis_text.add_argument("--min-samples", type=int, default=1)
    audit_synthesis_text.add_argument(
        "--allow-claim-evidence",
        action="store_true",
        help="allow rows marked claim_evidence_eligible; default rejects them",
    )
    audit_synthesis_text.set_defaults(func=cmd_audit_synthesis_text_manifest)

    audit_synthesis_promotion = subparsers.add_parser(
        "audit-synthesis-text-promotion",
        help="audit synthesis-text JSONL before rendering/training promotion",
    )
    audit_synthesis_promotion.add_argument("manifest")
    audit_synthesis_promotion.add_argument("--out", default="outputs/synthesis-text-promotion-audit")
    audit_synthesis_promotion.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="manifest, reference file, text file, or directory whose text hashes must not overlap; repeatable",
    )
    audit_synthesis_promotion.add_argument(
        "--require-reviewed-license",
        action="store_true",
        help="fail rows whose metadata.license_status is missing or pending_review",
    )
    audit_synthesis_promotion.set_defaults(func=cmd_audit_synthesis_text_promotion)

    split_synthesis = subparsers.add_parser(
        "split-synthesis-text",
        help="split synthesis-text JSONL into deterministic train/eval files with promotion audits",
    )
    split_synthesis.add_argument("manifest")
    split_synthesis.add_argument("--out", default="data/synthesis-text/splits")
    split_synthesis.add_argument("--eval-ratio", type=float, default=0.15)
    split_synthesis.add_argument("--seed", type=int, default=13)
    split_synthesis.add_argument(
        "--group-by",
        choices=["text_sha256", "source_path", "source_row_id", "sample_id"],
        default="text_sha256",
    )
    split_synthesis.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="manifest, reference file, text file, or directory whose text hashes must not overlap; repeatable",
    )
    split_synthesis.add_argument("--require-reviewed-license", action="store_true")
    split_synthesis.set_defaults(func=cmd_split_synthesis_text)

    render_lines = subparsers.add_parser("render-synthesis-text-lines", help="render synthesis-text JSONL rows into OCR line images")
    render_lines.add_argument("synthesis_text_manifest")
    render_lines.add_argument("--out", default="data/rendered-lines")
    render_lines.add_argument("--font-path")
    render_lines.add_argument("--limit", type=int, default=0)
    render_lines.add_argument("--script", action="append", default=[], help="script label to include; repeatable")
    render_lines.add_argument("--font-size", type=int, default=36)
    render_lines.add_argument("--padding", type=int, default=16)
    render_lines.add_argument(
        "--degradation-profile",
        action="append",
        default=[],
        help="rendered-image degradation profile; repeatable: clean, scan, low_light, uneven_light, glare_light, camera_left, camera_right, camera_top, camera_bottom, phone_photo",
    )
    render_lines.add_argument("--degradation-seed", type=int, default=13)
    render_lines.add_argument("--split", default="train")
    render_lines.add_argument("--dataset", default="synthesis-rendered-lines")
    render_lines.set_defaults(func=cmd_render_synthesis_text_lines)

    render_split = subparsers.add_parser(
        "render-synthesis-text-split",
        help="render train/eval synthesis-text JSONL splits into audited OCR line bundles",
    )
    render_split.add_argument("--train-manifest", required=True)
    render_split.add_argument("--eval-manifest", required=True)
    render_split.add_argument("--out", default="data/rendered-line-split")
    render_split.add_argument("--font-path")
    render_split.add_argument("--limit-per-split", type=int, default=0)
    render_split.add_argument("--script", action="append", default=[], help="script label to include; repeatable")
    render_split.add_argument("--font-size", type=int, default=36)
    render_split.add_argument("--padding", type=int, default=16)
    render_split.add_argument(
        "--degradation-profile",
        action="append",
        default=[],
        help="rendered-image degradation profile; repeatable: clean, scan, low_light, uneven_light, glare_light, camera_left, camera_right, camera_top, camera_bottom, phone_photo",
    )
    render_split.add_argument("--degradation-seed", type=int, default=13)
    render_split.add_argument("--dataset", default="synthesis-rendered-lines")
    render_split.add_argument("--require-font", action="store_true", help="require font provenance in rendered audits")
    render_split.add_argument("--font-readiness-report", help="passed font-asset-readiness.json proving font asset provenance")
    render_split.set_defaults(func=cmd_render_synthesis_text_split)

    audit_rendered_lines = subparsers.add_parser(
        "audit-rendered-synthesis-lines",
        help="audit rendered synthetic OCR line manifests for provenance and claim guardrails",
    )
    audit_rendered_lines.add_argument("manifest")
    audit_rendered_lines.add_argument("--out", default="outputs/rendered-line-manifest-audit")
    audit_rendered_lines.add_argument("--label-file", help="PaddleOCR label file to cross-check against the manifest")
    audit_rendered_lines.add_argument("--require-font", action="store_true", help="require font path and font SHA provenance")
    audit_rendered_lines.add_argument("--font-readiness-report", help="passed font-asset-readiness.json proving font asset provenance")
    audit_rendered_lines.set_defaults(func=cmd_audit_rendered_synthesis_lines)

    audit_degradation_split = subparsers.add_parser(
        "audit-rendered-degradation-split",
        help="audit augmented rendered train/eval manifests for degradation balance and text-hash leakage",
    )
    audit_degradation_split.add_argument("--train-manifest", required=True)
    audit_degradation_split.add_argument("--eval-manifest", required=True)
    audit_degradation_split.add_argument("--out", default="outputs/rendered-degradation-split-audit")
    audit_degradation_split.add_argument(
        "--expected-profile",
        action="append",
        default=[],
        help="expected degradation profile for every text hash; repeatable",
    )
    audit_degradation_split.set_defaults(func=cmd_audit_rendered_degradation_split)

    remote = subparsers.add_parser("audit-remote-host", help="audit a remote host for training and benchmark readiness")
    remote.add_argument("--host", required=True)
    remote.add_argument("--user")
    remote.add_argument("--port", type=int, default=22)
    remote.add_argument("--password-env")
    remote.add_argument("--min-python", default="3.11")
    remote.add_argument("--min-free-gb", type=float, default=20.0)
    remote.add_argument("--workdir")
    remote.add_argument("--require-gpu", action="store_true")
    remote.add_argument("--require-paddle-training", action="store_true", help="fail if the host is not a valid official Paddle training target")
    remote.add_argument("--python-candidate", action="append", default=[])
    remote.add_argument("--require-command", action="append", default=[])
    remote.add_argument("--optional-command", action="append", default=[])
    remote.add_argument("--require-path", action="append", default=[])
    remote.add_argument("--out", default="outputs/remote-audit")
    remote.set_defaults(func=cmd_audit_remote_host)

    bootstrap = subparsers.add_parser("bootstrap-remote-host", help="provision remote workdir and Python virtualenv for training")
    bootstrap.add_argument("--host", required=True)
    bootstrap.add_argument("--user")
    bootstrap.add_argument("--port", type=int, default=22)
    bootstrap.add_argument("--password-env")
    bootstrap.add_argument("--workdir", required=True)
    bootstrap.add_argument("--min-python", default="3.11")
    bootstrap.add_argument("--python-candidate", action="append", default=[])
    bootstrap.add_argument("--venv-name", default=".venv")
    bootstrap.add_argument("--recreate-venv", action="store_true")
    bootstrap.add_argument("--out", default="outputs/remote-bootstrap")
    bootstrap.set_defaults(func=cmd_bootstrap_remote_host)

    sync_remote = subparsers.add_parser("sync-remote-workspace", help="copy the current workspace to a remote host workdir")
    sync_remote.add_argument("--host", required=True)
    sync_remote.add_argument("--user")
    sync_remote.add_argument("--port", type=int, default=22)
    sync_remote.add_argument("--password-env")
    sync_remote.add_argument("--local-root", default=".")
    sync_remote.add_argument("--remote-workdir", required=True)
    sync_remote.add_argument("--exclude", action="append", default=[])
    sync_remote.add_argument("--out", default="outputs/remote-sync")
    sync_remote.set_defaults(func=cmd_sync_remote_workspace)

    review = subparsers.add_parser("review-claim", help="audit a full SOTA claim bundle across multiple experiment reports")
    review.add_argument("--experiment", action="append", required=True, help="experiment.json path; repeatable")
    review.add_argument("--config")
    review.add_argument("--model-config")
    review.add_argument("--out", default="outputs/claim-review")
    review.set_defaults(func=cmd_review_claim)

    ablation_audit = subparsers.add_parser("audit-augmentation-ablation", help="audit a synthetic augmentation ablation plan before model promotion")
    ablation_audit.add_argument("config", help="augmentation ablation config JSON")
    ablation_audit.add_argument("--real-review-audit", action="append", default=[], help="passed Gorkhapatra language-page review audit JSON; repeatable")
    ablation_audit.add_argument("--out", default="outputs/augmentation-ablation-audit")
    ablation_audit.set_defaults(func=cmd_audit_augmentation_ablation)

    validate = subparsers.add_parser("validate-claim", help="validate whether benchmark evidence supports a SOTA claim")
    validate.add_argument("--benchmark-report", required=True, help="benchmark report.json path")
    validate.add_argument("--out", default="outputs/validation", help="validation report directory")
    validate.add_argument("--config", help="validation gates JSON")
    validate.add_argument("--eval-manifest", help="held-out evaluation manifest")
    validate.add_argument("--train-manifest", action="append", default=[], help="training manifest to check for leakage; repeatable")
    validate.add_argument("--candidate-model-config", help="candidate model card path to record in validation provenance")
    validate.add_argument("--fallback-model-config", help="fallback model card path to record in validation provenance")
    validate.set_defaults(func=cmd_validate_claim)
    return parser


def cmd_prepare_data(args: argparse.Namespace) -> int:
    if args.hf:
        path = prepare_hf_dataset(
            args.source,
            args.out,
            dataset=args.dataset_name,
            split=args.split,
            image_field=args.image_field,
            text_field=args.text_field,
            hf_subset=args.hf_subset,
            limit=args.limit,
            skip_invalid=args.skip_invalid,
            strict_chars=args.strict_chars,
            slices=args.slice,
        )
    else:
        path = prepare_local_dataset(
            args.source,
            args.out,
            dataset=args.dataset_name,
            split=args.split,
            image_field=args.image_field,
            text_field=args.text_field,
            limit=args.limit,
            skip_invalid=args.skip_invalid,
            strict_chars=args.strict_chars,
            slices=args.slice,
        )
    print(path.manifest_path)
    print(path.summary_json_path)
    print(path.summary_md_path)
    if path.rejects_path:
        print(path.rejects_path)
    print(f"prepared={path.sample_count} rejected={path.rejected_count}")
    return 0


def cmd_inspect_hf_dataset(args: argparse.Namespace) -> int:
    report = inspect_hf_dataset(
        args.dataset_id,
        args.out,
        split=args.split,
        hf_subset=args.hf_subset,
        image_field=args.image_field,
        text_field=args.text_field,
        limit=args.limit,
    )
    print(report.output_json_path)
    print(report.output_md_path)
    print(f"passed={report.passed} inspected={report.inspected_count}")
    return 0 if report.passed else 1


def cmd_audit_manifest(args: argparse.Namespace) -> int:
    audit = audit_manifest(
        args.manifest,
        args.out,
        verify_hashes=not args.no_verify_hashes,
        require_slices=args.require_slice,
        strict_chars=args.strict_chars,
    )
    print(Path(args.out) / "manifest-audit.json")
    print(Path(args.out) / "manifest-audit.md")
    print(f"passed={audit.passed} samples={audit.sample_count}")
    return 0 if audit.passed else 1


def cmd_split_manifest(args: argparse.Namespace) -> int:
    summary = split_manifest(
        args.manifest,
        args.out,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        stratify_by=args.stratify_by,
        group_by=args.group_by,
    )
    print(summary.train_manifest)
    print(summary.eval_manifest)
    print(f"train={summary.train_count} eval={summary.eval_count} leakage_passed={summary.leakage_passed}")
    return 0


def cmd_rebalance_manifest(args: argparse.Namespace) -> int:
    summary = rebalance_manifest(args.manifest, args.out, slices=args.slice, target_count=args.target_count, seed=args.seed)
    out_path = Path(args.out)
    print(summary.output_manifest)
    print(out_path.with_name(f"{out_path.stem}-rebalance.json"))
    print(out_path.with_name(f"{out_path.stem}-rebalance.md"))
    print(f"original={summary.original_count} rebalanced={summary.rebalanced_count} target={summary.target_count}")
    return 0


def cmd_audit_recognizer_corpus(args: argparse.Namespace) -> int:
    audit = audit_recognizer_corpus(
        args.train_manifest,
        args.eval_manifest,
        args.out,
        min_train_samples=args.min_train_samples,
        min_eval_samples=args.min_eval_samples,
        min_train_english=args.min_train_english,
        min_train_devanagari=args.min_train_devanagari,
        min_train_mixed=args.min_train_mixed,
        min_eval_english=args.min_eval_english,
        min_eval_devanagari=args.min_eval_devanagari,
        min_eval_mixed=args.min_eval_mixed,
        min_train_latin_only=args.min_train_latin_only,
        min_train_devanagari_only=args.min_train_devanagari_only,
        min_eval_latin_only=args.min_eval_latin_only,
        min_eval_devanagari_only=args.min_eval_devanagari_only,
        min_train_real=args.min_train_real,
        min_train_synthetic=args.min_train_synthetic,
        require_eval_real=args.require_eval_real,
    )
    print(Path(args.out) / "recognizer-corpus-audit.json")
    print(Path(args.out) / "recognizer-corpus-audit.md")
    print(f"passed={audit.passed} train={audit.train_profile.sample_count} eval={audit.eval_profile.sample_count}")
    return 0 if audit.passed else 1


def cmd_build_recognizer_corpus(args: argparse.Namespace) -> int:
    summary = build_recognizer_corpus(
        args.manifests,
        args.out,
        target_latin_only=args.target_latin_only,
        target_devanagari_only=args.target_devanagari_only,
        target_mixed=args.target_mixed,
        target_real=args.target_real,
        target_synthetic=args.target_synthetic,
        seed=args.seed,
    )
    out_path = Path(args.out)
    print(summary.output_manifest)
    print(out_path.with_name(f"{out_path.stem}-recognizer-corpus-build.json"))
    print(out_path.with_name(f"{out_path.stem}-recognizer-corpus-build.md"))
    print(f"source={summary.source_count} output={summary.output_count}")
    return 0


def cmd_merge_manifests(args: argparse.Namespace) -> int:
    summary = merge_manifests(args.manifests, args.out)
    out_path = Path(args.out)
    print(summary.output_manifest)
    print(out_path.with_name(f"{out_path.stem}-merge.json"))
    print(out_path.with_name(f"{out_path.stem}-merge.md"))
    print(f"samples={summary.sample_count} inputs={len(summary.input_manifests)} warnings={len(summary.warnings)}")
    return 0


def cmd_filter_manifest(args: argparse.Namespace) -> int:
    summary = filter_manifest(
        args.manifest,
        args.out,
        sample_ids=args.sample_id,
        slices=args.slice,
        document_types=args.document_type,
        degradations=args.degradation,
        limit=args.limit,
    )
    out_path = Path(args.out)
    print(summary.output_manifest)
    print(out_path.with_name(f"{out_path.stem}-filter.json"))
    print(out_path.with_name(f"{out_path.stem}-filter.md"))
    print(f"selected={summary.selected_count} source={summary.source_count}")
    return 0


def cmd_materialize_references(args: argparse.Namespace) -> int:
    report = materialize_references(args.manifest, args.out, rewritten_manifest_path=args.rewritten_manifest)
    print(report.references_dir)
    if report.rewritten_manifest:
        print(report.rewritten_manifest)
    print(report.summary_json_path)
    print(report.summary_md_path)
    print(f"samples={report.sample_count}")
    return 0


def cmd_audit_references(args: argparse.Namespace) -> int:
    report = audit_references(
        args.manifest,
        args.out,
        require_claim_ready=args.require_claim_ready,
    )
    print(report.summary_json_path)
    print(report.summary_md_path)
    print(
        f"passed={report.passed} samples={report.sample_count} parsed={report.parsed_count} "
        f"issues={len(report.issues)} warnings={len(report.warnings)}"
    )
    return 0 if report.passed else 1


def cmd_normalize_manifest_images(args: argparse.Namespace) -> int:
    report = normalize_manifest_images(
        args.manifest,
        args.out,
        image_dir_name=args.image_dir_name,
        output_manifest_name=args.output_manifest_name,
    )
    print(report.manifest_path)
    print(report.summary_json_path)
    print(report.summary_md_path)
    print(f"samples={report.sample_count} copied={report.copied_count} converted={report.converted_count}")
    return 0


def cmd_create_eval_pack(args: argparse.Namespace) -> int:
    summary = create_eval_pack(
        args.out,
        count_per_template=args.count_per_template,
        input_format=args.input_format,
        degradations=args.degradation,
        seed=args.seed,
        font_path=args.font_path,
        templates=args.template or None,
        variant_offset=args.variant_offset,
    )
    print(summary.manifest_path)
    print(f"samples={summary.sample_count} claim_ready={summary.claim_ready}")
    return 0


def cmd_audit_eval_pack(args: argparse.Namespace) -> int:
    audit = audit_eval_pack(args.manifest, args.out)
    print(Path(args.out) / "eval-pack-audit.json")
    print(Path(args.out) / "eval-pack-audit.md")
    print(f"passed={audit.passed} claim_ready={audit.claim_ready}")
    if not audit.passed or (args.require_claim_ready and not audit.claim_ready):
        return 1
    return 0


def cmd_extract_crops(args: argparse.Namespace) -> int:
    summary = extract_crop_manifest(
        args.manifest,
        args.out,
        crop_types=args.crop_type or None,
        split=args.split,
        dataset_name=args.dataset_name,
        min_text_length=args.min_text_length,
    )
    print(summary.manifest_path)
    print(Path(args.out) / "crop-manifest-summary.json")
    print(Path(args.out) / "crop-manifest-summary.md")
    print(f"crops={summary.crop_count} skipped={summary.skipped_count}")
    return 0


def cmd_train_recognizer(args: argparse.Namespace) -> int:
    if args.backend == "hf-vision-encoder-decoder":
        if args.failure_review:
            assert_recognizer_family_not_stopped(args.failure_review, backend="hf_vision_encoder_decoder", base_model=args.base_model)
        path = write_hf_recognizer_recipe(
            args.train_manifest,
            args.eval_manifest,
            args.out,
            base_model=args.base_model,
            max_target_length=args.max_target_length,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            run=args.run,
        )
    else:
        path = write_recognizer_recipe(
            args.train_manifest,
            args.eval_manifest,
            args.out,
            base_model=args.base_model,
            dictionary_path=args.dictionary,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            warmup_epoch=args.warmup_epoch,
            use_gpu=args.use_gpu,
            train_num_workers=args.num_workers,
            eval_num_workers=args.num_workers,
            train_drop_last=args.train_drop_last,
            main_indicator=args.main_indicator,
            eval_batch_step=args.eval_batch_step,
            failure_review=args.failure_review,
            run=args.run,
        )
    print(path)
    return 0


def cmd_export_recognizer(args: argparse.Namespace) -> int:
    path = write_recognizer_export_recipe(
        args.training_config,
        args.checkpoint,
        args.out,
        inference_dir=args.inference_dir,
        paddleocr_dir=args.paddleocr_dir,
        export_options=args.export_option,
        run=args.run,
    )
    print(path)
    return 0


def cmd_package_recognizer(args: argparse.Namespace) -> int:
    path = package_paddleocr_model(
        args.inference_dir,
        args.out,
        model_id=args.model_id,
        dictionary_path=args.dictionary,
        base_model=args.base_model,
        paddle_lang=args.paddle_lang,
        text_recognition_model_name=args.text_recognition_model_name,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        training_summary=args.training_summary,
        metrics_report=args.metrics_report,
        admission_validation_report=args.admission_validation_report,
        recognition_mode=args.recognition_mode,
        line_mode_max_height=args.line_mode_max_height,
        line_mode_min_aspect_ratio=args.line_mode_min_aspect_ratio,
        source_archive=args.source_archive,
        source_checkpoint=args.source_checkpoint,
        source_training_config=args.source_training_config,
        export_recipe=args.export_recipe,
    )
    print(path)
    return 0


def cmd_bundle_recognizer_training(args: argparse.Namespace) -> int:
    bundle = bundle_recognizer_training(
        args.recipe_dir,
        args.out,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        archive_name=args.archive_name,
        base_dir=args.base_dir,
        require_corpus_audit=args.require_corpus_audit,
        min_train_samples=args.min_train_samples,
        min_eval_samples=args.min_eval_samples,
        min_train_english=args.min_train_english,
        min_train_devanagari=args.min_train_devanagari,
        min_train_mixed=args.min_train_mixed,
        min_eval_english=args.min_eval_english,
        min_eval_devanagari=args.min_eval_devanagari,
        min_eval_mixed=args.min_eval_mixed,
        min_train_latin_only=args.min_train_latin_only,
        min_train_devanagari_only=args.min_train_devanagari_only,
        min_eval_latin_only=args.min_eval_latin_only,
        min_eval_devanagari_only=args.min_eval_devanagari_only,
        min_train_real=args.min_train_real,
        min_train_synthetic=args.min_train_synthetic,
        require_eval_real=args.require_eval_real,
        failure_review=args.failure_review,
    )
    print(bundle.archive_path)
    print(bundle.manifest_path)
    print(bundle.readme_path)
    print(f"files={bundle.file_count} bytes={bundle.total_bytes}")
    return 0


def cmd_package_hf_recognizer(args: argparse.Namespace) -> int:
    path = package_hf_recognizer_model(
        args.model_dir,
        args.out,
        model_id=args.model_id,
        base_model=args.base_model,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        training_summary=args.training_summary,
        metrics_report=args.metrics_report,
        admission_validation_report=args.admission_validation_report,
        max_new_tokens=args.max_new_tokens,
    )
    print(path)
    return 0


def cmd_finalize_hf_recognizer_run(args: argparse.Namespace) -> int:
    result = finalize_hf_recognizer_run(
        args.run_dir,
        output_dir=args.out,
        model_id=args.model_id,
        base_model=args.base_model,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        training_summary=args.training_summary,
        admission_validation_report=args.admission_validation_report,
        max_new_tokens=args.max_new_tokens,
    )
    print(result.model_card_path)
    print(result.audit_path)
    print(result.report_json_path)
    print(result.report_md_path)
    print(f"audit_passed={result.audit_passed}")
    return 0 if result.audit_passed else 1


def cmd_package_hf_corrector(args: argparse.Namespace) -> int:
    path = package_hf_text_corrector_model(
        args.model_dir,
        args.out,
        model_id=args.model_id,
        base_model=args.base_model,
        base_engine=args.base_engine,
        base_engine_kwargs=_parse_key_values(args.base_engine_kwarg),
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        metrics_report=args.metrics_report,
        max_new_tokens=args.max_new_tokens,
    )
    print(path)
    return 0


def cmd_generate_correction_pairs(args: argparse.Namespace) -> int:
    path = generate_correction_pairs(
        args.manifest,
        args.out,
        engine_name=args.engine,
        model_config=args.model_config,
        limit=args.limit,
        only_errors=args.only_errors,
    )
    print(path)
    return 0


def cmd_train_corrector(args: argparse.Namespace) -> int:
    recipe_path = write_corrector_recipe(
        args.out,
        clean_manifest=args.clean_manifest,
        pairs_source=args.pairs_source,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        limit=args.limit,
        seed=args.seed,
    )
    if args.backend == "recipe":
        print(recipe_path)
        return 0
    path = write_hf_text_corrector_recipe(
        Path(args.out) / "correction_pairs.jsonl",
        args.out,
        base_model=args.base_model,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        run=args.run,
    )
    print(path)
    return 0


def cmd_create_model_card(args: argparse.Namespace) -> int:
    path = write_model_card(
        args.out,
        model_id=args.model_id,
        backend=args.backend,
        base_model=args.base_model,
        artifact_paths=args.artifact,
        backend_kwargs=_parse_key_values([*args.backend_kwarg, *args.paddleocr_kwarg]),
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        metrics_report=args.metrics_report,
        admission_validation_report=args.admission_validation_report,
        fallback_model_card=args.fallback_model_card,
        notes=args.notes,
    )
    print(path)
    return 0


def cmd_audit_model(args: argparse.Namespace) -> int:
    audit = audit_model_card(args.model_config, args.out)
    print(Path(args.out) / "model-audit.json")
    print(Path(args.out) / "model-audit.md")
    print(f"passed={audit.passed} model_id={audit.model_id}")
    return 0 if audit.passed else 1


def cmd_parse_doc(args: argparse.Namespace) -> int:
    document = parse_document(
        args.input,
        args.out,
        engine_name=args.engine,
        model_config=args.model_config,
        fallback_engine=args.fallback_engine,
        fallback_model_config=args.fallback_model_config,
        low_confidence_threshold=args.low_confidence_threshold,
        fallback_min_quality_score=args.fallback_min_quality_score,
    )
    print(Path(args.out) / "document.json")
    print(Path(args.out) / "document.md")
    print(f"pages={len(document.pages)} tables={len(document.tables)} figures={len(document.figures)}")
    return 0


def cmd_parse_limbu_doc(args: argparse.Namespace) -> int:
    document = parse_limbu_document(
        args.input,
        args.out,
        engine_name=args.engine,
        model_config=args.model_config,
        fallback_engine=args.fallback_engine,
        fallback_model_config=args.fallback_model_config,
        low_confidence_threshold=args.low_confidence_threshold,
        sirijonga_low_confidence_threshold=args.sirijonga_low_confidence_threshold,
        devanagari_low_confidence_threshold=args.devanagari_low_confidence_threshold,
        mixed_script_low_confidence_threshold=args.mixed_script_low_confidence_threshold,
        other_script_low_confidence_threshold=args.other_script_low_confidence_threshold,
        fallback_min_quality_score=args.fallback_min_quality_score,
        script_ratio_threshold=args.script_ratio_threshold,
        capture_prep_metadata=args.capture_prep_metadata,
        post_correction_profile=args.post_correction_profile,
        line_detection_mode=args.line_detection_mode,
        image_line_threshold=args.image_line_threshold,
        image_line_bbox_source=args.image_line_bbox_source,
        image_line_reading_order=args.image_line_reading_order,
        image_line_horizontal_kernel=args.image_line_horizontal_kernel,
        image_line_vertical_kernel=args.image_line_vertical_kernel,
        image_line_dilation_iterations=args.image_line_dilation_iterations,
        image_line_min_width=args.image_line_min_width,
        image_line_min_height=args.image_line_min_height,
        image_line_min_area=args.image_line_min_area,
        image_line_max_height=args.image_line_max_height,
        image_line_min_aspect_ratio=args.image_line_min_aspect_ratio,
        image_line_max_aspect_ratio=args.image_line_max_aspect_ratio,
        image_line_detector_padding=args.image_line_detector_padding,
        image_line_crop_padding=args.image_line_crop_padding,
        image_line_rescue_detector_passes=_parse_image_line_rescue_detector_passes(args.image_line_rescue_detector_pass),
        image_line_merge_iou_threshold=args.image_line_merge_iou_threshold,
        image_line_split_tall_components=args.image_line_split_tall_components,
        image_line_split_tall_row_min_ink=args.image_line_split_tall_row_min_ink,
        image_line_split_tall_max_row_gap=args.image_line_split_tall_max_row_gap,
        image_line_split_wide_components=args.image_line_split_wide_components,
        image_line_split_wide_col_min_ink=args.image_line_split_wide_col_min_ink,
        image_line_split_wide_max_col_gap=args.image_line_split_wide_max_col_gap,
        image_line_split_wide_min_width=args.image_line_split_wide_min_width,
        image_line_split_detected_row_components=args.image_line_split_detected_row_components,
        image_line_split_detected_row_col_min_ink=args.image_line_split_detected_row_col_min_ink,
        image_line_split_detected_row_max_col_gap=args.image_line_split_detected_row_max_col_gap,
        image_line_split_detected_row_min_width=args.image_line_split_detected_row_min_width,
        image_line_split_detected_row_min_segment_width=args.image_line_split_detected_row_min_segment_width,
        image_line_split_detected_tall_components=args.image_line_split_detected_tall_components,
        image_line_split_detected_tall_row_min_ink=args.image_line_split_detected_tall_row_min_ink,
        image_line_split_detected_tall_max_row_gap=args.image_line_split_detected_tall_max_row_gap,
        image_line_split_detected_tall_min_height=args.image_line_split_detected_tall_min_height,
        image_line_split_detected_tall_min_segment_height=args.image_line_split_detected_tall_min_segment_height,
        image_line_merge_same_row_components=args.image_line_merge_same_row_components,
        image_line_merge_same_row_y_tolerance=args.image_line_merge_same_row_y_tolerance,
        image_line_merge_same_row_max_gap=args.image_line_merge_same_row_max_gap,
        image_line_merge_same_row_max_center_delta=args.image_line_merge_same_row_max_center_delta,
        image_line_merge_same_row_max_width=args.image_line_merge_same_row_max_width,
        image_line_merge_same_row_auto_fragmented_top_to_bottom=args.image_line_merge_same_row_auto_fragmented_top_to_bottom,
        image_line_merge_same_row_auto_min_reduction_ratio=args.image_line_merge_same_row_auto_min_reduction_ratio,
        image_line_merge_same_row_auto_min_reduction_count=args.image_line_merge_same_row_auto_min_reduction_count,
        image_line_allow_empty_lines=args.image_line_allow_empty_lines,
        image_line_filter_drop_empty=args.image_line_filter_drop_empty,
        image_line_filter_min_confidence=args.image_line_filter_min_confidence,
        image_line_filter_require_script=args.image_line_filter_require_script,
        image_line_filter_min_width_ratio=args.image_line_filter_min_width_ratio,
        image_line_filter_max_width_ratio=args.image_line_filter_max_width_ratio,
        image_line_filter_min_height_ratio=args.image_line_filter_min_height_ratio,
        image_line_filter_max_height_ratio=args.image_line_filter_max_height_ratio,
        run_context=_cli_run_context(args),
    )
    audit = document.metadata.get("limbu_pipeline", {})
    review_count = audit.get("review_line_count") if isinstance(audit, dict) else None
    print(Path(args.out) / "document.json")
    print(Path(args.out) / "document.md")
    print(Path(args.out) / "limbu-pipeline-audit.json")
    print(Path(args.out) / "limbu-post-correction-audit.json")
    print(Path(args.out) / "limbu-review-queue.tsv")
    print(Path(args.out) / "limbu-review-dashboard.html")
    print(Path(args.out) / "limbu-output-manifest.json")
    print(f"pages={len(document.pages)} tables={len(document.tables)} figures={len(document.figures)} review_lines={review_count}")
    return 0


def cmd_prepare_limbu_capture(args: argparse.Namespace) -> int:
    result = prepare_limbu_capture(
        args.input,
        args.out,
        rotate_degrees=args.rotate_degrees,
        crop_box=parse_crop_box(args.crop_box),
        perspective_quad=parse_perspective_quad(args.perspective_quad),
        auto_detect_page=args.auto_detect_page,
        auto_deskew=args.auto_deskew,
        max_auto_deskew_degrees=args.max_auto_deskew_degrees,
        autocontrast=not args.no_autocontrast,
        grayscale=not args.no_grayscale,
        run_context=_cli_run_context(args),
    )
    print(result.prepared_image_path)
    print(result.metadata_path)
    return 0


def cmd_audit_limbu_capture(args: argparse.Namespace) -> int:
    report = audit_limbu_capture(
        args.capture_metadata,
        args.out,
        require_metadata_path_self=args.require_metadata_path_self,
        min_prepared_width=args.min_prepared_width,
        min_prepared_height=args.min_prepared_height,
        min_prepared_entropy=args.min_prepared_entropy,
        min_prepared_luminance_stddev=args.min_prepared_luminance_stddev,
        min_prepared_edge_stddev=args.min_prepared_edge_stddev,
    )
    report_dir = Path(args.out) if args.out else (Path(args.capture_metadata) if Path(args.capture_metadata).is_dir() else Path(args.capture_metadata).parent)
    print(report_dir / "limbu-capture-audit.json")
    print(report_dir / "limbu-capture-audit.md")
    print(f"passed={report['passed']}")
    return 0 if report["passed"] else 1


def cmd_preflight_limbu_pipeline(args: argparse.Namespace) -> int:
    report = preflight_limbu_pipeline_runtime(
        args.model_config,
        args.out,
        required_script=args.required_script,
        required_tesseract_languages=tuple(args.required_tesseract_language or ["nep", "eng"]),
        require_validated_component_admissions=args.require_validated_components,
        required_component_roles=tuple(args.required_component_role or ()),
    )
    print(Path(args.out) / "limbu-runtime-preflight.json")
    print(Path(args.out) / "limbu-runtime-preflight.md")
    print(f"passed={report.passed}")
    return 0 if report.passed else 1


def cmd_apply_limbu_review_corrections(args: argparse.Namespace) -> int:
    document = apply_limbu_review_corrections(
        args.document,
        args.review_queue,
        args.out,
        accepted_statuses=tuple(args.accepted_status or ("accepted", "approved", "corrected", "verified", "done")),
        run_context=_cli_run_context(args),
    )
    audit = document.metadata.get("limbu_post_correction", {})
    applied_count = audit.get("applied_count") if isinstance(audit, dict) else None
    print(Path(args.out) / "document.json")
    print(Path(args.out) / "document.md")
    print(Path(args.out) / "limbu-correction-audit.json")
    print(Path(args.out) / "limbu-correction-pairs.jsonl")
    print(Path(args.out) / "limbu-output-manifest.json")
    print(f"applied_corrections={applied_count}")
    return 0


def cmd_build_limbu_correction_pair_pack(args: argparse.Namespace) -> int:
    report = build_limbu_correction_pair_pack(
        args.correction_pairs,
        args.out,
        pack_id=args.pack_id,
        heldout_fraction=args.heldout_fraction,
        min_heldout=args.min_heldout,
    )
    print(Path(args.out) / "limbu-correction-pair-pack.json")
    print(Path(args.out) / "limbu-correction-pair-pack.md")
    print(report["train_pairs"])
    print(report["heldout_pairs"])
    print(f"pack_content_sha256={report['pack_content_sha256']}")
    print(f"train={report['train_count']} heldout={report['heldout_count']}")
    return 0


def cmd_audit_limbu_correction_pair_pack(args: argparse.Namespace) -> int:
    report = audit_limbu_correction_pair_pack(args.pack_dir, args.out)
    audit_dir = Path(args.out) if args.out else Path(args.pack_dir)
    print(audit_dir / "limbu-correction-pair-pack-audit.json")
    print(audit_dir / "limbu-correction-pair-pack-audit.md")
    print(f"passed={report['passed']}")
    return 0 if report["passed"] else 1


def cmd_derive_limbu_post_correction_profile(args: argparse.Namespace) -> int:
    result = derive_limbu_post_correction_profile(
        args.correction_pairs,
        args.out,
        profile_id=args.profile_id,
        min_support=args.min_support,
    )
    derivation = result.get("derivation", {})
    rule_count = derivation.get("rule_count") if isinstance(derivation, dict) else None
    print(Path(args.out) / "limbu-post-correction-profile.json")
    print(Path(args.out) / "limbu-post-correction-profile-derivation.json")
    print(Path(args.out) / "limbu-post-correction-profile-derivation.md")
    print(f"rules={rule_count}")
    return 0


def cmd_score_limbu_post_correction_profile(args: argparse.Namespace) -> int:
    report = score_limbu_post_correction_profile(
        args.profile,
        args.correction_pairs,
        args.out,
        frozen_eval_pack=args.frozen_eval_pack,
    )
    print(Path(args.out) / "limbu-post-correction-profile-eval.json")
    print(Path(args.out) / "limbu-post-correction-profile-eval.md")
    print(Path(args.out) / "limbu-post-correction-profile-eval-lines.jsonl")
    print(f"passed={report['passed']}")
    print(f"metric={report['metric']} before={report['before']} after={report['after']}")
    print(f"eval_run_sha256={report['eval_run_sha256']}")
    return 0 if report["passed"] else 1


def cmd_admit_limbu_post_correction_profile(args: argparse.Namespace) -> int:
    report = admit_limbu_post_correction_profile(
        args.profile,
        args.eval_run,
        args.out,
        output_filename=args.output_filename,
    )
    print(report["admitted_profile"])
    print(Path(args.out) / "limbu-post-correction-profile-admission.json")
    print(Path(args.out) / "limbu-post-correction-profile-admission.md")
    print(Path(args.out) / "audit" / "limbu-post-correction-profile-audit.json")
    print(f"audit_passed={report['audit_passed']} claim_ready={report['audit_claim_ready']}")
    return 0 if report["audit_passed"] and report["audit_claim_ready"] else 1


def cmd_audit_limbu_post_correction_profile(args: argparse.Namespace) -> int:
    report = audit_limbu_post_correction_profile(args.profile, args.out)
    print(Path(args.out) / "limbu-post-correction-profile-audit.json")
    print(Path(args.out) / "limbu-post-correction-profile-audit.md")
    print(f"passed={report['passed']} claim_ready={report['claim_ready']}")
    return 0 if report["passed"] else 1


def cmd_audit_limbu_output(args: argparse.Namespace) -> int:
    required_script_counts = _parse_limbu_required_script_counts(args.require_script_count or ())
    report = audit_limbu_output(
        args.output_dir,
        args.out,
        require_capture_prep=args.require_capture_prep,
        min_capture_prepared_width=args.min_capture_prepared_width,
        min_capture_prepared_height=args.min_capture_prepared_height,
        min_capture_prepared_entropy=args.min_capture_prepared_entropy,
        min_capture_prepared_luminance_stddev=args.min_capture_prepared_luminance_stddev,
        min_capture_prepared_edge_stddev=args.min_capture_prepared_edge_stddev,
        require_capture_metadata_path_self=args.require_capture_metadata_path_self,
        require_no_pending_review=args.require_no_pending_review,
        require_reviewer_for_corrections=args.require_reviewer_for_corrections,
        require_no_dropped_image_lines=args.require_no_dropped_image_lines,
        min_line_count=args.min_line_count,
        min_average_line_confidence=args.min_average_line_confidence,
        min_quality_score=args.min_quality_score,
        required_scripts=tuple(args.require_script or ()),
        required_script_counts=required_script_counts,
    )
    report_dir = Path(args.out) if args.out else Path(args.output_dir)
    print(report_dir / "limbu-output-audit.json")
    print(report_dir / "limbu-output-audit.md")
    print(f"passed={report['passed']}")
    return 0 if report["passed"] else 1


def _parse_limbu_required_script_counts(items: list[str] | tuple[str, ...]) -> dict[str, int]:
    allowed = {"limbu_sirijonga", "devanagari_limbu", "mixed_limbu_devanagari", "other"}
    parsed: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise OcrTechError(f"--require-script-count must be SCRIPT=COUNT, got {item!r}")
        script, raw_count = item.split("=", 1)
        script = script.strip()
        raw_count = raw_count.strip()
        if script not in allowed:
            raise OcrTechError(f"unsupported --require-script-count script {script!r}; allowed values are {sorted(allowed)}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise OcrTechError(f"--require-script-count count must be an integer, got {raw_count!r}") from exc
        if count < 1:
            raise OcrTechError(f"--require-script-count count must be positive, got {count}")
        parsed[script] = count
    return dict(sorted(parsed.items()))


def cmd_benchmark(args: argparse.Namespace) -> int:
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    inputs = list(args.inputs)
    if args.inputs_from_manifest:
        inputs.extend(_benchmark_inputs_from_manifest(args.inputs_from_manifest))
    if not inputs:
        raise OcrTechError("benchmark requires positional inputs or --inputs-from-manifest")
    eval_manifest = args.eval_manifest or args.inputs_from_manifest
    results = run_benchmark(
        inputs,
        args.out,
        baselines=baselines,
        references_dir=args.references_dir,
        eval_manifest=eval_manifest,
        candidate_model_config=args.candidate_model_config,
        fallback_engine=args.fallback_engine,
        fallback_model_config=args.fallback_model_config,
        low_confidence_threshold=args.low_confidence_threshold,
        fallback_min_quality_score=args.fallback_min_quality_score,
        resume_existing=args.resume_existing,
        sample_timeout_seconds=args.sample_timeout_seconds,
        capture_gpu_metrics=args.capture_gpu_metrics,
    )
    print(Path(args.out) / "report.json")
    print(Path(args.out) / "report.md")
    print(f"runs={len(results)}")
    return 0


def _benchmark_inputs_from_manifest(manifest_path: str | Path) -> list[str]:
    manifest = Path(manifest_path)
    entries = load_manifest(manifest)
    if not entries:
        raise OcrTechError(f"benchmark input manifest is empty: {manifest}")
    inputs: list[str] = []
    for entry in entries:
        raw_path = Path(entry.image_path)
        if raw_path.is_absolute():
            resolved = raw_path
        elif raw_path.exists():
            resolved = raw_path
        else:
            resolved = manifest.parent / raw_path
        if not resolved.exists():
            raise OcrTechError(f"benchmark input does not exist for sample {entry.sample_id}: {resolved}")
        if resolved.is_dir():
            raise OcrTechError(f"benchmark input is a directory for sample {entry.sample_id}: {resolved}")
        inputs.append(str(resolved))
    return inputs


def cmd_summarize_benchmark(args: argparse.Namespace) -> int:
    summary = summarize_benchmark_report(
        args.benchmark_report,
        args.out,
        candidate=args.candidate,
        baselines=args.baseline or None,
        metrics=args.metric or None,
    )
    print(Path(args.out) / "summary.json")
    print(Path(args.out) / "summary.md")
    print(f"candidate={summary.candidate} paired={len(summary.paired_metrics)}")
    return 0


def cmd_rescore_benchmark(args: argparse.Namespace) -> int:
    results = rescore_benchmark_report(
        args.benchmark_report,
        args.out,
        references_dir=args.references_dir,
        eval_manifest=args.eval_manifest,
    )
    print(Path(args.out) / "report.json")
    print(Path(args.out) / "report.md")
    print(f"runs={len(results)}")
    return 0


def cmd_derive_table_postprocess_benchmark(args: argparse.Namespace) -> int:
    results = derive_table_postprocess_benchmark(
        args.benchmark_report,
        args.out,
        source_baseline=args.source_baseline,
        candidate_baseline=args.candidate_baseline,
        references_dir=args.references_dir,
        eval_manifest=args.eval_manifest,
    )
    print(Path(args.out) / "report.json")
    print(Path(args.out) / "report.md")
    print(f"runs={len(results)}")
    return 0


def cmd_analyze_table_cells(args: argparse.Namespace) -> int:
    analysis = analyze_table_cells(
        args.benchmark_report,
        args.eval_manifest,
        args.out,
        candidate=args.candidate,
        baselines=args.baseline or None,
        top_n=args.top_n,
    )
    print(Path(args.out) / "table-cell-analysis.json")
    print(Path(args.out) / "table-cell-analysis.md")
    print(f"systems={len(analysis.systems)} paired={len(analysis.paired)} failures={len(analysis.top_failures)}")
    return 0


def cmd_paper_benchmark_table(args: argparse.Namespace) -> int:
    display_names: dict[str, str] = {}
    for item in args.label:
        key, separator, value = str(item).partition("=")
        if not separator or not key or not value:
            raise OcrTechError(f"invalid --label value {item!r}; expected system=Label")
        display_names[key] = value
    source_preferences: dict[str, str] = {}
    for item in args.source_preference:
        key, separator, value = str(item).partition("=")
        if not separator or not key or not value:
            raise OcrTechError(f"invalid --source-preference value {item!r}; expected system=path")
        source_preferences[key] = value
    table = export_paper_benchmark_table(
        args.summary,
        args.out,
        slice_name=args.slice,
        metrics=args.metric or None,
        systems=args.system or None,
        display_names=display_names,
        source_preferences=source_preferences,
        title=args.title,
    )
    print(Path(args.out) / "paper-table.json")
    print(Path(args.out) / "paper-table.md")
    print(f"rows={len(table.rows)} slice={table.slice_name}")
    return 0


def cmd_benchmark_oracle(args: argparse.Namespace) -> int:
    report, summary = build_benchmark_oracle(
        args.benchmark_report,
        args.out,
        metric=args.metric,
        systems=args.system or None,
        direction=args.direction,
        oracle_name=args.oracle_name,
    )
    print(Path(args.out) / "oracle-analysis.json")
    print(Path(args.out) / "oracle-analysis.md")
    print(Path(args.out) / "oracle-choices.json")
    print(Path(args.out) / "oracle-summary" / "summary.json")
    print(f"oracle={report.oracle_name} samples={report.sample_count} paired={len(summary.paired_metrics)}")
    return 0


def cmd_benchmark_error_analysis(args: argparse.Namespace) -> int:
    analysis = analyze_benchmark_errors(
        args.benchmark_report,
        args.out,
        system=args.system,
        baselines=args.baseline or None,
        metric=args.metric,
        direction=args.direction,
        top_n=args.top_n,
    )
    print(Path(args.out) / "error-analysis.json")
    print(Path(args.out) / "error-analysis.md")
    print(Path(args.out) / "top-failures.json")
    print(f"system={analysis.system} samples={analysis.sample_count} metric={analysis.metric}")
    return 0


def cmd_export_failure_manifest(args: argparse.Namespace) -> int:
    export = export_failure_manifest(
        args.source_manifest,
        args.top_failures,
        args.out,
        split=args.split,
    )
    print(export.output_manifest)
    print(export.summary_path)
    print(f"selected={export.selected_count} split={export.split or ''}")
    return 0


def cmd_calibrate_quality_router(args: argparse.Namespace) -> int:
    calibration, model_path = calibrate_quality_router(
        args.experiment_report,
        args.out,
        model_id=args.model_id,
        base_model=args.base_model,
        primary_engine=args.primary_engine,
        secondary_engine=args.secondary_engine,
        primary_engine_kwargs=_parse_key_values(args.primary_engine_kwarg),
        secondary_engine_kwargs=_parse_key_values(args.secondary_engine_kwarg),
        candidate_baseline=args.candidate_baseline,
        primary_baseline=args.primary_baseline,
        metric=args.metric,
        direction=args.direction,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
        notes=args.notes,
    )
    print(Path(args.out) / "calibration.json")
    print(Path(args.out) / "calibration.md")
    print(model_path)
    print(f"selected_threshold={calibration.selected_threshold:.6f}")
    return 0


def cmd_calibrate_script_router(args: argparse.Namespace) -> int:
    calibration, model_path = calibrate_script_router(
        args.experiment_report,
        args.out,
        model_id=args.model_id,
        base_model=args.base_model,
        primary_engine=args.primary_engine,
        secondary_engine=args.secondary_engine,
        primary_engine_kwargs=_parse_key_values(args.primary_engine_kwarg),
        secondary_engine_kwargs=_parse_key_values(args.secondary_engine_kwarg),
        primary_baseline=args.primary_baseline,
        secondary_baseline=args.secondary_baseline,
        metric=args.metric,
        direction=args.direction,
        script=args.script,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
        routing_granularity=args.routing_granularity,
        secondary_structure_backfill=args.secondary_structure_backfill,
        notes=args.notes,
    )
    print(Path(args.out) / "calibration.json")
    print(Path(args.out) / "calibration.md")
    print(model_path)
    print(f"selected_threshold={calibration.selected_threshold:.6f}")
    return 0


def cmd_calibrate_tesseract_psm_ensemble(args: argparse.Namespace) -> int:
    report, model_path = calibrate_tesseract_psm_ensemble(
        args.primary_experiment_report,
        args.out,
        model_id=args.model_id,
        variant_experiment_reports=_parse_labeled_paths(args.variant_experiment),
        metric=args.metric,
        direction=args.direction,
        bias_start=args.bias_start,
        bias_end=args.bias_end,
        bias_step=args.bias_step,
        language=args.language,
        selection_margin=args.selection_margin,
        notes=args.notes,
    )
    print(Path(args.out) / "calibration.json")
    print(Path(args.out) / "calibration.md")
    print(model_path)
    print(f"selected_biases={report.selected_biases}")
    return 0


def cmd_run_experiment(args: argparse.Namespace) -> int:
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    report = run_experiment(
        args.eval_manifest,
        args.out,
        baselines=baselines,
        references_dir=args.references_dir,
        validation_config=args.validation_config,
        candidate_model_config=args.candidate_model_config,
        fallback_engine=args.fallback_engine,
        fallback_model_config=args.fallback_model_config,
        low_confidence_threshold=args.low_confidence_threshold,
        fallback_min_quality_score=args.fallback_min_quality_score,
        train_manifests=args.train_manifest,
        gorkhapatra_review_audits=args.gorkhapatra_review_audit,
        run_validation=not args.skip_validation,
        run_preflight=args.preflight,
        require_claim_ready_eval_pack=not args.allow_smoke_eval_pack,
        require_trained_recognizer=args.require_trained_recognizer,
        require_model_admission=args.require_model_admission,
    )
    print(Path(args.out) / "experiment.json")
    print(Path(args.out) / "experiment.md")
    print(f"benchmark={report.benchmark_report}")
    print(f"validation={report.validation_status or 'not_run'}")
    if args.require_validated and report.validation_status != "validated":
        return 1
    return 0


def cmd_preflight_claim(args: argparse.Namespace) -> int:
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    report = run_claim_preflight(
        args.eval_manifest,
        args.out,
        baselines=baselines,
        references_dir=args.references_dir,
        candidate_model_config=args.candidate_model_config,
        fallback_model_config=args.fallback_model_config,
        train_manifests=args.train_manifest,
        validation_config=args.validation_config,
        gorkhapatra_review_audits=args.gorkhapatra_review_audit,
        require_claim_ready_eval_pack=not args.allow_smoke_eval_pack,
        require_trained_recognizer=args.require_trained_recognizer,
        require_model_admission=args.require_model_admission,
    )
    print(Path(args.out) / "preflight.json")
    print(Path(args.out) / "preflight.md")
    print(f"passed={report.passed} samples={report.sample_count}")
    return 0 if report.passed else 1


def cmd_audit_remote_host(args: argparse.Namespace) -> int:
    min_python = _parse_python_version(args.min_python)
    report = audit_remote_host(
        args.host,
        args.out,
        user=args.user,
        port=args.port,
        password_env=args.password_env,
        min_python=min_python,
        min_free_gb=args.min_free_gb,
        workdir=args.workdir,
        require_gpu=args.require_gpu,
        require_paddle_training=args.require_paddle_training,
        python_candidates=args.python_candidate or DEFAULT_PYTHON_CANDIDATES,
        require_commands=args.require_command or None,
        optional_commands=args.optional_command or None,
        require_paths=args.require_path,
    )
    print(Path(args.out) / "remote-audit.json")
    print(Path(args.out) / "remote-audit.md")
    print(f"passed={report.passed} host={report.host}")
    return 0 if report.passed else 1


def cmd_bootstrap_remote_host(args: argparse.Namespace) -> int:
    min_python = _parse_python_version(args.min_python)
    report = bootstrap_remote_host(
        args.host,
        args.out,
        user=args.user,
        port=args.port,
        password_env=args.password_env,
        workdir=args.workdir,
        min_python=min_python,
        python_candidates=args.python_candidate or DEFAULT_PYTHON_CANDIDATES,
        venv_name=args.venv_name,
        recreate_venv=args.recreate_venv,
    )
    print(Path(args.out) / "remote-bootstrap.json")
    print(Path(args.out) / "remote-bootstrap.md")
    print(f"passed={report.passed} workdir={report.workdir} venv={report.venv_dir}")
    return 0 if report.passed else 1


def cmd_sync_remote_workspace(args: argparse.Namespace) -> int:
    report = sync_remote_workspace(
        args.local_root,
        args.out,
        host=args.host,
        user=args.user,
        port=args.port,
        password_env=args.password_env,
        remote_workdir=args.remote_workdir,
        exclude_patterns=args.exclude or None,
    )
    print(Path(args.out) / "remote-sync.json")
    print(Path(args.out) / "remote-sync.md")
    print(f"passed={report.passed} remote_workdir={report.remote_workdir}")
    return 0 if report.passed else 1


def cmd_review_claim(args: argparse.Namespace) -> int:
    report = review_claim(
        args.experiment,
        args.out,
        config_path=args.config,
        model_config=args.model_config,
    )
    print(Path(args.out) / "claim-review.json")
    print(Path(args.out) / "claim-review.md")
    print(f"passed={report.passed} experiments={len(report.experiments)}")
    return 0 if report.passed else 1


def cmd_audit_augmentation_ablation(args: argparse.Namespace) -> int:
    report = audit_augmentation_ablation(
        args.config,
        args.out,
        real_review_audits=args.real_review_audit,
    )
    print(Path(args.out) / "augmentation-ablation-audit.json")
    print(Path(args.out) / "augmentation-ablation-audit.md")
    print(
        f"passed={report.passed} arms={report.arm_count} "
        f"real_evidence={report.real_evidence_count} verified_references={report.verified_reference_count} "
        f"issues={len(report.issues)} warnings={len(report.warnings)}"
    )
    return 0 if report.passed else 1


def cmd_validate_claim(args: argparse.Namespace) -> int:
    report = validate_claim(
        args.benchmark_report,
        args.out,
        config_path=args.config,
        eval_manifest=args.eval_manifest,
        train_manifests=args.train_manifest,
        candidate_model_config=args.candidate_model_config,
        fallback_model_config=args.fallback_model_config,
    )
    print(Path(args.out) / "validation.json")
    print(Path(args.out) / "validation.md")
    print(f"status={report.claim_status}")
    if not report.passed:
        return 1
    return 0


def cmd_recognizer_admission_decision(args: argparse.Namespace) -> int:
    decision = decide_recognizer_admission(args.validation_report, args.out)
    print(Path(args.out) / "recognizer-admission-decision.json")
    print(Path(args.out) / "recognizer-admission-decision.md")
    print(f"decision={decision.decision} status={decision.claim_status}")
    return 0 if decision.decision == "admit" else 1


def cmd_recognizer_failure_review(args: argparse.Namespace) -> int:
    review = review_recognizer_failures(
        args.admission_decision,
        args.validation_report,
        args.out,
        model_cards=args.model_card,
        min_rejections_to_stop=args.min_rejections_to_stop,
    )
    stopped_count = sum(1 for family in review.families if family.recommendation == "stop_family")
    print(Path(args.out) / "recognizer-failure-review.json")
    print(Path(args.out) / "recognizer-failure-review.md")
    print(f"families={len(review.families)} stopped={stopped_count}")
    return 1 if stopped_count else 0


def cmd_list_public_benchmarks(args: argparse.Namespace) -> int:
    _ = args
    for spec in list_public_benchmarks():
        print(f"{spec.dataset}\t{spec.title}\tmin_samples={spec.min_samples}\t{spec.url}")
    return 0


def cmd_prepare_public_benchmark(args: argparse.Namespace) -> int:
    report = prepare_public_benchmark(
        args.benchmark,
        args.out,
        limit=args.limit,
        language=args.language,
        subset=args.subset,
        data_source=args.data_source,
        require_tables=args.require_tables,
        annotation_path=args.annotation_path,
        local_image_root=args.local_image_root,
    )
    print(report.summary_json_path)
    print(report.summary_md_path)
    print(f"manifest={report.manifest_path} samples={report.sample_count} benchmark={report.benchmark}")
    return 0


def cmd_audit_public_benchmark(args: argparse.Namespace) -> int:
    audit = audit_public_benchmark_manifest(
        args.manifest,
        args.out,
        benchmark=args.benchmark,
        min_samples=args.min_samples,
        require_reference_paths=not args.allow_missing_reference_paths,
    )
    print(Path(args.out) / "public-benchmark-audit.json")
    print(Path(args.out) / "public-benchmark-audit.md")
    print(f"passed={audit.passed} samples={audit.sample_count} benchmark={audit.benchmark}")
    return 0 if audit.passed else 1


def cmd_audit_gorkhapatra_source(args: argparse.Namespace) -> int:
    category_urls = list(args.category_url)
    epaper_urls = list(args.epaper_url)
    if not args.no_default_urls:
        if not category_urls and not args.category_html:
            category_urls = list(DEFAULT_GORKHAPATRA_CATEGORY_URLS)
        if not epaper_urls and not args.epaper_html:
            epaper_urls = [DEFAULT_GORKHAPATRA_EPAPER_URL]
    audit = audit_gorkhapatra_source(
        args.out,
        category_urls=category_urls,
        epaper_urls=epaper_urls,
        category_html_paths=args.category_html,
        epaper_html_paths=args.epaper_html,
        download_assets=args.download_assets,
        max_article_images=args.max_article_images,
        max_epaper_pdfs=args.max_epaper_pdfs,
        timeout_seconds=args.timeout_seconds,
    )
    print(Path(args.out) / "gorkhapatra-source-audit.json")
    print(Path(args.out) / "gorkhapatra-source-audit.md")
    print(f"articles={audit.article_count} epapers={audit.epaper_count} downloaded={audit.downloaded_count} warnings={len(audit.warnings)}")
    return 0


def cmd_prepare_gorkhapatra_pack(args: argparse.Namespace) -> int:
    summary = prepare_gorkhapatra_pack(
        args.source_audit_json,
        args.out,
        include_article_images=not args.no_article_images,
        include_epaper_pages=not args.no_epaper_pages,
        max_article_images=args.max_article_images,
        max_epaper_pdfs=args.max_epaper_pdfs,
        max_pages_per_pdf=args.max_pages_per_pdf,
        split=args.split,
        dataset_name=args.dataset_name,
    )
    print(summary.manifest_path)
    print(Path(args.out) / "gorkhapatra-pack.json")
    print(Path(args.out) / "gorkhapatra-pack.md")
    print(f"samples={summary.sample_count} article_images={summary.article_image_count} epaper_pages={summary.epaper_page_count}")
    return 0


def cmd_audit_gorkhapatra_language_pages(args: argparse.Namespace) -> int:
    publication_urls = list(args.publication_url)
    epaper_urls = list(args.epaper_url)
    if not args.no_default_urls:
        if not publication_urls and not args.publication_html:
            publication_urls = [DEFAULT_NAYA_NEPAL_PUBLICATION_URL]
        if not epaper_urls and not args.epaper_html:
            epaper_urls = [DEFAULT_GORKHAPATRA_EPAPER_URL]
    publication_urls = _expand_publication_page_urls(publication_urls, args.publication_pages)
    audit = audit_gorkhapatra_language_pages(
        args.out,
        publication_urls=publication_urls,
        epaper_urls=epaper_urls,
        publication_html_paths=args.publication_html,
        epaper_html_paths=args.epaper_html,
        download_pdfs=args.download_pdfs,
        render_pages=args.render_pages,
        max_publication_items=args.max_publication_items,
        max_epaper_pdfs=args.max_epaper_pdfs,
        timeout_seconds=args.timeout_seconds,
    )
    print(Path(args.out) / "gorkhapatra-language-pages.json")
    print(Path(args.out) / "gorkhapatra-language-pages.md")
    print(
        f"publication_items={audit.publication_item_count} "
        f"aligned={audit.aligned_count} language_page_hits={audit.language_page_hit_count} warnings={len(audit.warnings)}"
    )
    return 0


def _expand_publication_page_urls(urls: list[str], page_count: int) -> list[str]:
    if page_count < 1:
        raise OcrTechError("--publication-pages must be >= 1")
    if page_count == 1:
        return urls
    expanded: list[str] = []
    for url in urls:
        expanded.extend(_publication_page_url(url, page) for page in range(1, page_count + 1))
    return expanded


def _publication_page_url(url: str, page: int) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if page == 1 and "page" not in query:
        return url
    query["page"] = str(page)
    return urlunparse(parsed._replace(query=urlencode(query)))


def cmd_prepare_gorkhapatra_language_page_pack(args: argparse.Namespace) -> int:
    summary = prepare_gorkhapatra_language_page_pack(
        args.language_page_audit_json,
        args.out,
        split=args.split,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
    )
    print(summary.manifest_path)
    print(Path(args.out) / "gorkhapatra-language-page-pack.json")
    print(Path(args.out) / "gorkhapatra-language-page-pack.md")
    print(f"samples={summary.sample_count} copied_pages={summary.copied_page_count} warnings={len(summary.warnings)}")
    return 0


def cmd_prepare_gorkhapatra_language_page_review(args: argparse.Namespace) -> int:
    summary = prepare_gorkhapatra_language_page_review(
        args.language_page_pack_manifest,
        args.out,
    )
    print(summary.review_json_path)
    print(summary.review_csv_path)
    print(summary.review_md_path)
    print(
        f"samples={summary.sample_count} pending_review={summary.pending_review_count} "
        f"claim_eligible={summary.claim_evidence_eligible_count} warnings={len(summary.warnings)}"
    )
    return 0


def cmd_audit_gorkhapatra_language_page_review(args: argparse.Namespace) -> int:
    summary = audit_gorkhapatra_language_page_review(
        args.language_page_pack_manifest,
        args.review_csv,
        args.out,
        require_verified_references=args.require_verified_references,
        candidate_languages=args.candidate_language,
    )
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} groups={summary.group_count} "
        f"accepted={summary.accepted_count} unresolved={summary.unresolved_count} issues={len(summary.issues)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_gorkhapatra_language_page_reference_templates(args: argparse.Namespace) -> int:
    summary = prepare_gorkhapatra_language_page_reference_templates(
        args.language_page_pack_manifest,
        args.out,
        review_csv_path=args.review_csv,
        accepted_only=args.accepted_only,
    )
    print(summary.output_dir)
    print(summary.index_csv_path)
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} templates={summary.sample_count} "
        f"skipped={summary.skipped_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_gorkhapatra_language_page_reviewer_bundle(args: argparse.Namespace) -> int:
    summary = prepare_gorkhapatra_language_page_reviewer_bundle(
        args.language_page_pack_manifest,
        args.review_csv,
        args.out,
        reference_template_dir=args.reference_template_dir,
    )
    print(summary.output_dir)
    print(summary.index_csv_path)
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} "
        f"images={summary.copied_image_count} references={summary.copied_reference_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_gorkhapatra_language_page_transcription_work_order(args: argparse.Namespace) -> int:
    summary = prepare_gorkhapatra_language_page_transcription_work_order(
        args.language_page_pack_manifest,
        args.review_csv,
        args.out,
        candidate_languages=args.candidate_language,
    )
    print(summary.summary_json_path)
    print(summary.index_csv_path)
    print(summary.transcription_html_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} "
        f"blocked={summary.blocked_count} verified={summary.verified_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_extract_gorkhapatra_language_page_pdf_text(args: argparse.Namespace) -> int:
    summary = extract_gorkhapatra_language_page_pdf_text(
        args.language_page_pack_manifest,
        args.out,
        candidate_languages=args.candidate_language,
        target_font=args.target_font,
    )
    print(summary.summary_json_path)
    print(summary.index_csv_path)
    print(summary.spans_csv_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.extracted_sample_count}/{summary.sample_count} "
        f"spans={summary.span_count} target_spans={summary.target_span_count} fonts={summary.font_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_convert_limbu_legacy(args: argparse.Namespace) -> int:
    if bool(args.text) == bool(args.pdf_text_json):
        raise OcrTechError("convert-limbu-legacy requires exactly one of --text or --pdf-text-json")
    if args.text:
        conversion = convert_limbu_legacy_text(args.text, args.map)
        print(conversion.unicode_text)
        print(
            f"limbu_chars={conversion.limbu_char_count} replacements={conversion.replacement_count} "
            f"unmapped={len(conversion.unmapped_codepoints)}"
        )
        return 0 if conversion.limbu_char_count > 0 else 1
    result = convert_gorkhapatra_limbu_pdf_text(args.pdf_text_json, args.out, map_path=args.map)
    print(result["json_path"])
    print(result["text_path"])
    print(
        f"sample={result['sample_id']} spans={result['target_span_count']} "
        f"limbu_chars={result['limbu_char_count']}"
    )
    return 0 if int(result["limbu_char_count"]) > 0 else 1


def cmd_prepare_gorkhapatra_language_page_assisted_references(args: argparse.Namespace) -> int:
    summary = prepare_gorkhapatra_language_page_assisted_references(
        args.language_page_pack_manifest,
        args.review_csv,
        args.out,
        candidate_languages=args.candidate_language,
        engine=args.engine,
        tesseract_language=args.tesseract_language,
        overwrite=args.overwrite,
    )
    print(summary.summary_json_path)
    print(summary.index_csv_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} assisted={summary.assisted_count} "
        f"failed={summary.failed_count} skipped={summary.skipped_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_gorkhapatra_language_page_pdf_native_references(args: argparse.Namespace) -> int:
    summary = prepare_gorkhapatra_language_page_pdf_native_references(
        args.language_page_pack_manifest,
        args.review_csv,
        args.pdf_text_dir,
        args.out,
        candidate_languages=args.candidate_language,
        converter_id=args.converter,
        map_path=args.map_path,
        overwrite=args.overwrite,
    )
    print(summary.summary_json_path)
    print(summary.index_csv_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} converted={summary.assisted_count} "
        f"failed={summary.failed_count} skipped={summary.skipped_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_gorkhapatra_language_page_recognizer_eval(args: argparse.Namespace) -> int:
    summary = prepare_gorkhapatra_language_page_recognizer_eval(
        args.language_page_pack_manifest,
        args.references_dir,
        args.out,
        candidate_languages=args.candidate_language,
        split=args.split,
        dataset_name=args.dataset_name,
        min_text_length=args.min_text_length,
        allow_draft=args.allow_draft,
    )
    print(summary.summary_json_path)
    print(summary.manifest_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} crops={summary.crop_count} "
        f"claim_eligible={summary.claim_eligible_crop_count} skipped={summary.skipped_count} "
        f"warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_gorkhapatra_language_page_ocr_sidecars(args: argparse.Namespace) -> int:
    summary = prepare_gorkhapatra_language_page_ocr_sidecars(
        args.language_page_pack_manifest,
        args.review_csv,
        args.out,
        candidate_languages=args.candidate_language,
        engine=args.engine,
        tesseract_language=args.tesseract_language,
        overwrite=args.overwrite,
        in_place=args.in_place,
    )
    print(summary.summary_json_path)
    print(summary.index_csv_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} sidecars={summary.sidecar_count} "
        f"failed={summary.failed_count} skipped={summary.skipped_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_gorkhapatra_language_page_verification_bundle(args: argparse.Namespace) -> int:
    summary = prepare_gorkhapatra_language_page_verification_bundle(
        args.language_page_pack_manifest,
        args.review_csv,
        args.assisted_references_dir,
        args.out,
        candidate_languages=args.candidate_language,
        overwrite=args.overwrite,
    )
    print(summary.summary_json_path)
    print(summary.index_csv_path)
    print(summary.review_html_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} lines={summary.line_count} "
        f"crops={summary.crop_count} missing_bboxes={summary.missing_bbox_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_audit_gorkhapatra_language_page_verification_csv(args: argparse.Namespace) -> int:
    summary = audit_gorkhapatra_language_page_verification_csv(
        args.verification_csv,
        args.out,
        candidate_languages=args.candidate_language,
        required_codepoint_ranges=args.require_codepoint_range,
    )
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} lines={summary.line_count} "
        f"ready={summary.ready_line_count} dropped={summary.dropped_line_count} "
        f"blocked={summary.blocked_line_count} issues={len(summary.issues)} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_split_gorkhapatra_language_page_verification_csv(args: argparse.Namespace) -> int:
    summary = split_gorkhapatra_language_page_verification_csv(
        args.verification_csv,
        args.out,
        batch_size=args.batch_size,
        candidate_languages=args.candidate_language,
    )
    print(summary.summary_json_path)
    print(summary.index_csv_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} lines={summary.line_count} "
        f"batches={summary.batch_count} batch_size={summary.batch_size} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_merge_gorkhapatra_language_page_verification_batches(args: argparse.Namespace) -> int:
    summary = merge_gorkhapatra_language_page_verification_batches(
        args.source_verification_csv,
        args.batches_dir,
        args.out,
    )
    print(summary.summary_json_path)
    print(summary.merged_csv_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} source_lines={summary.source_line_count} merged_lines={summary.merged_line_count} "
        f"batch_files={summary.batch_file_count} issues={len(summary.issues)} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_assign_gorkhapatra_language_page_verification_batches(args: argparse.Namespace) -> int:
    summary = assign_gorkhapatra_language_page_verification_batches(
        args.split_index_csv,
        args.out,
        reviewers=args.reviewer,
        due_date=args.due_date,
    )
    print(summary.summary_json_path)
    print(summary.assignment_csv_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} batches={summary.batch_count} assigned={summary.assigned_count} "
        f"reviewers={len(summary.reviewer_counts)} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_apply_gorkhapatra_language_page_verification_bundle(args: argparse.Namespace) -> int:
    summary = apply_gorkhapatra_language_page_verification_bundle(
        args.language_page_pack_manifest,
        args.review_csv,
        args.verification_csv,
        args.assisted_references_dir,
        args.out,
        candidate_languages=args.candidate_language,
        reviewer=args.reviewer,
        reviewed_at=args.reviewed_at,
        tables_reviewed=args.tables_reviewed,
        figures_reviewed=args.figures_reviewed,
        captions_reviewed=args.captions_reviewed,
        required_codepoint_ranges=args.require_codepoint_range,
        overwrite=args.overwrite,
    )
    print(summary.summary_json_path)
    print(summary.updated_review_csv_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} verified={summary.verified_reference_count} "
        f"blocked={summary.blocked_count} reviewed_lines={summary.reviewed_line_count} "
        f"dropped_lines={summary.dropped_line_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_finalize_gorkhapatra_language_page_review(args: argparse.Namespace) -> int:
    summary = finalize_gorkhapatra_language_page_review(
        args.language_page_pack_manifest,
        args.review_csv,
        args.out,
        split=args.split,
        dataset_name=args.dataset_name,
        candidate_languages=args.candidate_language,
    )
    print(summary.manifest_path)
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} "
        f"accepted={summary.accepted_review_count} skipped={summary.skipped_review_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_audit_language_registry(args: argparse.Namespace) -> int:
    audit = audit_language_registry(
        args.registry,
        args.out,
        min_nepal_languages=args.min_nepal_languages,
        require_limbu_first=not args.allow_non_limbu_first,
        verify_local_paths=not args.no_verify_local_paths,
    )
    print(Path(args.out) / "language-registry-audit.json")
    print(Path(args.out) / "language-registry-audit.md")
    print(
        "passed="
        f"{audit.passed} languages={audit.language_count} "
        f"nepal_languages={audit.counted_nepal_language_count} first_targets={','.join(audit.first_target_ids)}"
    )
    return 0 if audit.passed else 1


def cmd_audit_language_readiness(args: argparse.Namespace) -> int:
    audit = audit_language_readiness(
        args.registry,
        args.out,
        synthesis_text_audit_paths=args.synthesis_text_audit,
        min_target_languages=args.min_target_languages,
        min_synthesis_ready_languages=args.min_synthesis_ready_languages,
        require_limbu_synthesis=not args.no_require_limbu_synthesis,
    )
    print(Path(args.out) / "language-readiness-audit.json")
    print(Path(args.out) / "language-readiness-audit.md")
    print(
        "passed="
        f"{audit.passed} target_languages={audit.target_language_count} "
        f"synthesis_ready={audit.synthesis_ready_language_count}/{audit.required_synthesis_ready_language_count}"
    )
    return 0 if audit.passed else 1


def cmd_audit_synthesis_resources(args: argparse.Namespace) -> int:
    audit = audit_synthesis_resources(
        args.root,
        args.out,
        labels=args.label,
        max_files_per_root=args.max_files_per_root,
        max_text_samples_per_root=args.max_text_samples_per_root,
        sample_bytes=args.sample_bytes,
    )
    print(audit.output_json_path)
    print(audit.output_md_path)
    print(
        f"passed={audit.passed} roots={audit.existing_root_count}/{audit.root_count} "
        f"files={audit.total_files} warnings={len(audit.warnings)}"
    )
    return 0 if audit.passed else 1


def cmd_prepare_limbu_limdic_text(args: argparse.Namespace) -> int:
    summary = prepare_limbu_limdic_text(
        args.source,
        args.out,
        split=args.split,
        dataset=args.dataset,
        min_text_chars=args.min_text_chars,
    )
    print(summary.manifest_path)
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} "
        f"duplicates={summary.duplicate_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_tamang_text(args: argparse.Namespace) -> int:
    summary = prepare_tamang_text(
        args.source,
        args.out,
        split=args.split,
        dataset=args.dataset,
        min_text_chars=args.min_text_chars,
        limit_per_source=args.limit_per_source,
    )
    print(summary.manifest_path)
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} "
        f"duplicates={summary.duplicate_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_magar_text(args: argparse.Namespace) -> int:
    summary = prepare_magar_text(
        args.source,
        args.out,
        split=args.split,
        dataset=args.dataset,
        min_text_chars=args.min_text_chars,
        limit_per_source=args.limit_per_source,
    )
    print(summary.manifest_path)
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} "
        f"duplicates={summary.duplicate_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_bible_brain_text(args: argparse.Namespace) -> int:
    summary = prepare_bible_brain_text(
        args.manifest,
        args.out,
        languages=args.language,
        split=args.split,
        dataset=args.dataset,
        min_text_chars=args.min_text_chars,
        limit_per_manifest=args.limit_per_manifest,
    )
    print(summary.manifest_path)
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} "
        f"duplicates={summary.duplicate_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_limbu_unicode_text(args: argparse.Namespace) -> int:
    summary = prepare_limbu_unicode_text(
        args.source,
        args.out,
        split=args.split,
        dataset=args.dataset,
        min_text_chars=args.min_text_chars,
        min_limbu_chars=args.min_limbu_chars,
        limit_per_source=args.limit_per_source,
    )
    print(summary.manifest_path)
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} "
        f"duplicates={summary.duplicate_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_prepare_toolkit_parallel_text(args: argparse.Namespace) -> int:
    summary = prepare_toolkit_parallel_text(
        args.source,
        args.out,
        text_fields=args.text_field,
        split=args.split,
        dataset=args.dataset,
        min_text_chars=args.min_text_chars,
        limit_per_source=args.limit_per_source,
        row_id_field=args.row_id_field,
        metadata_fields=args.metadata_field,
        license_status=args.license_status,
        split_policy=args.split_policy,
    )
    print(summary.manifest_path)
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(
        f"passed={summary.passed} samples={summary.sample_count} "
        f"duplicates={summary.duplicate_count} warnings={len(summary.warnings)}"
    )
    return 0 if summary.passed else 1


def cmd_audit_synthesis_text_promotion(args: argparse.Namespace) -> int:
    audit = audit_synthesis_text_promotion(
        args.manifest,
        args.out,
        exclude_paths=args.exclude,
        require_reviewed_license=args.require_reviewed_license,
    )
    print(audit.output_json_path)
    print(audit.output_md_path)
    print(
        f"passed={audit.passed} samples={audit.sample_count} overlaps={audit.overlap_count} "
        f"issues={len(audit.issues)} warnings={len(audit.warnings)}"
    )
    return 0 if audit.passed else 1


def cmd_split_synthesis_text(args: argparse.Namespace) -> int:
    summary = split_synthesis_text_manifest(
        args.manifest,
        args.out,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        group_by=args.group_by,
        exclude_paths=args.exclude,
        require_reviewed_license=args.require_reviewed_license,
    )
    print(summary.train_manifest)
    print(summary.eval_manifest)
    print(
        f"train={summary.train_count} eval={summary.eval_count} "
        f"train_promotion={summary.train_promotion_passed} eval_promotion={summary.eval_promotion_passed}"
    )
    return 0 if summary.passed else 1


def cmd_audit_synthesis_text_manifest(args: argparse.Namespace) -> int:
    audit = audit_synthesis_text_manifest(
        args.manifest,
        args.out,
        require_languages=args.require_language,
        require_scripts=args.require_script,
        min_samples=args.min_samples,
        allow_claim_evidence=args.allow_claim_evidence,
    )
    print(audit.output_json_path)
    print(audit.output_md_path)
    print(
        f"passed={audit.passed} samples={audit.sample_count} "
        f"claim_eligible={audit.claim_evidence_eligible_count} issues={len(audit.issues)} warnings={len(audit.warnings)}"
    )
    return 0 if audit.passed else 1


def cmd_render_synthesis_text_lines(args: argparse.Namespace) -> int:
    summary = render_synthesis_text_lines(
        args.synthesis_text_manifest,
        args.out,
        font_path=args.font_path,
        limit=args.limit,
        scripts=args.script,
        font_size=args.font_size,
        padding=args.padding,
        degradation_profiles=args.degradation_profile,
        degradation_seed=args.degradation_seed,
        split=args.split,
        dataset=args.dataset,
    )
    print(summary.manifest_path)
    print(summary.label_path)
    print(summary.summary_json_path)
    print(summary.summary_md_path)
    print(f"passed={summary.passed} samples={summary.sample_count} skipped={summary.skipped_count} warnings={len(summary.warnings)}")
    return 0 if summary.passed else 1


def cmd_render_synthesis_text_split(args: argparse.Namespace) -> int:
    summary = render_synthesis_text_split(
        args.train_manifest,
        args.eval_manifest,
        args.out,
        font_path=args.font_path,
        limit_per_split=args.limit_per_split,
        scripts=args.script,
        font_size=args.font_size,
        padding=args.padding,
        degradation_profiles=args.degradation_profile,
        degradation_seed=args.degradation_seed,
        dataset=args.dataset,
        require_font=args.require_font,
        font_readiness_report=args.font_readiness_report,
    )
    print(summary.train_manifest)
    print(summary.train_label_path)
    print(summary.eval_manifest)
    print(summary.eval_label_path)
    print(
        f"passed={summary.passed} train={summary.train_count} eval={summary.eval_count} "
        f"train_audit={summary.train_audit_passed} eval_audit={summary.eval_audit_passed}"
    )
    return 0 if summary.passed else 1


def cmd_audit_rendered_synthesis_lines(args: argparse.Namespace) -> int:
    audit = audit_rendered_synthesis_lines(
        args.manifest,
        args.out,
        label_file=args.label_file,
        require_font=args.require_font,
        font_readiness_report=args.font_readiness_report,
    )
    print(audit.output_json_path)
    print(audit.output_md_path)
    print(
        f"passed={audit.passed} samples={audit.sample_count} "
        f"claim_eligible={audit.claim_evidence_eligible_count} issues={len(audit.issues)} warnings={len(audit.warnings)}"
    )
    return 0 if audit.passed else 1


def cmd_audit_rendered_degradation_split(args: argparse.Namespace) -> int:
    audit = audit_rendered_degradation_split(
        args.train_manifest,
        args.eval_manifest,
        args.out,
        expected_profiles=args.expected_profile,
    )
    print(audit.output_json_path)
    print(audit.output_md_path)
    print(
        f"passed={audit.passed} train={audit.train_count} eval={audit.eval_count} "
        f"train_texts={audit.train_text_hash_count} eval_texts={audit.eval_text_hash_count} "
        f"issues={len(audit.issues)} warnings={len(audit.warnings)}"
    )
    return 0 if audit.passed else 1


def cmd_audit_font_renderability(args: argparse.Namespace) -> int:
    audit = audit_font_renderability(
        args.registry,
        args.out,
        primary_only=not args.all_scripts,
        min_coverage_ratio=args.min_coverage_ratio,
        render_samples=args.render_samples,
    )
    print(Path(args.out) / "font-renderability-audit.json")
    print(Path(args.out) / "font-renderability-audit.md")
    print(
        "passed="
        f"{audit.passed} scripts={audit.script_count} candidates={audit.candidate_count} "
        f"rendered={sum(1 for candidate in audit.candidates if candidate.render_status == 'rendered')}"
    )
    return 0 if audit.passed else 1


def cmd_inventory_fonts(args: argparse.Namespace) -> int:
    audit = inventory_fonts(
        args.registry,
        args.root,
        args.out,
        min_script_coverage_ratio=args.min_script_coverage_ratio,
    )
    print(Path(args.out) / "font-inventory.json")
    print(Path(args.out) / "font-inventory.csv")
    print(Path(args.out) / "font-inventory.md")
    print(
        f"passed={audit.passed} fonts={audit.font_count} readable={audit.readable_font_count} "
        f"archives={audit.archive_count} duplicates={audit.duplicate_sha_count} warnings={len(audit.warnings)}"
    )
    return 0 if audit.passed else 1


def cmd_prepare_font_assets(args: argparse.Namespace) -> int:
    summary = prepare_font_assets(
        args.manifest,
        asset_root=args.asset_root,
        out_dir=args.out,
        force=args.force,
        timeout_seconds=args.timeout_seconds,
    )
    print(Path(args.out) / "font-assets-preparation.json")
    print(Path(args.out) / "font-assets-preparation.md")
    print(
        f"passed={summary.passed} assets={summary.asset_count} downloaded={summary.downloaded_count} "
        f"reused={summary.reused_count} failed={summary.failed_count}"
    )
    return 0 if summary.passed else 1


def cmd_audit_font_asset_readiness(args: argparse.Namespace) -> int:
    report = audit_font_asset_readiness(
        args.manifest,
        args.preparation_report,
        args.inventory_report,
        args.renderability_report,
        args.out,
        asset_root=args.asset_root,
    )
    print(Path(args.out) / "font-asset-readiness.json")
    print(Path(args.out) / "font-asset-readiness.md")
    print(
        f"passed={report.passed} assets={report.asset_count} "
        f"checked={report.checked_file_count} issues={len(report.issues)}"
    )
    return 0 if report.passed else 1


def cmd_audit_paddle_dictionary(args: argparse.Namespace) -> int:
    audit = audit_paddle_dictionary(
        args.dictionary,
        args.out,
        required_ranges=args.required_range,
        label_file_paths=args.label_file,
        allow_space_char=not args.no_allow_space_char,
    )
    print(Path(args.out) / "paddle-dictionary-audit.json")
    print(Path(args.out) / "paddle-dictionary-audit.md")
    print(
        "passed="
        f"{audit.passed} chars={audit.character_count} "
        f"missing_required={len(audit.missing_required_characters)} "
        f"missing_labels={len(audit.missing_label_characters)}"
    )
    return 0 if audit.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    setattr(args, "_raw_argv", raw_argv)
    try:
        return int(args.func(args))
    except OcrTechError as exc:
        print(f"ocrtech: error: {exc}", file=sys.stderr)
        return 2


def _cli_run_context(args: argparse.Namespace) -> dict[str, object]:
    raw_argv = getattr(args, "_raw_argv", None)
    if not isinstance(raw_argv, list):
        raw_argv = []
    return {
        "command": "ocrtech " + " ".join(str(item) for item in raw_argv),
        "argv": [str(item) for item in raw_argv],
    }


def _parse_key_values(values: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in values:
        if "=" not in item:
            raise OcrTechError(f"expected key=value for --paddleocr-kwarg, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise OcrTechError("paddleocr kwarg key cannot be empty")
        parsed[key] = _coerce_scalar(value)
    return parsed


def _parse_image_line_rescue_detector_passes(values: list[str]) -> tuple[dict[str, object], ...]:
    allowed = {
        "threshold",
        "bbox_source",
        "horizontal_kernel",
        "vertical_kernel",
        "dilation_iterations",
        "min_width",
        "min_height",
        "min_area",
        "max_height",
        "min_aspect_ratio",
        "max_aspect_ratio",
        "detector_padding",
        "split_tall_components",
        "split_tall_row_min_ink",
        "split_tall_max_row_gap",
        "split_wide_components",
        "split_wide_col_min_ink",
        "split_wide_max_col_gap",
        "split_wide_min_width",
        "split_detected_row_components",
        "split_detected_row_col_min_ink",
        "split_detected_row_max_col_gap",
        "split_detected_row_min_width",
        "split_detected_row_min_segment_width",
        "split_detected_tall_components",
        "split_detected_tall_row_min_ink",
        "split_detected_tall_max_row_gap",
        "split_detected_tall_min_height",
        "split_detected_tall_min_segment_height",
        "region_min_x_ratio",
        "region_max_x_ratio",
        "region_min_y_ratio",
        "region_max_y_ratio",
    }
    passes: list[dict[str, object]] = []
    for raw_pass in values:
        parsed: dict[str, object] = {}
        for raw_item in raw_pass.split(","):
            item = raw_item.strip()
            if not item:
                continue
            if "=" not in item:
                raise OcrTechError(f"expected key=value in --image-line-rescue-detector-pass, got {item!r}")
            key, value = (part.strip() for part in item.split("=", 1))
            if key not in allowed:
                raise OcrTechError(
                    f"unsupported image-line rescue detector key {key!r}; allowed keys are {sorted(allowed)}"
                )
            if not value:
                raise OcrTechError(f"image-line rescue detector key {key!r} cannot be empty")
            parsed[key] = _coerce_scalar(value)
        if not parsed:
            raise OcrTechError("--image-line-rescue-detector-pass must contain at least one key=value override")
        passes.append(parsed)
    return tuple(passes)


def _parse_labeled_paths(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise OcrTechError(f"expected label=path, got {item!r}")
        label, path = item.split("=", 1)
        if not label or not path:
            raise OcrTechError(f"expected non-empty label=path, got {item!r}")
        parsed[label] = path
    return parsed


def _coerce_scalar(value: str) -> object:
    stripped = value.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_python_version(value: str) -> tuple[int, int]:
    parts = value.strip().split(".")
    if len(parts) != 2:
        raise OcrTechError(f"expected MINOR python version like 3.11, got {value!r}")
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except ValueError as exc:
        raise OcrTechError(f"python version must be numeric, got {value!r}") from exc
    if major < 1 or minor < 0:
        raise OcrTechError(f"invalid python version: {value!r}")
    return major, minor


if __name__ == "__main__":
    raise SystemExit(main())
