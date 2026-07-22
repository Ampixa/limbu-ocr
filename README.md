# LimbuOCR — page-to-book OCR for Limbu and the Nepal language pack

LimbuOCR turns scanned Limbu pages, image batches, and PDFs into editable
Unicode text with page and line structure, reading order, and per-line crops
that trace every prediction back to its source. Limbu is written in the
Sirijonga script (U+1900–U+194F); this repository ships the recognizer,
detection and reading-order pipeline, book runner, and evaluation tools behind
the paper below, plus the registry that extends the same pipeline across the
wider Nepal language pack.

**Paper:** [`paper/draft.pdf`](paper/draft.pdf) — *LimbuOCR: An End-to-End
Page-to-Book OCR System for Limbu, and a Pipeline Toward the Nepal Language
Pack* (Ashish Thapa, Ampixa Labs).

**Hosted demo and weights:** the Gradio/Docker app in [`space/`](space/) runs
as a Hugging Face Space (`ampixa/limbu-ocr`); trained model weights are
distributed through Hugging Face rather than this repository.

## Results at a glance

Measured on a date-disjoint test built from real newspaper print, with
machine-converted (silver) references — engineering baselines, not
human-graded accuracy:

| Diagnostic | Result |
|---|---|
| Sirijonga line recognition (902 held-out lines) | 0.474% codepoint CER · 0.979% grapheme CER · 85.9% exact lines |
| Full-page pipeline, non-oracle (3 held-out pages) | 2.575% page CER · 96.85% line-detection F1 |
| Second language (Newar-Devanagari, 1,139 lines) | 4.66% codepoint CER (diagnostic; pending review) |

Page-level CER is only meaningful under a declared serialization: the same
predictions score 41.34% or 4.33% depending on the reading-order contract.
The paper covers this and the per-script status of the wider pack (Tirhuta,
Ol Chiki, Sunuwar, Kirat Rai, Lepcha, Newa, Gurung Khema).

## Layout

```
src/ocrtech/   pipeline library: detection, script routing, recognition,
               reading order, legacy-font converters, export, review artifacts
tools/         book runner and evaluation CLIs
configs/       nepal-language-pack-pipeline-v1.json — 50-language registry
space/         the deployable OCR app (Docker + Gradio); weights on HF
paper/         the paper (PDF, LaTeX source, figures)
```

## Quickstart

Python 3.11+.

```bash
pip install -e .            # core library (no heavy deps)
pip install -e ".[engines]" # + PaddleOCR runtime for actual recognition
```

Run OCR over a PDF or a directory of page images:

```bash
python tools/run_book_ocr.py <book.pdf|pages/> \
  --out out/my-book \
  --config configs/nepal-language-pack-pipeline-v1.json \
  --language limbu
```

Outputs: per-page `document.json`, `book.txt`, `book.md`, a review queue of
low-confidence lines, and a run summary with input/output hashes.

Score a recognizer against a labeled line pack:

```bash
python tools/eval_metrics.py --help
python tools/run_page_ocr_eval.py --help
```

The registry config maps each language to its script, recognizer, converter,
and status; routes whose data or rights review is incomplete are refused at
runtime rather than silently executed.

## Data and rights

No publisher page images or crop packs are included. Training and evaluation
labels are silver (machine-converted from legacy-font PDF text layers), and the
paper states per script what may be claimed from them. The legacy-font
converter maps under `src/ocrtech/maps/` are project-created mappings grounded
in the relevant Unicode proposals.

## Citation

```bibtex
@misc{thapa2026limbuocr,
  title  = {LimbuOCR: An End-to-End Page-to-Book OCR System for Limbu,
            and a Pipeline Toward the Nepal Language Pack},
  author = {Thapa, Ashish},
  year   = {2026},
  note   = {Ampixa Labs},
  url    = {https://github.com/ampixa/limbu-ocr}
}
```

## License

Apache-2.0. Model weights and datasets distributed via Hugging Face carry
their own licenses noted in their model cards.
