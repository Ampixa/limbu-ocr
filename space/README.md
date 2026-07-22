---
title: Limbu OCR
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# Limbu OCR

Limbu-focused OCR for cropped lines, full page images, and small book/PDF batches.
It includes a Limbu-specific Sirijonga recognizer and a generic Devanagari route
intended for the Devanagari-written Limbu branch. Limbu-specific quality for that
Devanagari branch has not yet been validated.

This Space bundles:

- Sirijonga line recognizer: `models/limbu-sirijonga-ft-v2/`
- Generic Devanagari line route: `models/deva-v2/`
- PaddleOCR full-page text detection for primary page and book OCR.
- Deterministic image-line segmentation as a fallback page/book path.
- Optional YOLO probe: `models/line-detector/best.pt`

The operational page/book path does not depend on YOLO. The primary route uses
PaddleOCR's full-page detector, orders the detected text lines, routes each crop
through the bundled Sirijonga/Devanagari Paddle recognizers, and returns editable
Unicode text with crop and overlay evidence. The Paddle Book OCR tab accepts page
images or PDFs and returns a ZIP with `book.txt`, `book.json`, annotated pages, and
line crops.

This is an operational OCR system for producing text and review evidence. It is not
yet a claim-safe benchmark result. The Sirijonga recognizer has a date-separated
silver diagnostic; the bundled Devanagari model has not yet been validated on a
human-reviewed set of visibly Devanagari-written Limbu pages.

## Local Run

```bash
cd spaces/limbu-ocr
docker build -t limbu-ocr-space .
docker run --rm -p 7860:7860 limbu-ocr-space
```

Open `http://127.0.0.1:7860`.

## Deploy To Hugging Face

This Space has been uploaded to `https://huggingface.co/spaces/ampixa/limbu-ocr`.
See `DEPLOY.md` for the repeatable `uvx` upload command.

```bash
cd spaces/limbu-ocr
git init
git add .
git commit -m "Add Limbu OCR Space"
git remote add origin https://huggingface.co/spaces/ampixa/limbu-ocr
git push origin main
```

Use a CPU Space for basic demos. Upgrade hardware if page OCR latency is too high.
