#!/usr/bin/env bash
#
# download_oscar.sh — Download the Turkish subset of the OSCAR corpus.
#
# OSCAR (Open Super-large Crawled Aggregated coRpus) is a large-scale
# multilingual corpus extracted from Common Crawl. The Turkish subset
# is ~30GB of cleaned text, ideal for BPE tokenizer training.
#
# Source: https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
#
# Requirements:
#   - Python 3.8+
#   - huggingface_hub (`pip install huggingface_hub`)
#   - ~35GB free disk space (compressed + extracted)
#   - A HuggingFace account (OSCAR requires agreeing to terms)
#
# Usage:
#   ./scripts/download_oscar.sh [output_dir]
#
# The script downloads, extracts, and concatenates into a single
# text file suitable for tokenizer training.
#

set -euo pipefail

OUTPUT_DIR="${1:-data}"
DATASET="oscar-corpus/OSCAR-2301"
LANGUAGE="tr"
FINAL_FILE="${OUTPUT_DIR}/oscar_tr.txt"

echo "═══════════════════════════════════════════"
echo "  OSCAR Turkish Corpus Downloader"
echo "═══════════════════════════════════════════"
echo ""
echo "  Dataset:  ${DATASET}"
echo "  Language: ${LANGUAGE}"
echo "  Output:   ${FINAL_FILE}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ── Check dependencies ───────────────────────────────────────────────

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is required but not found."
    exit 1
fi

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub datasets
fi

# ── Check HuggingFace authentication ────────────────────────────────

if ! python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo ""
    echo "You need to be logged in to HuggingFace to access OSCAR."
    echo "Run: huggingface-cli login"
    echo ""
    echo "You also need to accept the dataset terms at:"
    echo "  https://huggingface.co/datasets/${DATASET}"
    echo ""
    exit 1
fi

# ── Download using Python/datasets ──────────────────────────────────

echo "Downloading OSCAR Turkish split..."
echo "This may take a while (~30GB)..."
echo ""

python3 << 'PYEOF'
import os
import sys

output_dir = os.environ.get("OUTPUT_DIR", "data")
final_file = os.path.join(output_dir, "oscar_tr.txt")

try:
    from datasets import load_dataset

    print("Loading dataset (streaming mode)...")
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        language="tr",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    print(f"Writing to {final_file}...")
    written = 0
    with open(final_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            text = example.get("text", "").strip()
            if text:
                f.write(text + "\n")
                written += 1

            if (i + 1) % 100000 == 0:
                size_mb = os.path.getsize(final_file) / (1024 * 1024)
                print(f"  {i+1:,} documents processed, {size_mb:.0f} MB written...")

    size_mb = os.path.getsize(final_file) / (1024 * 1024)
    print(f"\nDone! {written:,} documents, {size_mb:.1f} MB total.")

except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    print("\nAlternative: download manually from HuggingFace:", file=sys.stderr)
    print("  https://huggingface.co/datasets/oscar-corpus/OSCAR-2301", file=sys.stderr)
    sys.exit(1)
PYEOF

# ── Summary ──────────────────────────────────────────────────────────

if [ -f "${FINAL_FILE}" ]; then
    SIZE=$(du -sh "${FINAL_FILE}" | cut -f1)
    LINES=$(wc -l < "${FINAL_FILE}")
    echo ""
    echo "═══════════════════════════════════════════"
    echo "  Download complete!"
    echo "  File:  ${FINAL_FILE}"
    echo "  Size:  ${SIZE}"
    echo "  Lines: ${LINES}"
    echo "═══════════════════════════════════════════"
    echo ""
    echo "Next step: preprocess the corpus:"
    echo "  python3 scripts/preprocess.py ${FINAL_FILE} ${OUTPUT_DIR}/oscar_tr_clean.txt"
    echo ""
    echo "Then train the tokenizer:"
    echo "  ./tools/train -i ${OUTPUT_DIR}/oscar_tr_clean.txt -o models/turkish.tkmodel"
else
    echo "ERROR: download failed."
    exit 1
fi