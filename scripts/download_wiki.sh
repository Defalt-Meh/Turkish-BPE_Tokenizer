#!/usr/bin/env bash
#
# download_wiki.sh — Download and extract Turkish Wikipedia as plain text.
#
# Turkish Wikipedia (~1-2GB extracted) is smaller than OSCAR but very
# clean. Useful as a validation set or supplementary training data.
#
# Uses WikiExtractor to convert the XML dump to plain text.
#
# Requirements:
#   - Python 3.8+
#   - wikiextractor (`pip install wikiextractor`)
#   - ~5GB free disk space (dump + extracted)
#
# Usage:
#   ./scripts/download_wiki.sh [output_dir]
#

set -euo pipefail

OUTPUT_DIR="${1:-data}"
DUMP_URL="https://dumps.wikimedia.org/trwiki/latest/trwiki-latest-pages-articles.xml.bz2"
DUMP_FILE="${OUTPUT_DIR}/trwiki-latest-pages-articles.xml.bz2"
EXTRACT_DIR="${OUTPUT_DIR}/wiki_extracted"
FINAL_FILE="${OUTPUT_DIR}/wiki_tr.txt"

echo "═══════════════════════════════════════════"
echo "  Turkish Wikipedia Downloader"
echo "═══════════════════════════════════════════"
echo ""
echo "  URL:    ${DUMP_URL}"
echo "  Output: ${FINAL_FILE}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ── Check dependencies ───────────────────────────────────────────────

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is required."
    exit 1
fi

if ! python3 -c "import wikiextractor" 2>/dev/null; then
    echo "Installing wikiextractor..."
    pip install wikiextractor
fi

# ── Download dump ────────────────────────────────────────────────────

if [ -f "${DUMP_FILE}" ]; then
    echo "Dump file already exists, skipping download."
else
    echo "Downloading Turkish Wikipedia dump..."
    echo "This is ~800MB, may take a while..."
    echo ""

    if command -v wget &>/dev/null; then
        wget -c -O "${DUMP_FILE}" "${DUMP_URL}"
    elif command -v curl &>/dev/null; then
        curl -L -C - -o "${DUMP_FILE}" "${DUMP_URL}"
    else
        echo "ERROR: wget or curl required."
        exit 1
    fi
fi

# ── Extract with WikiExtractor ──────────────────────────────────────

echo ""
echo "Extracting articles with WikiExtractor..."
echo "This processes the XML and strips markup..."
echo ""

python3 -m wikiextractor.WikiExtractor \
    "${DUMP_FILE}" \
    --output "${EXTRACT_DIR}" \
    --bytes 50M \
    --no-templates \
    --min_text_length 50 \
    --processes 4 \
    --quiet

# ── Concatenate into single file ────────────────────────────────────

echo "Concatenating extracted files..."

# WikiExtractor creates AA/wiki_00, AA/wiki_01, AB/wiki_00, etc.
# Each file has <doc> tags we need to strip.

find "${EXTRACT_DIR}" -name 'wiki_*' -type f | sort | while read -r f; do
    # Strip <doc> and </doc> tags, skip empty lines
    sed -e '/<doc /d' -e '/<\/doc>/d' "$f"
done | grep -v '^$' > "${FINAL_FILE}"

# ── Cleanup extracted directory (optional, keep dump for re-extraction) ──

rm -rf "${EXTRACT_DIR}"

# ── Summary ──────────────────────────────────────────────────────────

if [ -f "${FINAL_FILE}" ]; then
    SIZE=$(du -sh "${FINAL_FILE}" | cut -f1)
    LINES=$(wc -l < "${FINAL_FILE}")
    WORDS=$(wc -w < "${FINAL_FILE}")
    echo ""
    echo "═══════════════════════════════════════════"
    echo "  Extraction complete!"
    echo "  File:   ${FINAL_FILE}"
    echo "  Size:   ${SIZE}"
    echo "  Lines:  ${LINES}"
    echo "  Words:  ~${WORDS}"
    echo "═══════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "  # Preprocess:"
    echo "  python3 scripts/preprocess.py ${FINAL_FILE} ${OUTPUT_DIR}/wiki_tr_clean.txt"
    echo ""
    echo "  # Use as validation set alongside OSCAR for training:"
    echo "  ./tools/train -i ${OUTPUT_DIR}/oscar_tr_clean.txt -o models/turkish.tkmodel"
else
    echo "ERROR: extraction failed."
    exit 1
fi