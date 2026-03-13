// ./bench <model.tkmodel> <input.txt> [iterations]
// ./bench models/trial.tkmodel data/trial.txt
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tokenizer.h"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int read_file(const char *path, uint8_t **out_data, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long end = ftell(f);
    if (end < 0)                      { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0)  { fclose(f); return -1; }

    size_t len = (size_t)end;
    uint8_t *buf = (uint8_t *)malloc(len + 1);
    if (!buf) { fclose(f); return -1; }

    if (len > 0 && fread(buf, 1, len, f) != len) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);
    buf[len] = '\0';
    *out_data = buf;
    *out_len  = len;
    return 0;
}

static size_t count_words(const uint8_t *data, size_t len) {
    size_t words  = 0;
    int    in_word = 0;
    for (size_t i = 0; i < len; i++) {
        int ws = isspace((unsigned char)data[i]);
        if (!ws && !in_word) { words++; in_word = 1; }
        else if (ws)          { in_word = 0; }
    }
    return words;
}

/* Escape-print a window of bytes around `pos`. The byte at pos is bracketed. */
static void print_context(FILE *out, const uint8_t *buf, size_t len,
                           size_t pos, size_t window) {
    size_t start = (pos >= window) ? pos - window : 0;
    size_t end   = (pos + window < len) ? pos + window : len;
    for (size_t i = start; i < end; i++) {
        if (i == pos) fputc('[', out);
        uint8_t b = buf[i];
        if (b >= 0x20 && b < 0x7F) fputc(b, out);
        else                        fprintf(out, "\\x%02X", b);
        if (i == pos) fputc(']', out);
    }
}

static void diagnose_roundtrip(const tk_tokenizer_t *tk,
                                const uint8_t *orig,    size_t orig_len,
                                const uint8_t *decoded, size_t decoded_len,
                                const uint32_t *ids,    size_t n_tokens) {
    fprintf(stderr, "\n--- Roundtrip Failure Diagnosis ---\n");
    fprintf(stderr, "  Original  : %zu bytes\n", orig_len);
    fprintf(stderr, "  Decoded   : %zu bytes\n", decoded_len);

    /* Find first byte mismatch */
    size_t min_len   = (orig_len < decoded_len) ? orig_len : decoded_len;
    size_t mismatch  = min_len; /* default: length differs, no byte diff found */
    for (size_t i = 0; i < min_len; i++) {
        if (orig[i] != decoded[i]) { mismatch = i; break; }
    }

    if (mismatch < min_len) {
        fprintf(stderr, "  First mismatch at byte %zu:\n", mismatch);
        fprintf(stderr, "    orig[%zu]    = 0x%02X\n", mismatch, (unsigned)orig[mismatch]);
        fprintf(stderr, "    decoded[%zu] = 0x%02X\n", mismatch, (unsigned)decoded[mismatch]);
        fprintf(stderr, "  Context (orig)   : ...");
        print_context(stderr, orig,    orig_len,    mismatch, 20);
        fprintf(stderr, "...\n");
        fprintf(stderr, "  Context (decoded): ...");
        print_context(stderr, decoded, decoded_len, mismatch, 20);
        fprintf(stderr, "...\n");

        /* Walk decoded token stream to find the token that covers `mismatch` */
        size_t byte_cursor = 0;
        for (size_t i = 0; i < n_tokens && byte_cursor <= mismatch; i++) {
            uint16_t       tlen   = 0;
            const uint8_t *tbytes = tk_token_bytes(tk, ids[i], &tlen);
            if (!tbytes) { byte_cursor++; continue; }
            if (byte_cursor + tlen > mismatch) {
                fprintf(stderr, "  Token covering mismatch: id=%-6u  \"", ids[i]);
                for (uint16_t j = 0; j < tlen; j++) {
                    uint8_t b = tbytes[j];
                    if (b >= 0x20 && b < 0x7F) fputc(b, stderr);
                    else                        fprintf(stderr, "\\x%02X", b);
                }
                fprintf(stderr, "\"  (%u B)\n", tlen);
                break;
            }
            byte_cursor += tlen;
        }
    } else {
        fprintf(stderr, "  No byte-level mismatch; lengths differ.\n");
    }

    fprintf(stderr, "\n  Most likely cause: normalization is active.\n");
    fprintf(stderr, "  tk_encode normalizes the input (e.g. collapse whitespace,\n");
    fprintf(stderr, "  turkish lowercase), so tk_decode reproduces the *normalized*\n");
    fprintf(stderr, "  text, not the original raw bytes.\n");
    fprintf(stderr, "  Roundtrip is only lossless when norm_flags == 0.\n");
    fprintf(stderr, "-----------------------------------\n");
}

static void print_norm_flags(uint32_t flags) {
    if (flags == 0) { printf("none"); return; }
    const char *sep = "";
    if (flags & TK_NORM_LOWERCASE)     { printf("%sLOWERCASE",     sep); sep = "|"; }
    if (flags & TK_NORM_NFC)           { printf("%sNFC",            sep); sep = "|"; }
    if (flags & TK_NORM_STRIP_ACCENTS) { printf("%sSTRIP_ACCENTS",  sep); sep = "|"; }
    if (flags & TK_NORM_WHITESPACE)    { printf("%sWHITESPACE",     sep);            }
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <model.tkmodel> <input.txt> [iterations]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *input_path = argv[2];
    size_t      iterations = 10;

    if (argc == 4) {
        char *endptr = NULL;
        unsigned long it = strtoul(argv[3], &endptr, 10);
        if (!endptr || *endptr != '\0' || it == 0) {
            fprintf(stderr, "Invalid iterations: %s\n", argv[3]);
            return 1;
        }
        iterations = (size_t)it;
    }

    /* ── Load model ────────────────────────────────────────────────── */
    tk_tokenizer_t tk;
    if (tk_load(&tk, model_path) != 0) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    /* ── Model stats (printed to stderr by tk_print_stats) ─────────── */
    tk_print_stats(&tk);

    /* ── Load input ────────────────────────────────────────────────── */
    uint8_t *data = NULL;
    size_t   len  = 0;
    if (read_file(input_path, &data, &len) != 0) {
        fprintf(stderr, "Failed to read input file: %s (%s)\n",
                input_path, strerror(errno));
        tk_free(&tk);
        return 1;
    }

    if (len == 0) {
        fprintf(stderr, "Input file is empty: %s\n", input_path);
        free(data); tk_free(&tk);
        return 1;
    }

    /* ── Input analysis ─────────────────────────────────────────────── */
    size_t word_count = count_words(data, len);
    size_t codepoints = utf8_strlen(data, len);
    bool   utf8_ok    = utf8_validate(data, len);

    printf("\n=== Input ===\n");
    printf("Path           : %s\n", input_path);
    printf("Size           : %zu bytes (%.2f KB)\n", len, (double)len / 1024.0);
    printf("Codepoints     : %zu\n", codepoints);
    printf("Words          : %zu\n", word_count);
    printf("UTF-8 valid    : %s\n", utf8_ok ? "yes" : "NO (malformed — results may be wrong)");

    /* ── Active config ────────���─────────────────────────────────────── */
    printf("\n=== Config ===\n");
    printf("Vocab size     : %u\n",  tk_vocab_size_of(&tk));
    printf("Merge rules    : %u\n",  tk.vocab.num_merges);
    printf("Pretokenize    : %s\n",  tk.config.pretokenize ? "yes" : "no");
    printf("Norm flags     : ");     print_norm_flags(tk.config.norm_flags); printf("\n");
    if (tk.config.norm_flags != 0) {
        printf("  *** Normalization is active — roundtrip will NOT be lossless ***\n");
    }
    printf("Iterations     : %zu  (+3 warmup)\n", iterations);

    /* ── Buffers ────────────────────────────────────────────────────── */
    uint32_t *ids     = (uint32_t *)malloc(len * sizeof(uint32_t));
    uint8_t  *decoded = (uint8_t  *)malloc(len + 1);
    if (!ids || !decoded) {
        fprintf(stderr, "Out of memory.\n");
        free(decoded); free(ids); free(data); tk_free(&tk);
        return 1;
    }

    /* ── Encode warmup ──────────────────────────────────────────────── */
    size_t n_tokens = 0;
    for (size_t i = 0; i < 3; i++) {
        n_tokens = tk_encode(&tk, data, len, ids, len);
        if (n_tokens == (size_t)-1) {
            fprintf(stderr, "tk_encode failed during warmup\n");
            free(decoded); free(ids); free(data); tk_free(&tk);
            return 1;
        }
    }

    /* ── Encode benchmark ───────────────────────────────────────────── */
    double enc_start = now_seconds();
    for (size_t i = 0; i < iterations; i++) {
        n_tokens = tk_encode(&tk, data, len, ids, len);
        if (n_tokens == (size_t)-1) {
            fprintf(stderr, "tk_encode failed on iteration %zu\n", i + 1);
            free(decoded); free(ids); free(data); tk_free(&tk);
            return 1;
        }
    }
    double enc_elapsed = now_seconds() - enc_start;

    /* ── Decode benchmark ───────────────────────────────────────────── */
    size_t decoded_len = 0;
    double dec_start = now_seconds();
    for (size_t i = 0; i < iterations; i++) {
        decoded_len = tk_decode(&tk, ids, n_tokens, decoded, len);
        if (decoded_len == (size_t)-1) {
            fprintf(stderr, "tk_decode failed on iteration %zu\n", i + 1);
            free(decoded); free(ids); free(data); tk_free(&tk);
            return 1;
        }
    }
    double dec_elapsed = now_seconds() - dec_start;

    /* ── Roundtrip check ────────────────────────────────────────────── */
    int roundtrip_ok = (decoded_len == len && memcmp(data, decoded, len) == 0);

    /* ── Derived metrics ────────────────────────────────────────────── */
    double total_mb      = ((double)len * (double)iterations) / (1024.0 * 1024.0);
    double enc_mb_s      = (enc_elapsed > 0.0) ? total_mb / enc_elapsed : 0.0;
    double enc_avg_ms    = (enc_elapsed * 1000.0) / (double)iterations;
    double enc_ktok_s    = (enc_elapsed > 0.0)
                           ? ((double)n_tokens * (double)iterations) / enc_elapsed / 1000.0
                           : 0.0;
    double dec_mb_s      = (dec_elapsed > 0.0) ? total_mb / dec_elapsed : 0.0;
    double dec_avg_ms    = (dec_elapsed * 1000.0) / (double)iterations;
    double bytes_per_tok = n_tokens ? (double)len / (double)n_tokens : 0.0;
    double tok_per_word  = word_count ? (double)n_tokens / (double)word_count : 0.0;
    double tok_per_cp    = codepoints ? (double)n_tokens / (double)codepoints : 0.0;

    /* ── Print results ──────────────────────────────────────────────── */
    printf("\n=== Tokens ===\n");
    printf("Count          : %zu\n",   n_tokens);
    printf("Bytes/token    : %.4f\n",  bytes_per_tok);
    printf("Tokens/word    : %.4f\n",  tok_per_word);
    printf("Tokens/cp      : %.4f\n",  tok_per_cp);

    printf("\n=== Encode ===\n");
    printf("Total time     : %.3f ms\n",  enc_elapsed * 1000.0);
    printf("Avg per call   : %.3f ms\n",  enc_avg_ms);
    printf("Throughput     : %.2f MB/s\n", enc_mb_s);
    printf("Tokens/sec     : %.1f k\n",    enc_ktok_s);

    printf("\n=== Decode ===\n");
    printf("Total time     : %.3f ms\n",  dec_elapsed * 1000.0);
    printf("Avg per call   : %.3f ms\n",  dec_avg_ms);
    printf("Throughput     : %.2f MB/s\n", dec_mb_s);
    printf("Decoded size   : %zu bytes\n", decoded_len);

    printf("\n=== Roundtrip ===\n");
    printf("Result         : %s\n", roundtrip_ok ? "PASS" : "FAIL");
    if (!roundtrip_ok) {
        diagnose_roundtrip(&tk, data, len, decoded, decoded_len, ids, n_tokens);
    }

    /* ── First 20 tokens ────────────────────────────────────────────── */
    printf("\n=== First 20 tokens ===\n");
    size_t show = (n_tokens < 20) ? n_tokens : 20;
    for (size_t i = 0; i < show; i++) {
        uint16_t       tlen   = 0;
        const uint8_t *tbytes = tk_token_bytes(&tk, ids[i], &tlen);
        printf("  [%3zu] id=%-6u  \"", i, ids[i]);
        if (tbytes) {
            for (uint16_t j = 0; j < tlen; j++) {
                uint8_t b = tbytes[j];
                if (b >= 0x20 && b < 0x7F) fputc(b, stdout);
                else                        printf("\\x%02X", b);
            }
        }
        printf("\"  (%u B)\n", tlen);
    }

    free(decoded);
    free(ids);
    free(data);
    tk_free(&tk);
    return roundtrip_ok ? 0 : 2;
}
