/*
 * decode.c — Decode BPE token IDs back into UTF-8 text.
 *
 * Usage:
 *   echo "312 45 1023 88" | ./tools/decode -m <model.tkmodel>
 *   ./tools/decode -m <model.tkmodel> -i <input.ids> -o <output.txt>
 *
 * Options:
 *   -m <path>     Model file (required)
 *   -i <path>     Input IDs file (default: stdin)
 *   -o <path>     Output text file (default: stdout)
 *   -h            Show help
 *
 * Input format: whitespace or comma-separated integer token IDs.
 */

#include "tokenizer.h"
#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -m <model> [-i input.ids] [-o output.txt]\n"
        "\n"
        "Options:\n"
        "  -m <path>   Trained .tkmodel file (required)\n"
        "  -i <path>   Input file with token IDs (default: stdin)\n"
        "  -o <path>   Output text file (default: stdout)\n"
        "  -h          Show this help\n"
        "\n"
        "Input: whitespace or comma-separated integers.\n"
        "Example:\n"
        "  echo '312 45 1023 88' | %s -m models/turkish.tkmodel\n",
        prog, prog);
}

/* Read all of stdin into a heap buffer. */
static char *read_stdin(size_t *out_len) {
    size_t cap = 8192;
    size_t len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;

    for (;;) {
        size_t n = fread(buf + len, 1, cap - len, stdin);
        len += n;
        if (n == 0) break;
        if (len == cap) {
            cap *= 2;
            char *grown = (char *)realloc(buf, cap);
            if (!grown) { free(buf); return NULL; }
            buf = grown;
        }
    }

    *out_len = len;
    return buf;
}

/* Parse whitespace/comma-separated unsigned integers from text. */
static uint32_t *parse_ids(const char *text, size_t text_len,
                           size_t *out_count) {
    /* Upper bound: at most one ID per two characters ("N "). */
    size_t cap = (text_len / 2) + 16;
    if (cap < 64) cap = 64;

    uint32_t *ids = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!ids) return NULL;

    size_t count = 0;
    const char *p   = text;
    const char *end = text + text_len;

    while (p < end) {
        while (p < end && (isspace((unsigned char)*p) || *p == ','))
            p++;
        if (p >= end) break;

        char *endptr;
        unsigned long val = strtoul(p, &endptr, 10);
        if (endptr == p) { p++; continue; }

        if (count >= cap) {
            cap *= 2;
            uint32_t *grown = (uint32_t *)realloc(ids, cap * sizeof(uint32_t));
            if (!grown) { free(ids); return NULL; }
            ids = grown;
        }

        ids[count++] = (uint32_t)val;
        p = endptr;
    }

    *out_count = count;
    return ids;
}

int main(int argc, char **argv) {
    const char *model_path  = NULL;
    const char *input_path  = NULL;
    const char *output_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "error: -m <model> is required\n");
        return 1;
    }

    /* Load model. */
    tk_tokenizer_t tk;
    if (tk_load(&tk, model_path) < 0) {
        fprintf(stderr, "error: failed to load '%s'\n", model_path);
        return 1;
    }

    /* Read input: mmap if file, buffered read if stdin. */
    char *text      = NULL;
    size_t text_len = 0;
    tk_mmap_t mmap_handle;
    int used_mmap = 0;

    if (input_path) {
        if (tk_mmap_open(&mmap_handle, input_path) == 0) {
            text     = (char *)mmap_handle.data;
            text_len = mmap_handle.size;
            used_mmap = 1;
        } else {
            fprintf(stderr, "error: cannot open '%s'\n", input_path);
            tk_free(&tk);
            return 1;
        }
    } else {
        text = read_stdin(&text_len);
        if (!text) {
            fprintf(stderr, "error: failed to read stdin\n");
            tk_free(&tk);
            return 1;
        }
    }

    /* Parse token IDs. */
    size_t num_ids;
    uint32_t *ids = parse_ids(text, text_len, &num_ids);

    if (used_mmap) tk_mmap_close(&mmap_handle);
    else free(text);
    text = NULL;

    if (!ids || num_ids == 0) {
        if (!ids) fprintf(stderr, "error: failed to parse IDs\n");
        else      fprintf(stderr, "warning: no token IDs in input\n");
        free(ids);
        tk_free(&tk);
        return ids ? 0 : 1;
    }

    /* Decode. Output bound: each token is at most TK_MAX_TOKEN_LEN bytes. */
    size_t out_cap = num_ids * TK_MAX_TOKEN_LEN;
    uint8_t *out_buf = (uint8_t *)malloc(out_cap);
    if (!out_buf) {
        fprintf(stderr, "error: allocation failed (%zu bytes)\n", out_cap);
        free(ids);
        tk_free(&tk);
        return 1;
    }

    size_t out_len = tk_decode(&tk, ids, num_ids, out_buf, out_cap);
    free(ids);

    if (out_len == (size_t)-1) {
        fprintf(stderr, "error: decode failed (invalid token ID?)\n");
        free(out_buf);
        tk_free(&tk);
        return 1;
    }

    /* Write output. */
    FILE *out = stdout;
    if (output_path) {
        out = fopen(output_path, "wb");
        if (!out) {
            fprintf(stderr, "error: cannot open '%s' for writing\n", output_path);
            free(out_buf);
            tk_free(&tk);
            return 1;
        }
    }

    fwrite(out_buf, 1, out_len, out);
    if (out_len > 0 && out_buf[out_len - 1] != '\n')
        fputc('\n', out);

    if (output_path) fclose(out);

    fprintf(stderr, "decoded %zu tokens -> %zu bytes\n", num_ids, out_len);

    free(out_buf);
    tk_free(&tk);
    return 0;
}