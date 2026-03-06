/*
 * decode.c — CLI tool to decode BPE token IDs back into UTF-8 text.
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
        "Decode BPE token IDs back into UTF-8 text.\n"
        "\n"
        "Options:\n"
        "  -m <path>   Trained .tkmodel file (required)\n"
        "  -i <path>   Input file with token IDs (default: stdin)\n"
        "  -o <path>   Output text file (default: stdout)\n"
        "  -h          Show this help\n"
        "\n"
        "Input format: whitespace or comma-separated integers.\n"
        "Example:\n"
        "  echo '312 45 1023 88' | %s -m models/turkish.tkmodel\n",
        prog, prog);
}

/* Parse a stream of whitespace/comma-separated integers */
static uint32_t *parse_ids(const char *text, size_t text_len, size_t *out_count) {
    size_t cap = 1024;
    uint32_t *ids = malloc(cap * sizeof(uint32_t));
    if (!ids) return NULL;

    size_t count = 0;
    const char *p = text;
    const char *end = text + text_len;

    while (p < end) {
        /* Skip whitespace, commas, newlines */
        while (p < end && (isspace((unsigned char)*p) || *p == ','))
            p++;

        if (p >= end) break;

        /* Parse an integer */
        char *endptr;
        unsigned long val = strtoul(p, &endptr, 10);

        if (endptr == p) {
            /* Not a valid number — skip this character */
            p++;
            continue;
        }

        if (count >= cap) {
            cap *= 2;
            uint32_t *tmp = realloc(ids, cap * sizeof(uint32_t));
            if (!tmp) { free(ids); return NULL; }
            ids = tmp;
        }

        ids[count++] = (uint32_t)val;
        p = endptr;
    }

    *out_count = count;
    return ids;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *input_path = NULL;
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
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "error: model path required (-m)\n");
        usage(argv[0]);
        return 1;
    }

    /* Load model */
    tk_tokenizer_t tk;
    if (tk_load(&tk, model_path) < 0) {
        fprintf(stderr, "error: failed to load model from '%s'\n", model_path);
        return 1;
    }

    /* Read input */
    char *text = NULL;
    size_t text_len = 0;

    if (input_path) {
        text = (char *)tk_read_file(input_path, &text_len);
        if (!text) {
            fprintf(stderr, "error: failed to read '%s'\n", input_path);
            tk_free(&tk);
            return 1;
        }
    } else {
        /* Read from stdin */
        size_t cap = 4096;
        text = malloc(cap);
        if (!text) { tk_free(&tk); return 1; }

        int c;
        while ((c = fgetc(stdin)) != EOF) {
            if (text_len >= cap) {
                cap *= 2;
                char *tmp = realloc(text, cap);
                if (!tmp) { free(text); tk_free(&tk); return 1; }
                text = tmp;
            }
            text[text_len++] = (char)c;
        }
    }

    /* Parse token IDs */
    size_t num_ids;
    uint32_t *ids = parse_ids(text, text_len, &num_ids);
    free(text);

    if (!ids) {
        fprintf(stderr, "error: failed to parse token IDs\n");
        tk_free(&tk);
        return 1;
    }

    if (num_ids == 0) {
        fprintf(stderr, "warning: no token IDs found in input\n");
        free(ids);
        tk_free(&tk);
        return 0;
    }

    /* Decode — worst case each token is TK_MAX_TOKEN_LEN bytes */
    size_t out_cap = num_ids * TK_MAX_TOKEN_LEN;
    uint8_t *out_buf = malloc(out_cap);
    if (!out_buf) {
        free(ids);
        tk_free(&tk);
        return 1;
    }

    size_t out_len = tk_decode(&tk, ids, num_ids, out_buf, out_cap);
    if (out_len == (size_t)-1) {
        fprintf(stderr, "error: decoding failed (invalid token ID?)\n");
        free(out_buf);
        free(ids);
        tk_free(&tk);
        return 1;
    }

    /* Output */
    FILE *out = stdout;
    if (output_path) {
        out = fopen(output_path, "wb");
        if (!out) {
            fprintf(stderr, "error: cannot open '%s' for writing\n", output_path);
            free(out_buf);
            free(ids);
            tk_free(&tk);
            return 1;
        }
    }

    fwrite(out_buf, 1, out_len, out);

    /* Ensure trailing newline for terminal readability */
    if (out_len > 0 && out_buf[out_len - 1] != '\n') {
        fputc('\n', out);
    }

    fprintf(stderr, "decoded %zu tokens -> %zu bytes\n", num_ids, out_len);

    if (output_path) fclose(out);
    free(out_buf);
    free(ids);
    tk_free(&tk);
    return 0;
}