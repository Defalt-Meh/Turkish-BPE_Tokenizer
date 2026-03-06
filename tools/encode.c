/*
 * encode.c — CLI tool to encode UTF-8 text into BPE token IDs.
 *
 * Usage:
 *   echo "Merhaba dünya" | ./tools/encode -m <model.tkmodel>
 *   ./tools/encode -m <model.tkmodel> -i <input.txt> -o <output.ids>
 *
 * Options:
 *   -m <path>     Model file (required)
 *   -i <path>     Input text file (default: stdin)
 *   -o <path>     Output file (default: stdout)
 *   -s            Show token strings alongside IDs
 *   -c            Output as comma-separated (default: one ID per line)
 *   -h            Show help
 */

#include "tokenizer.h"
#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -m <model> [-i input] [-o output] [options]\n"
        "\n"
        "Encode UTF-8 text into BPE token IDs.\n"
        "\n"
        "Options:\n"
        "  -m <path>   Trained .tkmodel file (required)\n"
        "  -i <path>   Input text file (default: stdin)\n"
        "  -o <path>   Output file (default: stdout)\n"
        "  -s          Show token byte strings alongside IDs\n"
        "  -c          Comma-separated output (default: one per line)\n"
        "  -h          Show this help\n"
        "\n"
        "Example:\n"
        "  echo 'İstanbul çok güzel bir şehir.' | %s -m models/turkish.tkmodel -s\n",
        prog, prog);
}

/* Print a token's bytes in a human-readable way */
static void print_token_str(FILE *out, const tk_tokenizer_t *tk, uint32_t id) {
    uint16_t len;
    const uint8_t *bytes = tk_token_bytes(tk, id, &len);
    if (!bytes) {
        fprintf(out, "<UNK>");
        return;
    }

    fputc('"', out);
    for (uint16_t i = 0; i < len; i++) {
        uint8_t b = bytes[i];
        if (b == '"') fprintf(out, "\\\"");
        else if (b == '\\') fprintf(out, "\\\\");
        else if (b == '\n') fprintf(out, "\\n");
        else if (b == '\t') fprintf(out, "\\t");
        else if (b >= 0x20 && b < 0x7F) fputc(b, out);
        else fprintf(out, "\\x%02X", b);
    }
    fputc('"', out);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *input_path = NULL;
    const char *output_path = NULL;
    bool show_strings = false;
    bool csv_mode = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0) {
            show_strings = true;
        } else if (strcmp(argv[i], "-c") == 0) {
            csv_mode = true;
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
    uint8_t *text = NULL;
    size_t text_len = 0;

    if (input_path) {
        text = tk_read_file(input_path, &text_len);
        if (!text) {
            fprintf(stderr, "error: failed to read '%s'\n", input_path);
            tk_free(&tk);
            return 1;
        }
    } else {
        /* Read from stdin into a growing buffer */
        size_t cap = 4096;
        text = malloc(cap);
        if (!text) { tk_free(&tk); return 1; }

        int c;
        while ((c = fgetc(stdin)) != EOF) {
            if (text_len >= cap) {
                cap *= 2;
                uint8_t *tmp = realloc(text, cap);
                if (!tmp) { free(text); tk_free(&tk); return 1; }
                text = tmp;
            }
            text[text_len++] = (uint8_t)c;
        }
    }

    /* Encode */
    size_t max_tokens = text_len + 1; /* worst case: 1 byte = 1 token */
    uint32_t *ids = malloc(max_tokens * sizeof(uint32_t));
    if (!ids) {
        free(text);
        tk_free(&tk);
        return 1;
    }

    size_t num_tokens = tk_encode(&tk, text, text_len, ids, max_tokens);
    if (num_tokens == (size_t)-1) {
        fprintf(stderr, "error: encoding failed\n");
        free(ids);
        free(text);
        tk_free(&tk);
        return 1;
    }

    /* Output */
    FILE *out = stdout;
    if (output_path) {
        out = fopen(output_path, "w");
        if (!out) {
            fprintf(stderr, "error: cannot open '%s' for writing\n", output_path);
            free(ids);
            free(text);
            tk_free(&tk);
            return 1;
        }
    }

    for (size_t i = 0; i < num_tokens; i++) {
        if (csv_mode && i > 0) fprintf(out, ",");

        fprintf(out, "%u", ids[i]);

        if (show_strings) {
            fprintf(out, "\t");
            print_token_str(out, &tk, ids[i]);
        }

        if (!csv_mode) fprintf(out, "\n");
    }

    if (csv_mode) fprintf(out, "\n");

    /* Stats to stderr */
    fprintf(stderr, "tokens: %zu  bytes: %zu  ratio: %.2f bytes/token\n",
            num_tokens, text_len,
            num_tokens > 0 ? (double)text_len / num_tokens : 0.0);

    /* Cleanup */
    if (output_path) fclose(out);
    free(ids);
    free(text);
    tk_free(&tk);
    return 0;
}