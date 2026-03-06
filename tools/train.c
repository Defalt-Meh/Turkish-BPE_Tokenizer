/*
 * train.c — CLI tool to train a byte-level BPE tokenizer on a Turkish corpus.
 *
 * Usage:
 *   ./tools/train -i <corpus.txt> -o <output.tkmodel> [options]
 *
 * Options:
 *   -i <path>     Input corpus file (UTF-8 text, one doc per line)
 *   -o <path>     Output model file (default: models/turkish.tkmodel)
 *   -v <size>     Target vocabulary size (default: 32000)
 *   -n <flags>    Normalization: w=whitespace, l=lowercase, a=strip-accents
 *   -p <int>      Print progress every N merges (default: 100, 0=quiet)
 *   -h            Show help
 */

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -i <corpus> -o <model> [options]\n"
        "\n"
        "Train a byte-level BPE tokenizer on a UTF-8 text corpus.\n"
        "\n"
        "Options:\n"
        "  -i <path>   Input corpus file (required)\n"
        "  -o <path>   Output .tkmodel file (default: models/turkish.tkmodel)\n"
        "  -v <int>    Target vocabulary size (default: 32000)\n"
        "  -n <flags>  Normalization flags (combinable):\n"
        "                w = collapse whitespace\n"
        "                l = Turkish-aware lowercase\n"
        "                a = strip accents/diacritics\n"
        "              (default: w)\n"
        "  -p <int>    Progress interval in merges (default: 100, 0=quiet)\n"
        "  -h          Show this help\n"
        "\n"
        "Example:\n"
        "  %s -i data/oscar_tr.txt -o models/turkish_32k.tkmodel -v 32000 -n wl\n",
        prog, prog);
}

static unsigned parse_norm_flags(const char *s) {
    unsigned flags = 0;
    for (; *s; s++) {
        switch (*s) {
            case 'w': flags |= TK_NORM_WHITESPACE; break;
            case 'l': flags |= TK_NORM_LOWERCASE; break;
            case 'a': flags |= TK_NORM_STRIP_ACCENTS; break;
            default:
                fprintf(stderr, "warning: unknown normalization flag '%c'\n", *s);
                break;
        }
    }
    return flags;
}

int main(int argc, char **argv) {
    const char *input_path = NULL;
    const char *output_path = "models/turkish.tkmodel";
    uint32_t vocab_size = 32000;
    unsigned norm_flags = TK_NORM_WHITESPACE;
    int verbose = 100;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            vocab_size = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            norm_flags = parse_norm_flags(argv[++i]);
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            verbose = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!input_path) {
        fprintf(stderr, "error: input corpus path required (-i)\n");
        usage(argv[0]);
        return 1;
    }

    if (vocab_size < 257) {
        fprintf(stderr, "error: vocab size must be > 256 (byte tokens)\n");
        return 1;
    }

    /* Configure */
    tk_config_t config = tk_config_default();
    config.vocab_size = vocab_size;
    config.norm_flags = norm_flags;
    config.verbose = verbose;

    /* Initialize */
    tk_tokenizer_t tk;
    if (tk_init(&tk, &config) < 0) {
        fprintf(stderr, "error: failed to initialize tokenizer\n");
        return 1;
    }

    /* Train */
    fprintf(stderr, "Training tokenizer:\n");
    fprintf(stderr, "  corpus:     %s\n", input_path);
    fprintf(stderr, "  vocab size: %u\n", vocab_size);
    fprintf(stderr, "  norm flags: 0x%x\n", norm_flags);
    fprintf(stderr, "\n");

    clock_t start = clock();

    if (tk_train_file(&tk, input_path) < 0) {
        fprintf(stderr, "error: training failed\n");
        tk_free(&tk);
        return 1;
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    fprintf(stderr, "\nTraining completed in %.2f seconds.\n", elapsed);
    tk_print_stats(&tk);

    /* Save */
    fprintf(stderr, "Saving model to: %s\n", output_path);
    if (tk_save(&tk, output_path) < 0) {
        fprintf(stderr, "error: failed to save model\n");
        tk_free(&tk);
        return 1;
    }

    fprintf(stderr, "Done.\n");
    tk_free(&tk);
    return 0;
}