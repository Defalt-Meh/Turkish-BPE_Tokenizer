/*
 * train.c — Train a byte-level BPE tokenizer on a Turkish corpus.
 *
 * Usage:
 *   ./tools/train -i <corpus.txt> -o <output.tkmodel> [options]
 *
 * Options:
 *   -i <path>     Input corpus file (UTF-8 text)
 *   -o <path>     Output model file (default: models/turkish.tkmodel)
 *   -v <size>     Target vocabulary size (default: 32000)
 *   -c <bytes>    Training chunk size in MiB (default: 64)
 *   -n <flags>    Normalization: w=whitespace, l=lowercase, a=strip-accents
 *   -p <int>      Print progress every N merges (default: 100, 0=quiet)
 *   -h            Show help
 */

#include "tokenizer.h"
#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -i <corpus> -o <model> [options]\n"
        "\n"
        "Options:\n"
        "  -i <path>   Input corpus file (required)\n"
        "  -o <path>   Output .tkmodel file (default: models/turkish.tkmodel)\n"
        "  -v <int>    Vocabulary size (default: 32000, min: 257)\n"
        "  -c <int>    Chunk size in MiB for streaming trainer (default: 64)\n"
        "  -n <flags>  Normalization (combinable): w=whitespace l=lowercase a=strip-accents\n"
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
        case 'w': flags |= TK_NORM_WHITESPACE;    break;
        case 'l': flags |= TK_NORM_LOWERCASE;     break;
        case 'a': flags |= TK_NORM_STRIP_ACCENTS; break;
        default:
            fprintf(stderr, "warning: unknown normalization flag '%c'\n", *s);
            break;
        }
    }
    return flags;
}

static const char *format_size(size_t bytes, char *buf, size_t buf_len) {
    if (bytes >= (size_t)1 << 30)
        snprintf(buf, buf_len, "%.2f GiB", (double)bytes / (1 << 30));
    else if (bytes >= (size_t)1 << 20)
        snprintf(buf, buf_len, "%.2f MiB", (double)bytes / (1 << 20));
    else if (bytes >= (size_t)1 << 10)
        snprintf(buf, buf_len, "%.2f KiB", (double)bytes / (1 << 10));
    else
        snprintf(buf, buf_len, "%zu B", bytes);
    return buf;
}

static double elapsed_sec(struct timespec *t0, struct timespec *t1) {
    return (double)(t1->tv_sec - t0->tv_sec)
         + (double)(t1->tv_nsec - t0->tv_nsec) * 1e-9;
}

int main(int argc, char **argv) {
    const char *input_path  = NULL;
    const char *output_path = "models/turkish.tkmodel";
    uint32_t vocab_size     = 32000;
    unsigned norm_flags     = TK_NORM_WHITESPACE;
    int verbose             = 100;
    size_t chunk_mib        = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            vocab_size = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            chunk_mib = (size_t)atoi(argv[++i]);
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
        fprintf(stderr, "error: -i <corpus> is required\n");
        return 1;
    }

    if (vocab_size < 257) {
        fprintf(stderr, "error: vocab size must be > 256\n");
        return 1;
    }

    /* Verify input file exists and is non-empty. */
    struct stat st;
    if (stat(input_path, &st) < 0) {
        fprintf(stderr, "error: cannot stat '%s'\n", input_path);
        return 1;
    }
    if (st.st_size == 0) {
        fprintf(stderr, "error: corpus file is empty\n");
        return 1;
    }

    char size_buf[32];
    fprintf(stderr, "Training configuration:\n");
    fprintf(stderr, "  corpus     : %s (%s)\n",
            input_path, format_size((size_t)st.st_size, size_buf, sizeof(size_buf)));
    fprintf(stderr, "  output     : %s\n", output_path);
    fprintf(stderr, "  vocab size : %u\n", vocab_size);
    fprintf(stderr, "  chunk size : %zu MiB\n", chunk_mib);
    fprintf(stderr, "  norm flags : 0x%x", norm_flags);
    if (norm_flags & TK_NORM_WHITESPACE)    fprintf(stderr, " [whitespace]");
    if (norm_flags & TK_NORM_LOWERCASE)     fprintf(stderr, " [lowercase]");
    if (norm_flags & TK_NORM_STRIP_ACCENTS) fprintf(stderr, " [strip-accents]");
    fprintf(stderr, "\n  verbose    : %d\n\n", verbose);

    /* Configure and initialize. */
    tk_config_t config = tk_config_default();
    config.vocab_size  = vocab_size;
    config.norm_flags  = norm_flags;
    config.verbose     = (uint32_t)verbose;
    config.chunk_size  = chunk_mib * 1024 * 1024;

    tk_tokenizer_t tk;
    if (tk_init(&tk, &config) < 0) {
        fprintf(stderr, "error: tokenizer init failed\n");
        return 1;
    }

    /* Train. */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int ret = tk_train_file(&tk, input_path);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = elapsed_sec(&t0, &t1);

    if (ret < 0) {
        fprintf(stderr, "error: training failed\n");
        tk_free(&tk);
        return 1;
    }

    double throughput = (double)st.st_size / secs;
    fprintf(stderr, "\nTraining completed in %.2f s (%s/s)\n",
            secs, format_size((size_t)throughput, size_buf, sizeof(size_buf)));
    tk_print_stats(&tk);

    /* Save. */
    fprintf(stderr, "Saving model to %s\n", output_path);
    if (tk_save(&tk, output_path) < 0) {
        fprintf(stderr, "error: save failed\n");
        tk_free(&tk);
        return 1;
    }

    /* Report output size. */
    if (stat(output_path, &st) == 0) {
        fprintf(stderr, "Model size: %s\n",
                format_size((size_t)st.st_size, size_buf, sizeof(size_buf)));
    }

    fprintf(stderr, "Done.\n");
    tk_free(&tk);
    return 0;
}