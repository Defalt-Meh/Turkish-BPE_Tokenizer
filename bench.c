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

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long end = ftell(f);
    if (end < 0) {
        fclose(f);
        return -1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }

    size_t len = (size_t)end;
    uint8_t *buf = (uint8_t *)malloc(len + 1);
    if (!buf) {
        fclose(f);
        return -1;
    }

    if (len > 0 && fread(buf, 1, len, f) != len) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);

    buf[len] = '\0';
    *out_data = buf;
    *out_len = len;
    return 0;
}

static size_t count_words(const uint8_t *data, size_t len) {
    size_t words = 0;
    int in_word = 0;

    for (size_t i = 0; i < len; i++) {
        int ws = isspace((unsigned char)data[i]);
        if (!ws && !in_word) {
            words++;
            in_word = 1;
        } else if (ws) {
            in_word = 0;
        }
    }
    return words;
}

static void print_mismatch(const uint8_t *orig, size_t orig_len,
                           const uint8_t *decoded, size_t decoded_len) {
    size_t min_len = (orig_len < decoded_len) ? orig_len : decoded_len;
    for (size_t i = 0; i < min_len; i++) {
        if (orig[i] != decoded[i]) {
            fprintf(stderr, "Roundtrip mismatch at byte %zu (orig=0x%02X, dec=0x%02X)\n",
                    i, (unsigned)orig[i], (unsigned)decoded[i]);
            return;
        }
    }
    if (orig_len != decoded_len) {
        fprintf(stderr, "Roundtrip length mismatch (orig=%zu, dec=%zu)\n",
                orig_len, decoded_len);
    }
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <model.tkmodel> <input.txt> [iterations]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *input_path = argv[2];
    size_t iterations = 10;

    if (argc == 4) {
        char *endptr = NULL;
        unsigned long it = strtoul(argv[3], &endptr, 10);
        if (!endptr || *endptr != '\0' || it == 0) {
            fprintf(stderr, "Invalid iterations: %s\n", argv[3]);
            return 1;
        }
        iterations = (size_t)it;
    }

    tk_tokenizer_t tk;
    if (tk_load(&tk, model_path) != 0) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    uint8_t *data = NULL;
    size_t len = 0;
    if (read_file(input_path, &data, &len) != 0) {
        fprintf(stderr, "Failed to read input file: %s (%s)\n", input_path, strerror(errno));
        tk_free(&tk);
        return 1;
    }

    if (len == 0) {
        fprintf(stderr, "Input file is empty: %s\n", input_path);
        free(data);
        tk_free(&tk);
        return 1;
    }

    size_t word_count = count_words(data, len);
    uint32_t *ids = (uint32_t *)malloc(len * sizeof(uint32_t));
    uint8_t *decoded = (uint8_t *)malloc(len);
    if (!ids || !decoded) {
        fprintf(stderr, "Out of memory.\n");
        free(decoded);
        free(ids);
        free(data);
        tk_free(&tk);
        return 1;
    }

    size_t n_tokens = 0;
    double start = now_seconds();
    for (size_t i = 0; i < iterations; i++) {
        n_tokens = tk_encode(&tk, data, len, ids, len);
        if (n_tokens == (size_t)-1) {
            fprintf(stderr, "tk_encode failed on iteration %zu\n", i + 1);
            free(decoded);
            free(ids);
            free(data);
            tk_free(&tk);
            return 1;
        }
    }
    double elapsed = now_seconds() - start;

    size_t decoded_len = tk_decode(&tk, ids, n_tokens, decoded, len);
    int roundtrip_ok = (decoded_len == len && memcmp(data, decoded, len) == 0);

    double total_mb = ((double)len * (double)iterations) / (1024.0 * 1024.0);
    double throughput = (elapsed > 0.0) ? (total_mb / elapsed) : 0.0;
    double avg_ms = (elapsed * 1000.0) / (double)iterations;

    printf("Model:            %s\n", model_path);
    printf("Input:            %s\n", input_path);
    printf("Input Size:       %zu bytes (%.2f KB)\n", len, (double)len / 1024.0);
    printf("Iterations:       %zu\n", iterations);
    printf("Tokens:           %zu\n", n_tokens);
    printf("Bytes/Token:      %.4f\n", n_tokens ? (double)len / (double)n_tokens : 0.0);
    printf("Tokens/Word:      %.4f\n", word_count ? (double)n_tokens / (double)word_count : 0.0);
    printf("Total Time:       %.3f ms\n", elapsed * 1000.0);
    printf("Avg Encode Time:  %.3f ms\n", avg_ms);
    printf("Throughput:       %.2f MB/s\n", throughput);
    printf("Roundtrip:        %s\n", roundtrip_ok ? "PASS" : "FAIL");

    if (!roundtrip_ok) {
        print_mismatch(data, len, decoded, decoded_len);
    }

    free(decoded);
    free(ids);
    free(data);
    tk_free(&tk);
    return roundtrip_ok ? 0 : 2;
}
