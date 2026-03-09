#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <ctype.h>
#include "tokenizer.h"

long long get_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000 + tv.tv_usec;
}

// Helper to debug exactly where the byte-stream diverged
void analyze_mismatch(const uint8_t *orig, const uint8_t *repro, size_t orig_len, size_t repro_len) {
    printf("\n--- Roundtrip Error Analysis ---\n");
    if (orig_len != repro_len) {
        printf("Length Mismatch: Original %zu bytes vs Decoded %zu bytes\n", orig_len, repro_len);
    }

    size_t min_len = orig_len < repro_len ? orig_len : repro_len;
    for (size_t i = 0; i < min_len; i++) {
        if (orig[i] != repro[i]) {
            printf("First divergence at byte index %zu:\n", i);
            printf("  Original: 0x%02X ('%c')\n", orig[i], isprint(orig[i]) ? orig[i] : '.');
            printf("  Decoded:  0x%02X ('%c')\n", repro[i], isprint(repro[i]) ? repro[i] : '.');
            
            // Show context window of 10 characters around the error
            printf("  Context (Original): ");
            for(int j = -5; j <= 5; j++) {
                int idx = (int)i + j;
                if (idx >= 0 && idx < (int)orig_len) 
                    printf("%c", isprint(orig[idx]) ? orig[idx] : '.');
            }
            printf("\n");
            return;
        }
    }
    if (repro_len > orig_len) {
        printf("Decoded output has %zu extra bytes at the end.\n", repro_len - orig_len);
    }
}

int main() {
    tk_tokenizer_t tk;
    if (tk_load(&tk, "models/trial.tkmodel") != 0) {
        fprintf(stderr, "Error: Could not load models/trial.tkmodel\n");
        return 1;
    }

    // 1. Load data and count words for baseline
    FILE *f = fopen("data/trial.txt", "rb");
    if (!f) { perror("data/trial.txt"); return 1; }
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    uint8_t *data = malloc(len + 1);
    fread(data, 1, len, f);
    data[len] = '\0';
    fclose(f);

    size_t word_count = 0;
    for (size_t i = 0; i < len; i++) if (isspace(data[i])) word_count++;

    // 2. Performance Benchmark
    uint32_t *ids = malloc(len * sizeof(uint32_t));
    long long start = get_us();
    size_t n_tokens = tk_encode(&tk, data, len, ids, len);
    long long end = get_us();

    double ms = (end - start) / 1000.0;
    
    printf("--- Performance & Compression ---\n");
    printf("Input size:      %.2f KB\n", len / 1024.0);
    printf("Latency:         %.2f ms\n", ms);
    printf("Throughput:      %.2f MB/s\n", (len / 1024.0 / 1024.0) / (ms / 1000.0));
    printf("Token Count:     %zu\n", n_tokens);
    printf("Compression Ratio: %.2f (Bytes/Token)\n", (double)len / n_tokens);
    printf("Density:         %.2f (Tokens/Word)\n", (double)n_tokens / (word_count ? word_count : 1));

    // 3. Integrity Check
    uint8_t *decoded = malloc(len * 2); // Over-allocate for safety
    size_t decoded_len = tk_decode(&tk, ids, n_tokens, decoded, len * 2);

    if (decoded_len == len && memcmp(data, decoded, len) == 0) {
        printf("\nRoundtrip:      [PASSED]\n");
    } else {
        printf("\nRoundtrip:      [FAILED]\n");
        analyze_mismatch(data, decoded, len, decoded_len);
    }

    free(data); free(ids); free(decoded);
    tk_free(&tk);
    return 0;
}