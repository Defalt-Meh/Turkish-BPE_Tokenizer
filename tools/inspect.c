/*
 * inspect.c — CLI tool to inspect a trained .tkmodel file.
 *
 * Usage:
 *   ./tools/inspect -m <model.tkmodel> [options]
 *
 * Options:
 *   -m <path>     Model file (required)
 *   -t <int>      Show top N most-merged tokens (default: 50)
 *   -r <int>      Show first N merge rules (default: 20)
 *   -s <string>   Search for tokens containing this byte substring
 *   -a            Show ALL vocab entries (can be very long)
 *   -h            Show help
 */

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -m <model> [options]\n"
        "\n"
        "Inspect a trained .tkmodel file.\n"
        "\n"
        "Options:\n"
        "  -m <path>     Trained .tkmodel file (required)\n"
        "  -t <int>      Show top N tokens beyond byte-level (default: 50)\n"
        "  -r <int>      Show first N merge rules (default: 20)\n"
        "  -s <string>   Search for tokens containing this substring\n"
        "  -a            Dump ALL vocab entries\n"
        "  -h            Show this help\n"
        "\n"
        "Example:\n"
        "  %s -m models/turkish.tkmodel -t 100 -r 30\n"
        "  %s -m models/turkish.tkmodel -s \"ler\"\n",
        prog, prog, prog);
}

/* Print token bytes in a human-readable format */
static void print_token(FILE *out, const uint8_t *bytes, uint16_t len) {
    fputc('"', out);
    for (uint16_t i = 0; i < len; i++) {
        uint8_t b = bytes[i];
        if (b == '"')       fprintf(out, "\\\"");
        else if (b == '\\') fprintf(out, "\\\\");
        else if (b == '\n') fprintf(out, "\\n");
        else if (b == '\r') fprintf(out, "\\r");
        else if (b == '\t') fprintf(out, "\\t");
        else if (b == ' ')  fprintf(out, "\xC2\xB7"); /* middle dot for visibility */
        else if (b >= 0x20 && b < 0x7F) fputc(b, out);
        else fprintf(out, "\\x%02X", b);
    }
    fputc('"', out);
}

/* Check if `needle` appears in `haystack` */
static int contains(const uint8_t *haystack, uint16_t hlen,
                    const uint8_t *needle, size_t nlen) {
    if (nlen > hlen) return 0;
    for (uint16_t i = 0; i <= hlen - nlen; i++) {
        if (memcmp(haystack + i, needle, nlen) == 0)
            return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    int top_n = 50;
    int rule_n = 20;
    const char *search_str = NULL;
    int show_all = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            top_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            rule_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            search_str = argv[++i];
        } else if (strcmp(argv[i], "-a") == 0) {
            show_all = 1;
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

    uint32_t vs = tk_vocab_size(&tk);
    uint32_t nm = tk.vocab.num_merges;

    /* ── Header ──────────────────────────────────────────────────── */
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  Model: %-32s ║\n", model_path);
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Vocab size:    %-8u                 ║\n", vs);
    printf("║  Merge rules:   %-8u                 ║\n", nm);
    printf("║  Byte tokens:   256                      ║\n");
    printf("║  Merged tokens: %-8u                 ║\n", vs > 256 ? vs - 256 : 0);
    printf("╚══════════════════════════════════════════╝\n\n");

    /* ── Token length distribution ───────────────────────────────── */
    printf("── Token length distribution ──\n");
    int len_hist[TK_MAX_TOKEN_LEN + 1];
    memset(len_hist, 0, sizeof(len_hist));

    for (uint32_t i = 0; i < vs; i++) {
        const tk_token_t *t = tk_vocab_get(&tk.vocab, i);
        if (t && t->len <= TK_MAX_TOKEN_LEN)
            len_hist[t->len]++;
    }

    int max_hist = 0;
    for (int i = 1; i <= TK_MAX_TOKEN_LEN; i++) {
        if (len_hist[i] > max_hist) max_hist = len_hist[i];
    }

    for (int i = 1; i <= 20 && i <= TK_MAX_TOKEN_LEN; i++) {
        if (len_hist[i] == 0) continue;
        int bar = (max_hist > 0) ? (len_hist[i] * 40) / max_hist : 0;
        printf("  %2d bytes: %5d  ", i, len_hist[i]);
        for (int j = 0; j < bar; j++) putchar('#');
        putchar('\n');
    }
    printf("\n");

    /* ── Merge rules ─────────────────────────────────────────────── */
    if (rule_n > 0 && nm > 0) {
        uint32_t show = (uint32_t)rule_n;
        if (show > nm) show = nm;

        printf("── First %u merge rules (highest priority) ──\n", show);
        for (uint32_t i = 0; i < show; i++) {
            const tk_merge_t *m = &tk.vocab.merges[i];
            const tk_token_t *lt = tk_vocab_get(&tk.vocab, m->left);
            const tk_token_t *rt = tk_vocab_get(&tk.vocab, m->right);
            const tk_token_t *res = tk_vocab_get(&tk.vocab, m->result);

            printf("  [%4u] ", i);
            if (lt) print_token(stdout, lt->bytes, lt->len);
            else    printf("<%u>", m->left);
            printf(" + ");
            if (rt) print_token(stdout, rt->bytes, rt->len);
            else    printf("<%u>", m->right);
            printf(" -> ");
            if (res) print_token(stdout, res->bytes, res->len);
            else     printf("<%u>", m->result);
            printf("  (id=%u)\n", m->result);
        }
        printf("\n");
    }

    /* ── Top merged tokens ───────────────────────────────────────── */
    if (top_n > 0 && vs > 256) {
        uint32_t show = (uint32_t)top_n;
        if (show > vs - 256) show = vs - 256;

        printf("── First %u merged tokens (by creation order) ──\n", show);
        for (uint32_t i = 0; i < show; i++) {
            uint32_t id = 256 + i;
            const tk_token_t *t = tk_vocab_get(&tk.vocab, id);
            if (!t) break;

            printf("  [%5u] ", id);
            print_token(stdout, t->bytes, t->len);
            printf("  (%u bytes)\n", t->len);
        }
        printf("\n");
    }

    /* ── Search ──────────────────────────────────────────────────── */
    if (search_str) {
        size_t slen = strlen(search_str);
        printf("── Tokens containing \"%s\" ──\n", search_str);

        int found = 0;
        for (uint32_t i = 0; i < vs; i++) {
            const tk_token_t *t = tk_vocab_get(&tk.vocab, i);
            if (!t) continue;

            if (contains(t->bytes, t->len, (const uint8_t *)search_str, slen)) {
                printf("  [%5u] ", t->id);
                print_token(stdout, t->bytes, t->len);
                printf("  (%u bytes)\n", t->len);
                found++;
            }
        }

        if (found == 0) printf("  (none found)\n");
        else            printf("  %d token(s) found.\n", found);
        printf("\n");
    }

    /* ── Full dump ───────────────────────────────────────────────── */
    if (show_all) {
        printf("── All vocab entries ──\n");
        for (uint32_t i = 0; i < vs; i++) {
            const tk_token_t *t = tk_vocab_get(&tk.vocab, i);
            if (!t) continue;

            printf("  [%5u] ", t->id);
            print_token(stdout, t->bytes, t->len);
            printf("\n");
        }
        printf("\n");
    }

    tk_free(&tk);
    return 0;
}