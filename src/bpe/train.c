/*
 * bpe/train.c — BPE training (morphology-aware) + file-based entry point.
 */
#include "bpe.h"
#include "internal.h"
#include "morphology.h"
#include "unicode.h"
#include "io.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── pre-tokenize callback context ────────────────────────────────── */

typedef struct {
    tk_sequence_t **seqs;
    size_t          num_seqs;
    size_t          seqs_cap;
    size_t          total_bytes;
} pretok_ctx_t;

static void pretok_callback(const uint8_t *start, size_t len, void *ud) {
    pretok_ctx_t *ctx = (pretok_ctx_t *)ud;
    if (len == 0) return;

    if (ctx->num_seqs >= ctx->seqs_cap) {
        size_t new_cap = ctx->seqs_cap * 2;
        if (new_cap < 256) new_cap = 256;
        tk_sequence_t **grown = (tk_sequence_t **)realloc(
            ctx->seqs, new_cap * sizeof(tk_sequence_t *));
        if (!grown) return;
        ctx->seqs     = grown;
        ctx->seqs_cap = new_cap;
    }

    tk_sequence_t *seq = (tk_sequence_t *)malloc(sizeof(tk_sequence_t));
    if (!seq) return;
    if (tk_sequence_init(seq, len) < 0) { free(seq); return; }

    for (size_t i = 0; i < len; i++)
        tk_sequence_append(seq, (uint32_t)start[i]);

    ctx->seqs[ctx->num_seqs++] = seq;
    ctx->total_bytes += len;
}

static void pretok_ctx_free(pretok_ctx_t *ctx) {
    for (size_t i = 0; i < ctx->num_seqs; i++) {
        tk_sequence_free(ctx->seqs[i]);
        free(ctx->seqs[i]);
    }
    free(ctx->seqs);
    memset(ctx, 0, sizeof(*ctx));
}

/* ── BPE training (morphology-aware) ──────────────────────────────── */

int tk_bpe_train(const uint8_t *text, size_t text_len,
                 tk_vocab_t *vocab, const tk_train_config_t *config) {
    tk_vocab_add_byte_tokens(vocab);

    pretok_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.seqs_cap = 2048;
    ctx.seqs = (tk_sequence_t **)malloc(ctx.seqs_cap * sizeof(tk_sequence_t *));
    if (!ctx.seqs) return -1;

    tk_pretokenize(text, text_len, pretok_callback, &ctx);

    if (ctx.num_seqs == 0) {
        pretok_ctx_free(&ctx);
        return 0;
    }

    tk_pair_table_t pt;
    uint32_t pt_size = 1 << 18;
    if (ctx.total_bytes > (1 << 20)) pt_size = 1 << 20;
    if (tk_pair_table_init(&pt, pt_size) < 0) {
        pretok_ctx_free(&ctx);
        return -1;
    }

    uint32_t target = config->target_vocab_size;
    if (target <= 256) target = 256;

    int merge_num = 0;

    while (vocab->vocab_size < target) {
        tk_pair_table_clear(&pt);
        for (size_t i = 0; i < ctx.num_seqs; i++)
            tk_sequence_count_pairs(ctx.seqs[i], &pt);

        scored_pair_t best = find_best_scored(&pt, vocab);
        if (best.score < 0 || best.raw_count < 2) break;

        const tk_token_t *lt = tk_vocab_get(vocab, best.left);
        const tk_token_t *rt = tk_vocab_get(vocab, best.right);
        if (!lt || !rt) break;

        uint16_t new_len = lt->len + rt->len;
        if (new_len > TK_MAX_TOKEN_LEN) break;

        uint8_t new_bytes[TK_MAX_TOKEN_LEN];
        memcpy(new_bytes, lt->bytes, lt->len);
        memcpy(new_bytes + lt->len, rt->bytes, rt->len);

        uint32_t new_id = tk_vocab_add(vocab, new_bytes, new_len);
        if (new_id == (uint32_t)-1) break;

        tk_vocab_add_merge(vocab, best.left, best.right, new_id);

        size_t total_merged = 0;
        for (size_t i = 0; i < ctx.num_seqs; i++)
            total_merged += tk_sequence_apply_merge(
                ctx.seqs[i], best.left, best.right, new_id);

        merge_num++;
        if (config->verbose > 0 && merge_num % (int)config->verbose == 0) {
            byte_span_t ls = { lt->bytes, lt->len };
            byte_span_t rs = { rt->bytes, rt->len };
            bool harmony = pair_respects_vowel_harmony(ls, rs);
            bool suffix  = is_morphological_suffix(rs);

            fprintf(stderr,
                "merge %d: (%u,%u)->%u  cnt=%ld  score=%.1f  "
                "applied=%zu  vocab=%u%s%s\n",
                merge_num, best.left, best.right, new_id,
                (long)best.raw_count, best.score,
                total_merged, vocab->vocab_size,
                harmony ? " [harmony]" : "",
                suffix  ? " [suffix]"  : "");
        }
    }

    tk_pair_table_free(&pt);
    pretok_ctx_free(&ctx);
    return 0;
}

/* ── file-based training ──────────────────────────────────────────── */

int tk_bpe_train_file(const char *path, tk_vocab_t *vocab,
                      const tk_train_config_t *config, size_t chunk_size) {
    tk_mmap_t mmap;
    if (tk_mmap_open(&mmap, path) < 0) return -1;

    size_t train_len  = mmap.size;
    size_t max_train  = (size_t)2 * 1024 * 1024 * 1024;
    if (train_len > max_train) train_len = max_train;

    (void)chunk_size;
    int ret = tk_bpe_train(mmap.data, train_len, vocab, config);

    tk_mmap_close(&mmap);
    return ret;
}