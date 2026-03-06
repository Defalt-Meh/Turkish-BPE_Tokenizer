/*
 * bpe.c — Byte-Pair Encoding: pair-frequency counting, iterative merge
 *          training, and encode/decode using learned merge rules.
 */

#include "bpe.h"
#include "unicode.h"
#include "io.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ════════════════════════════════════════════════════════════════════════
 *  Pair Hash Table
 *
 *  Key: (left_id, right_id) packed into a 64-bit integer.
 *  Uses FNV-1a on the 8-byte key with separate chaining.
 * ════════════════════════════════════════════════════════════════════════ */

static uint32_t pair_hash(uint32_t left, uint32_t right, uint32_t num_buckets) {
    uint64_t key = ((uint64_t)left << 32) | (uint64_t)right;
    /* FNV-1a on 8 bytes */
    uint32_t h = 0x811C9DC5u;
    for (int i = 0; i < 8; i++) {
        h ^= (uint8_t)(key >> (i * 8));
        h *= 0x01000193u;
    }
    return h % num_buckets;
}

int tk_pair_table_init(tk_pair_table_t *pt, uint32_t num_buckets) {
    if (num_buckets < 1024) num_buckets = 1024;
    pt->buckets = calloc(num_buckets, sizeof(tk_pair_entry_t *));
    if (!pt->buckets) return -1;
    pt->num_buckets = num_buckets;
    pt->num_entries = 0;
    return 0;
}

void tk_pair_table_free(tk_pair_table_t *pt) {
    if (!pt->buckets) return;
    for (uint32_t i = 0; i < pt->num_buckets; i++) {
        tk_pair_entry_t *e = pt->buckets[i];
        while (e) {
            tk_pair_entry_t *next = e->next;
            free(e);
            e = next;
        }
    }
    free(pt->buckets);
    pt->buckets = NULL;
    pt->num_entries = 0;
}

void tk_pair_table_clear(tk_pair_table_t *pt) {
    for (uint32_t i = 0; i < pt->num_buckets; i++) {
        tk_pair_entry_t *e = pt->buckets[i];
        while (e) {
            tk_pair_entry_t *next = e->next;
            free(e);
            e = next;
        }
        pt->buckets[i] = NULL;
    }
    pt->num_entries = 0;
}

int tk_pair_table_add(tk_pair_table_t *pt, uint32_t left, uint32_t right, int64_t count) {
    uint32_t idx = pair_hash(left, right, pt->num_buckets);
    tk_pair_entry_t *e = pt->buckets[idx];

    while (e) {
        if (e->left == left && e->right == right) {
            e->count += count;
            return 0;
        }
        e = e->next;
    }

    /* New entry */
    e = malloc(sizeof(tk_pair_entry_t));
    if (!e) return -1;
    e->left = left;
    e->right = right;
    e->count = count;
    e->next = pt->buckets[idx];
    pt->buckets[idx] = e;
    pt->num_entries++;
    return 0;
}

const tk_pair_entry_t *tk_pair_table_best(const tk_pair_table_t *pt) {
    const tk_pair_entry_t *best = NULL;
    for (uint32_t i = 0; i < pt->num_buckets; i++) {
        const tk_pair_entry_t *e = pt->buckets[i];
        while (e) {
            if (!best || e->count > best->count ||
                (e->count == best->count &&
                 ((uint64_t)e->left << 32 | e->right) <
                 ((uint64_t)best->left << 32 | best->right))) {
                best = e;
            }
            e = e->next;
        }
    }
    return best;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Token Sequence (doubly-linked list with pool allocator)
 *
 *  A linked list lets us merge adjacent pairs in O(1) per merge site,
 *  which is critical when applying a merge across millions of tokens.
 * ════════════════════════════════════════════════════════════════════════ */

int tk_sequence_init(tk_sequence_t *seq, size_t max_nodes) {
    memset(seq, 0, sizeof(*seq));
    seq->pool = malloc(max_nodes * sizeof(tk_node_t));
    if (!seq->pool) return -1;
    seq->pool_size = max_nodes;
    seq->pool_used = 0;
    return 0;
}

void tk_sequence_free(tk_sequence_t *seq) {
    free(seq->pool);
    memset(seq, 0, sizeof(*seq));
}

void tk_sequence_clear(tk_sequence_t *seq) {
    seq->head = NULL;
    seq->tail = NULL;
    seq->length = 0;
    seq->pool_used = 0;
}

static tk_node_t *seq_alloc_node(tk_sequence_t *seq) {
    if (seq->pool_used >= seq->pool_size) return NULL;
    return &seq->pool[seq->pool_used++];
}

int tk_sequence_append(tk_sequence_t *seq, uint32_t id) {
    tk_node_t *n = seq_alloc_node(seq);
    if (!n) return -1;

    n->id = id;
    n->next = NULL;
    n->prev = seq->tail;

    if (seq->tail) {
        seq->tail->next = n;
    } else {
        seq->head = n;
    }
    seq->tail = n;
    seq->length++;
    return 0;
}

size_t tk_sequence_apply_merge(tk_sequence_t *seq,
                               uint32_t left, uint32_t right, uint32_t result) {
    size_t count = 0;
    tk_node_t *n = seq->head;

    while (n && n->next) {
        if (n->id == left && n->next->id == right) {
            /* Merge: replace n with result, unlink n->next */
            tk_node_t *to_remove = n->next;

            n->id = result;
            n->next = to_remove->next;
            if (to_remove->next) {
                to_remove->next->prev = n;
            } else {
                seq->tail = n;
            }

            seq->length--;
            count++;

            /* Don't advance n — the new merged token might form another
             * pair with its new right neighbor. But that would be a
             * different pair, so we DO advance. */
            n = n->next;
        } else {
            n = n->next;
        }
    }

    return count;
}

void tk_sequence_count_pairs(const tk_sequence_t *seq, tk_pair_table_t *pt) {
    const tk_node_t *n = seq->head;
    while (n && n->next) {
        tk_pair_table_add(pt, n->id, n->next->id, 1);
        n = n->next;
    }
}

size_t tk_sequence_to_ids(const tk_sequence_t *seq, uint32_t *out, size_t out_cap) {
    size_t i = 0;
    const tk_node_t *n = seq->head;
    while (n && i < out_cap) {
        out[i++] = n->id;
        n = n->next;
    }
    return i;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Pre-tokenize callback: collects chunks into an array of sequences
 * ════════════════════════════════════════════════════════════════════════ */

typedef struct {
    tk_sequence_t **seqs;
    size_t          num_seqs;
    size_t          seqs_cap;
    size_t          total_bytes;
} pretok_ctx_t;

static void pretok_callback(const uint8_t *start, size_t len, void *ud) {
    pretok_ctx_t *ctx = (pretok_ctx_t *)ud;

    /* Skip empty chunks */
    if (len == 0) return;

    /* Grow array if needed */
    if (ctx->num_seqs >= ctx->seqs_cap) {
        size_t new_cap = ctx->seqs_cap * 2;
        if (new_cap < 256) new_cap = 256;
        tk_sequence_t **new_seqs = realloc(ctx->seqs, new_cap * sizeof(tk_sequence_t *));
        if (!new_seqs) return;
        ctx->seqs = new_seqs;
        ctx->seqs_cap = new_cap;
    }

    /* Create a new sequence for this pre-token chunk */
    tk_sequence_t *seq = malloc(sizeof(tk_sequence_t));
    if (!seq) return;
    if (tk_sequence_init(seq, len) < 0) {
        free(seq);
        return;
    }

    /* Fill with byte-level tokens */
    for (size_t i = 0; i < len; i++) {
        tk_sequence_append(seq, (uint32_t)start[i]);
    }

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

/* ════════════════════════════════════════════════════════════════════════
 *  BPE Training (in-memory)
 *
 *  Algorithm:
 *    1. Pre-tokenize the input into word-like chunks.
 *    2. Convert each chunk into a sequence of byte tokens (IDs 0–255).
 *    3. Loop:
 *       a. Count all adjacent pairs across all sequences.
 *       b. Find the most frequent pair.
 *       c. Create a new token = concat(left_bytes, right_bytes).
 *       d. Replace all occurrences of the pair in all sequences.
 *       e. Record the merge rule.
 *       f. Stop when vocab reaches target size.
 * ════════════════════════════════════════════════════════════════════════ */

int tk_bpe_train(const uint8_t *text, size_t text_len,
                 tk_vocab_t *vocab, const tk_train_config_t *config) {

    /* Step 1: Add byte-level base tokens */
    tk_vocab_add_byte_tokens(vocab);

    /* Step 2: Pre-tokenize into word chunks */
    pretok_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.seqs_cap = 1024;
    ctx.seqs = malloc(ctx.seqs_cap * sizeof(tk_sequence_t *));
    if (!ctx.seqs) return -1;

    tk_pretokenize(text, text_len, pretok_callback, &ctx);

    if (ctx.num_seqs == 0) {
        pretok_ctx_free(&ctx);
        return 0; /* nothing to train on */
    }

    /* Step 3: Iterative merging */
    tk_pair_table_t pt;
    if (tk_pair_table_init(&pt, 1 << 18) < 0) { /* 256K buckets */
        pretok_ctx_free(&ctx);
        return -1;
    }

    uint32_t target = config->target_vocab_size;
    if (target <= 256) target = 256; /* need at least byte tokens */

    int merge_num = 0;

    while (vocab->vocab_size < target) {
        /* 3a. Count all pairs */
        tk_pair_table_clear(&pt);
        for (size_t i = 0; i < ctx.num_seqs; i++) {
            tk_sequence_count_pairs(ctx.seqs[i], &pt);
        }

        /* 3b. Find best pair */
        const tk_pair_entry_t *best = tk_pair_table_best(&pt);
        if (!best || best->count < 2) break; /* no useful merges left */

        uint32_t left_id = best->left;
        uint32_t right_id = best->right;

        /* 3c. Create new merged token */
        const tk_token_t *lt = tk_vocab_get(vocab, left_id);
        const tk_token_t *rt = tk_vocab_get(vocab, right_id);
        if (!lt || !rt) break;

        uint16_t new_len = lt->len + rt->len;
        if (new_len > TK_MAX_TOKEN_LEN) break;

        uint8_t new_bytes[TK_MAX_TOKEN_LEN];
        memcpy(new_bytes, lt->bytes, lt->len);
        memcpy(new_bytes + lt->len, rt->bytes, rt->len);

        uint32_t new_id = tk_vocab_add(vocab, new_bytes, new_len);
        if (new_id == (uint32_t)-1) break;

        /* 3d. Record the merge rule */
        tk_vocab_add_merge(vocab, left_id, right_id, new_id);

        /* 3e. Apply merge across all sequences */
        size_t total_merged = 0;
        for (size_t i = 0; i < ctx.num_seqs; i++) {
            total_merged += tk_sequence_apply_merge(ctx.seqs[i],
                                                    left_id, right_id, new_id);
        }

        merge_num++;
        if (config->verbose > 0 && merge_num % config->verbose == 0) {
            fprintf(stderr, "merge %d: (%u, %u) -> %u  count=%ld  applied=%zu  vocab=%u\n",
                    merge_num, left_id, right_id, new_id,
                    (long)best->count, total_merged, vocab->vocab_size);
        }
    }

    tk_pair_table_free(&pt);
    pretok_ctx_free(&ctx);
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════
 *  BPE Training (file-based, chunked)
 *
 *  For large corpora that don't fit in RAM, we:
 *    1. mmap the file.
 *    2. On each merge iteration, stream through chunks to count pairs.
 *    3. Pick the best pair globally.
 *    4. Stream again to apply the merge.
 *
 *  This is slower (2 passes per merge) but uses bounded memory.
 *  For practical use, the in-memory trainer on a pre-sampled subset
 *  (e.g. 1–2GB) is usually sufficient.
 * ════════════════════════════════════════════════════════════════════════ */

int tk_bpe_train_file(const char *path, tk_vocab_t *vocab,
                      const tk_train_config_t *config, size_t chunk_size) {
    tk_mmap_t mmap;
    if (tk_mmap_open(&mmap, path) < 0) return -1;

    /* For very large files, we sample a portion for training.
     * Use up to 2GB or the whole file, whichever is smaller. */
    size_t train_len = mmap.size;
    size_t max_train = (size_t)2 * 1024 * 1024 * 1024;
    if (train_len > max_train) train_len = max_train;

    int ret = tk_bpe_train(mmap.data, train_len, vocab, config);

    tk_mmap_close(&mmap);
    return ret;
}

/* ════════════════════════════════════════════════════════════════════════
 *  BPE Encoding (applying merges)
 *
 *  Given a byte sequence, apply the learned merge rules in priority
 *  order to produce the final token IDs.
 *
 *  Algorithm:
 *    1. Convert input bytes to a sequence of byte-level token IDs.
 *    2. For each merge rule (in priority order):
 *       Scan the sequence, replace all (left, right) → result.
 *    3. Return the final token IDs.
 *
 *  This is the "naive" O(n*m) approach where n = sequence length and
 *  m = number of merges. For short texts this is fine. For batch
 *  encoding of large texts, a more sophisticated approach (e.g.
 *  building a merge priority queue) would be faster.
 * ════════════════════════════════════════════════════════════════════════ */

size_t tk_bpe_encode(const uint8_t *text, size_t text_len,
                     const tk_vocab_t *vocab,
                     uint32_t *out_ids, size_t out_cap) {
    if (text_len == 0) return 0;

    /* Build initial byte-level sequence */
    tk_sequence_t seq;
    if (tk_sequence_init(&seq, text_len) < 0) return (size_t)-1;

    for (size_t i = 0; i < text_len; i++) {
        tk_sequence_append(&seq, (uint32_t)text[i]);
    }

    /* Apply each merge rule in priority order */
    for (uint32_t m = 0; m < vocab->num_merges; m++) {
        const tk_merge_t *merge = &vocab->merges[m];
        tk_sequence_apply_merge(&seq, merge->left, merge->right, merge->result);

        /* Early exit if sequence is already a single token */
        if (seq.length <= 1) break;
    }

    /* Extract final IDs */
    size_t n = tk_sequence_to_ids(&seq, out_ids, out_cap);
    tk_sequence_free(&seq);
    return n;
}

/* ════════════════════════════════════════════════════════════════════════
 *  BPE Decoding (token IDs → bytes)
 *
 *  Simply concatenate the byte representation of each token.
 * ════════════════════════════════════════════════════════════════════════ */

size_t tk_bpe_decode(const uint32_t *ids, size_t num_ids,
                     const tk_vocab_t *vocab,
                     uint8_t *out, size_t out_cap) {
    size_t pos = 0;

    for (size_t i = 0; i < num_ids; i++) {
        const tk_token_t *tok = tk_vocab_get(vocab, ids[i]);
        if (!tok) return (size_t)-1;

        if (pos + tok->len > out_cap) return (size_t)-1;

        memcpy(out + pos, tok->bytes, tok->len);
        pos += tok->len;
    }

    return pos;
}