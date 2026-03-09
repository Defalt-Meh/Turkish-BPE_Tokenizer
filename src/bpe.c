#include "bpe.h"
#include "unicode.h"
#include "io.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if defined(__GNUC__) || defined(__clang__)
#   define TK_LIKELY(x)   __builtin_expect(!!(x), 1)
#   define TK_UNLIKELY(x) __builtin_expect(!!(x), 0)
#   define TK_HOT          __attribute__((hot))
#else
#   define TK_LIKELY(x)   (x)
#   define TK_UNLIKELY(x) (x)
#   define TK_HOT
#endif

#define PAIR_EMPTY  ((uint64_t)-1)
#define PAIR_TOMB   ((uint64_t)-2)

/* ── pair key packing ─────────────────────────────────────────────── */

static inline uint64_t pack_pair(uint32_t left, uint32_t right) {
    return ((uint64_t)left << 32) | (uint64_t)right;
}

static inline uint32_t pair_left(uint64_t key) {
    return (uint32_t)(key >> 32);
}

static inline uint32_t pair_right(uint64_t key) {
    return (uint32_t)(key & 0xFFFFFFFF);
}

static inline uint32_t pair_hash64(uint64_t key) {
    key ^= key >> 33;
    key *= 0xFF51AFD7ED558CCDull;
    key ^= key >> 33;
    key *= 0xC4CEB9FE1A85EC53ull;
    key ^= key >> 33;
    return (uint32_t)key;
}

/* ── open-addressing pair table ───────────────────────────────────── */

static uint32_t next_pow2_32(uint32_t v) {
    v--; v |= v >> 1; v |= v >> 2;
    v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

int tk_pair_table_init(tk_pair_table_t *pt, uint32_t min_slots) {
    uint32_t n = next_pow2_32(min_slots < 1024 ? 1024 : min_slots);
    pt->keys = (uint64_t *)malloc(n * sizeof(uint64_t));
    pt->counts = (int64_t *)malloc(n * sizeof(int64_t));
    if (!pt->keys || !pt->counts) {
        free(pt->keys); free(pt->counts);
        pt->keys = NULL; pt->counts = NULL;
        return -1;
    }
    memset(pt->keys, 0xFF, n * sizeof(uint64_t));
    pt->num_slots = n;
    pt->mask = n - 1;
    pt->num_entries = 0;
    return 0;
}

void tk_pair_table_free(tk_pair_table_t *pt) {
    free(pt->keys);
    free(pt->counts);
    pt->keys = NULL;
    pt->counts = NULL;
    pt->num_entries = 0;
}

void tk_pair_table_clear(tk_pair_table_t *pt) {
    memset(pt->keys, 0xFF, pt->num_slots * sizeof(uint64_t));
    pt->num_entries = 0;
}

static int pair_table_resize(tk_pair_table_t *pt) {
    uint32_t old_n = pt->num_slots;
    uint64_t *old_keys = pt->keys;
    int64_t *old_counts = pt->counts;

    uint32_t new_n = old_n * 2;
    uint32_t new_mask = new_n - 1;

    uint64_t *new_keys = (uint64_t *)malloc(new_n * sizeof(uint64_t));
    int64_t *new_counts = (int64_t *)malloc(new_n * sizeof(int64_t));
    if (!new_keys || !new_counts) {
        free(new_keys); free(new_counts);
        return -1;
    }
    memset(new_keys, 0xFF, new_n * sizeof(uint64_t));

    for (uint32_t i = 0; i < old_n; i++) {
        if (old_keys[i] == PAIR_EMPTY || old_keys[i] == PAIR_TOMB) continue;
        uint32_t idx = pair_hash64(old_keys[i]) & new_mask;
        while (new_keys[idx] != PAIR_EMPTY)
            idx = (idx + 1) & new_mask;
        new_keys[idx] = old_keys[i];
        new_counts[idx] = old_counts[i];
    }

    free(old_keys);
    free(old_counts);
    pt->keys = new_keys;
    pt->counts = new_counts;
    pt->num_slots = new_n;
    pt->mask = new_mask;
    return 0;
}

int TK_HOT tk_pair_table_add(tk_pair_table_t *pt, uint32_t left,
                              uint32_t right, int64_t count) {
    if (TK_UNLIKELY(pt->num_entries * 2 >= pt->num_slots)) {
        if (pair_table_resize(pt) < 0) return -1;
    }

    uint64_t key = pack_pair(left, right);
    uint32_t idx = pair_hash64(key) & pt->mask;

    for (;;) {
        uint64_t k = pt->keys[idx];
        if (k == key) {
            pt->counts[idx] += count;
            return 0;
        }
        if (k == PAIR_EMPTY || k == PAIR_TOMB) {
            pt->keys[idx] = key;
            pt->counts[idx] = count;
            pt->num_entries++;
            return 0;
        }
        idx = (idx + 1) & pt->mask;
    }
}

const tk_pair_entry_t *tk_pair_table_best(const tk_pair_table_t *pt) {
    static tk_pair_entry_t result;
    int64_t best_count = 0;
    uint64_t best_key = PAIR_EMPTY;
    uint32_t best_idx = 0;
    (void)best_idx;

    for (uint32_t i = 0; i < pt->num_slots; i++) {
        uint64_t k = pt->keys[i];
        if (k == PAIR_EMPTY || k == PAIR_TOMB) continue;
        int64_t c = pt->counts[i];
        if (c > best_count || (c == best_count && k < best_key)) {
            best_count = c;
            best_key = k;
            best_idx = i;
        }
    }

    if (best_key == PAIR_EMPTY) return NULL;

    result.left = pair_left(best_key);
    result.right = pair_right(best_key);
    result.count = best_count;
    return &result;
}

/* ── Turkish morphological scoring ────────────────────────────────── */

typedef struct {
    const uint8_t *bytes;
    uint16_t       len;
} byte_span_t;

static bool span_eq(byte_span_t a, const char *s) {
    size_t slen = strlen(s);
    if (a.len != slen) return false;
    return memcmp(a.bytes, s, slen) == 0;
}

static bool span_ends_with(byte_span_t span, const char *suffix) {
    size_t slen = strlen(suffix);
    if (span.len < slen) return false;
    return memcmp(span.bytes + span.len - slen, suffix, slen) == 0;
}

static bool is_turkish_vowel_byte(uint8_t b) {
    switch (b) {
    case 'a': case 'e': case 'o': case 'u':
    case 'A': case 'E': case 'O': case 'U':
    case 'i': case 'I':
        return true;
    default:
        return false;
    }
}

static bool is_turkish_vowel_cp(uint32_t cp) {
    switch (cp) {
    case 'a': case 'e': case 'i': case 'o': case 'u':
    case 'A': case 'E': case 'I': case 'O': case 'U':
    case 0x00F6: case 0x00FC: case 0x0131: case 0x00D6:
    case 0x00DC: case 0x0130:
        return true;
    default:
        return false;
    }
}

static bool is_back_vowel(uint32_t cp) {
    switch (cp) {
    case 'a': case 'A':
    case 0x0131:
    case 'o': case 'O':
    case 'u': case 'U':
        return true;
    default:
        return false;
    }
}

static bool is_front_vowel(uint32_t cp) {
    switch (cp) {
    case 'e': case 'E':
    case 'i': case 0x0130:
    case 0x00F6: case 0x00D6:
    case 0x00FC: case 0x00DC:
        return true;
    default:
        return false;
    }
}

static uint32_t last_vowel_in_span(byte_span_t span) {
    const uint8_t *p = span.bytes;
    const uint8_t *end = p + span.len;
    uint32_t last_v = 0;
    while (p < end) {
        uint32_t cp = utf8_decode(&p, end);
        if (is_turkish_vowel_cp(cp)) last_v = cp;
    }
    return last_v;
}

static bool pair_respects_vowel_harmony(byte_span_t left_span,
                                        byte_span_t right_span) {
    uint32_t lv = last_vowel_in_span(left_span);
    if (lv == 0) return true;

    const uint8_t *p = right_span.bytes;
    const uint8_t *end = p + right_span.len;
    uint32_t first_rv = 0;
    while (p < end) {
        uint32_t cp = utf8_decode(&p, end);
        if (is_turkish_vowel_cp(cp)) { first_rv = cp; break; }
    }
    if (first_rv == 0) return true;

    if (is_back_vowel(lv) && is_back_vowel(first_rv)) return true;
    if (is_front_vowel(lv) && is_front_vowel(first_rv)) return true;

    return false;
}

static const char *turkish_suffix_table[] = {
    /* case suffixes */
    "de", "da", "den", "dan", "te", "ta", "ten", "tan",
    "ye", "ya", "e", "a",
    "in", "un", "\xC4\xB1n", "\xC3\xBCn",
    "nin", "nun", "n\xC4\xB1n", "n\xC3\xBCn",
    /* plural */
    "ler", "lar",
    /* possessive */
    "im", "\xC4\xB1m", "um", "\xC3\xBCm",
    "in", "\xC4\xB1n", "un", "\xC3\xBCn",
    "si", "s\xC4\xB1",
    "miz", "m\xC4\xB1z", "muz", "m\xC3\xBCz",
    "niz", "n\xC4\xB1z", "nuz", "n\xC3\xBCz",
    "leri", "lar\xC4\xB1",
    /* tense */
    "di", "d\xC4\xB1", "du", "d\xC3\xBC",
    "ti", "t\xC4\xB1", "tu", "t\xC3\xBC",
    "mis", "m\xC4\xB1s", "mus", "m\xC3\xBCs",
    "mi\xC5\x9F", "m\xC4\xB1\xC5\x9F",
    "yor", "iyor", "\xC4\xB1yor", "uyor", "\xC3\xBCyor",
    "ecek", "acak",
    "ir", "\xC4\xB1r", "ur", "\xC3\xBCr",
    "er", "ar",
    /* verbal noun / participle */
    "mek", "mak",
    "en", "an",
    "dik", "d\xC4\xB1k", "duk", "d\xC3\xBCk",
    "tik", "t\xC4\xB1k", "tuk", "t\xC3\xBCk",
    /* ability */
    "ebil", "abil",
    /* negative */
    "me", "ma",
    /* copula */
    "dir", "d\xC4\xB1r", "dur", "d\xC3\xBCr",
    "tir", "t\xC4\xB1r", "tur", "t\xC3\xBCr",
    /* question */
    "mi", "m\xC4\xB1", "mu", "m\xC3\xBC",
    NULL
};

static bool is_morphological_suffix(byte_span_t span) {
    for (int i = 0; turkish_suffix_table[i]; i++) {
        if (span_eq(span, turkish_suffix_table[i])) return true;
    }
    return false;
}

static bool merge_forms_suffix(byte_span_t left, byte_span_t right) {
    if (left.len + right.len > TK_MAX_TOKEN_LEN) return false;
    uint8_t combined[TK_MAX_TOKEN_LEN];
    memcpy(combined, left.bytes, left.len);
    memcpy(combined + left.len, right.bytes, right.len);
    byte_span_t comb = { combined, (uint16_t)(left.len + right.len) };
    return is_morphological_suffix(comb);
}

static bool is_consonant_cluster_break(byte_span_t left, byte_span_t right) {
    if (left.len == 0 || right.len == 0) return false;
    uint8_t lb = left.bytes[left.len - 1];
    uint8_t rb = right.bytes[0];
    if (!is_turkish_vowel_byte(lb) && !is_turkish_vowel_byte(rb))
        return true;
    return false;
}

static double morphological_score(byte_span_t left_span,
                                  byte_span_t right_span,
                                  int64_t raw_count) {
    double score = (double)raw_count;

    if (pair_respects_vowel_harmony(left_span, right_span))
        score *= 1.15;

    if (is_morphological_suffix(right_span))
        score *= 1.25;

    if (merge_forms_suffix(left_span, right_span))
        score *= 1.30;

    if (is_consonant_cluster_break(left_span, right_span))
        score *= 0.85;

    bool left_has_vowel = (last_vowel_in_span(left_span) != 0);
    bool right_has_vowel = (last_vowel_in_span(right_span) != 0);
    if (left_has_vowel && right_has_vowel)
        score *= 1.05;

    if (left_span.len >= 2 && right_span.len >= 2)
        score *= 1.02;

    return score;
}

/* ── scored best-pair selection ───────────────────────────────────── */

typedef struct {
    uint32_t left;
    uint32_t right;
    int64_t  raw_count;
    double   score;
} scored_pair_t;

static scored_pair_t find_best_scored(const tk_pair_table_t *pt,
                                      const tk_vocab_t *vocab) {
    scored_pair_t best = { 0, 0, 0, -1.0 };

    for (uint32_t i = 0; i < pt->num_slots; i++) {
        uint64_t k = pt->keys[i];
        if (k == PAIR_EMPTY || k == PAIR_TOMB) continue;

        int64_t count = pt->counts[i];
        if (count < 2) continue;

        uint32_t l = pair_left(k);
        uint32_t r = pair_right(k);

        const tk_token_t *lt = tk_vocab_get(vocab, l);
        const tk_token_t *rt = tk_vocab_get(vocab, r);
        if (!lt || !rt) continue;

        byte_span_t ls = { lt->bytes, lt->len };
        byte_span_t rs = { rt->bytes, rt->len };

        double sc = morphological_score(ls, rs, count);

        if (sc > best.score ||
            (sc == best.score && k < pack_pair(best.left, best.right))) {
            best.left = l;
            best.right = r;
            best.raw_count = count;
            best.score = sc;
        }
    }

    return best;
}

/* ── token sequence (doubly-linked, pool-allocated) ───────────────── */

int tk_sequence_init(tk_sequence_t *seq, size_t max_nodes) {
    memset(seq, 0, sizeof(*seq));
    seq->pool = (tk_node_t *)malloc(max_nodes * sizeof(tk_node_t));
    if (!seq->pool) return -1;
    seq->pool_size = max_nodes;
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

static inline tk_node_t *seq_alloc(tk_sequence_t *seq) {
    if (TK_UNLIKELY(seq->pool_used >= seq->pool_size)) return NULL;
    return &seq->pool[seq->pool_used++];
}

int tk_sequence_append(tk_sequence_t *seq, uint32_t id) {
    tk_node_t *n = seq_alloc(seq);
    if (!n) return -1;
    n->id = id;
    n->next = NULL;
    n->prev = seq->tail;
    if (seq->tail) seq->tail->next = n;
    else seq->head = n;
    seq->tail = n;
    seq->length++;
    return 0;
}

size_t TK_HOT tk_sequence_apply_merge(tk_sequence_t *seq,
                                       uint32_t left, uint32_t right,
                                       uint32_t result) {
    size_t count = 0;
    tk_node_t *n = seq->head;

    while (n && n->next) {
        if (n->id == left && n->next->id == right) {
            tk_node_t *dead = n->next;
            n->id = result;
            n->next = dead->next;
            if (dead->next) dead->next->prev = n;
            else seq->tail = n;
            seq->length--;
            count++;
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

size_t tk_sequence_to_ids(const tk_sequence_t *seq, uint32_t *out,
                          size_t out_cap) {
    size_t i = 0;
    const tk_node_t *n = seq->head;
    while (n && i < out_cap) {
        out[i++] = n->id;
        n = n->next;
    }
    return i;
}

/* ── pre-tokenize callback for training ───────────────────────────── */

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
        ctx->seqs = grown;
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
            bool suffix = is_morphological_suffix(rs);

            fprintf(stderr,
                "merge %d: (%u,%u)->%u  cnt=%ld  score=%.1f  "
                "applied=%zu  vocab=%u%s%s\n",
                merge_num, best.left, best.right, new_id,
                (long)best.raw_count, best.score,
                total_merged, vocab->vocab_size,
                harmony ? " [harmony]" : "",
                suffix ? " [suffix]" : "");
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

    size_t train_len = mmap.size;
    size_t max_train = (size_t)2 * 1024 * 1024 * 1024;
    if (train_len > max_train) train_len = max_train;

    (void)chunk_size;
    int ret = tk_bpe_train(mmap.data, train_len, vocab, config);

    tk_mmap_close(&mmap);
    return ret;
}

/* ── priority-queue BPE encoding ──────────────────────────────────── */

typedef struct {
    uint32_t pos;
    uint32_t merge_idx;
    int64_t  priority;
} pq_entry_t;

typedef struct {
    pq_entry_t *data;
    size_t       len;
    size_t       cap;
} min_heap_t;

static void heap_swap(pq_entry_t *a, pq_entry_t *b) {
    pq_entry_t tmp = *a; *a = *b; *b = tmp;
}

static void heap_push(min_heap_t *h, pq_entry_t e) {
    if (h->len >= h->cap) {
        h->cap = h->cap ? h->cap * 2 : 64;
        h->data = (pq_entry_t *)realloc(h->data, h->cap * sizeof(pq_entry_t));
    }
    h->data[h->len] = e;
    size_t i = h->len++;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (h->data[parent].priority <= h->data[i].priority) break;
        heap_swap(&h->data[parent], &h->data[i]);
        i = parent;
    }
}

static pq_entry_t heap_pop(min_heap_t *h) {
    pq_entry_t top = h->data[0];
    h->data[0] = h->data[--h->len];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = 2 * i + 2, smallest = i;
        if (l < h->len && h->data[l].priority < h->data[smallest].priority)
            smallest = l;
        if (r < h->len && h->data[r].priority < h->data[smallest].priority)
            smallest = r;
        if (smallest == i) break;
        heap_swap(&h->data[i], &h->data[smallest]);
        i = smallest;
    }
    return top;
}

static uint32_t find_merge_rank(const tk_vocab_t *vocab,
                                uint32_t left, uint32_t right) {
    for (uint32_t i = 0; i < vocab->num_merges; i++) {
        if (vocab->merges[i].left == left && vocab->merges[i].right == right)
            return i;
    }
    return (uint32_t)-1;
}

size_t TK_HOT tk_bpe_encode(const uint8_t *text, size_t text_len,
                              const tk_vocab_t *vocab,
                              uint32_t *out_ids, size_t out_cap) {
    if (text_len == 0) return 0;

    if (vocab->num_merges == 0) {
        if (text_len > out_cap) return (size_t)-1;
        for (size_t i = 0; i < text_len; i++) {
            uint32_t byte_id = (uint32_t)text[i];
            // Verify byte exists in vocab, fallback to space if missing
            if (!tk_vocab_get(vocab, byte_id)) {
                byte_id = (uint32_t)' ';
            }
            out_ids[i] = byte_id;
        }
        return text_len;
    }

    if (text_len <= 128) {
        tk_sequence_t seq;
        if (tk_sequence_init(&seq, text_len) < 0) return (size_t)-1;
        for (size_t i = 0; i < text_len; i++) {
            uint32_t byte_id = (uint32_t)text[i];
            // Verify byte exists in vocab, fallback to space if missing
            if (!tk_vocab_get(vocab, byte_id)) {
                byte_id = (uint32_t)' ';
            }
            tk_sequence_append(&seq, byte_id);
        }
        for (uint32_t m = 0; m < vocab->num_merges; m++) {
            const tk_merge_t *mg = &vocab->merges[m];
            tk_sequence_apply_merge(&seq, mg->left, mg->right, mg->result);
            if (seq.length <= 1) break;
        }
        size_t n = tk_sequence_to_ids(&seq, out_ids, out_cap);
        tk_sequence_free(&seq);
        return n;
    }

    uint32_t *ids = (uint32_t *)malloc(text_len * sizeof(uint32_t));
    uint32_t *next_arr = (uint32_t *)malloc(text_len * sizeof(uint32_t));
    uint32_t *prev_arr = (uint32_t *)malloc(text_len * sizeof(uint32_t));
    if (!ids || !next_arr || !prev_arr) {
        free(ids); free(next_arr); free(prev_arr);
        return (size_t)-1;
    }

    for (size_t i = 0; i < text_len; i++) {
        uint32_t byte_id = (uint32_t)text[i];
        // Verify byte exists in vocab, fallback to space if missing
        if (!tk_vocab_get(vocab, byte_id)) {
            byte_id = (uint32_t)' ';
        }
        ids[i] = byte_id;
        next_arr[i] = (uint32_t)(i + 1);
        prev_arr[i] = (uint32_t)(i - 1);
    }
    next_arr[text_len - 1] = (uint32_t)-1;
    prev_arr[0] = (uint32_t)-1;

    min_heap_t heap = { NULL, 0, 0 };

    for (size_t i = 0; i + 1 < text_len; i++) {
        uint32_t rank = find_merge_rank(vocab, ids[i], ids[i + 1]);
        if (rank != (uint32_t)-1) {
            pq_entry_t e = { (uint32_t)i, rank, (int64_t)rank };
            heap_push(&heap, e);
        }
    }

    size_t remaining = text_len;

    while (heap.len > 0 && remaining > 1) {
        pq_entry_t top = heap_pop(&heap);
        uint32_t pos = top.pos;
        uint32_t mi = top.merge_idx;

        if (ids[pos] == (uint32_t)-1) continue;
        uint32_t nxt = next_arr[pos];
        if (nxt == (uint32_t)-1 || ids[nxt] == (uint32_t)-1) continue;

        if (ids[pos] != vocab->merges[mi].left ||
            ids[nxt] != vocab->merges[mi].right) continue;

        ids[pos] = vocab->merges[mi].result;
        ids[nxt] = (uint32_t)-1;
        next_arr[pos] = next_arr[nxt];
        if (next_arr[nxt] != (uint32_t)-1)
            prev_arr[next_arr[nxt]] = pos;
        remaining--;

        if (prev_arr[pos] != (uint32_t)-1) {
            uint32_t p = prev_arr[pos];
            uint32_t rank = find_merge_rank(vocab, ids[p], ids[pos]);
            if (rank != (uint32_t)-1) {
                pq_entry_t e = { p, rank, (int64_t)rank };
                heap_push(&heap, e);
            }
        }

        if (next_arr[pos] != (uint32_t)-1) {
            uint32_t n2 = next_arr[pos];
            uint32_t rank = find_merge_rank(vocab, ids[pos], ids[n2]);
            if (rank != (uint32_t)-1) {
                pq_entry_t e = { pos, rank, (int64_t)rank };
                heap_push(&heap, e);
            }
        }
    }

    size_t out_len = 0;
    uint32_t cur = 0;
    while (cur != (uint32_t)-1 && out_len < out_cap) {
        if (ids[cur] != (uint32_t)-1)
            out_ids[out_len++] = ids[cur];
        cur = next_arr[cur];
    }

    free(ids);
    free(next_arr);
    free(prev_arr);
    free(heap.data);
    return out_len;
}

/* ── BPE decoding ─────────────────────────────────────────────────── */

size_t TK_HOT tk_bpe_decode(const uint32_t *ids, size_t num_ids,
                              const tk_vocab_t *vocab,
                              uint8_t *out, size_t out_cap) {
    size_t pos = 0;
    for (size_t i = 0; i < num_ids; i++) {
        const tk_token_t *tok = tk_vocab_get(vocab, ids[i]);
        if (TK_UNLIKELY(!tok)) return (size_t)-1;
        if (TK_UNLIKELY(pos + tok->len > out_cap)) return (size_t)-1;
        memcpy(out + pos, tok->bytes, tok->len);
        pos += tok->len;
    }
    return pos;
}