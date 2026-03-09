/*
 * vocab.c — Vocabulary table, merge rules, and .tkmodel serialization.
 *
 * Open-addressing hash table with linear probing. No linked lists,
 * no per-entry malloc, no pointer chasing. One contiguous slab.
 */

#include "vocab.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(__GNUC__) || defined(__clang__)
#   define TK_LIKELY(x)   __builtin_expect(!!(x), 1)
#   define TK_UNLIKELY(x) __builtin_expect(!!(x), 0)
#   define TK_HOT          __attribute__((hot))
#else
#   define TK_LIKELY(x)   (x)
#   define TK_UNLIKELY(x) (x)
#   define TK_HOT
#endif

/* Sentinel value for empty slots in the open-addressing table. */
#define HT_EMPTY  ((uint32_t)-1)
#define HT_TOMBSTONE ((uint32_t)-2)


/* ── FNV-1a hash ──────────────────────────────────────────────────── */

static uint32_t TK_HOT fnv1a(const uint8_t *data, uint16_t len) {
    uint32_t h = 0x811C9DC5u;
    for (uint16_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193u;
    }
    return h;
}


/* ── Open-addressing hash table ───────────────────────────────────── */

/*
 * Each slot stores a token ID. The actual key (byte sequence) lives
 * in the dense token array. Probing fetches the key from there.
 * This keeps the slot array compact — one uint32_t per slot — and
 * the probe loop cache-friendly.
 */

/* Slot count must be power of two for masking. */
static uint32_t next_pow2(uint32_t v) {
    v--;
    v |= v >> 1;  v |= v >> 2;
    v |= v >> 4;  v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

static int ht_init(tk_vocab_t *v, uint32_t min_slots) {
    uint32_t n = next_pow2(min_slots < 256 ? 256 : min_slots);
    v->slots = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!v->slots) return -1;
    memset(v->slots, 0xFF, n * sizeof(uint32_t)); /* fill with HT_EMPTY */
    v->num_slots = n;
    v->slot_mask = n - 1;
    return 0;
}

static void ht_free(tk_vocab_t *v) {
    free(v->slots);
    v->slots = NULL;
    v->num_slots = 0;
}

/* Rebuild the table into a larger allocation. */
static int ht_resize(tk_vocab_t *v, uint32_t new_count) {
    uint32_t n = next_pow2(new_count);
    uint32_t *new_slots = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!new_slots) return -1;
    memset(new_slots, 0xFF, n * sizeof(uint32_t));

    uint32_t mask = n - 1;
    for (uint32_t i = 0; i < v->num_slots; i++) {
        uint32_t id = v->slots[i];
        if (id == HT_EMPTY || id == HT_TOMBSTONE) continue;
        const tk_token_t *t = &v->tokens[id];
        uint32_t idx = fnv1a(t->bytes, t->len) & mask;
        while (new_slots[idx] != HT_EMPTY)
            idx = (idx + 1) & mask;
        new_slots[idx] = id;
    }

    free(v->slots);
    v->slots = new_slots;
    v->num_slots = n;
    v->slot_mask = mask;
    return 0;
}

/* Find slot containing a matching key, or the first empty slot. */
static TK_HOT uint32_t ht_probe(const tk_vocab_t *v,
                                  const uint8_t *key, uint16_t len) {
    uint32_t idx = fnv1a(key, len) & v->slot_mask;
    for (;;) {
        uint32_t id = v->slots[idx];
        if (id == HT_EMPTY) return idx;
        if (id != HT_TOMBSTONE) {
            const tk_token_t *t = &v->tokens[id];
            if (t->len == len && memcmp(t->bytes, key, len) == 0)
                return idx;
        }
        idx = (idx + 1) & v->slot_mask;
    }
}

/* Insert a token id at its hashed position. Caller ensures capacity. */
static void ht_insert_id(tk_vocab_t *v, const uint8_t *key,
                          uint16_t len, uint32_t id) {
    uint32_t idx = fnv1a(key, len) & v->slot_mask;
    while (v->slots[idx] != HT_EMPTY && v->slots[idx] != HT_TOMBSTONE)
        idx = (idx + 1) & v->slot_mask;
    v->slots[idx] = id;
}


/* ── Vocabulary lifecycle ─────────────────────────────────────────── */

int tk_vocab_init(tk_vocab_t *v, uint32_t initial_cap) {
    memset(v, 0, sizeof(*v));
    if (initial_cap < 512) initial_cap = 512;

    v->tokens = (tk_token_t *)malloc(initial_cap * sizeof(tk_token_t));
    if (!v->tokens) return -1;
    v->vocab_cap = initial_cap;

    /* Table at 4x token capacity keeps load factor under 0.25. */
    if (ht_init(v, initial_cap * 4) < 0) {
        free(v->tokens);
        return -1;
    }

    v->merges_cap = initial_cap;
    v->merges = (tk_merge_t *)malloc(v->merges_cap * sizeof(tk_merge_t));
    if (!v->merges) {
        ht_free(v);
        free(v->tokens);
        return -1;
    }

    return 0;
}

void tk_vocab_free(tk_vocab_t *v) {
    ht_free(v);
    free(v->tokens);
    free(v->merges);
    memset(v, 0, sizeof(*v));
}


/* ── Building the vocabulary ──────────────────────────────────────── */

uint32_t tk_vocab_add(tk_vocab_t *v, const uint8_t *bytes, uint16_t len) {
    if (TK_UNLIKELY(len > TK_MAX_TOKEN_LEN)) return (uint32_t)-1;

    /* Check existence via probe. */
    uint32_t slot = ht_probe(v, bytes, len);
    if (v->slots[slot] != HT_EMPTY && v->slots[slot] != HT_TOMBSTONE)
        return v->slots[slot];

    /* Grow token array if full. */
    if (TK_UNLIKELY(v->vocab_size >= v->vocab_cap)) {
        uint32_t new_cap = v->vocab_cap * 2;
        tk_token_t *grown = (tk_token_t *)realloc(
            v->tokens, new_cap * sizeof(tk_token_t));
        if (!grown) return (uint32_t)-1;
        v->tokens = grown;
        v->vocab_cap = new_cap;
    }

    /* Resize table if load factor exceeds 0.5. */
    if (TK_UNLIKELY(v->vocab_size * 2 >= v->num_slots)) {
        if (ht_resize(v, v->num_slots * 2) < 0) return (uint32_t)-1;
    }

    uint32_t id = v->vocab_size;
    v->tokens[id].id  = id;
    v->tokens[id].len = len;
    memcpy(v->tokens[id].bytes, bytes, len);

    ht_insert_id(v, bytes, len, id);
    v->vocab_size++;
    return id;
}

void tk_vocab_add_byte_tokens(tk_vocab_t *v) {
    for (unsigned i = 0; i < 256; i++) {
        uint8_t b = (uint8_t)i;
        tk_vocab_add(v, &b, 1);
    }
}

int tk_vocab_add_merge(tk_vocab_t *v, uint32_t left,
                       uint32_t right, uint32_t result) {
    if (TK_UNLIKELY(v->num_merges >= v->merges_cap)) {
        uint32_t new_cap = v->merges_cap * 2;
        tk_merge_t *grown = (tk_merge_t *)realloc(
            v->merges, new_cap * sizeof(tk_merge_t));
        if (!grown) return -1;
        v->merges = grown;
        v->merges_cap = new_cap;
    }

    tk_merge_t *m = &v->merges[v->num_merges++];
    m->left   = left;
    m->right  = right;
    m->result = result;
    return 0;
}


/* ── Lookup ───────────────────────────────────────────────────────── */

uint32_t TK_HOT tk_vocab_lookup(const tk_vocab_t *v,
                                 const uint8_t *bytes, uint16_t len) {
    uint32_t idx = fnv1a(bytes, len) & v->slot_mask;
    for (;;) {
        uint32_t id = v->slots[idx];
        if (id == HT_EMPTY) return (uint32_t)-1;
        if (id != HT_TOMBSTONE) {
            const tk_token_t *t = &v->tokens[id];
            if (t->len == len && memcmp(t->bytes, bytes, len) == 0)
                return id;
        }
        idx = (idx + 1) & v->slot_mask;
    }
}

const tk_token_t *tk_vocab_get(const tk_vocab_t *v, uint32_t id) {
    if (TK_UNLIKELY(id >= v->vocab_size)) return NULL;
    return &v->tokens[id];
}


/* ── Serialization helpers (little-endian, unaligned-safe) ────────── */

static inline void write_u32(FILE *f, uint32_t val) {
    uint8_t b[4] = { (uint8_t)val, (uint8_t)(val >> 8),
                      (uint8_t)(val >> 16), (uint8_t)(val >> 24) };
    fwrite(b, 1, 4, f);
}

static inline void write_u16(FILE *f, uint16_t val) {
    uint8_t b[2] = { (uint8_t)val, (uint8_t)(val >> 8) };
    fwrite(b, 1, 2, f);
}

static inline uint32_t read_u32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static inline uint16_t read_u16(FILE *f) {
    uint8_t b[2];
    if (fread(b, 1, 2, f) != 2) return 0;
    return (uint16_t)b[0] | ((uint16_t)b[1] << 8);
}


/* ── Save ─────────────────────────────────────────────────────────── */

int tk_vocab_save(const tk_vocab_t *v, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* 16-byte header. */
    write_u32(f, TK_MAGIC);
    write_u32(f, TK_VERSION);
    write_u32(f, v->vocab_size);
    write_u32(f, v->num_merges);

    /* Token entries: id (4) + len (2) + bytes (len). */
    for (uint32_t i = 0; i < v->vocab_size; i++) {
        write_u32(f, v->tokens[i].id);
        write_u16(f, v->tokens[i].len);
        fwrite(v->tokens[i].bytes, 1, v->tokens[i].len, f);
    }

    /* Merge rules: left (4) + right (4) + result (4). */
    for (uint32_t i = 0; i < v->num_merges; i++) {
        write_u32(f, v->merges[i].left);
        write_u32(f, v->merges[i].right);
        write_u32(f, v->merges[i].result);
    }

    int err = ferror(f);
    fclose(f);
    return err ? -1 : 0;
}


/* ── Load ─────────────────────────────────────────────────────────── */

int tk_vocab_load(tk_vocab_t *v, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic   = read_u32(f);
    uint32_t version = read_u32(f);
    uint32_t vs      = read_u32(f);
    uint32_t nm      = read_u32(f);

    if (magic != TK_MAGIC || version != TK_VERSION) {
        fclose(f);
        return -1;
    }

    if (tk_vocab_init(v, vs + 256) < 0) {
        fclose(f);
        return -1;
    }

    for (uint32_t i = 0; i < vs; i++) {
        uint32_t id  = read_u32(f);
        uint16_t len = read_u16(f);
        uint8_t bytes[TK_MAX_TOKEN_LEN];
        if (len > TK_MAX_TOKEN_LEN || fread(bytes, 1, len, f) != len) {
            tk_vocab_free(v);
            fclose(f);
            return -1;
        }
        if (tk_vocab_add(v, bytes, len) != id) {
            tk_vocab_free(v);
            fclose(f);
            return -1;
        }
    }

    for (uint32_t i = 0; i < nm; i++) {
        uint32_t left   = read_u32(f);
        uint32_t right  = read_u32(f);
        uint32_t result = read_u32(f);
        if (tk_vocab_add_merge(v, left, right, result) < 0) {
            tk_vocab_free(v);
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}