/*
 * vocab.c — Vocabulary hash table, merge-rule storage, and binary
 *           serialization for the .tkmodel format.
 */

#include "vocab.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ════════════════════════════════════════════════════════════════════════
 *  Hashing (FNV-1a — simple, fast, good distribution for short keys)
 * ════════════════════════════════════════════════════════════════════════ */

static uint32_t fnv1a(const uint8_t *data, uint16_t len) {
    uint32_t h = 0x811C9DC5u;
    for (uint16_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193u;
    }
    return h;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Hash Table (separate chaining)
 * ════════════════════════════════════════════════════════════════════════ */

static int ht_init(tk_vocab_t *v, uint32_t num_buckets) {
    v->num_buckets = num_buckets;
    v->buckets = calloc(num_buckets, sizeof(tk_ht_entry_t *));
    return v->buckets ? 0 : -1;
}

static void ht_free(tk_vocab_t *v) {
    if (!v->buckets) return;
    for (uint32_t i = 0; i < v->num_buckets; i++) {
        tk_ht_entry_t *e = v->buckets[i];
        while (e) {
            tk_ht_entry_t *next = e->next;
            free(e->key);
            free(e);
            e = next;
        }
    }
    free(v->buckets);
    v->buckets = NULL;
}

/* Resize when load factor > 0.75 */
static int ht_maybe_resize(tk_vocab_t *v) {
    if (v->vocab_size < (v->num_buckets * 3) / 4) return 0;

    uint32_t new_nb = v->num_buckets * 2;
    tk_ht_entry_t **new_buckets = calloc(new_nb, sizeof(tk_ht_entry_t *));
    if (!new_buckets) return -1;

    for (uint32_t i = 0; i < v->num_buckets; i++) {
        tk_ht_entry_t *e = v->buckets[i];
        while (e) {
            tk_ht_entry_t *next = e->next;
            uint32_t idx = fnv1a(e->key, e->key_len) % new_nb;
            e->next = new_buckets[idx];
            new_buckets[idx] = e;
            e = next;
        }
    }

    free(v->buckets);
    v->buckets = new_buckets;
    v->num_buckets = new_nb;
    return 0;
}

static tk_ht_entry_t *ht_find(const tk_vocab_t *v, const uint8_t *key, uint16_t len) {
    uint32_t idx = fnv1a(key, len) % v->num_buckets;
    tk_ht_entry_t *e = v->buckets[idx];
    while (e) {
        if (e->key_len == len && memcmp(e->key, key, len) == 0)
            return e;
        e = e->next;
    }
    return NULL;
}

static int ht_insert(tk_vocab_t *v, const uint8_t *key, uint16_t len, uint32_t value) {
    if (ht_maybe_resize(v) < 0) return -1;

    tk_ht_entry_t *e = malloc(sizeof(tk_ht_entry_t));
    if (!e) return -1;

    e->key = malloc(len);
    if (!e->key) { free(e); return -1; }
    memcpy(e->key, key, len);
    e->key_len = len;
    e->value = value;

    uint32_t idx = fnv1a(key, len) % v->num_buckets;
    e->next = v->buckets[idx];
    v->buckets[idx] = e;
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Vocabulary Lifecycle
 * ════════════════════════════════════════════════════════════════════════ */

int tk_vocab_init(tk_vocab_t *v, uint32_t initial_cap) {
    memset(v, 0, sizeof(*v));

    if (initial_cap < 512) initial_cap = 512;

    v->tokens = malloc(initial_cap * sizeof(tk_token_t));
    if (!v->tokens) return -1;
    v->vocab_cap = initial_cap;
    v->vocab_size = 0;

    if (ht_init(v, initial_cap * 2) < 0) {
        free(v->tokens);
        return -1;
    }

    v->merges_cap = initial_cap;
    v->merges = malloc(v->merges_cap * sizeof(tk_merge_t));
    if (!v->merges) {
        ht_free(v);
        free(v->tokens);
        return -1;
    }
    v->num_merges = 0;

    return 0;
}

void tk_vocab_free(tk_vocab_t *v) {
    ht_free(v);
    free(v->tokens);
    free(v->merges);
    memset(v, 0, sizeof(*v));
}

/* ════════════════════════════════════════════════════════════════════════
 *  Building the Vocabulary
 * ════════════════════════════════════════════════════════════════════════ */

uint32_t tk_vocab_add(tk_vocab_t *v, const uint8_t *bytes, uint16_t len) {
    if (len > TK_MAX_TOKEN_LEN) return (uint32_t)-1;

    /* Check if already exists */
    tk_ht_entry_t *existing = ht_find(v, bytes, len);
    if (existing) return existing->value;

    /* Grow token array if needed */
    if (v->vocab_size >= v->vocab_cap) {
        uint32_t new_cap = v->vocab_cap * 2;
        tk_token_t *new_tokens = realloc(v->tokens, new_cap * sizeof(tk_token_t));
        if (!new_tokens) return (uint32_t)-1;
        v->tokens = new_tokens;
        v->vocab_cap = new_cap;
    }

    uint32_t id = v->vocab_size;

    /* Store in dense array */
    v->tokens[id].id = id;
    v->tokens[id].len = len;
    memcpy(v->tokens[id].bytes, bytes, len);

    /* Insert into hash table */
    if (ht_insert(v, bytes, len, id) < 0) return (uint32_t)-1;

    v->vocab_size++;
    return id;
}

void tk_vocab_add_byte_tokens(tk_vocab_t *v) {
    for (int i = 0; i < 256; i++) {
        uint8_t byte = (uint8_t)i;
        tk_vocab_add(v, &byte, 1);
    }
}

int tk_vocab_add_merge(tk_vocab_t *v, uint32_t left, uint32_t right, uint32_t result) {
    if (v->num_merges >= v->merges_cap) {
        uint32_t new_cap = v->merges_cap * 2;
        tk_merge_t *new_merges = realloc(v->merges, new_cap * sizeof(tk_merge_t));
        if (!new_merges) return -1;
        v->merges = new_merges;
        v->merges_cap = new_cap;
    }

    v->merges[v->num_merges].left = left;
    v->merges[v->num_merges].right = right;
    v->merges[v->num_merges].result = result;
    v->num_merges++;
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Lookup
 * ════════════════════════════════════════════════════════════════════════ */

uint32_t tk_vocab_lookup(const tk_vocab_t *v, const uint8_t *bytes, uint16_t len) {
    tk_ht_entry_t *e = ht_find(v, bytes, len);
    return e ? e->value : (uint32_t)-1;
}

const tk_token_t *tk_vocab_get(const tk_vocab_t *v, uint32_t id) {
    if (id >= v->vocab_size) return NULL;
    return &v->tokens[id];
}

/* ════════════════════════════════════════════════════════════════════════
 *  Serialization — .tkmodel binary format
 *
 *  All integers are stored little-endian.
 * ════════════════════════════════════════════════════════════════════════ */

static void write_u32(FILE *f, uint32_t val) {
    uint8_t buf[4] = {
        (uint8_t)(val),
        (uint8_t)(val >> 8),
        (uint8_t)(val >> 16),
        (uint8_t)(val >> 24)
    };
    fwrite(buf, 1, 4, f);
}

static void write_u16(FILE *f, uint16_t val) {
    uint8_t buf[2] = { (uint8_t)(val), (uint8_t)(val >> 8) };
    fwrite(buf, 1, 2, f);
}

static uint32_t read_u32(FILE *f) {
    uint8_t buf[4];
    if (fread(buf, 1, 4, f) != 4) return 0;
    return (uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
           ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24);
}

static uint16_t read_u16(FILE *f) {
    uint8_t buf[2];
    if (fread(buf, 1, 2, f) != 2) return 0;
    return (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
}

int tk_vocab_save(const tk_vocab_t *v, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Header */
    write_u32(f, TK_MAGIC);
    write_u32(f, TK_VERSION);
    write_u32(f, v->vocab_size);
    write_u32(f, v->num_merges);

    /* Vocab entries */
    for (uint32_t i = 0; i < v->vocab_size; i++) {
        write_u32(f, v->tokens[i].id);
        write_u16(f, v->tokens[i].len);
        fwrite(v->tokens[i].bytes, 1, v->tokens[i].len, f);
    }

    /* Merge rules */
    for (uint32_t i = 0; i < v->num_merges; i++) {
        write_u32(f, v->merges[i].left);
        write_u32(f, v->merges[i].right);
        write_u32(f, v->merges[i].result);
    }

    fclose(f);
    return 0;
}

int tk_vocab_load(tk_vocab_t *v, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic = read_u32(f);
    uint32_t version = read_u32(f);
    uint32_t vs = read_u32(f);
    uint32_t nm = read_u32(f);

    if (magic != TK_MAGIC || version != TK_VERSION) {
        fclose(f);
        return -1;
    }

    if (tk_vocab_init(v, vs + 256) < 0) {
        fclose(f);
        return -1;
    }

    /* Read vocab entries */
    for (uint32_t i = 0; i < vs; i++) {
        uint32_t id = read_u32(f);
        uint16_t len = read_u16(f);
        uint8_t bytes[TK_MAX_TOKEN_LEN];
        if (len > TK_MAX_TOKEN_LEN || fread(bytes, 1, len, f) != len) {
            tk_vocab_free(v);
            fclose(f);
            return -1;
        }
        uint32_t added_id = tk_vocab_add(v, bytes, len);
        if (added_id != id) {
            /* IDs should match since we add in order */
            tk_vocab_free(v);
            fclose(f);
            return -1;
        }
    }

    /* Read merge rules */
    for (uint32_t i = 0; i < nm; i++) {
        uint32_t left = read_u32(f);
        uint32_t right = read_u32(f);
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