/*
 * test_bpe.c — Tests for pair counting, merge application,
 *              BPE training, encoding, and decoding.
 */

#include "bpe.h"
#include "vocab.h"
#include "tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL [%d]: %s\n", tests_run, msg); \
    } else { \
        tests_passed++; \
    } \
} while(0)

/* ── Pair Table ──────────────────────────────────────────────────────── */

static void test_pair_table(void) {
    fprintf(stderr, "  pair table...\n");

    tk_pair_table_t pt;
    ASSERT(tk_pair_table_init(&pt, 256) == 0, "pair table init");

    /* Add some pairs */
    tk_pair_table_add(&pt, 'a', 'b', 5);
    tk_pair_table_add(&pt, 'c', 'd', 3);
    tk_pair_table_add(&pt, 'a', 'b', 2); /* increment existing */

    const tk_pair_entry_t *best = tk_pair_table_best(&pt);
    ASSERT(best != NULL, "best pair not NULL");
    ASSERT(best->left == 'a' && best->right == 'b', "best pair is (a,b)");
    ASSERT(best->count == 7, "best pair count is 5+2=7");

    /* Clear and verify */
    tk_pair_table_clear(&pt);
    best = tk_pair_table_best(&pt);
    ASSERT(best == NULL, "empty table has no best pair");

    tk_pair_table_free(&pt);
}

/* ── Sequence & Merge ────────────────────────────────────────────────── */

static void test_sequence_merge(void) {
    fprintf(stderr, "  sequence merge...\n");

    tk_sequence_t seq;
    ASSERT(tk_sequence_init(&seq, 16) == 0, "seq init");

    /* Build: a b a b c a b */
    tk_sequence_append(&seq, 'a');
    tk_sequence_append(&seq, 'b');
    tk_sequence_append(&seq, 'a');
    tk_sequence_append(&seq, 'b');
    tk_sequence_append(&seq, 'c');
    tk_sequence_append(&seq, 'a');
    tk_sequence_append(&seq, 'b');
    ASSERT(seq.length == 7, "seq length is 7");

    /* Merge (a, b) -> 256 */
    size_t merged = tk_sequence_apply_merge(&seq, 'a', 'b', 256);
    ASSERT(merged == 3, "3 merges of (a,b)");
    ASSERT(seq.length == 4, "seq length after merge is 4");

    /* Verify: 256 256 c 256 */
    uint32_t ids[8];
    size_t n = tk_sequence_to_ids(&seq, ids, 8);
    ASSERT(n == 4, "4 ids after merge");
    ASSERT(ids[0] == 256, "id[0] = 256");
    ASSERT(ids[1] == 256, "id[1] = 256");
    ASSERT(ids[2] == 'c',  "id[2] = 'c'");
    ASSERT(ids[3] == 256, "id[3] = 256");

    /* Count pairs: (256,256), (256,c), (c,256) */
    tk_pair_table_t pt;
    tk_pair_table_init(&pt, 64);
    tk_sequence_count_pairs(&seq, &pt);

    const tk_pair_entry_t *best = tk_pair_table_best(&pt);
    ASSERT(best != NULL, "pairs exist");
    /* All pairs have count 1, so any could be "best" */
    ASSERT(best->count == 1, "max pair count is 1");

    tk_pair_table_free(&pt);
    tk_sequence_free(&seq);
}

/* ── Small BPE Training ──────────────────────────────────────────────── */

static void test_bpe_train_small(void) {
    fprintf(stderr, "  bpe train (small corpus)...\n");

    /* A tiny repeated corpus to ensure BPE finds merges */
    const char *corpus =
        "aaabdaaabac "
        "aaabdaaabac "
        "aaabdaaabac "
        "aaabdaaabac ";

    tk_vocab_t vocab;
    ASSERT(tk_vocab_init(&vocab, 512) == 0, "vocab init");

    tk_train_config_t config;
    config.target_vocab_size = 270; /* 256 bytes + 14 merges */
    config.verbose = 0;

    int ret = tk_bpe_train((const uint8_t *)corpus, strlen(corpus),
                           &vocab, &config);
    ASSERT(ret == 0, "training succeeded");
    ASSERT(vocab.vocab_size > 256, "vocab grew beyond byte tokens");
    ASSERT(vocab.num_merges > 0, "at least one merge learned");

    fprintf(stderr, "    vocab=%u merges=%u\n", vocab.vocab_size, vocab.num_merges);

    tk_vocab_free(&vocab);
}

/* ── Encode / Decode ─────────────────────────────────────────────────── */

static void test_encode_decode(void) {
    fprintf(stderr, "  encode/decode...\n");

    /* Train on a small corpus first */
    const char *corpus =
        "merhaba merhaba merhaba dünya dünya "
        "merhaba merhaba merhaba dünya dünya "
        "merhaba merhaba merhaba dünya dünya "
        "merhaba merhaba merhaba dünya dünya "
        "merhaba merhaba merhaba dünya dünya ";

    tk_tokenizer_t tk;
    tk_config_t config = tk_config_default();
    config.vocab_size = 280;
    config.verbose = 0;
    config.norm_flags = 0; /* no normalization for this test */

    ASSERT(tk_init(&tk, &config) == 0, "tokenizer init");
    ASSERT(tk_train(&tk, (const uint8_t *)corpus, strlen(corpus)) == 0,
           "training succeeded");

    /* Encode a string */
    const char *input = "merhaba";
    uint32_t ids[64];
    size_t num = tk_encode(&tk, (const uint8_t *)input, strlen(input),
                           ids, 64);
    ASSERT(num != (size_t)-1, "encoding succeeded");
    ASSERT(num > 0, "at least 1 token");
    ASSERT(num <= strlen(input), "compressed or equal to byte count");

    fprintf(stderr, "    '%s' -> %zu tokens\n", input, num);

    /* Decode back */
    uint8_t decoded[256];
    size_t dec_len = tk_decode(&tk, ids, num, decoded, 256);
    ASSERT(dec_len != (size_t)-1, "decoding succeeded");
    ASSERT(dec_len == strlen(input), "decoded length matches input");
    ASSERT(memcmp(decoded, input, dec_len) == 0, "decoded bytes match input");

    tk_free(&tk);
}

/* ── Vocab Save / Load ───────────────────────────────────────────────── */

static void test_save_load(void) {
    fprintf(stderr, "  save/load...\n");

    /* Train a small tokenizer */
    const char *corpus =
        "test test test data data data "
        "test test test data data data "
        "test test test data data data ";

    tk_tokenizer_t tk1;
    tk_config_t config = tk_config_default();
    config.vocab_size = 270;
    config.verbose = 0;
    config.norm_flags = 0;

    ASSERT(tk_init(&tk1, &config) == 0, "tk1 init");
    ASSERT(tk_train(&tk1, (const uint8_t *)corpus, strlen(corpus)) == 0,
           "tk1 training");

    /* Save */
    const char *path = "/tmp/test_bpe_model.tkmodel";
    ASSERT(tk_save(&tk1, path) == 0, "save model");

    /* Load into a new tokenizer */
    tk_tokenizer_t tk2;
    ASSERT(tk_load(&tk2, path) == 0, "load model");

    /* Verify same vocab size and merge count */
    ASSERT(tk_vocab_size(&tk1) == tk_vocab_size(&tk2),
           "vocab sizes match after load");
    ASSERT(tk1.vocab.num_merges == tk2.vocab.num_merges,
           "merge counts match after load");

    /* Encode with both and compare */
    const char *input = "test data";
    uint32_t ids1[64], ids2[64];
    size_t n1 = tk_encode(&tk1, (const uint8_t *)input, strlen(input), ids1, 64);
    size_t n2 = tk_encode(&tk2, (const uint8_t *)input, strlen(input), ids2, 64);

    ASSERT(n1 == n2, "same number of tokens from both");
    if (n1 == n2) {
        ASSERT(memcmp(ids1, ids2, n1 * sizeof(uint32_t)) == 0,
               "identical token IDs from both");
    }

    tk_free(&tk1);
    tk_free(&tk2);

    /* Clean up */
    remove(path);
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void) {
    fprintf(stderr, "test_bpe:\n");

    test_pair_table();
    test_sequence_merge();
    test_bpe_train_small();
    test_encode_decode();
    test_save_load();

    fprintf(stderr, "\n  %d/%d tests passed.\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}