/*
 * test_turkish.c — Turkish language-specific tests.
 *
 * Focuses on:
 *   - Agglutinative word forms (long suffixed words)
 *   - İ/I/i/ı correctness through the full encode/decode pipeline
 *   - Turkish-specific punctuation and formatting
 *   - Fertility (tokens-per-word) sanity checks
 */

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

/* ── Shared tokenizer trained on Turkish text ────────────────────────── */

static tk_tokenizer_t *make_turkish_tokenizer(void) {
    static tk_tokenizer_t tk;

    /* A moderately sized Turkish corpus with diverse morphology */
    const char *corpus =
        /* Common words and phrases, repeated for frequency */
        "Merhaba dünya! İstanbul çok güzel bir şehir. "
        "Ankara Türkiye'nin başkentidir. "
        "Çocuklar okuldan eve geldiler. "
        "Öğretmenler sınıfta ders anlatıyorlar. "
        "Bugün hava çok güzel, dışarı çıkalım mı? "
        "Kediler ve köpekler en sevilen evcil hayvanlardır. "
        "Evlerinizdekilerden haber aldık. "
        "Güneş doğudan doğar, batıdan batar. "
        "Kitaplarımızı okumamız gerekiyor. "
        "Arkadaşlarımla birlikte sinemaya gittik. "

        /* Agglutinative forms */
        "gelebilecekmiydiler görebileceklermiş "
        "yapabileceklerimizdendi anlayamayacaklarımızdandır "
        "oturabileceklerimizden söyleyemediklerimizden "
        "değiştirebileceklerinizden başarabileceklerimizdir "

        /* İ/I variations */
        "İstanbul İzmir İğdır İnegöl "
        "ışık ılık ırmak ısı "
        "ilk için ise ile "
        "İstanbul'dan İzmir'e gittik. "

        /* Numbers and mixed content */
        "2024 yılında 150 milyon kişi ziyaret etti. "
        "Saat 14:30'da toplantımız var. "
        "Fiyatı 1.250,00 TL'dir. "

        /* Repeat for frequency building */
        "Merhaba dünya! İstanbul çok güzel bir şehir. "
        "Ankara Türkiye'nin başkentidir. "
        "Çocuklar okuldan eve geldiler. "
        "Öğretmenler sınıfta ders anlatıyorlar. "
        "Bugün hava çok güzel, dışarı çıkalım mı? "
        "Kediler ve köpekler en sevilen evcil hayvanlardır. "
        "Evlerinizdekilerden haber aldık. "
        "Güneş doğudan doğar, batıdan batar. "
        "Kitaplarımızı okumamız gerekiyor. "
        "Arkadaşlarımla birlikte sinemaya gittik. "
        "gelebilecekmiydiler görebileceklermiş "
        "yapabileceklerimizdendi anlayamayacaklarımızdandır "
        "İstanbul İzmir İğdır İnegöl "
        "ışık ılık ırmak ısı "

        /* More repetitions */
        "Merhaba dünya! İstanbul çok güzel bir şehir. "
        "Ankara Türkiye'nin başkentidir. "
        "Çocuklar okuldan eve geldiler. "
        "Evlerinizdekilerden haber aldık. "
        "Kitaplarımızı okumamız gerekiyor. "
        "gelebilecekmiydiler görebileceklermiş "
        "yapabileceklerimizdendi anlayamayacaklarımızdandır "
        "İstanbul İzmir İğdır İnegöl "
        "ışık ılık ırmak ısı ";

    tk_config_t config = tk_config_default();
    config.vocab_size = 400;  /* enough to learn Turkish subwords */
    config.verbose = 0;
    config.norm_flags = 0;    /* no normalization for exact roundtrip */

    if (tk_init(&tk, &config) != 0) return NULL;
    if (tk_train(&tk, (const uint8_t *)corpus, strlen(corpus)) != 0) return NULL;

    fprintf(stderr, "    trained: vocab=%u merges=%u\n",
            tk_vocab_size(&tk), tk.vocab.num_merges);

    return &tk;
}

/* Helper: encode and return token count */
static size_t count_tokens(tk_tokenizer_t *tk, const char *text) {
    uint32_t ids[1024];
    return tk_encode(tk, (const uint8_t *)text, strlen(text), ids, 1024);
}

/* Helper: encode-decode roundtrip check */
static int roundtrip_ok(tk_tokenizer_t *tk, const char *text) {
    size_t len = strlen(text);
    uint32_t ids[1024];
    uint8_t decoded[4096];

    size_t nt = tk_encode(tk, (const uint8_t *)text, len, ids, 1024);
    if (nt == (size_t)-1) return 0;

    size_t dl = tk_decode(tk, ids, nt, decoded, 4096);
    if (dl == (size_t)-1) return 0;

    return (dl == len && memcmp(text, decoded, len) == 0);
}

/* ── Agglutination Tests ─────────────────────────────────────────────── */

static void test_agglutination(tk_tokenizer_t *tk) {
    fprintf(stderr, "  agglutinative forms...\n");

    /* These long agglutinated words should roundtrip correctly */
    const char *words[] = {
        "evlerinizdekilerden",        /* from those in your houses */
        "gelebilecekmiydiler",        /* were they going to be able to come */
        "görebileceklermiş",          /* apparently they will be able to see */
        "yapabileceklerimizdendi",     /* it was from what we could do */
        "anlayamayacaklarımızdandır", /* it is from those of ours who won't understand */
        "oturabileceklerimizden",     /* from those of ours who can sit */
        "söyleyemediklerimizden",     /* from those things we couldn't say */
        "değiştirebileceklerinizden", /* from those you could change */
        "başarabileceklerimizdir",    /* they are what we can achieve */
    };

    for (int i = 0; i < (int)(sizeof(words) / sizeof(words[0])); i++) {
        char label[128];
        snprintf(label, sizeof(label), "roundtrip: %s", words[i]);
        ASSERT(roundtrip_ok(tk, words[i]), label);
    }

    /* Agglutinated words should compress — fewer tokens than bytes.
     * With a small vocab this won't be dramatic, but common suffixes
     * like "ler", "den", "lar" should be merged. */
    size_t tokens_long = count_tokens(tk, "evlerinizdekilerden");
    size_t bytes_long = strlen("evlerinizdekilerden");
    fprintf(stderr, "    'evlerinizdekilerden': %zu tokens / %zu bytes = %.2f bytes/tok\n",
            tokens_long, bytes_long, (double)bytes_long / tokens_long);

    ASSERT(tokens_long < bytes_long, "agglutinated word compresses");
}

/* ── İ / I / i / ı Pipeline Tests ────────────────────────────────────── */

static void test_dotted_dotless_i(tk_tokenizer_t *tk) {
    fprintf(stderr, "  İ/I/i/ı handling...\n");

    /* All four variants must survive roundtrip */
    ASSERT(roundtrip_ok(tk, "İstanbul"), "roundtrip İstanbul");
    ASSERT(roundtrip_ok(tk, "istanbul"), "roundtrip istanbul (lowercase)");
    ASSERT(roundtrip_ok(tk, "ISTANBUL"), "roundtrip ISTANBUL (ASCII I)");

    /* ı (dotless lowercase) in isolation and in words */
    ASSERT(roundtrip_ok(tk, "ışık"),   "roundtrip ışık");
    ASSERT(roundtrip_ok(tk, "ılık"),   "roundtrip ılık");
    ASSERT(roundtrip_ok(tk, "ırmak"), "roundtrip ırmak");

    /* İ (dotted uppercase) at various positions */
    ASSERT(roundtrip_ok(tk, "İzmir"),    "roundtrip İzmir");
    ASSERT(roundtrip_ok(tk, "İğdır"),    "roundtrip İğdır");
    ASSERT(roundtrip_ok(tk, "İnegöl"),   "roundtrip İnegöl");

    /* Mixed in a sentence */
    ASSERT(roundtrip_ok(tk, "İstanbul'dan İzmir'e gittik."),
           "roundtrip sentence with İ");

    /* Ensure İ and I encode to DIFFERENT token sequences */
    uint32_t ids_dotted[64], ids_ascii[64];
    /* İ = C4 B0 in UTF-8 */
    uint8_t dotted_i[] = {0xC4, 0xB0};
    /* I = 0x49 in UTF-8 */
    uint8_t ascii_I[] = {0x49};

    size_t n1 = tk_encode(tk, dotted_i, 2, ids_dotted, 64);
    size_t n2 = tk_encode(tk, ascii_I, 1, ids_ascii, 64);

    ASSERT(n1 != (size_t)-1 && n2 != (size_t)-1, "both encode");

    /* They must differ since they're different byte sequences */
    int differ = (n1 != n2);
    if (!differ && n1 > 0) {
        differ = (memcmp(ids_dotted, ids_ascii, n1 * sizeof(uint32_t)) != 0);
    }
    ASSERT(differ, "İ and I produce different token sequences");
}

/* ── Turkish Special Characters ──────────────────────────────────────── */

static void test_special_chars(tk_tokenizer_t *tk) {
    fprintf(stderr, "  special Turkish characters...\n");

    const char *words[] = {
        "çay",       /* tea — ç */
        "şeker",     /* sugar — ş */
        "güzel",     /* beautiful — ü */
        "görmek",    /* to see — ö */
        "çiçek",     /* flower — double ç */
        "düşünce",   /* thought — ü, ş */
        "göğüs",     /* chest — ö, ğ, ü */
        "müdür",     /* director — ü, ü */
        "öğretmen",  /* teacher — ö, ğ */
        "büyükçe",   /* largish — ü, ü, ç */
    };

    for (int i = 0; i < (int)(sizeof(words) / sizeof(words[0])); i++) {
        char label[128];
        snprintf(label, sizeof(label), "roundtrip special: %s", words[i]);
        ASSERT(roundtrip_ok(tk, words[i]), label);
    }
}

/* ── Fertility (Tokens per Word) ─────────────────────────────────────── */

static void test_fertility(tk_tokenizer_t *tk) {
    fprintf(stderr, "  fertility checks...\n");

    /* Common short Turkish words should tokenize reasonably.
     * With a 400-token vocab trained on our small corpus,
     * we check that high-frequency words get some compression. */

    struct { const char *word; size_t max_tokens; } cases[] = {
        {"bir",       3},  /* very common, 3 bytes → should be ≤3 tokens */
        {"ve",        2},  /* 2 bytes → ≤2 tokens */
        {"bu",        2},
        {"çok",       4},  /* ç is 2 bytes, so "çok" is 4 bytes → ≤4 */
        {"güzel",     7},  /* 7 bytes (ü is 2B) → ≤7 tokens */
        {"İstanbul", 10},  /* İ(2B)+stanbul = 9 bytes → ≤10 tokens */
    };

    for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
        size_t nt = count_tokens(tk, cases[i].word);
        char label[128];
        snprintf(label, sizeof(label), "'%s' ≤ %zu tokens (got %zu)",
                 cases[i].word, cases[i].max_tokens, nt);
        ASSERT(nt != (size_t)-1 && nt <= cases[i].max_tokens, label);
    }

    /* Sentence-level: tokens should be fewer than bytes */
    const char *sentence = "Bugün hava çok güzel, dışarı çıkalım mı?";
    size_t sent_tokens = count_tokens(tk, sentence);
    size_t sent_bytes = strlen(sentence);

    fprintf(stderr, "    sentence: %zu tokens / %zu bytes = %.2f bytes/tok\n",
            sent_tokens, sent_bytes, (double)sent_bytes / sent_tokens);

    ASSERT(sent_tokens < sent_bytes, "sentence compresses vs byte count");
}

/* ── Mixed Content ───────────────────────────────────────────────────── */

static void test_mixed_content(tk_tokenizer_t *tk) {
    fprintf(stderr, "  mixed content...\n");

    /* Numbers embedded in Turkish text */
    ASSERT(roundtrip_ok(tk, "2024 yılında 150 milyon kişi ziyaret etti."),
           "roundtrip numbers in Turkish");

    /* Punctuation-heavy */
    ASSERT(roundtrip_ok(tk, "Evet! Hayır? Tamam... Güle güle!"),
           "roundtrip punctuation-heavy");

    /* Time and currency formatting */
    ASSERT(roundtrip_ok(tk, "Saat 14:30'da toplantımız var."),
           "roundtrip time format");
    ASSERT(roundtrip_ok(tk, "Fiyatı 1.250,00 TL'dir."),
           "roundtrip currency format");

    /* Empty and whitespace */
    ASSERT(roundtrip_ok(tk, ""), "roundtrip empty string");
    ASSERT(roundtrip_ok(tk, " "), "roundtrip single space");
    ASSERT(roundtrip_ok(tk, "\n\n\n"), "roundtrip newlines");

    /* Very short */
    ASSERT(roundtrip_ok(tk, "a"), "roundtrip single char 'a'");
    ASSERT(roundtrip_ok(tk, "ı"), "roundtrip single char 'ı'");
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void) {
    fprintf(stderr, "test_turkish:\n");

    tk_tokenizer_t *tk = make_turkish_tokenizer();
    if (!tk) {
        fprintf(stderr, "  FAIL: could not create Turkish tokenizer\n");
        return 1;
    }

    test_agglutination(tk);
    test_dotted_dotless_i(tk);
    test_special_chars(tk);
    test_fertility(tk);
    test_mixed_content(tk);

    fprintf(stderr, "\n  %d/%d tests passed.\n", tests_passed, tests_run);

    tk_free(tk);
    return (tests_passed == tests_run) ? 0 : 1;
}