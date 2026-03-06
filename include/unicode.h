#ifndef TK_UNICODE_H
#define TK_UNICODE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * unicode.h — UTF-8 decoding/encoding and Turkish-aware text normalization.
 *
 * Turkish has special casing rules:
 *   - Dotted uppercase İ (U+0130) <-> lowercase i (U+0069)
 *   - Dotless uppercase I (U+0049) <-> lowercase ı (U+0131)
 *
 * This is the opposite of English and breaks naive tolower()/toupper().
 * We handle it explicitly throughout.
 */

/* ── UTF-8 codec ─────────────────────────────────────────────────────── */

/* Decode one codepoint from a UTF-8 byte stream.
 * Returns the codepoint (or 0xFFFD on error) and advances *pos.
 * `end` is one-past-last valid byte. */
uint32_t utf8_decode(const uint8_t **pos, const uint8_t *end);

/* Encode one codepoint to UTF-8. Writes up to 4 bytes into `out`.
 * Returns number of bytes written (1–4), or 0 on invalid codepoint. */
int utf8_encode(uint32_t cp, uint8_t *out);

/* Return the byte length of the UTF-8 sequence starting at `byte`.
 * Returns 1–4 on valid lead bytes, 1 on invalid (treat as single byte). */
int utf8_seq_len(uint8_t byte);

/* Count the number of Unicode codepoints in a UTF-8 string. */
size_t utf8_strlen(const uint8_t *s, size_t byte_len);

/* Validate that `s` is well-formed UTF-8. Returns true if valid. */
bool utf8_validate(const uint8_t *s, size_t byte_len);

/* ── Turkish-aware casing ────────────────────────────────────────────── */

/* Turkish-locale lowercase of a single codepoint.
 *   I (U+0049) -> ı (U+0131)    [NOT i]
 *   İ (U+0130) -> i (U+0069)
 *   All others: standard Unicode lowercase. */
uint32_t turkish_tolower(uint32_t cp);

/* Turkish-locale uppercase of a single codepoint.
 *   i (U+0069) -> İ (U+0130)    [NOT I]
 *   ı (U+0131) -> I (U+0049)
 *   All others: standard Unicode uppercase. */
uint32_t turkish_toupper(uint32_t cp);

/* ── Normalization ───────────────────────────────────────────────────── */

/* Normalization flags (bitfield). */
#define TK_NORM_LOWERCASE   (1 << 0)  /* Turkish-aware lowercasing       */
#define TK_NORM_NFC         (1 << 1)  /* NFC normalization (İ composed)  */
#define TK_NORM_STRIP_ACCENTS (1 << 2) /* strip combining marks (optional)*/
#define TK_NORM_WHITESPACE  (1 << 3)  /* collapse runs of whitespace     */

/* Normalize a UTF-8 string in-place (may shrink, never grows for the
 * flags we support). Returns new byte length.
 * `buf` must be writable and `len` bytes long. */
size_t tk_normalize(uint8_t *buf, size_t len, unsigned flags);

/* ── Character classification ────────────────────────────────────────── */

bool tk_is_whitespace(uint32_t cp);
bool tk_is_punctuation(uint32_t cp);
bool tk_is_letter(uint32_t cp);
bool tk_is_digit(uint32_t cp);

/* ── Pre-tokenization (byte-level BPE prep) ──────────────────────────── */

/* Split text into pre-token chunks following GPT-style regex:
 *   'ler|'t|'re|... | \p{L}+ | \p{N}+ | [^\s\p{L}\p{N}]+ | \s+
 *
 * Calls `callback` for each chunk with (start, length, user_data).
 * This gives BPE smaller, linguistically-sensible pieces to work on. */
typedef void (*tk_pretok_cb)(const uint8_t *start, size_t len, void *ud);

void tk_pretokenize(const uint8_t *text, size_t len, tk_pretok_cb cb, void *ud);

#endif /* TK_UNICODE_H */