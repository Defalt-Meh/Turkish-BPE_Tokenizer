/*
 * encode.c — Encode UTF-8 text into BPE token IDs.
 *
 * Usage:
 *   echo "Merhaba dünya" | ./tools/encode -m <model.tkmodel>
 *   ./tools/encode -m <model.tkmodel> -i <input.txt> -o <output.ids>
 *
 * Options:
 *   -m <path>     Model file (required)
 *   -i <path>     Input text file (default: stdin)
 *   -o <path>     Output file (default: stdout)
 *   -s            Show token strings alongside IDs
 *   -c            Comma-separated output (default: one per line)
 *   -b            Bare binary output (4-byte LE per token, no text)
 *   -q            Quiet — suppress stats on stderr
 *   -h            Show help
 */

#include "tokenizer.h"
#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ── Output buffer ────────────────────────────────────────────────────
 *
 * fprintf per token is catastrophically slow on large inputs. We
 * format into a 256 KiB buffer and flush with a single write().
 * On a 30 MB corpus this is the difference between 2 seconds and
 * 200 milliseconds.
 */

#define OUT_BUF_SIZE (256 * 1024)

typedef struct {
    uint8_t  buf[OUT_BUF_SIZE];
    size_t   pos;
    int      fd;
} out_buf_t;

static void ob_init(out_buf_t *ob, int fd) {
    ob->pos = 0;
    ob->fd  = fd;
}

static void ob_flush(out_buf_t *ob) {
    if (ob->pos == 0) return;
    size_t total = 0;
    while (total < ob->pos) {
        ssize_t n = write(ob->fd, ob->buf + total, ob->pos - total);
        if (n <= 0) break;
        total += (size_t)n;
    }
    ob->pos = 0;
}

static inline void ob_ensure(out_buf_t *ob, size_t need) {
    if (__builtin_expect(ob->pos + need > OUT_BUF_SIZE, 0))
        ob_flush(ob);
}

/* Write raw bytes. */
static inline void ob_write(out_buf_t *ob, const void *data, size_t len) {
    ob_ensure(ob, len);
    memcpy(ob->buf + ob->pos, data, len);
    ob->pos += len;
}

/* Write a uint32 as decimal ASCII. Hand-rolled, no sprintf overhead. */
static inline void ob_write_u32(out_buf_t *ob, uint32_t v) {
    ob_ensure(ob, 12);
    char tmp[12];
    int len = 0;
    if (v == 0) {
        tmp[len++] = '0';
    } else {
        while (v > 0) {
            tmp[len++] = '0' + (char)(v % 10);
            v /= 10;
        }
    }
    /* Reverse into output buffer directly. */
    uint8_t *dst = ob->buf + ob->pos;
    for (int i = len - 1; i >= 0; i--)
        *dst++ = (uint8_t)tmp[i];
    ob->pos += (size_t)len;
}

/* Write a single byte. */
static inline void ob_putc(out_buf_t *ob, uint8_t c) {
    ob_ensure(ob, 1);
    ob->buf[ob->pos++] = c;
}

/* Write a token's bytes in escaped human-readable form. */
static void ob_write_token_str(out_buf_t *ob, const tk_tokenizer_t *tk,
                                uint32_t id) {
    uint16_t len;
    const uint8_t *bytes = tk_token_bytes(tk, id, &len);
    if (!bytes) {
        ob_write(ob, "<UNK>", 5);
        return;
    }

    ob_putc(ob, '"');
    for (uint16_t i = 0; i < len; i++) {
        ob_ensure(ob, 4);
        uint8_t b = bytes[i];
        if (b == '"')       { ob->buf[ob->pos++] = '\\'; ob->buf[ob->pos++] = '"'; }
        else if (b == '\\') { ob->buf[ob->pos++] = '\\'; ob->buf[ob->pos++] = '\\'; }
        else if (b == '\n') { ob->buf[ob->pos++] = '\\'; ob->buf[ob->pos++] = 'n'; }
        else if (b == '\t') { ob->buf[ob->pos++] = '\\'; ob->buf[ob->pos++] = 't'; }
        else if (b >= 0x20 && b < 0x7F) { ob->buf[ob->pos++] = b; }
        else {
            static const char hex[] = "0123456789ABCDEF";
            ob->buf[ob->pos++] = '\\';
            ob->buf[ob->pos++] = 'x';
            ob->buf[ob->pos++] = (uint8_t)hex[b >> 4];
            ob->buf[ob->pos++] = (uint8_t)hex[b & 0xF];
        }
    }
    ob_putc(ob, '"');
}


/* ── Stdin reader ─────────────────────────────────────────────────── */

static uint8_t *read_stdin(size_t *out_len) {
    size_t cap = 64 * 1024;
    size_t len = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return NULL;

    for (;;) {
        ssize_t n = read(STDIN_FILENO, buf + len, cap - len);
        if (n < 0)  { free(buf); return NULL; }
        if (n == 0) break;
        len += (size_t)n;
        if (len == cap) {
            cap *= 2;
            uint8_t *tmp = (uint8_t *)realloc(buf, cap);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
    }

    *out_len = len;
    return buf;
}


/* ── Usage ────────────────────────────────────────────────────────── */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -m <model> [-i input] [-o output] [options]\n"
        "\n"
        "Options:\n"
        "  -m <path>   Trained .tkmodel file (required)\n"
        "  -i <path>   Input text file (default: stdin)\n"
        "  -o <path>   Output file (default: stdout)\n"
        "  -s          Show token byte strings alongside IDs\n"
        "  -c          Comma-separated output\n"
        "  -b          Binary output (4-byte LE uint32 per token)\n"
        "  -q          Quiet — no stats on stderr\n"
        "  -h          Show this help\n"
        "\n"
        "Example:\n"
        "  echo 'Istanbul cok guzel.' | %s -m models/turkish.tkmodel -s\n",
        prog, prog);
}


/* ── Main ─────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *model_path  = NULL;
    const char *input_path  = NULL;
    const char *output_path = NULL;
    bool show_strings = false;
    bool csv_mode     = false;
    bool binary_mode  = false;
    bool quiet        = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0) {
            show_strings = true;
        } else if (strcmp(argv[i], "-c") == 0) {
            csv_mode = true;
        } else if (strcmp(argv[i], "-b") == 0) {
            binary_mode = true;
        } else if (strcmp(argv[i], "-q") == 0) {
            quiet = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "error: -m <model> is required\n");
        return 1;
    }

    /* Load model. */
    tk_tokenizer_t tk;
    if (tk_load(&tk, model_path) < 0) {
        fprintf(stderr, "error: failed to load '%s'\n", model_path);
        return 1;
    }

    /* Read input — mmap for files, buffered read() for stdin. */
    tk_mmap_t mmap_file;
    uint8_t *stdin_buf    = NULL;
    const uint8_t *text   = NULL;
    size_t text_len       = 0;
    bool using_mmap       = false;

    if (input_path) {
        if (tk_mmap_open(&mmap_file, input_path) < 0) {
            fprintf(stderr, "error: cannot open '%s'\n", input_path);
            tk_free(&tk);
            return 1;
        }
        text      = mmap_file.data;
        text_len  = mmap_file.size;
        using_mmap = true;
    } else {
        stdin_buf = read_stdin(&text_len);
        if (!stdin_buf) {
            fprintf(stderr, "error: failed to read stdin\n");
            tk_free(&tk);
            return 1;
        }
        text = stdin_buf;
    }

    /* Allocate output token buffer. Worst case: one token per byte. */
    uint32_t *ids = (uint32_t *)malloc((text_len + 1) * sizeof(uint32_t));
    if (!ids) {
        fprintf(stderr, "error: allocation failed\n");
        goto fail;
    }

    /* Encode. */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    size_t num_tokens = tk_encode(&tk, text, text_len, ids, text_len + 1);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    if (num_tokens == (size_t)-1) {
        fprintf(stderr, "error: encoding failed\n");
        goto fail;
    }

    /* Open output. */
    int out_fd = STDOUT_FILENO;
    FILE *out_file = NULL;
    if (output_path) {
        out_file = fopen(output_path, binary_mode ? "wb" : "w");
        if (!out_file) {
            fprintf(stderr, "error: cannot open '%s'\n", output_path);
            goto fail;
        }
        out_fd = fileno(out_file);
    }

    /* Write output. */
    if (binary_mode) {
        /* Raw LE uint32 — write the entire array in one call. */
        size_t total = num_tokens * sizeof(uint32_t);
        size_t written = 0;
        while (written < total) {
            ssize_t n = write(out_fd, (uint8_t *)ids + written, total - written);
            if (n <= 0) break;
            written += (size_t)n;
        }
    } else {
        /* Text output via buffered writer. */
        out_buf_t ob;
        ob_init(&ob, out_fd);

        for (size_t i = 0; i < num_tokens; i++) {
            if (csv_mode && i > 0) ob_putc(&ob, ',');

            ob_write_u32(&ob, ids[i]);

            if (show_strings) {
                ob_putc(&ob, '\t');
                ob_write_token_str(&ob, &tk, ids[i]);
            }

            if (!csv_mode) ob_putc(&ob, '\n');
        }

        if (csv_mode) ob_putc(&ob, '\n');

        ob_flush(&ob);
    }

    /* Stats. */
    if (!quiet) {
        double secs = (double)(t1.tv_sec - t0.tv_sec)
                    + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
        double mb_per_sec = secs > 0.0
            ? ((double)text_len / (1024.0 * 1024.0)) / secs : 0.0;

        fprintf(stderr, "tokens: %zu  bytes: %zu  ratio: %.2f B/tok",
                num_tokens, text_len,
                num_tokens > 0 ? (double)text_len / (double)num_tokens : 0.0);
        fprintf(stderr, "  time: %.3f ms  throughput: %.1f MiB/s\n",
                secs * 1000.0, mb_per_sec);
    }

    /* Cleanup. */
    if (out_file) fclose(out_file);
    free(ids);
    if (using_mmap) tk_mmap_close(&mmap_file);
    free(stdin_buf);
    tk_free(&tk);
    return 0;

fail:
    free(ids);
    if (using_mmap) tk_mmap_close(&mmap_file);
    free(stdin_buf);
    tk_free(&tk);
    return 1;
}