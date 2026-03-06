/*
 * io.c — Memory-mapped file I/O, line iteration, and chunked corpus
 *         reading for BPE training on large text files.
 */

#include "io.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ════════════════════════════════════════════════════════════════════════
 *  Memory-Mapped File
 * ════════════════════════════════════════════════════════════════════════ */

int tk_mmap_open(tk_mmap_t *m, const char *path) {
    memset(m, 0, sizeof(*m));

    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return -1;
    }

    size_t size = (size_t)st.st_size;
    if (size == 0) {
        close(fd);
        return -1;
    }

    void *ptr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        return -1;
    }

    /* Hint the kernel we'll read sequentially */
    madvise(ptr, size, MADV_SEQUENTIAL);

    m->data = (uint8_t *)ptr;
    m->size = size;
    m->fd = fd;
    return 0;
}

void tk_mmap_close(tk_mmap_t *m) {
    if (m->data) {
        munmap(m->data, m->size);
        m->data = NULL;
    }
    if (m->fd >= 0) {
        close(m->fd);
        m->fd = -1;
    }
    m->size = 0;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Line Iterator
 * ════════════════════════════════════════════════════════════════════════ */

void tk_line_iter_init(tk_line_iter_t *it, const tk_mmap_t *m) {
    it->data = m->data;
    it->size = m->size;
    it->offset = 0;
}

bool tk_line_iter_next(tk_line_iter_t *it, const uint8_t **line, size_t *len) {
    if (it->offset >= it->size) return false;

    const uint8_t *start = it->data + it->offset;
    const uint8_t *end = it->data + it->size;

    /* Find the next newline */
    const uint8_t *nl = memchr(start, '\n', (size_t)(end - start));

    if (nl) {
        *line = start;
        *len = (size_t)(nl - start);
        it->offset = (size_t)(nl - it->data) + 1; /* skip past \n */
    } else {
        /* Last line without trailing newline */
        *line = start;
        *len = (size_t)(end - start);
        it->offset = it->size;
    }

    return true;
}

void tk_line_iter_reset(tk_line_iter_t *it) {
    it->offset = 0;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Chunked Corpus Reader
 *
 *  Yields chunks of approximately `chunk_size` bytes, always ending
 *  on a newline boundary so we never split a line (or a UTF-8 sequence
 *  within a line) across chunks.
 * ════════════════════════════════════════════════════════════════════════ */

void tk_chunk_iter_init(tk_chunk_iter_t *it, const tk_mmap_t *m, size_t chunk_size) {
    it->data = m->data;
    it->size = m->size;
    it->offset = 0;
    it->chunk_size = chunk_size;
}

bool tk_chunk_iter_next(tk_chunk_iter_t *it, const uint8_t **chunk, size_t *len) {
    if (it->offset >= it->size) return false;

    const uint8_t *start = it->data + it->offset;
    size_t remaining = it->size - it->offset;

    if (remaining <= it->chunk_size) {
        /* Last chunk: take everything */
        *chunk = start;
        *len = remaining;
        it->offset = it->size;
        return true;
    }

    /* Find the last newline within chunk_size bytes */
    size_t boundary = it->chunk_size;
    const uint8_t *search_start = start;

    /* Scan backward from the target boundary to find a newline */
    while (boundary > 0 && search_start[boundary] != '\n') {
        boundary--;
    }

    if (boundary == 0) {
        /* No newline found within chunk_size — scan forward instead.
         * This handles pathologically long lines. */
        const uint8_t *end = it->data + it->size;
        const uint8_t *nl = memchr(start, '\n', (size_t)(end - start));
        if (nl) {
            boundary = (size_t)(nl - start);
        } else {
            boundary = remaining; /* no newline at all, take the rest */
        }
    }

    /* Include the newline itself in this chunk */
    boundary++;

    *chunk = start;
    *len = boundary;
    it->offset += boundary;
    return true;
}

/* ════════════════════════════════════════════════════════════════════════
 *  File Utilities
 * ════════════════════════════════════════════════════════════════════════ */

size_t tk_file_size(const char *path) {
    struct stat st;
    if (stat(path, &st) < 0) return (size_t)-1;
    return (size_t)st.st_size;
}

uint8_t *tk_read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);

    uint8_t *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }

    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);

    if (rd != (size_t)sz) {
        free(buf);
        return NULL;
    }

    *out_len = (size_t)sz;
    return buf;
}

int tk_write_file(const char *path, const uint8_t *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    size_t written = fwrite(data, 1, len, f);
    fclose(f);

    return (written == len) ? 0 : -1;
}