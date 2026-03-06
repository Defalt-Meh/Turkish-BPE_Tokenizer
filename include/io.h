#ifndef TK_IO_H
#define TK_IO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * io.h — Memory-mapped file reading and corpus streaming.
 *
 * For BPE training we need to iterate over multi-GB text files without
 * loading them entirely into RAM. We use mmap for random access and
 * provide a line iterator for sequential processing.
 */

/* ── Memory-mapped file ──────────────────────────────────────────────── */

typedef struct {
    uint8_t *data;     /* mapped region */
    size_t   size;     /* file size in bytes */
    int      fd;       /* file descriptor (kept open until unmap) */
} tk_mmap_t;

/* Map a file read-only. Returns 0 on success, -1 on error. */
int tk_mmap_open(tk_mmap_t *m, const char *path);

/* Unmap and close. */
void tk_mmap_close(tk_mmap_t *m);

/* ── Line iterator over mmap'd file ──────────────────────────────────── */

typedef struct {
    const uint8_t *data;
    size_t         size;
    size_t         offset;  /* current read position */
} tk_line_iter_t;

/* Initialize a line iterator from a mapped file. */
void tk_line_iter_init(tk_line_iter_t *it, const tk_mmap_t *m);

/* Get the next line. Sets *line and *len. Returns true if a line was
 * read, false at EOF. Lines do NOT include the trailing newline. */
bool tk_line_iter_next(tk_line_iter_t *it, const uint8_t **line, size_t *len);

/* Reset iterator to the beginning. */
void tk_line_iter_reset(tk_line_iter_t *it);

/* ── Chunked corpus reader ───────────────────────────────────────────── */

/*
 * For BPE training we want to process the corpus in chunks (e.g. 64MB)
 * to keep pair-counting hash tables at a manageable size. The chunk
 * reader yields aligned chunks that don't split UTF-8 sequences.
 */

typedef struct {
    const uint8_t *data;
    size_t         size;
    size_t         offset;
    size_t         chunk_size;  /* target chunk size in bytes */
} tk_chunk_iter_t;

/* Initialize a chunk iterator. `chunk_size` is the target size per chunk. */
void tk_chunk_iter_init(tk_chunk_iter_t *it, const tk_mmap_t *m, size_t chunk_size);

/* Get the next chunk. Sets *chunk and *len. Returns true if data remains.
 * Chunks are guaranteed to end on a newline boundary (not mid-line). */
bool tk_chunk_iter_next(tk_chunk_iter_t *it, const uint8_t **chunk, size_t *len);

/* ── Utility ─────────────────────────────────────────────────────────── */

/* Get file size without opening. Returns (size_t)-1 on error. */
size_t tk_file_size(const char *path);

/* Read an entire file into a malloc'd buffer. Caller frees.
 * Sets *out_len to byte count. Returns NULL on error. */
uint8_t *tk_read_file(const char *path, size_t *out_len);

/* Write a buffer to a file. Returns 0 on success. */
int tk_write_file(const char *path, const uint8_t *data, size_t len);

#endif /* TK_IO_H */