/*
 * ╔═══════════════════════════════════════════════════════════════════════╗
 * ║  io.h — Public I/O interface for the Turkish BPE Tokenizer          ║
 * ║                                                                     ║
 * ║  Memory-mapped files, zero-copy line iteration, ring-buffered       ║
 * ║  streaming, and chunked corpus reading.                             ║
 * ╚═══════════════════════════════════════════════════════════════════════╝
 */

#ifndef TK_IO_H
#define TK_IO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <unistd.h>   /* ssize_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────────────────────────────────────────────────────────────────
 *  Memory-Mapped File
 * ───────────────────────────────────────────────────────────────────── */

typedef struct {
    uint8_t *data;       /* Pointer to mapped region (read-only content) */
    size_t   size;       /* File size in bytes                           */
    int      fd;         /* Underlying file descriptor                   */
} tk_mmap_t;

int  tk_mmap_open(tk_mmap_t *m, const char *path);
void tk_mmap_close(tk_mmap_t *m);

/* ─────────────────────────────────────────────────────────────────────
 *  Line Iterator (zero-copy over mmap)
 * ───────────────────────────────────────────────────────────────────── */

typedef struct {
    const uint8_t *data;     /* Pointer to mmap data (not owned)    */
    size_t         size;     /* Total mapped size                   */
    size_t         offset;   /* Current read position               */
} tk_line_iter_t;

void tk_line_iter_init(tk_line_iter_t *it, const tk_mmap_t *m);
bool tk_line_iter_next(tk_line_iter_t *it, const uint8_t **line, size_t *len);
void tk_line_iter_reset(tk_line_iter_t *it);

/* ─────────────────────────────────────────────────────────────────────
 *  Ring Buffer
 *
 *  Power-of-two ring buffer with optional mmap-mirror trick for
 *  zero-copy contiguous reads across the wrap boundary.
 * ───────────────────────────────────────────────────────────────────── */

typedef struct {
    uint8_t *buf;        /* Buffer memory (heap or mmap-mirrored)       */
    size_t   capacity;   /* Always a power of two                       */
    size_t   mask;       /* capacity - 1, for branchless wrapping       */
    size_t   rd_pos;     /* Monotonically increasing read position      */
    size_t   wr_pos;     /* Monotonically increasing write position     */
    bool     mirrored;   /* True if using the mmap-mirror trick         */
} tk_ring_t;

int           tk_ring_init(tk_ring_t *r, size_t min_capacity);
void          tk_ring_free(tk_ring_t *r);
size_t        tk_ring_readable(const tk_ring_t *r);
size_t        tk_ring_writable(const tk_ring_t *r);
uint8_t      *tk_ring_write_ptr(tk_ring_t *r, size_t *out_len);
void          tk_ring_commit_write(tk_ring_t *r, size_t nbytes);
const uint8_t*tk_ring_read_ptr(const tk_ring_t *r, size_t *out_len);
void          tk_ring_commit_read(tk_ring_t *r, size_t nbytes);
size_t        tk_ring_peek(const tk_ring_t *r, uint8_t *dst, size_t nbytes);
void          tk_ring_discard(tk_ring_t *r, size_t nbytes);
void          tk_ring_reset(tk_ring_t *r);

/* ─────────────────────────────────────────────────────────────────────
 *  Buffered File Reader (ring-backed, for non-mmap sources)
 * ───────────────────────────────────────────────────────────────────── */

typedef struct {
    tk_ring_t ring;      /* Underlying ring buffer                      */
    int       fd;        /* Source file descriptor                      */
    bool      eof;       /* True once read() returns 0                  */
} tk_buffered_reader_t;

int     tk_buffered_reader_init(tk_buffered_reader_t *br, int fd,
                                size_t buffer_size);
void    tk_buffered_reader_free(tk_buffered_reader_t *br);
ssize_t tk_buffered_reader_fill(tk_buffered_reader_t *br);
ssize_t tk_buffered_reader_getline(tk_buffered_reader_t *br,
                                    uint8_t *dst, size_t dst_cap);

/* ─────────────────────────────────────────────────────────────────────
 *  Chunked Corpus Reader (zero-copy over mmap)
 * ───────────────────────────────────────────────────────────────────── */

typedef struct {
    const uint8_t *data;         /* Pointer to mmap data (not owned) */
    size_t         size;         /* Total mapped size                */
    size_t         offset;       /* Current read position            */
    size_t         chunk_size;   /* Approximate chunk size target    */
} tk_chunk_iter_t;

void tk_chunk_iter_init(tk_chunk_iter_t *it, const tk_mmap_t *m,
                        size_t chunk_size);
bool tk_chunk_iter_next(tk_chunk_iter_t *it, const uint8_t **chunk,
                        size_t *len);

/* ─────────────────────────────────────────────────────────────────────
 *  File Utilities
 * ───────────────────────────────────────────────────────────────────── */

size_t   tk_file_size(const char *path);
uint8_t *tk_read_file(const char *path, size_t *out_len);
int      tk_write_file(const char *path, const uint8_t *data, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* TK_IO_H */