/*
 * ╔═══════════════════════════════════════════════════════════════════════╗
 * ║  io.c — High-Performance I/O for Turkish BPE Tokenizer                ║
 * ║                                                                       ║
 * ║  Memory-mapped file I/O, zero-copy line iteration, ring-buffered      ║
 * ║  corpus streaming with kernel prefetch hints, and chunked reading     ║
 * ║  that never breaks a line or a UTF-8 sequence at a boundary.          ║
 * ║                                                                       ║
 * ║  "In the beginning was the Word, and the Word was with bytes,         ║
 * ║   and the Word was bytes." — John 1:1, Systems Programming Edition    ║
 * ╚═══════════════════════════════════════════════════════════════════════╝
 *
 * Design principles:
 *   1. Zero-copy wherever possible. We do not copy data the kernel
 *      already placed in our address space. That would be coveting
 *      thy neighbor's buffer.
 *   2. Cache-friendly access patterns. Sequential reads, prefetched
 *      pages, and ring buffers that keep the TLB happy.
 *   3. No allocation on the hot path. The ring buffer is allocated
 *      once and reused until the end of days (or tk_ring_free).
 *   4. Fail loudly. Every function returns a status or NULL. We do
 *      not silently produce garbage — that way lies madness and
 *      off-by-one Unicode errors.
 */

#include "io.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ─────────────────────────────────────────────────────────────────────
 *  Platform detection & intrinsics
 *
 *  We conditionally pull in SIMD headers for newline scanning.
 *  If your platform has neither SSE2 nor NEON, we fall back to
 *  memchr, which is still plenty fast — the Lord provides.
 * ───────────────────────────────────────────────────────────────────── */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#   include <immintrin.h>
#   define TK_HAS_SSE2 1
#elif defined(__aarch64__) || defined(__ARM_NEON)
#   include <arm_neon.h>
#   define TK_HAS_NEON 1
#endif

/* Prefetch hint macros — whisper to the CPU what cometh next */
#if defined(__GNUC__) || defined(__clang__)
#   define TK_PREFETCH_READ(addr)   __builtin_prefetch((addr), 0, 3)
#   define TK_PREFETCH_WRITE(addr)  __builtin_prefetch((addr), 1, 3)
#   define TK_LIKELY(x)             __builtin_expect(!!(x), 1)
#   define TK_UNLIKELY(x)           __builtin_expect(!!(x), 0)
#else
#   define TK_PREFETCH_READ(addr)   ((void)(addr))
#   define TK_PREFETCH_WRITE(addr)  ((void)(addr))
#   define TK_LIKELY(x)             (x)
#   define TK_UNLIKELY(x)           (x)
#endif

/* Page size — used for madvise alignment and ring buffer granularity.
 * 4096 on every platform we care about, but let us not be presumptuous. */
#ifndef TK_PAGE_SIZE
#   define TK_PAGE_SIZE 4096
#endif

/* Default prefetch distance in bytes — how far ahead we tell the
 * kernel to read. Two pages is conservative; four pages is aggressive.
 * "For where your prefetch is, there your cache hit will be also." */
#define TK_PREFETCH_DISTANCE (TK_PAGE_SIZE * 4)

/* Align `x` up to the nearest multiple of `align` (must be power of 2) */
#define TK_ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align) - 1))


/* ═══════════════════════════════════════════════════════════════════════
 *  § 1. SIMD Newline Scanner
 *
 *  Finding newlines in bulk is the inner loop of every line iterator
 *  and chunk boundary finder. memchr is good. Sixteen bytes at a time
 *  is better. The serpent tempted us with premature optimization, and
 *  for once, the serpent was right.
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * tk_find_newline — Locate the first '\n' in [start, start+len).
 *
 * Returns a pointer to the newline, or NULL if none found.
 * Uses SSE2/NEON when available, falls back to memchr otherwise.
 *
 * On a 12KB Dostoyevsky chapter this makes no measurable difference.
 * On a 30GB OSCAR corpus, it shaves real seconds off chunk scanning.
 */
static inline const uint8_t *tk_find_newline(const uint8_t *start, size_t len) {
#if defined(TK_HAS_SSE2)
    /*
     * SSE2 path: compare 16 bytes at a time against '\n'.
     * Process the unaligned head with memchr, then go wide.
     */
    const uint8_t *p = start;
    const uint8_t *end = start + len;

    /* Handle unaligned prefix up to 16-byte boundary */
    size_t prefix = (16 - ((uintptr_t)p & 15)) & 15;
    if (prefix > len) prefix = len;
    if (prefix > 0) {
        const uint8_t *found = (const uint8_t *)memchr(p, '\n', prefix);
        if (found) return found;
        p += prefix;
    }

    /* Main SIMD loop — 16 bytes per iteration, 64 bytes per unrolled pass */
    __m128i needle = _mm_set1_epi8('\n');
    while (p + 64 <= end) {
        /* Prefetch the next cache line while we chew on this one.
         * "Give us this day our daily cache line." */
        TK_PREFETCH_READ(p + 256);

        __m128i v0 = _mm_load_si128((const __m128i *)(p));
        __m128i v1 = _mm_load_si128((const __m128i *)(p + 16));
        __m128i v2 = _mm_load_si128((const __m128i *)(p + 32));
        __m128i v3 = _mm_load_si128((const __m128i *)(p + 48));

        __m128i c0 = _mm_cmpeq_epi8(v0, needle);
        __m128i c1 = _mm_cmpeq_epi8(v1, needle);
        __m128i c2 = _mm_cmpeq_epi8(v2, needle);
        __m128i c3 = _mm_cmpeq_epi8(v3, needle);

        /* OR all comparison results — if any byte matched, the
         * combined mask is nonzero. Check the forest before the trees. */
        __m128i any = _mm_or_si128(_mm_or_si128(c0, c1),
                                   _mm_or_si128(c2, c3));
        if (_mm_movemask_epi8(any)) {
            /* At least one newline in this 64-byte block. Narrow it down. */
            int m0 = _mm_movemask_epi8(c0);
            if (m0) return p + __builtin_ctz(m0);
            int m1 = _mm_movemask_epi8(c1);
            if (m1) return p + 16 + __builtin_ctz(m1);
            int m2 = _mm_movemask_epi8(c2);
            if (m2) return p + 32 + __builtin_ctz(m2);
            int m3 = _mm_movemask_epi8(c3);
            if (m3) return p + 48 + __builtin_ctz(m3);
        }
        p += 64;
    }

    /* Remaining 0–63 bytes: single-vector passes */
    while (p + 16 <= end) {
        __m128i v = _mm_loadu_si128((const __m128i *)p);
        __m128i c = _mm_cmpeq_epi8(v, needle);
        int mask = _mm_movemask_epi8(c);
        if (mask) return p + __builtin_ctz(mask);
        p += 16;
    }

    /* Tail bytes — too few for SIMD */
    return (const uint8_t *)memchr(p, '\n', (size_t)(end - p));

#elif defined(TK_HAS_NEON)
    /*
     * NEON path: 16 bytes at a time, same principle.
     * ARM doesn't have movemask, so we reduce with horizontal max.
     */
    const uint8_t *p = start;
    const uint8_t *end = start + len;
    uint8x16_t needle = vdupq_n_u8('\n');

    while (p + 16 <= end) {
        TK_PREFETCH_READ(p + 256);
        uint8x16_t v = vld1q_u8(p);
        uint8x16_t cmp = vceqq_u8(v, needle);
        /* vmaxvq_u8 returns max across all lanes — nonzero iff match */
        if (vmaxvq_u8(cmp)) {
            /* Scalar scan to find exact position */
            for (size_t i = 0; i < 16 && p + i < end; i++) {
                if (p[i] == '\n') return p + i;
            }
        }
        p += 16;
    }

    return (const uint8_t *)memchr(p, '\n', (size_t)(end - p));

#else
    /* Scalar fallback. memchr is typically well-optimized by libc.
     * "The race is not to the swift, but memchr is still pretty fast." */
    return (const uint8_t *)memchr(start, '\n', len);
#endif
}

/*
 * tk_rfind_newline — Locate the LAST '\n' in [start, start+len).
 *
 * Scans backward from the end. Used for finding chunk boundaries
 * without scanning the entire chunk forward. We do not have SIMD
 * for the reverse scan — the chunks are small enough that scalar
 * suffices. Premature optimization in both directions would be gluttony.
 */
static inline const uint8_t *tk_rfind_newline(const uint8_t *start, size_t len) {
    const uint8_t *p = start + len;
    while (p > start) {
        p--;
        if (*p == '\n') return p;
    }
    return NULL;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 2. Memory-Mapped File
 *
 *  mmap is the covenant between userspace and kernel: "I shall not
 *  call read(), and thou shalt page in my data when I touch it."
 *  We add MADV_SEQUENTIAL because BPE training reads corpora front
 *  to back, and MADV_WILLNEED on the first few megabytes to prime
 *  the pump.
 * ═══════════════════════════════════════════════════════════════════════ */

int tk_mmap_open(tk_mmap_t *m, const char *path) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;  /* Initialize to invalid so tk_mmap_close is safe */

    int fd = open(path, O_RDONLY);
    if (TK_UNLIKELY(fd < 0)) {
        return -1;
    }

    struct stat st;
    if (TK_UNLIKELY(fstat(fd, &st) < 0)) {
        close(fd);
        return -1;
    }

    /* Reject empty files. There is nothing to tokenize in the void,
     * and mmap(0) is undefined behavior on some platforms. */
    size_t size = (size_t)st.st_size;
    if (TK_UNLIKELY(size == 0)) {
        close(fd);
        return -1;
    }

    void *ptr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (TK_UNLIKELY(ptr == MAP_FAILED)) {
        close(fd);
        return -1;
    }

    /* Sequential access hint — tells the kernel to read-ahead
     * aggressively and drop pages behind us. */
    madvise(ptr, size, MADV_SEQUENTIAL);

    /* Prime the first chunk into page cache. On spinning rust this
     * eliminates a cold-start seek penalty. On SSD it is harmless. */
    size_t prime = (size < (4 * 1024 * 1024)) ? size : (4 * 1024 * 1024);
    madvise(ptr, prime, MADV_WILLNEED);

    m->data = (uint8_t *)ptr;
    m->size = size;
    m->fd   = fd;
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


/* ═══════════════════════════════════════════════════════════════════════
 *  § 3. Line Iterator (Zero-Copy)
 *
 *  Yields lines from a memory-mapped file without copying a single
 *  byte. The returned pointers are directly into the mmap region.
 *  Do not free them. Do not write to them. "Thou shalt not covet
 *  thy mmap's bytes; thou shalt only read them."
 *
 *  The prefetch on each iteration tells the CPU to start loading
 *  the next line's cache lines while we process the current one.
 * ═══════════════════════════════════════════════════════════════════════ */

void tk_line_iter_init(tk_line_iter_t *it, const tk_mmap_t *m) {
    it->data   = m->data;
    it->size   = m->size;
    it->offset = 0;
}

bool tk_line_iter_next(tk_line_iter_t *it, const uint8_t **line, size_t *len) {
    if (TK_UNLIKELY(it->offset >= it->size)) return false;

    const uint8_t *start     = it->data + it->offset;
    const size_t   remaining = it->size - it->offset;

    /* Prefetch ahead — by the time we return this line to the caller
     * and they come back for the next one, the data will be warm. */
    if (remaining > TK_PREFETCH_DISTANCE) {
        TK_PREFETCH_READ(start + TK_PREFETCH_DISTANCE);
    }

    /* Use our SIMD-accelerated newline finder */
    const uint8_t *nl = tk_find_newline(start, remaining);

    if (nl) {
        *line = start;
        *len  = (size_t)(nl - start);
        it->offset = (size_t)(nl - it->data) + 1;  /* skip past '\n' */
    } else {
        /* Final line, no trailing newline.
         * "It is finished." — John 19:30 */
        *line = start;
        *len  = remaining;
        it->offset = it->size;
    }

    return true;
}

void tk_line_iter_reset(tk_line_iter_t *it) {
    it->offset = 0;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 4. Ring Buffer
 *
 *  A power-of-two ring buffer for streaming I/O. Used by the
 *  buffered reader (§ 5) to decouple file reads from consumer
 *  iteration, and by the chunk iterator for zero-copy windowing
 *  over data that doesn't come from mmap.
 *
 *  Memory layout:
 *
 *    ┌──────────────┬────────────┬──────────────┐
 *    │  consumed    │  readable  │  free space   │
 *    │  (garbage)   │  (valid)   │  (writable)   │
 *    └──────────────┴────────────┴──────────────┘
 *    ^              ^            ^                ^
 *    buf         rd_pos       wr_pos          capacity
 *
 *  The mask trick: `pos & (capacity - 1)` wraps without a branch.
 *  Works because capacity is always a power of two. As the Pharaoh
 *  learned, you cannot fight the mathematics of modular arithmetic.
 *
 *  We optionally use the mmap-mirror trick on Linux to create a
 *  virtually contiguous view of the ring buffer: the same physical
 *  pages are mapped twice, back-to-back. This means a read that
 *  wraps around the ring boundary can be done with a single memcpy
 *  or pointer dereference — no split handling needed.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Round up to next power of two */
static size_t next_pow2(size_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if __SIZEOF_SIZE_T__ >= 8
    v |= v >> 32;
#endif
    v++;
    return v;
}

/*
 * tk_ring_init_mirrored — Try to create a mirrored ring buffer.
 *
 * The mmap-mirror trick: we create an anonymous mapping of 2*capacity
 * bytes, then remap the second half to alias the same physical pages
 * as the first half. This means data at offset (capacity + x) is the
 * same physical memory as offset x. Reads that span the wrap point
 * Just Work™ without any memcpy gymnastics.
 *
 * "And the LORD said, let there be a contiguous address space,
 *  and there was a contiguous address space. And it was good."
 *
 * Returns 0 on success, -1 if the kernel denies our request
 * (in which case we fall back to a plain malloc buffer).
 */
static int tk_ring_init_mirrored(tk_ring_t *r, size_t capacity) {
#ifdef __linux__
    /* We need a file descriptor for the shared mapping. memfd_create
     * gives us an anonymous fd backed by RAM. */
    int fd = -1;

    #if defined(__NR_memfd_create) || defined(SYS_memfd_create)
        fd = memfd_create("tk_ring", 0);
    #endif

    if (fd < 0) {
        /* memfd_create unavailable (old kernel). Try shm_open fallback. */
        char name[64];
        snprintf(name, sizeof(name), "/tk_ring_%d_%p", getpid(), (void *)r);
        fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) shm_unlink(name);  /* Unlink immediately; fd keeps it alive */
    }

    if (fd < 0) return -1;

    if (ftruncate(fd, (off_t)capacity) < 0) {
        close(fd);
        return -1;
    }

    /* Reserve 2*capacity of contiguous virtual address space */
    void *base = mmap(NULL, capacity * 2, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        close(fd);
        return -1;
    }

    /* Map the first half to our fd */
    void *first = mmap(base, capacity, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED, fd, 0);
    if (first == MAP_FAILED) {
        munmap(base, capacity * 2);
        close(fd);
        return -1;
    }

    /* Map the second half to the SAME fd — the mirror.
     * Same physical pages, different virtual addresses.
     * This is the miracle of the loaves and fishes, but for RAM. */
    void *second = mmap((uint8_t *)base + capacity, capacity,
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_FIXED, fd, 0);
    if (second == MAP_FAILED) {
        munmap(base, capacity * 2);
        close(fd);
        return -1;
    }

    close(fd);  /* Pages stay mapped; fd no longer needed */

    r->buf      = (uint8_t *)base;
    r->capacity = capacity;
    r->mask     = capacity - 1;
    r->rd_pos   = 0;
    r->wr_pos   = 0;
    r->mirrored = true;
    return 0;

#else
    /* Non-Linux: mirrored mapping is platform-specific.
     * macOS could use mach_vm_remap, but we keep it simple. */
    (void)r; (void)capacity;
    return -1;
#endif
}

int tk_ring_init(tk_ring_t *r, size_t min_capacity) {
    memset(r, 0, sizeof(*r));

    /* Enforce minimum and round to power of two for masking */
    size_t capacity = next_pow2(min_capacity < TK_PAGE_SIZE
                                 ? TK_PAGE_SIZE
                                 : min_capacity);

    /* Try the mirrored mapping first — it is strictly superior */
    if (tk_ring_init_mirrored(r, capacity) == 0) {
        return 0;
    }

    /* Fallback: plain heap buffer. Still fast, just needs careful
     * wrap handling on reads that span the boundary. */
    r->buf = (uint8_t *)malloc(capacity);
    if (!r->buf) return -1;

    r->capacity = capacity;
    r->mask     = capacity - 1;
    r->rd_pos   = 0;
    r->wr_pos   = 0;
    r->mirrored = false;
    return 0;
}

void tk_ring_free(tk_ring_t *r) {
    if (!r->buf) return;

    if (r->mirrored) {
        munmap(r->buf, r->capacity * 2);
    } else {
        free(r->buf);
    }

    memset(r, 0, sizeof(*r));
}

/*
 * Ring buffer query functions — inlined for the hot path.
 * "Ask, and the available bytes shall be given unto you." — Matthew 7:7
 */
size_t tk_ring_readable(const tk_ring_t *r) {
    return r->wr_pos - r->rd_pos;
}

size_t tk_ring_writable(const tk_ring_t *r) {
    return r->capacity - (r->wr_pos - r->rd_pos);
}

/*
 * tk_ring_write_ptr — Get a pointer where new data can be written.
 *
 * For mirrored buffers, you can always write `writable` bytes
 * contiguously. For non-mirrored buffers, you can write up to
 * min(writable, capacity - (wr_pos & mask)) bytes before wrapping.
 *
 * Returns the number of contiguous bytes available via *out_len.
 */
uint8_t *tk_ring_write_ptr(tk_ring_t *r, size_t *out_len) {
    size_t avail = tk_ring_writable(r);
    size_t offset = r->wr_pos & r->mask;

    if (r->mirrored) {
        /* Mirror guarantees contiguity across the wrap point */
        *out_len = avail;
    } else {
        /* Non-mirrored: can only go to end of physical buffer */
        size_t to_end = r->capacity - offset;
        *out_len = (avail < to_end) ? avail : to_end;
    }

    return r->buf + offset;
}

/*
 * tk_ring_commit_write — Advance the write position after data has
 * been written into the region returned by tk_ring_write_ptr.
 */
void tk_ring_commit_write(tk_ring_t *r, size_t nbytes) {
    r->wr_pos += nbytes;
}

/*
 * tk_ring_read_ptr — Get a pointer to readable data.
 *
 * Same contiguity semantics as write_ptr. For mirrored buffers,
 * you always get a contiguous view of all readable data.
 */
const uint8_t *tk_ring_read_ptr(const tk_ring_t *r, size_t *out_len) {
    size_t avail = tk_ring_readable(r);
    size_t offset = r->rd_pos & r->mask;

    if (r->mirrored) {
        *out_len = avail;
    } else {
        size_t to_end = r->capacity - offset;
        *out_len = (avail < to_end) ? avail : to_end;
    }

    return r->buf + offset;
}

/*
 * tk_ring_commit_read — Advance the read position (consume data).
 * The consumed region becomes writable space.
 */
void tk_ring_commit_read(tk_ring_t *r, size_t nbytes) {
    r->rd_pos += nbytes;
}

/*
 * tk_ring_peek — Read data from the ring into a linear output buffer
 * without advancing the read position.
 *
 * Handles wrap-around for non-mirrored buffers by doing two memcpys.
 * Returns the number of bytes actually peeked (may be less than
 * requested if the ring doesn't have enough data).
 */
size_t tk_ring_peek(const tk_ring_t *r, uint8_t *dst, size_t nbytes) {
    size_t avail = tk_ring_readable(r);
    if (nbytes > avail) nbytes = avail;
    if (nbytes == 0) return 0;

    size_t offset = r->rd_pos & r->mask;

    if (r->mirrored) {
        /* Mirrored: always contiguous. One memcpy to rule them all. */
        memcpy(dst, r->buf + offset, nbytes);
    } else {
        size_t first = r->capacity - offset;
        if (first >= nbytes) {
            memcpy(dst, r->buf + offset, nbytes);
        } else {
            /* Split read: end of buffer, then beginning.
             * "A house divided against itself cannot stand,
             *  but a ring buffer divided can, with two memcpys." */
            memcpy(dst, r->buf + offset, first);
            memcpy(dst + first, r->buf, nbytes - first);
        }
    }

    return nbytes;
}

/*
 * tk_ring_discard — Drop bytes from the readable region without
 * copying them anywhere. Used to skip past consumed data.
 */
void tk_ring_discard(tk_ring_t *r, size_t nbytes) {
    size_t avail = tk_ring_readable(r);
    if (nbytes > avail) nbytes = avail;
    r->rd_pos += nbytes;
}

/*
 * tk_ring_reset — Empty the ring buffer, restoring it to its
 * initial state. The allocation is preserved.
 * "Behold, I make all things new." — Revelation 21:5
 */
void tk_ring_reset(tk_ring_t *r) {
    r->rd_pos = 0;
    r->wr_pos = 0;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 5. Buffered File Reader
 *
 *  For when mmap is unavailable or undesirable (pipes, stdin,
 *  network streams, files on exotic filesystems), this reader
 *  uses the ring buffer to provide a streaming interface with
 *  configurable read-ahead.
 *
 *  The pattern: fill the ring from the fd, scan for line/chunk
 *  boundaries in the readable region, yield data, consume it,
 *  repeat until EOF.
 * ═══════════════════════════════════════════════════════════════════════ */

int tk_buffered_reader_init(tk_buffered_reader_t *br, int fd,
                            size_t buffer_size) {
    memset(br, 0, sizeof(*br));

    if (tk_ring_init(&br->ring, buffer_size) < 0) {
        return -1;
    }

    br->fd  = fd;
    br->eof = false;
    return 0;
}

void tk_buffered_reader_free(tk_buffered_reader_t *br) {
    tk_ring_free(&br->ring);
    br->fd  = -1;
    br->eof = true;
}

/*
 * tk_buffered_reader_fill — Pull data from the fd into the ring.
 *
 * Reads as much as the ring can hold in one syscall. Returns the
 * number of bytes read, 0 on EOF, or -1 on error.
 *
 * "Man shall not live by mmap alone, but by every read() that
 *  proceeds from the file descriptor." — Deuteronomy 8:3, abridged
 */
ssize_t tk_buffered_reader_fill(tk_buffered_reader_t *br) {
    if (br->eof) return 0;

    size_t space;
    uint8_t *dst = tk_ring_write_ptr(&br->ring, &space);

    if (space == 0) return 0;  /* Ring is full, consumer must drain it */

    ssize_t n = read(br->fd, dst, space);
    if (n < 0) return -1;      /* Error — the plagues of Egypt */
    if (n == 0) {
        br->eof = true;
        return 0;              /* EOF — we have reached the Promised Land */
    }

    tk_ring_commit_write(&br->ring, (size_t)n);
    return n;
}

/*
 * tk_buffered_reader_getline — Read one line from the buffered reader.
 *
 * Searches the ring for a newline. If found, copies the line (without
 * the newline) into `dst` and returns the line length. If no newline
 * is present, fills the ring further and retries.
 *
 * Returns:
 *   > 0 : line length (not including newline)
 *     0 : EOF reached with no more data
 *    -1 : error (line exceeds dst_cap, or read error)
 */
ssize_t tk_buffered_reader_getline(tk_buffered_reader_t *br,
                                    uint8_t *dst, size_t dst_cap) {
    for (;;) {
        /* Scan the readable region for a newline */
        size_t readable_len;
        const uint8_t *readable = tk_ring_read_ptr(&br->ring, &readable_len);

        if (readable_len > 0) {
            const uint8_t *nl = tk_find_newline(readable, readable_len);
            if (nl) {
                size_t line_len = (size_t)(nl - readable);
                if (line_len > dst_cap) return -1;  /* Line too long */
                memcpy(dst, readable, line_len);
                tk_ring_commit_read(&br->ring, line_len + 1);  /* +1 for '\n' */
                return (ssize_t)line_len;
            }
        }

        /* No newline found yet. Try to read more data. */
        if (br->eof) {
            /* No more data coming. Yield what remains as the last line. */
            if (readable_len == 0) return 0;
            if (readable_len > dst_cap) return -1;
            memcpy(dst, readable, readable_len);
            tk_ring_commit_read(&br->ring, readable_len);
            return (ssize_t)readable_len;
        }

        ssize_t filled = tk_buffered_reader_fill(br);
        if (filled < 0) return -1;
        if (filled == 0 && !br->eof) continue;  /* Spurious empty read */
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 6. Chunked Corpus Reader
 *
 *  Yields chunks of approximately `chunk_size` bytes from a memory-
 *  mapped file, always splitting on newline boundaries so we never
 *  bisect a line (or a UTF-8 sequence within a line). This is the
 *  primary interface for BPE training on large corpora.
 *
 *  Algorithm:
 *    1. From current offset, look ahead chunk_size bytes.
 *    2. Scan BACKWARD from that point to find the last newline.
 *    3. If no newline found (pathologically long line), scan FORWARD.
 *    4. Include the newline in the chunk.
 *
 *  The backward scan uses tk_rfind_newline (scalar). The forward
 *  scan uses tk_find_newline (SIMD). In practice, the backward
 *  scan almost always finds a newline within a few hundred bytes
 *  because natural text has short lines. The forward scan is the
 *  "in case of emergency break glass" path.
 *
 *  "Divide the corpus into portions for seven, yes for eight,
 *   for you do not know what disaster may come upon the heap."
 *   — Ecclesiastes 11:2, Systems Programming Edition
 * ═══════════════════════════════════════════════════════════════════════ */

void tk_chunk_iter_init(tk_chunk_iter_t *it, const tk_mmap_t *m,
                        size_t chunk_size) {
    it->data       = m->data;
    it->size       = m->size;
    it->offset     = 0;
    it->chunk_size = chunk_size;
}

bool tk_chunk_iter_next(tk_chunk_iter_t *it, const uint8_t **chunk,
                        size_t *len) {
    if (TK_UNLIKELY(it->offset >= it->size)) return false;

    const uint8_t *start     = it->data + it->offset;
    const size_t   remaining = it->size - it->offset;

    /* Prefetch the next chunk while the caller processes this one */
    if (remaining > it->chunk_size + TK_PREFETCH_DISTANCE) {
        TK_PREFETCH_READ(start + it->chunk_size);
    }

    /* Last chunk: take everything that remains */
    if (remaining <= it->chunk_size) {
        *chunk     = start;
        *len       = remaining;
        it->offset = it->size;
        return true;
    }

    /*
     * Find the last newline within chunk_size bytes. Scan backward
     * from the boundary — this is typically very fast because text
     * lines are short (mean ~40 bytes in Turkish prose).
     */
    const uint8_t *boundary_ptr = tk_rfind_newline(start, it->chunk_size);

    size_t boundary;
    if (boundary_ptr) {
        /* Found. Include the newline in this chunk. */
        boundary = (size_t)(boundary_ptr - start) + 1;
    } else {
        /*
         * No newline in chunk_size bytes. This is a pathologically
         * long line — possibly a base64 blob or a single-paragraph
         * document. Scan forward to find the next newline.
         *
         * "Woe to you who append line to line without a newline
         *  character." — Isaiah 5:8, approximately
         */
        const uint8_t *nl = tk_find_newline(start + it->chunk_size,
                                             remaining - it->chunk_size);
        if (nl) {
            boundary = (size_t)(nl - start) + 1;
        } else {
            /* No newline at all in the entire remaining data.
             * Take everything. This is the end times. */
            boundary = remaining;
        }
    }

    *chunk      = start;
    *len        = boundary;
    it->offset += boundary;
    return true;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 7. File Utilities
 *
 *  Simple helpers for read-all-at-once and write-all-at-once
 *  operations. These are not on the hot path — they are for loading
 *  model files and writing test fixtures.
 *
 *  "There is a time for mmap, and a time for fread."
 *   — Ecclesiastes 3:1, give or take
 * ═══════════════════════════════════════════════════════════════════════ */

size_t tk_file_size(const char *path) {
    struct stat st;
    if (TK_UNLIKELY(stat(path, &st) < 0)) return (size_t)-1;
    return (size_t)st.st_size;
}

/*
 * tk_read_file — Slurp an entire file into a heap-allocated buffer.
 *
 * Returns NULL on failure. Caller owns the returned pointer and must
 * free() it. The file size is written to *out_len.
 *
 * We use a single fread rather than a read loop because fread handles
 * short reads internally. For files under 2GB this is fine. For files
 * over 2GB, you should be using mmap anyway, and if you are using
 * fread on a 30GB file, may God have mercy on your page cache.
 */
uint8_t *tk_read_file(const char *path, size_t *out_len) {
    int fd = open(path, O_RDONLY);
    if (TK_UNLIKELY(fd < 0)) return NULL;

    struct stat st;
    if (TK_UNLIKELY(fstat(fd, &st) < 0)) {
        close(fd);
        return NULL;
    }

    size_t sz = (size_t)st.st_size;
    if (TK_UNLIKELY(sz == 0)) {
        close(fd);
        return NULL;
    }

    uint8_t *buf = (uint8_t *)malloc(sz);
    if (TK_UNLIKELY(!buf)) {
        close(fd);
        return NULL;
    }

    /* Read in a loop to handle partial reads correctly.
     * "The wise man builds his reader on the rock of EINTR handling." */
    size_t total = 0;
    while (total < sz) {
        ssize_t n = read(fd, buf + total, sz - total);
        if (n < 0) {
            if (errno == EINTR) continue;  /* Interrupted — try again */
            free(buf);
            close(fd);
            return NULL;
        }
        if (n == 0) break;  /* Unexpected EOF (file truncated?) */
        total += (size_t)n;
    }

    close(fd);

    if (TK_UNLIKELY(total != sz)) {
        free(buf);
        return NULL;
    }

    *out_len = sz;
    return buf;
}

/*
 * tk_write_file — Write a buffer to a file atomically.
 *
 * We write to a temporary file and rename, so readers never see a
 * partially written model file. "Let your writes be 'yes' complete
 * or 'no' absent. Anything in between comes from the evil one."
 *
 * Falls back to direct write if rename fails (cross-device, etc.).
 */
int tk_write_file(const char *path, const uint8_t *data, size_t len) {
    /* Build temp path: <path>.tmp.XXXXXX */
    size_t path_len = strlen(path);
    char *tmp_path = (char *)malloc(path_len + 12);
    if (!tmp_path) return -1;
    snprintf(tmp_path, path_len + 12, "%s.XXXXXX", path);

    int fd = mkstemp(tmp_path);
    if (fd < 0) {
        free(tmp_path);
        /* Fallback: direct write without atomicity */
        goto direct_write;
    }

    /* Write in a loop to handle partial writes */
    size_t total = 0;
    while (total < len) {
        ssize_t n = write(fd, data + total, len - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            close(fd);
            unlink(tmp_path);
            free(tmp_path);
            return -1;
        }
        total += (size_t)n;
    }

    /* fsync to ensure data hits the disk before rename.
     * "Trust in the LORD with all your heart, and lean not on
     *  the OS buffer cache." — Proverbs 3:5, for the paranoid */
    fsync(fd);
    close(fd);

    /* Atomic rename */
    if (rename(tmp_path, path) < 0) {
        unlink(tmp_path);
        free(tmp_path);
        goto direct_write;
    }

    free(tmp_path);
    return 0;

direct_write:
    {
        FILE *f = fopen(path, "wb");
        if (!f) return -1;
        size_t written = fwrite(data, 1, len, f);
        fclose(f);
        return (written == len) ? 0 : -1;
    }
}