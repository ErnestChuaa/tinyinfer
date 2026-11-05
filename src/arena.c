#include "tinyinfer.h"
#include <stdlib.h>
#include <string.h>

int ti_arena_init(ti_arena *a, size_t cap) {
    /* 64-byte aligned base so every allocation can start on a cache line. */
    a->base = aligned_alloc(64, (cap + 63) & ~(size_t)63);
    if (!a->base) return -1;
    a->cap = cap;
    a->used = a->peak = 0;
    return 0;
}

void ti_arena_free(ti_arena *a) {
    free(a->base);
    memset(a, 0, sizeof *a);
}

void *ti_arena_alloc(ti_arena *a, size_t bytes, size_t align) {
    if (align == 0 || (align & (align - 1))) return NULL;      /* power of two only */
    size_t off = (a->used + align - 1) & ~(align - 1);
    if (off > a->cap || bytes > a->cap - off) return NULL;     /* overflow-safe */
    a->used = off + bytes;
    if (a->used > a->peak) a->peak = a->used;
    return a->base + off;
}

size_t ti_arena_mark(const ti_arena *a) { return a->used; }

void ti_arena_reset(ti_arena *a, size_t mark) {
    if (mark <= a->used) a->used = mark;
}
