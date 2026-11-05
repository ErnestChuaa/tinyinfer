/* tinyinfer: a small, dependency-free neural-network inference runtime in C11.
 *
 * Design goals
 *   - Zero allocations on the hot path: every buffer comes from an arena that is
 *     sized once by a memory planner at load time.
 *   - Two numeric paths: fp32 and int8 (per-output-channel weight scales,
 *     dynamic per-row activation scales, int32 accumulation).
 *   - Data-parallel kernels on a persistent thread pool (no thread creation per call).
 */
#ifndef TINYINFER_H
#define TINYINFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- Arena allocator ---------------- */
typedef struct {
    uint8_t *base;
    size_t   cap;
    size_t   used;
    size_t   peak;
} ti_arena;

int   ti_arena_init(ti_arena *a, size_t cap);
void  ti_arena_free(ti_arena *a);
void *ti_arena_alloc(ti_arena *a, size_t bytes, size_t align);
size_t ti_arena_mark(const ti_arena *a);
void  ti_arena_reset(ti_arena *a, size_t mark);

/* ---------------- Thread pool ---------------- */
typedef struct ti_pool ti_pool;
typedef void (*ti_task_fn)(void *ctx, int task_index);

ti_pool *ti_pool_create(int n_threads);          /* n_threads includes the caller */
void     ti_pool_destroy(ti_pool *p);
int      ti_pool_size(const ti_pool *p);
/* Runs fn(ctx, i) for i in [0, n_tasks). Caller participates; returns when all done. */
void     ti_pool_run(ti_pool *p, ti_task_fn fn, void *ctx, int n_tasks);

/* ---------------- Kernels ---------------- */
/* y[b][o] = act( sum_i x[b][i] * W[o][i] + bias[o] ),  W row-major [out][in] */
void ti_dense_f32(const float *x, const float *W, const float *bias, float *y,
                  int batch, int in, int out, int relu, ti_pool *pool);

/* int8 weights with per-output-channel scale; activations quantised per row on the fly. */
void ti_dense_i8(const float *x, const int8_t *Wq, const float *w_scale, const float *bias,
                 float *y, int8_t *xq_scratch, float *x_scale_scratch,
                 int batch, int in, int out, int relu, ti_pool *pool);

void ti_softmax_rows(float *x, int batch, int dim);

/* ---------------- Model ---------------- */
enum { TI_LAYER_DENSE = 1, TI_LAYER_RELU = 2, TI_LAYER_SOFTMAX = 3 };
enum { TI_F32 = 0, TI_I8 = 1 };

typedef struct {
    uint32_t type, in, out, dtype;
    int      fuse_relu;           /* set by the graph pass: DENSE followed by RELU */
    const float  *W;              /* fp32 weights  */
    const int8_t *Wq;             /* int8 weights  */
    const float  *w_scale;
    const float  *bias;
} ti_layer;

typedef struct {
    ti_layer *layers;
    int       n_layers;
    int       in_dim, out_dim, max_dim;
    int       max_batch;
    /* planned buffers */
    float    *act[2];             /* ping-pong activations, max_batch * max_dim each */
    int8_t   *xq;                 /* int8 scratch, max_batch * max_dim */
    float    *xs;                 /* per-row scales, max_batch */
    size_t    weight_bytes, act_bytes;
} ti_model;

/* Loads a .tinf file into the arena and runs the fusion pass. Returns NULL on error. */
ti_model *ti_model_load(const char *path, ti_arena *a);
/* Allocates all activation/scratch memory for batches up to max_batch. */
int ti_model_plan(ti_model *m, int max_batch, ti_arena *a);
/* Runs inference. Returns pointer to planned output buffer [batch][out_dim]. */
const float *ti_model_run(ti_model *m, const float *x, int batch, ti_pool *pool);

#ifdef __cplusplus
}
#endif
#endif
