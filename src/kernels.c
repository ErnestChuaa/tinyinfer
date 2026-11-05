/* Dense-layer kernels.
 *
 * Layout choice: weights are stored [out][in] so that each output is a dot
 * product over two contiguous arrays (one input row, one weight row). That makes
 * the inner loop a straight unit-stride reduction the compiler can vectorise,
 * and the 4x4 register tile below reuses every loaded input and weight value
 * four times, cutting memory traffic by ~4x versus a naive triple loop.
 *
 * Parallelism: the (batch x out) output space is cut into tiles; each tile is a
 * pool task. Tiles never overlap, so there is no synchronisation inside kernels.
 */
#include "tinyinfer.h"
#include <math.h>
#include <stdint.h>

#define TILE_B 16
#define TILE_O 64

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

/* ------------------------------------------------------------------ fp32 */
typedef struct {
    const float *x, *W, *bias;
    float *y;
    int batch, in, out, relu, tiles_o;
} f32_job;

static void f32_tile(void *vctx, int t) {
    const f32_job *j = vctx;
    const int b0 = (t / j->tiles_o) * TILE_B, o0 = (t % j->tiles_o) * TILE_O;
    const int b1 = b0 + TILE_B < j->batch ? b0 + TILE_B : j->batch;
    const int o1 = o0 + TILE_O < j->out ? o0 + TILE_O : j->out;
    const int in = j->in;

    for (int b = b0; b < b1; b += 4) {
        const int nb = b1 - b < 4 ? b1 - b : 4;
        for (int o = o0; o < o1; o += 4) {
            const int no = o1 - o < 4 ? o1 - o : 4;
            float acc[4][4] = {{0}};
            if (nb == 4 && no == 4) {
                /* register-tiled 4x4 micro-kernel */
                const float *x0 = j->x + (size_t)b * in, *x1 = x0 + in, *x2 = x1 + in, *x3 = x2 + in;
                const float *w0 = j->W + (size_t)o * in, *w1 = w0 + in, *w2 = w1 + in, *w3 = w2 + in;
                for (int i = 0; i < in; i++) {
                    const float a0 = x0[i], a1 = x1[i], a2 = x2[i], a3 = x3[i];
                    const float c0 = w0[i], c1 = w1[i], c2 = w2[i], c3 = w3[i];
                    acc[0][0] += a0 * c0; acc[0][1] += a0 * c1; acc[0][2] += a0 * c2; acc[0][3] += a0 * c3;
                    acc[1][0] += a1 * c0; acc[1][1] += a1 * c1; acc[1][2] += a1 * c2; acc[1][3] += a1 * c3;
                    acc[2][0] += a2 * c0; acc[2][1] += a2 * c1; acc[2][2] += a2 * c2; acc[2][3] += a2 * c3;
                    acc[3][0] += a3 * c0; acc[3][1] += a3 * c1; acc[3][2] += a3 * c2; acc[3][3] += a3 * c3;
                }
            } else {
                /* edge tiles */
                for (int bb = 0; bb < nb; bb++)
                    for (int oo = 0; oo < no; oo++) {
                        const float *xr = j->x + (size_t)(b + bb) * in, *wr = j->W + (size_t)(o + oo) * in;
                        float s = 0.f;
                        for (int i = 0; i < in; i++) s += xr[i] * wr[i];
                        acc[bb][oo] = s;
                    }
            }
            for (int bb = 0; bb < nb; bb++)
                for (int oo = 0; oo < no; oo++) {
                    float v = acc[bb][oo] + (j->bias ? j->bias[o + oo] : 0.f);
                    if (j->relu && v < 0.f) v = 0.f;          /* fused epilogue */
                    j->y[(size_t)(b + bb) * j->out + o + oo] = v;
                }
        }
    }
}

void ti_dense_f32(const float *x, const float *W, const float *bias, float *y,
                  int batch, int in, int out, int relu, ti_pool *pool) {
    f32_job j = {x, W, bias, y, batch, in, out, relu, ceil_div(out, TILE_O)};
    ti_pool_run(pool, f32_tile, &j, ceil_div(batch, TILE_B) * j.tiles_o);
}

/* ------------------------------------------------------------------ int8 */
/* Symmetric quantisation: x ~= xq * sx with xq in [-127, 127]. Per-row (per
 * sample) scales are computed at run time so activations need no calibration
 * set; weights use per-output-channel scales computed offline. Accumulation is
 * int32: with |xq|,|wq| <= 127 that is safe for in < 133,000. */
typedef struct {
    const float *x; int8_t *xq; float *xs;
    const int8_t *Wq; const float *ws, *bias;
    float *y;
    int batch, in, out, relu, tiles_o;
} i8_job;

static void quantise_row(void *vctx, int b) {
    i8_job *j = vctx;
    const float *xr = j->x + (size_t)b * j->in;
    float m = 0.f;
    for (int i = 0; i < j->in; i++) { float a = fabsf(xr[i]); if (a > m) m = a; }
    const float s = m > 0.f ? m / 127.f : 1.f, inv = 1.f / s;
    int8_t *q = j->xq + (size_t)b * j->in;
    for (int i = 0; i < j->in; i++) q[i] = (int8_t)lrintf(xr[i] * inv);
    j->xs[b] = s;
}

static void i8_tile(void *vctx, int t) {
    const i8_job *j = vctx;
    const int b0 = (t / j->tiles_o) * TILE_B, o0 = (t % j->tiles_o) * TILE_O;
    const int b1 = b0 + TILE_B < j->batch ? b0 + TILE_B : j->batch;
    const int o1 = o0 + TILE_O < j->out ? o0 + TILE_O : j->out;
    const int in = j->in;
    for (int b = b0; b < b1; b++) {
        const int8_t *xr = j->xq + (size_t)b * in;
        for (int o = o0; o < o1; o++) {
            const int8_t *wr = j->Wq + (size_t)o * in;
            int32_t acc = 0;
            for (int i = 0; i < in; i++) acc += (int32_t)xr[i] * (int32_t)wr[i];
            float v = (float)acc * j->xs[b] * j->ws[o] + (j->bias ? j->bias[o] : 0.f);
            if (j->relu && v < 0.f) v = 0.f;
            j->y[(size_t)b * j->out + o] = v;
        }
    }
}

void ti_dense_i8(const float *x, const int8_t *Wq, const float *w_scale, const float *bias,
                 float *y, int8_t *xq_scratch, float *x_scale_scratch,
                 int batch, int in, int out, int relu, ti_pool *pool) {
    i8_job j = {x, xq_scratch, x_scale_scratch, Wq, w_scale, bias, y, batch, in, out, relu,
                ceil_div(out, TILE_O)};
    ti_pool_run(pool, quantise_row, &j, batch);
    ti_pool_run(pool, i8_tile, &j, ceil_div(batch, TILE_B) * j.tiles_o);
}

/* ------------------------------------------------------------------ softmax */
void ti_softmax_rows(float *x, int batch, int dim) {
    for (int b = 0; b < batch; b++) {
        float *r = x + (size_t)b * dim, m = r[0], s = 0.f;
        for (int i = 1; i < dim; i++) if (r[i] > m) m = r[i];   /* subtract max: no overflow */
        for (int i = 0; i < dim; i++) { r[i] = expf(r[i] - m); s += r[i]; }
        for (int i = 0; i < dim; i++) r[i] /= s;
    }
}
