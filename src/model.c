/* .tinf model loader, graph pass, memory planner and executor.
 *
 * File format (little-endian):
 *   char[4] "TINF" | u32 version (=1) | u32 n_layers
 *   per layer: u32 type | u32 in | u32 out | u32 dtype
 *     DENSE f32 : f32 W[out*in] | f32 bias[out]
 *     DENSE i8  : f32 scale[out] | i8 W[out*in] | pad to 4 bytes | f32 bias[out]
 *     RELU / SOFTMAX : no payload (in == out)
 */
#include "tinyinfer.h"
#include <stdio.h>
#include <string.h>

static int rd(FILE *f, void *dst, size_t n) { return fread(dst, 1, n, f) == n ? 0 : -1; }

static void *rd_alloc(FILE *f, ti_arena *a, size_t n, size_t *counter) {
    void *p = ti_arena_alloc(a, n, 64);
    if (!p || rd(f, p, n)) return NULL;
    *counter += n;
    return p;
}

ti_model *ti_model_load(const char *path, ti_arena *a) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    const size_t mark = ti_arena_mark(a);
    ti_model *m = NULL;
    char magic[4];
    uint32_t version, n;
    if (rd(f, magic, 4) || memcmp(magic, "TINF", 4) || rd(f, &version, 4) || version != 1 ||
        rd(f, &n, 4) || n == 0 || n > 1024)
        goto fail;

    m = ti_arena_alloc(a, sizeof *m, 64);
    ti_layer *raw = ti_arena_alloc(a, n * sizeof *raw, 64);
    if (!m || !raw) goto fail;
    memset(m, 0, sizeof *m);
    memset(raw, 0, n * sizeof *raw);

    for (uint32_t k = 0; k < n; k++) {
        ti_layer *L = &raw[k];
        uint32_t hdr[4];
        if (rd(f, hdr, sizeof hdr)) goto fail;
        L->type = hdr[0]; L->in = hdr[1]; L->out = hdr[2]; L->dtype = hdr[3];
        if (L->in == 0 || L->out == 0 || L->in > (1u << 20) || L->out > (1u << 20)) goto fail;
        if (k > 0 && L->in != raw[k - 1].out) goto fail;              /* shape check */
        if (L->type == TI_LAYER_DENSE) {
            const size_t nw = (size_t)L->in * L->out;
            if (L->dtype == TI_F32) {
                if (!(L->W = rd_alloc(f, a, nw * 4, &m->weight_bytes))) goto fail;
            } else if (L->dtype == TI_I8) {
                if (!(L->w_scale = rd_alloc(f, a, L->out * 4, &m->weight_bytes))) goto fail;
                if (!(L->Wq = rd_alloc(f, a, nw, &m->weight_bytes))) goto fail;
                size_t pad = (4 - nw % 4) % 4; uint8_t junk[3];
                if (pad && rd(f, junk, pad)) goto fail;
            } else goto fail;
            if (!(L->bias = rd_alloc(f, a, L->out * 4, &m->weight_bytes))) goto fail;
        } else if (L->type == TI_LAYER_RELU || L->type == TI_LAYER_SOFTMAX) {
            if (L->in != L->out) goto fail;
        } else goto fail;
    }
    fclose(f);
    f = NULL;

    /* Graph pass: fuse DENSE -> RELU into the dense epilogue (saves one full
     * read+write of the activation tensor per hidden layer). */
    int out_n = 0;
    for (uint32_t k = 0; k < n; k++) {
        raw[out_n] = raw[k];
        if (raw[k].type == TI_LAYER_DENSE && k + 1 < n && raw[k + 1].type == TI_LAYER_RELU) {
            raw[out_n].fuse_relu = 1;
            k++;
        }
        out_n++;
    }
    m->layers = raw;
    m->n_layers = out_n;
    m->in_dim = (int)raw[0].in;
    m->out_dim = (int)raw[out_n - 1].out;
    for (int k = 0; k < out_n; k++) {
        if ((int)raw[k].in > m->max_dim) m->max_dim = (int)raw[k].in;
        if ((int)raw[k].out > m->max_dim) m->max_dim = (int)raw[k].out;
    }
    return m;

fail:
    if (f) fclose(f);
    ti_arena_reset(a, mark);
    return NULL;
}

/* Memory plan: a sequential network only ever needs the current input and
 * output alive, so two ping-pong buffers of max_batch * max_dim suffice no
 * matter how deep the network is. All scratch is reserved here; ti_model_run
 * performs no allocation. */
int ti_model_plan(ti_model *m, int max_batch, ti_arena *a) {
    const size_t elems = (size_t)max_batch * (size_t)m->max_dim;
    m->act[0] = ti_arena_alloc(a, elems * sizeof(float), 64);
    m->act[1] = ti_arena_alloc(a, elems * sizeof(float), 64);
    m->xq = ti_arena_alloc(a, elems, 64);
    m->xs = ti_arena_alloc(a, (size_t)max_batch * sizeof(float), 64);
    if (!m->act[0] || !m->act[1] || !m->xq || !m->xs) return -1;
    m->max_batch = max_batch;
    m->act_bytes = elems * (2 * sizeof(float) + 1) + (size_t)max_batch * sizeof(float);
    return 0;
}

const float *ti_model_run(ti_model *m, const float *x, int batch, ti_pool *pool) {
    if (batch <= 0 || batch > m->max_batch) return NULL;
    const float *cur = x;
    int which = 0;
    for (int k = 0; k < m->n_layers; k++) {
        const ti_layer *L = &m->layers[k];
        float *dst = m->act[which];
        switch (L->type) {
        case TI_LAYER_DENSE:
            if (L->dtype == TI_F32)
                ti_dense_f32(cur, L->W, L->bias, dst, batch, (int)L->in, (int)L->out, L->fuse_relu, pool);
            else
                ti_dense_i8(cur, L->Wq, L->w_scale, L->bias, dst, m->xq, m->xs,
                            batch, (int)L->in, (int)L->out, L->fuse_relu, pool);
            break;
        case TI_LAYER_RELU: {
            const size_t ne = (size_t)batch * L->out;
            for (size_t i = 0; i < ne; i++) dst[i] = cur[i] > 0.f ? cur[i] : 0.f;
            break;
        }
        case TI_LAYER_SOFTMAX:
            if (cur != dst) memcpy(dst, cur, (size_t)batch * L->out * sizeof(float));
            ti_softmax_rows(dst, batch, (int)L->out);
            break;
        }
        cur = dst;
        which ^= 1;
    }
    return cur;
}
