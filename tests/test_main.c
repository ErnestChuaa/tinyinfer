#include "tinyinfer.h"
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); } } while (0)

static float frand(unsigned *s) { *s = *s * 1103515245u + 12345u; return ((*s >> 8) & 0xFFFF) / 32768.f - 1.f; }

static void ref_dense(const float *x, const float *W, const float *b, float *y, int B, int I, int O, int relu) {
    for (int bb = 0; bb < B; bb++) for (int o = 0; o < O; o++) {
        double s = b ? b[o] : 0;
        for (int i = 0; i < I; i++) s += (double)x[bb * I + i] * W[o * I + i];
        y[bb * O + o] = (float)(relu && s < 0 ? 0 : s);
    }
}

static void test_arena(void) {
    ti_arena a;
    CHECK(ti_arena_init(&a, 1024) == 0, "init");
    void *p = ti_arena_alloc(&a, 3, 1);
    void *q = ti_arena_alloc(&a, 8, 64);
    CHECK(p && q && ((uintptr_t)q % 64) == 0, "alignment");
    size_t m = ti_arena_mark(&a);
    CHECK(ti_arena_alloc(&a, 2000, 8) == NULL, "over-capacity must fail");
    CHECK(ti_arena_alloc(&a, (size_t)-1, 8) == NULL, "size overflow must fail");
    CHECK(ti_arena_alloc(&a, 8, 3) == NULL, "non power-of-two alignment must fail");
    ti_arena_alloc(&a, 100, 16);
    ti_arena_reset(&a, m);
    CHECK(ti_arena_mark(&a) == m && a.peak >= m + 100, "reset keeps peak");
    ti_arena_free(&a);
}

static atomic_int hits[1000];
static void count_task(void *ctx, int i) { (void)ctx; atomic_fetch_add(&hits[i], 1); }

static void test_pool(void) {
    ti_pool *p = ti_pool_create(4);
    for (int round = 0; round < 2000; round++) {          /* many back-to-back jobs of varying size */
        int n = 1 + (round * 37) % 997;
        memset(hits, 0, sizeof hits);
        ti_pool_run(p, count_task, NULL, n);
        int ok = 1;
        for (int i = 0; i < 1000; i++) if (atomic_load(&hits[i]) != (i < n)) ok = 0;
        CHECK(ok, "every task must run exactly once (round %d, n=%d)", round, n);
        if (!ok) break;
    }
    ti_pool_destroy(p);
}

static void test_dense(void) {
    ti_pool *p4 = ti_pool_create(4);
    unsigned s = 7;
    const int shapes[][3] = {{1, 1, 1}, {3, 5, 7}, {17, 64, 65}, {33, 100, 130}, {64, 256, 128}};
    for (size_t k = 0; k < sizeof shapes / sizeof *shapes; k++) {
        int B = shapes[k][0], I = shapes[k][1], O = shapes[k][2];
        float *x = malloc(sizeof(float) * B * I), *W = malloc(sizeof(float) * O * I), *b = malloc(sizeof(float) * O);
        float *y0 = malloc(sizeof(float) * B * O), *y1 = malloc(sizeof(float) * B * O), *y2 = malloc(sizeof(float) * B * O);
        int8_t *xq = malloc((size_t)B * I); float *xs = malloc(sizeof(float) * B);
        for (int i = 0; i < B * I; i++) x[i] = frand(&s);
        for (int i = 0; i < O * I; i++) W[i] = frand(&s);
        for (int i = 0; i < O; i++) b[i] = frand(&s);
        for (int relu = 0; relu < 2; relu++) {
            ref_dense(x, W, b, y0, B, I, O, relu);
            ti_dense_f32(x, W, b, y1, B, I, O, relu, NULL);
            ti_dense_f32(x, W, b, y2, B, I, O, relu, p4);
            double maxe = 0, same = 1;
            for (int i = 0; i < B * O; i++) {
                maxe = fmax(maxe, fabs(y0[i] - y1[i]));
                if (y1[i] != y2[i]) same = 0;
            }
            CHECK(maxe < 1e-3 * sqrt(I), "f32 %dx%dx%d err %g", B, I, O, maxe);
            CHECK(same, "threaded result must be bit-identical to single-threaded");

            /* int8: quantise W per row, compare against fp32 reference */
            int8_t *Wq = malloc((size_t)O * I); float *ws = malloc(sizeof(float) * O);
            for (int o = 0; o < O; o++) {
                float m = 0; for (int i = 0; i < I; i++) m = fmaxf(m, fabsf(W[o * I + i]));
                ws[o] = m > 0 ? m / 127.f : 1.f;
                for (int i = 0; i < I; i++) Wq[o * I + i] = (int8_t)lrintf(W[o * I + i] / ws[o]);
            }
            ti_dense_i8(x, Wq, ws, b, y1, xq, xs, B, I, O, relu, p4);
            double rel = 0, norm = 0;
            for (int i = 0; i < B * O; i++) { rel += (y0[i] - y1[i]) * (y0[i] - y1[i]); norm += y0[i] * y0[i]; }
            CHECK(norm == 0 || sqrt(rel / norm) < 0.03, "int8 relative error %g too high", sqrt(rel / (norm + 1e-12)));
            free(Wq); free(ws);
        }
        free(x); free(W); free(b); free(y0); free(y1); free(y2); free(xq); free(xs);
    }
    ti_pool_destroy(p4);
}

static int load_test(const char *path, int *n, int *dim, float **X, uint32_t **y, float **ref) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t h[2];
    if (fread(h, 4, 2, f) != 2) { fclose(f); return -1; }
    *n = (int)h[0]; *dim = (int)h[1];
    *X = malloc(sizeof(float) * *n * *dim); *y = malloc(4u * *n); *ref = malloc(sizeof(float) * *n * 10);
    size_t ok = fread(*X, 4, (size_t)*n * *dim, f) + fread(*y, 4, (size_t)*n, f) + fread(*ref, 4, (size_t)*n * 10, f);
    fclose(f);
    return ok == (size_t)*n * *dim + *n + (size_t)*n * 10 ? 0 : -1;
}

static void test_model(const char *dir) {
    char path[512];
    snprintf(path, sizeof path, "%s/digits_test.bin", dir);
    int n, dim; float *X, *ref; uint32_t *y;
    if (load_test(path, &n, &dim, &X, &y, &ref)) { printf("SKIP model tests (run tools/train_export.py)\n"); return; }
    ti_softmax_rows(ref, n, 10);
    ti_pool *pool = ti_pool_create(2);
    const char *names[] = {"digits_f32.tinf", "digits_i8.tinf"};
    for (int k = 0; k < 2; k++) {
        ti_arena a; ti_arena_init(&a, 4 << 20);
        snprintf(path, sizeof path, "%s/%s", dir, names[k]);
        ti_model *m = ti_model_load(path, &a);
        CHECK(m != NULL, "load %s", names[k]);
        if (!m) { ti_arena_free(&a); continue; }
        CHECK(m->n_layers == 4, "fusion pass should leave 4 layers (dense+relu, dense+relu, dense, softmax), got %d", m->n_layers);
        CHECK(ti_model_plan(m, n, &a) == 0, "plan");
        size_t before = a.used;
        const float *out = ti_model_run(m, X, n, pool);
        CHECK(a.used == before, "inference must not allocate");
        int correct = 0, agree = 0; double maxe = 0;
        for (int i = 0; i < n; i++) {
            int am = 0, ar = 0;
            for (int c = 1; c < 10; c++) {
                if (out[i * 10 + c] > out[i * 10 + am]) am = c;
                if (ref[i * 10 + c] > ref[i * 10 + ar]) ar = c;
                maxe = fmax(maxe, fabs(out[i * 10 + c] - ref[i * 10 + c]));
            }
            correct += am == (int)y[i]; agree += am == ar;
        }
        printf("  %-16s acc %.4f  agree-with-numpy %.4f  max|dp| %.2e  weights %zu B  activations %zu B\n",
               names[k], (double)correct / n, (double)agree / n, maxe, m->weight_bytes, m->act_bytes);
        if (k == 0) CHECK(maxe < 1e-4, "fp32 runtime must match numpy");
        else CHECK((double)agree / n > 0.98, "int8 must agree with fp32 on >98%% of samples");
        ti_arena_free(&a);
    }
    /* corrupted file must be rejected cleanly */
    snprintf(path, sizeof path, "%s/digits_f32.tinf", dir);
    FILE *src = fopen(path, "rb"), *dst = fopen("/tmp/ti_trunc.tinf", "wb");
    char buf[300]; size_t r = fread(buf, 1, sizeof buf, src); fwrite(buf, 1, r, dst); fclose(src); fclose(dst);
    ti_arena a; ti_arena_init(&a, 1 << 20);
    CHECK(ti_model_load("/tmp/ti_trunc.tinf", &a) == NULL && a.used == 0, "truncated model rejected, arena rolled back");
    ti_arena_free(&a);
    ti_pool_destroy(pool);
    free(X); free(y); free(ref);
}

int main(int argc, char **argv) {
    test_arena();
    test_pool();
    test_dense();
    test_model(argc > 1 ? argv[1] : "models");
    printf(failures ? "%d FAILURE(S)\n" : "all tests passed\n", failures);
    return failures != 0;
}
