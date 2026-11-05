/* Benchmarks: naive vs tiled fp32, 1 vs N threads, fp32 vs int8, plus
 * end-to-end digits model throughput. Usage: ./build/bench [threads] */
#include "tinyinfer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void naive(const float *x, const float *W, const float *b, float *y, int B, int I, int O) {
    for (int bb = 0; bb < B; bb++) for (int o = 0; o < O; o++) {
        float s = b[o];
        for (int i = 0; i < I; i++) s += x[bb * I + i] * W[o * I + i];
        y[bb * O + o] = s;
    }
}

#define TIME(label, flops, reps, stmt) do { stmt; double t0 = now(); for (int r_ = 0; r_ < reps; r_++) { stmt; } \
    double dt = (now() - t0) / reps; printf("  %-34s %8.2f ms  %7.2f GFLOP/s\n", label, dt * 1e3, flops / dt * 1e-9); } while (0)

int main(int argc, char **argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 2;
    const int B = 256, I = 1024, O = 1024;
    const double flops = 2.0 * B * I * O;
    float *x = malloc(sizeof(float) * B * I), *W = malloc(sizeof(float) * O * I), *b = calloc(O, sizeof(float));
    float *y = malloc(sizeof(float) * B * O), *ws = malloc(sizeof(float) * O), *xs = malloc(sizeof(float) * B);
    int8_t *Wq = malloc((size_t)O * I), *xq = malloc((size_t)B * I);
    srand(1);
    for (int i = 0; i < B * I; i++) x[i] = rand() / (float)RAND_MAX - .5f;
    for (int i = 0; i < O * I; i++) { W[i] = rand() / (float)RAND_MAX - .5f; Wq[i] = (int8_t)(W[i] * 254); }
    for (int o = 0; o < O; o++) ws[o] = 1.f / 254;
    ti_pool *p1 = ti_pool_create(1), *pn = ti_pool_create(nt);
    char lbl[64];

    printf("dense layer %dx%d -> %d (batch %d)\n", I, O, O, B);
    TIME("naive triple loop, 1 thread", flops, 3, naive(x, W, b, y, B, I, O));
    TIME("tiled fp32, 1 thread", flops, 5, ti_dense_f32(x, W, b, y, B, I, O, 0, p1));
    snprintf(lbl, sizeof lbl, "tiled fp32, %d threads", nt);
    TIME(lbl, flops, 5, ti_dense_f32(x, W, b, y, B, I, O, 0, pn));
    TIME("int8, 1 thread", flops, 5, ti_dense_i8(x, Wq, ws, b, y, xq, xs, B, I, O, 0, p1));
    snprintf(lbl, sizeof lbl, "int8, %d threads", nt);
    TIME(lbl, flops, 5, ti_dense_i8(x, Wq, ws, b, y, xq, xs, B, I, O, 0, pn));
    printf("  weight memory: fp32 %zu KiB, int8 %zu KiB\n", (size_t)O * I * 4 / 1024, (size_t)(O * I + O * 4) / 1024);

    const char *models[] = {"models/digits_f32.tinf", "models/digits_i8.tinf"};
    float *batch = malloc(sizeof(float) * 1024 * 64);
    for (int i = 0; i < 1024 * 64; i++) batch[i] = rand() / (float)RAND_MAX;
    printf("digits MLP 64-128-64-10 end to end\n");
    for (int k = 0; k < 2; k++) {
        ti_arena a; ti_arena_init(&a, 4 << 20);
        ti_model *m = ti_model_load(models[k], &a);
        if (!m) { printf("  (missing %s, run tools/train_export.py)\n", models[k]); ti_arena_free(&a); continue; }
        ti_model_plan(m, 1024, &a);
        const int reps = 200;
        double t0 = now();
        for (int r = 0; r < reps; r++) ti_model_run(m, batch, 1, p1);
        double lat = (now() - t0) / reps;
        t0 = now();
        for (int r = 0; r < reps / 10; r++) ti_model_run(m, batch, 1024, pn);
        double thr = 1024.0 * (reps / 10) / (now() - t0);
        printf("  %-24s batch-1 latency %6.2f us   batch-1024 throughput %9.0f samples/s   arena peak %zu KiB\n",
               models[k] + 7, lat * 1e6, thr, a.peak / 1024);
        ti_arena_free(&a);
    }
    ti_pool_destroy(p1); ti_pool_destroy(pn);
    return 0;
}
