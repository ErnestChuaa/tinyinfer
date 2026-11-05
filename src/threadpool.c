/* Persistent thread pool.
 *
 * Each ti_pool_run() call publishes a job descriptor that lives on the caller's
 * stack. Workers claim task indices from the job with an atomic fetch-add, so
 * tasks of uneven cost balance themselves without a central queue, and the
 * caller thread works too (a pool of size 1 has no threads and no locking).
 *
 * Lifetime rule (the subtle part): a worker may only touch a job it attached to
 * under the mutex while that job was still published, and the caller does not
 * return (or retract the job) until every attached worker has detached. A
 * worker that wakes late finds no published job and goes back to sleep, so it
 * can never steal indices from, or decrement counters of, a later job.
 */
#include "tinyinfer.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

typedef struct {
    ti_task_fn fn;
    void *ctx;
    int n_tasks;
    atomic_int next;
    int attached;               /* workers inside this job; guarded by pool mutex */
} job_t;

struct ti_pool {
    int n_threads;
    pthread_t *threads;
    pthread_mutex_t mu;
    pthread_cond_t work_cv, done_cv;
    unsigned long generation;
    job_t *job;                 /* published job or NULL; guarded by mu */
    int stop;
};

static void drain(job_t *j) {
    int i;
    while ((i = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed)) < j->n_tasks)
        j->fn(j->ctx, i);
}

static void *worker(void *arg) {
    ti_pool *p = arg;
    unsigned long seen = 0;
    pthread_mutex_lock(&p->mu);
    for (;;) {
        while (!p->stop && (p->generation == seen || !p->job)) {
            if (!p->job) seen = p->generation;     /* job already retracted: skip it */
            pthread_cond_wait(&p->work_cv, &p->mu);
        }
        if (p->stop) break;
        seen = p->generation;
        job_t *j = p->job;
        j->attached++;
        pthread_mutex_unlock(&p->mu);

        drain(j);                                  /* lock-free part */

        pthread_mutex_lock(&p->mu);
        if (--j->attached == 0) pthread_cond_signal(&p->done_cv);
    }
    pthread_mutex_unlock(&p->mu);
    return NULL;
}

ti_pool *ti_pool_create(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    ti_pool *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->n_threads = n_threads;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->work_cv, NULL);
    pthread_cond_init(&p->done_cv, NULL);
    p->threads = calloc((size_t)n_threads, sizeof(pthread_t));
    for (int i = 1; i < n_threads; i++) pthread_create(&p->threads[i], NULL, worker, p);
    return p;
}

void ti_pool_destroy(ti_pool *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    p->stop = 1;
    pthread_cond_broadcast(&p->work_cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n_threads; i++) pthread_join(p->threads[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->work_cv);
    pthread_cond_destroy(&p->done_cv);
    free(p->threads);
    free(p);
}

int ti_pool_size(const ti_pool *p) { return p ? p->n_threads : 1; }

void ti_pool_run(ti_pool *p, ti_task_fn fn, void *ctx, int n_tasks) {
    if (n_tasks <= 0) return;
    if (!p || p->n_threads == 1 || n_tasks == 1) {
        for (int i = 0; i < n_tasks; i++) fn(ctx, i);
        return;
    }
    job_t j = {.fn = fn, .ctx = ctx, .n_tasks = n_tasks, .attached = 0};
    atomic_init(&j.next, 0);

    pthread_mutex_lock(&p->mu);
    p->job = &j;
    p->generation++;
    pthread_cond_broadcast(&p->work_cv);
    pthread_mutex_unlock(&p->mu);

    drain(&j);

    /* All indices are claimed once drain() returns here; retract the job so no
     * new worker attaches, then wait for attached workers to finish theirs. */
    pthread_mutex_lock(&p->mu);
    p->job = NULL;
    while (j.attached > 0) pthread_cond_wait(&p->done_cv, &p->mu);
    pthread_mutex_unlock(&p->mu);
}
