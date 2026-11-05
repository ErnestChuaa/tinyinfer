# tinyinfer

A small, dependency-free neural-network inference runtime in C11. It loads a
compact binary model format, plans all memory up front, and runs fp32 or int8
dense networks on a persistent thread pool. The hot path performs no allocation.

I wrote this to understand what an inference engine actually does underneath
frameworks like PyTorch or ONNX Runtime: memory planning, quantisation, kernel
tiling and work scheduling.

```
make test        # build, train + export the demo model, run tests
make sanitize    # same tests under ASan+UBSan and under ThreadSanitizer
make bench       # kernel and end-to-end benchmarks
```

Requires a C11 compiler and pthreads. The model exporter needs Python 3 with NumPy and scikit-learn.

## Architecture

```
tools/train_export.py   NumPy MLP training (sklearn digits) -> .tinf fp32 + int8
src/arena.c             bump allocator: aligned, overflow-safe, mark/reset, peak tracking
src/threadpool.c        persistent pool, atomic work claiming, caller participates
src/kernels.c           fp32 register-tiled dense, int8 dense, softmax
src/model.c             loader + validation, DENSE->RELU fusion pass, memory planner, executor
tests/test_main.c       unit + integration tests
bench/bench.c           benchmarks
```

### Memory: one arena, planned at load time
Weights and every runtime buffer come from a single 64-byte-aligned arena.
`ti_model_plan()` reserves two ping-pong activation buffers of
`max_batch x max_dim`. A sequential network only ever needs its current input
and output alive, so memory stays flat however deep the network is. A test
asserts that `ti_model_run()` leaves the arena untouched. If loading fails
part-way, for example on a truncated file, the arena is rolled back to its
mark, so a bad model cannot leak memory.

### Graph pass: operator fusion
At load time, a `DENSE` followed by `RELU` is folded into the dense kernel's
epilogue. This saves one full write and re-read of the activation tensor per
hidden layer.

### Kernels
- **fp32.** Weights are stored `[out][in]`, so every output is a unit-stride dot
  product. A 4x4 register tile reuses each loaded input and weight four times.
- **int8.** Weights are quantised offline with symmetric per-output-channel
  scales. Activations are quantised per row at run time, so no calibration set
  is needed. Accumulation is in int32 and the result is rescaled in the
  epilogue. Weight memory shrinks 4x, and integer dot products vectorise much
  better than strict-IEEE float reductions.
- Output space is split into (batch x out) tiles. Each tile is an independent
  pool task, so kernels need no synchronisation. Threaded results are
  bit-identical to single-threaded results, and this is tested.

### Thread pool
Threads are created once. Each `ti_pool_run()` publishes a job descriptor, and
workers claim indices with `atomic_fetch_add`, so uneven tasks balance
themselves. The tricky part is lifetime: a worker that wakes late must never
touch a job that has finished. Workers attach to a job only while it is
published under the mutex, and the caller retracts the job and waits for
attached workers before returning. An earlier version without this rule raced
under a 2,000-job stress test; that test is still in the suite, and the suite
passes under ThreadSanitizer.

## Results

Measured on a 2-vCPU cloud Linux VM with `gcc -O3 -march=native`. The numbers
will differ on your machine; run `make bench`.

**Dense layer, batch 256, 1024 -> 1024**

| kernel | time | GFLOP/s |
|---|---:|---:|
| naive triple loop, 1 thread | 153.4 ms | 3.5 |
| tiled fp32, 1 thread | 113.2 ms | 4.7 |
| tiled fp32, 2 threads | 62.2 ms | 8.6 |
| int8, 1 thread | 31.1 ms | 17.3 |
| int8, 2 threads | 18.9 ms | 28.5 |

**Digits MLP (64-128-64-10), 450 held-out test images**

| model | accuracy | agrees with fp32 | weights | batch-1 latency | batch-1024 throughput |
|---|---:|---:|---:|---:|---:|
| fp32 | 97.1% | matches NumPy to 4e-7 | 68.9 KB | 11.9 us | 213k samples/s |
| int8 | 97.3% | 99.8% of predictions | 18.6 KB | 2.8 us | 536k samples/s |

## Trade-offs and limits
- The fp32 kernel is modest, about 1.35x over naive. Without `-ffast-math` the
  compiler may not reorder float reductions, so the inner loop does not fully
  vectorise. Explicit SIMD intrinsics or packing weights into panels are the
  next steps. I kept the code portable and the results reproducible instead.
- Dynamic per-row activation quantisation costs an extra pass over the input.
  Static, calibrated scales would remove it at the cost of a calibration step.
- Only sequential MLP graphs are supported: dense, relu and softmax. Conv and
  attention layers would need a real liveness-based planner rather than ping-pong
  buffers.
