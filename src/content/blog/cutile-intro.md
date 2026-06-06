---
title: 'cuTile Intro'
description: 'Starting from naive matrix multiplication, this post records my first CUDA kernel, WMMA, and cuTile experiments.'
pubDate: 'Jun 06 2026'
---

It rained, so the afternoon travel plan was gone. I picked up the half-written cuTile post from before and finished it.

The more I write Pegainfer, the more I feel that kernels are the heart of a model. If I do not understand how to efficiently write, maintain, and manage kernels, Pegainfer may not be able to reach another level. So recently I have read a lot of DSLs, and also started trying to write kernels.

This post is written in the old way. Some parts I learned from chatting with AI are all marked.

## Preface

First, my understanding of the GPU is that it wins speed through sufficiently large parallelism. My own feeling is that it is similar to SIMD, Single Instruction, Multiple Data. In plain words, there are a huge number of threads doing the same computation, only on different data.

But NV defines itself as SIMT (Single Instruction, Multiple Threads).

SIMD may emphasize things like "vector operations" more. One instruction really computes multiple pieces of data. For example, in the past one instruction could only compute one int plus one int. Later the hardware supported one instruction computing 8 ints plus 8 ints. This is the so-called SIMD, and can also be called vectorization.

NV's SIMT is slightly different. We still write from the single-thread perspective of one int plus one int (although NV should also have vectorization), but there can be a huge number of threads computing at the same time.

So it is equivalent to this: when computation is needed, in C++ we write a compute function, and then NV automatically has a thread pool of 1000 threads, automatically splits the task, and executes it.

This is my understanding of GPU. I have also never seriously written a CUDA kernel before, not even one.

CUDA 13.3 released cuTile C++, changing to another abstraction for writing GPU kernels. This abstraction is `tile`. Is `tile` essentially an IR? Now that DSLs keep getting hotter, and may become even hotter in the future, I also want to learn what these IRs or DSLs can really bring.

Still start from a classic matrix multiplication. A and B are both `(1024, 1024)` matrices, and we want to multiply them to get a matrix C.

That is:

```text
C[i][j] = sum_k A[i][k] * B[k][j]
```

On CPU, written in C++, it is roughly like this (this is f32, matrices flattened into one-dimensional arrays, row-major):

```cpp
constexpr int N = 1024;

void matmul_naive(const std::vector<float>& a,
                  const std::vector<float>& b,
                  std::vector<float>& c) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) {
                acc += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = acc;
        }
    }
}
```

This is the CPU baseline for comparison with GPU later. On my machine (AMD EPYC 7402), with `g++ -O3 -march=native`, single-thread measured:

```text
avg_time_ms=1054.120
throughput_gflops=2.04
```

One run is close to 1 second. Here is a question for the reader: can the order of these three loops be changed? After changing it, is the result still correct? What about the time cost?

Following CUDA beginner tutorials, I roughly wrote this (also my first kernel):

```cpp
__global__ void matmul_naive(const __half* __restrict__ a,
                             const __half* __restrict__ b,
                             float* __restrict__ c,
                             int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        acc += __half2float(a[row * n + k]) * __half2float(b[k * n + col]);
    }
    c[row * n + col] = acc;
}
```

Compared with the three nested loops on CPU above, here there is only one loop, which is the innermost loop. The remaining two loops have been moved to the template at launch time.

According to my abstraction above, what GPU gives us is a huge thread pool. What we need to define is the task (the inner loop). Then how should the task be split and distributed?

What is the smallest-granularity task in matrix multiplication? It is:

```cpp
acc += a[i][kk] * b[kk][j];
```

But this granularity is a bit too fine. Even in the example above, it may require `1024 * 1024 * 1024` threads to complete the task. Even for GPU threads, this granularity is a little too fine.

So we need "batching": batch processing, letting one thread do slightly more work. In the example above, the task is defined as computing one element in matrix C (that is, taking over the innermost loop), so the number of tasks is only `1024 * 1024`.

But this is still not enough. A kernel launch still needs two parameters: `<<<grid, block>>>`. Because all threads are not equal. They run on different physical units (SMs), and have different cache and shared memory. Therefore NV provides an abstraction called "block", used to describe a group of threads that can cooperate.

So this kernel is "distributed" to the GPU like this:

```cpp
dim3 block(16, 16);
dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
matmul_naive<<<grid, block>>>(d_a, d_b, d_c, N);
```

- Block: we define one block as `16 * 16 = 256` threads.
- Grid: the calculated total number of blocks. The purpose is to make blocks cover the whole `1024 * 1024` matrix. The result is `64 * 64` blocks.

In total: `64 x 64` blocks x `256` threads/block = `1048576` threads. This exactly corresponds to every element in matrix C.

For the naive version above, first record the actual baseline time on my 5070 Ti:

```text
kernel=matmul_naive
atype=fp16 btype=fp16 ctype=fp32
block=16x16 grid=64x64
warmup_iters=10 bench_iters=100
avg_kernel_time_ms=0.717803
throughput_gflops=2991.75
verification=passed
```

The CPU baseline is close to 1s (1054ms), while this GPU naive kernel is about 710us, roughly 1500x different.

But first go back to the previous question: can those three loops be reordered?

Yes. `C[i][j] = sum_k A[i][k] * B[k][j]` only requires accumulating the k dimension for each `(i, j)`. No matter how the loops are arranged, every element is the same set of products added in increasing k order, so the result is exactly the same (even the floating-point rounding order is the same, and the checksum is bitwise identical). But the time cost is worlds apart. Swap j and k, turning it into i-k-j (note that c must be zeroed first, because it changes from "write after finishing one element" to "repeatedly accumulate"):

```cpp
for (int i = 0; i < N; ++i) {
    for (int k = 0; k < N; ++k) {
        float aik = a[i * N + k];
        for (int j = 0; j < N; ++j) {
            c[i * N + j] += aik * b[k * N + j];
        }
    }
}
```

On the same machine, with the same compilation parameters, measured 65ms, 32.8 GFLOPS, 16x faster.

The whole difference is in the memory access pattern of the inner loop:

- i-j-k: the inner loop walks k. `b[k * N + j]` is column-wise access, and two adjacent accesses are separated by an entire row (4KB). Every multiply-add steps onto a new cache line. Also, 4KB is a power of 2, so all accesses fall into the same L1 set, making conflict misses full.
- i-k-j: the inner loop walks j. `b[k * N + j]` and `c[i * N + j]` are both continuous accesses, and all 16 floats in one cache line are used.

The same CPU, the same floating-point units, only changing the loop order gives a 16x difference. What is slow is not the hardware, nor the compiler.

It is that hardware is not "transparent and equal". In the example above, we need to know the CPU cache line abstraction, even some internal working principles (SIMD), and also need to make sure the compiler can correctly generate such assembly code.

**To write sufficiently good code on hardware, you must understand the hardware and compiler sufficiently well.**

Why is the compiler also part of this? I compiled the previous GPU naive kernel separately with nvcc 12.8 / 13.1 / 13.3. The generated SASS (GPU assembly) is indeed different. The instruction count and address calculation strategies are different, but the runtime is all ~0.71ms. When the bottleneck is the memory access pattern, instruction-level differences between compiler versions are completely covered up. The compiler is responsible for translating code into good-enough instructions, but the memory access pattern is decided by the person writing the code. It cannot save this. Under some kernels, different compiler versions do have obvious performance optimizations.

Back to the GPU kernel. It seems 710us is much faster than the CPU's 65ms. Then the question is: is 710us fast or slow? As mentioned above, this example is a naive thing that can be written after reading a beginner tutorial. But in the information flow, there are many kernel optimization experts, and NV's own black-magic kernels. Where is the gap?

Fast and slow are relative concepts. The same kernel only needs 510us on H200. Is it slow or fast then?

Here we need to introduce the Roofline model (https://jax-ml.github.io/scaling-book/roofline/). As the blog says:

> why does an algorithm take 50ms instead of 50s or 5ms? What is actually happening within the model that takes substantial time and how long should we expect it to take?

It defines a metric called arithmetic intensity (AI):

> Definition: the arithmetic intensity of an algorithm is given by the ratio of the total FLOPs it performs to the number of bytes it needs to communicate — either within a chip or between chips.

Simply put, it is the ratio between compute amount and memory access amount. Why define it this way? Because these are the two physical limits of GPU:

1. Compute limit: compute cores are finite, and the amount of data a GPU can compute per second under different precisions has an upper limit.
2. Bandwidth limit: HBM bandwidth also has a physical upper limit.

A kernel will eventually reach a limit (of course there are other limits, to discuss later). Is it limited by compute or bandwidth? Take the matrix multiplication example in this post.

In each loop:

1. Compute amount: one multiplication and one addition, 2 flops.
2. Memory access amount: read one element from A and one element from B, two elements, both fp16, so 4 bytes.

So the AI of the naive version is 0.5 flops / byte. Take my 5070 Ti:

1. Memory bandwidth: theoretical 896GB/s = 28 Gbps/pin x 256 pin / 8 bit/byte. Measured d2d copy bandwidth is only around 770GB/s.
2. FP32 compute: 44 TFLOPS. FP16 Tensor compute: 88 TFLOPS.

If estimated by this theory, our performance limit should be `896 GB/s x 0.5 FLOPs/Byte = 448 GFLOPS`, but the measured result is 2991 GFLOPS, more than 6x higher.

Is it cache? The cache was not cleared between 100 iterations. A, B, plus result C for `1024 x 1024` are only 8MB total (2+2+4), and the 5070 Ti has 48MB L2, easily fitting all of them.

CUDA has no public interface for "clearing the entire L2" (`cudaCtxResetPersistingL2Cache` only handles that small persisting part), but Claude said it can be done this way: before each launch, memset a dirty buffer larger than L2, pushing all old data out.

After flushing L2, the result was 2950 GFLOPS, only 2% slower. This 2% can also be accounted for: after flushing, each launch needs to move A+B again from memory and write C back, about 12MB. At the measured 770 GB/s, that is ~15us, exactly the time difference between flush and no flush.

Then since there is no cache between iterations, why is it still 2950? This requires going back to 448. It assumes a world with "no cache at all", where every number is fetched from memory on the spot, then discarded after use. Reverse-calculate under this assumption: the inner loop executes `1024^3 ~= 1.07` billion times. Each time reads 4 bytes, so one launch would need to read **4GB** from memory. This is how 448 GFLOPS comes from: moving 4.3GB takes `4.3GB / 896GB/s ~= 4.8ms`, and `2.1 GFLOP / 4.8ms ~= 448`.

Once the kernel starts, matrices A, B, and C are read in for the first time and remain in L1/L2. All later reads will hit cache.

Assumptions are assumptions. When we propose an assumption, we should use objective evidence as much as possible to measure it. At this point, NCU should appear.

By default, ncu flushes cache itself before every replay (`--cache-control all`). At the same time, ncu profiling brings some overhead, so the exact time may not be accurate.

```bash
# Take three samples each from the warm segment (launches 60-62)
# and the cold segment (launches 150-152, each preceded by memset flushing).
sudo ncu --cache-control none --clock-control none \
    --metrics lts__t_sector_hit_rate.pct \
    --launch-skip 60 --launch-count 3 ./build/matmul_baseline
```

- `lts__t_sector_hit_rate.pct`: NCU's L2 cache hit ratio metric.

You can use this command to query its meaning:

```bash
sudo /usr/local/cuda-13.3/bin/ncu --query-metrics-mode suffix    --metrics lts__t_sector_hit_rate
--------------------------------------------------------------------------- --------------- --------------- ----------------------------------------------------------------------
Metric Name                                                                 Metric Type     Metric Unit     Metric Description
--------------------------------------------------------------------------- --------------- --------------- ----------------------------------------------------------------------
lts__t_sector_hit_rate.max_rate                                             Ratio                           proportion of L2 sector lookups that hit (This ratio metric
                                                                                                            represents the ratio's maximum value across all sub-unit instances)
lts__t_sector_hit_rate.pct                                                  Ratio           %               proportion of L2 sector lookups that hit (This ratio metric
                                                                                                            represents the value expressed as a percentage across all sub-unit
                                                                                                            instances)
lts__t_sector_hit_rate.ratio                                                Ratio                           proportion of L2 sector lookups that hit (This ratio metric
                                                                                                            represents the value expressed as a ratio across all sub-unit
                                                                                                            instances)
```

The three warm runs were 99.89 / 99.94 / 99.87, and the three cold runs were 99.88 / 99.83 / 99.86. This shows that most memory accesses indeed hit L2 cache.

Then look at DRAM read/write amount. We look at the second metric (`dram__bytes_op_read.sum`, this needs `--replay-mode application`, otherwise replay will remeasure on warm cache): the warm launch reads literally **0** bytes from memory, and the cold launch reads 4.2-4.4MB, exactly the size of A+B.

So it already fully hits L2. Why is it only 3 TFLOPS, still far from FP32's 44 TFLOPS? We continue using NCU to explore why compute efficiency is still so low.

```bash
sudo /usr/local/cuda-13.3/bin/ncu --cache-control none --clock-control none --section SpeedOfLight --section WarpStateStats --section SchedulerStats --launch-skip 60 --launch-count 1 ./build/matmul_baseline
```

The result is as follows.

First look at `Section: GPU Speed Of Light Throughput`:

```text
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz        13.79
SM Frequency                    Ghz         2.81
Elapsed Cycles                cycle    2,008,828
Memory Throughput                 %        95.63
DRAM Throughput                   %         0.06
Duration                         us       715.81
L1/TEX Cache Throughput           %        96.74
L2 Cache Throughput               %        21.25
SM Active Cycles              cycle 1,984,872.54
Compute (SM) Throughput           %        95.63
```

DRAM is not doing much work, and L2 is also average, with a utilization of more than 20%. L1 / TEX is working crazily.

SM throughput is also very high, 95%.

But from this, L1 also seems full, and SM also seems full.

Then look at `Section: Scheduler Statistics`:

```text
---------------------------- ----------- ------------
 Metric Name                  Metric Unit Metric Value
 ---------------------------- ----------- ------------
 One or More Eligible                   %        49.29
 Issued Warp Per Scheduler                        0.49
 No Eligible                            %        50.71
 Active Warps Per Scheduler          warp        11.54
 Eligible Warps Per Scheduler        warp         2.48
 ---------------------------- ----------- ------------
```

Avg. Active Threads Per Warp: every scheduler has 11 active warps, so occupancy is relatively high.

No Eligible (50.71%): it means that in more than half of cycles, the scheduler looks around its 11 warps and finds none are ready.

A possible guess is that they are all waiting for L1 cache to be ready.

Finally go through `Section: Warp State Statistics`:

```text
---------------------------------------- ----------- ------------
  Metric Name                              Metric Unit Metric Value
  ---------------------------------------- ----------- ------------
  Warp Cycles Per Issued Instruction             cycle        23.40
  Warp Cycles Per Executed Instruction           cycle        23.40
  Avg. Active Threads Per Warp                                   32
  Avg. Not Predicated Off Threads Per Warp                    32.00
  ---------------------------------------- ----------- ------------
```

Warp Cycles Per Issued Instruction: 23.40. On average, for each instruction executed, the warp stalls for 23 cycles. In theory, for a compute-intensive kernel, it should be close to 1, or even lower (Gemini said so)?

Putting these together, the guess is that L1 is blown up, causing computation to be unable to proceed efficiently.

So we need to continue from GPU hardware architecture. For example, look at Blackwell architecture (https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/).

Blackwell has 160 Streaming Multiprocessors and provides 640 fifth-generation Tensor Cores.

Streaming Multiprocessors can be expanded further. Each SM can be viewed as an independent compute engine, including:

1. 128 CUDA Cores
2. 4 fifth-generation Tensor Cores
3. 256KB Tensor Memory, used by warps to synchronously store intermediate results
4. Special Function Units (SFU): according to NV itself, used for transcendental math, maybe reciprocal or something. I do not really understand, so ignore it.

NV also gave a diagram inside a single SM:

1. Warp Scheduler: 32 thread/clk
2. Dispatch Unit: 32 thread/clk
3. Register File: 16KB * 32bit
4. 64KB Tensor Memory
5. Tensor Memory Accelerator (TMA)
6. L1 Data cache / Shared Memory

At this point, we need to unpack quite a few concepts. In NV's own PTX (Parallel Thread Execution) documentation (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), these concepts are defined like this.

NV defines its own abstraction this way. First is grid:

> The batch of threads that executes a kernel is organized as a grid.

A group of threads executing one kernel is organized into a grid. A grid is actually composed of CTA (cooperative thread arrays) or clusters.

> Cooperative thread arrays (CTAs) implement CUDA thread blocks and clusters implement CUDA thread block clusters.

CTA can be simply understood as thread blocks, but the key is **Cooperative**, meaning they must cooperate together. Threads inside one CTA have shared memory and can synchronize together, and a CTA must land inside the same SM.

cluster is a concept introduced from Hopper. Simply put, it is a CTA of CTAs. It tries to solve the problem that physically neighboring CTAs can share shared memory. This capability is called distributed shared memory, DSMEM.

Threads inside a CTA execute like this:

> Threads within a CTA execute in SIMT (single-instruction, multiple-thread) fashion in groups called warps.

Threads inside a CTA execute in SIMT fashion inside warps. Threads inside a warp are numbered in order. Generally, one warp has 32 threads, and they execute the same instruction.

And warp is the unit scheduled and managed by SM, handled by the warp scheduler in the diagram above (https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html).

Reviewing the naive version we wrote, although logically we divided it into `<<<grid, block>>>`, each thread, even if adjacent and sharing some memory, still chooses to hard-fetch the row and column data it needs from global memory. It does not use the CTA-level Cooperative feature at all (for example, first moving data together into Shared Memory for reuse). This causes L1 Cache to become extremely crowded.

So at this point I let Codex and Claude write a professional version of matrix multiplication. They said they used something called WMMA (Warp Matrix Multiply Accumulate). It no longer thinks in the model of one thread computing one unit. It batches a little more, switching to one warp computing one `16 x 16` tile.

Later I swept another round of parameters. The currently better combination is: one warp computes one `64 x 64` C tile, which is `4 x 4` small tiles, and the K direction goes `16` each time.

The core code looks like this:

```cpp
__global__ void matmul_wmma_2d_regtile(const __half* __restrict__ a,
                                       const __half* __restrict__ b,
                                       float* __restrict__ c,
                                       int n) {
    using namespace nvcuda;

    int row = blockIdx.y * 64;
    int col = blockIdx.x * 64;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    for (int k = 0; k < n; k += 16) {
        for (int i = 0; i < 4; ++i) {
            wmma::load_matrix_sync(a_frag[i], a + (row + i * 16) * n + k, n);
        }
        for (int j = 0; j < 4; ++j) {
            wmma::load_matrix_sync(b_frag[j], b + k * n + col + j * 16, n);
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
            }
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            wmma::store_matrix_sync(c + (row + i * 16) * n + col + j * 16,
                                    acc[i][j], n, wmma::mem_row_major);
        }
    }
}
```

Result:

```text
atype=fp16 btype=fp16 ctype=fp32
tile=64x64 row_tiles=4 col_tiles=4 warps_per_block=1
avg_kernel_time_ms=0.040986
throughput_gflops=52395.64
verification=passed
```

I cannot read it. Talk about it later.

Then run one matrix multiplication from NV, that is cuBLAS:

```text
N=1024
kernel=cublasGemmEx
atype=fp16 btype=fp16 ctype=fp32
warmup_iters=10 bench_iters=100
avg_kernel_time_ms=0.028763
throughput_gflops=74662.46
verification=passed
```

You can see that even after Codex and Opus carefully optimized and tuned for a long time, there may still be a gap: 40us vs 28us.

## cuTile

Finally, the main character, cuTile. cuTile's core idea is: think directly at the Tile level. Personally, I feel it is more like thinking from the perspective of data instead of the perspective of computation.

The main document is this one: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html

No more nonsense, put the kernel directly:

```cpp
__tile_global__ void matmul_cutile_naive(const __half* __restrict__ a,
                                         const __half* __restrict__ b,
                                         float* __restrict__ c,
                                         std::size_t m,
                                         std::size_t k,
                                         std::size_t n) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    using f32_acc = ct::tile<float, ct::shape<32, 32>>;

    constexpr auto tm = 32_ic;
    constexpr auto tn = 32_ic;
    constexpr auto tk = 16_ic;

    auto a_view = ct::partition_view{ct::tensor_span{a, ct::extents{m, k}},
                                     ct::shape{tm, tk}};
    auto b_view = ct::partition_view{ct::tensor_span{b, ct::extents{k, n}},
                                     ct::shape{tk, tn}};
    auto c_view = ct::partition_view{ct::tensor_span{c, ct::extents{m, n}},
                                     ct::shape{tm, tn}};

    auto [bx, by, bz] = ct::bid();
    auto acc = ct::full<f32_acc>(0.0f);

    std::size_t k_tiles = (k + tk - 1) / tk;
    for (auto kk : ct::irange(std::size_t{0}, k_tiles)) {
        acc = ct::mma(a_view.load(by, kk),
                      b_view.load(kk, bx),
                      acc);
    }

    c_view.store(acc, by, bx);
}
```

Use `__tile_global__` to replace `__global__` as the tile kernel entry point.

The launch parameters also need a little change:

```cpp
matmul_cutile_naive<<<grid, 1>>>(d_a, d_b, d_c, N, N, N);
```

The first dimension is still grid, but the second block dimension needs to be 1, because the compiler will decide the thread count internally.

So what exactly is tile? According to NV's documentation:

> a fixed-size, multidimensional array of scalar elements whose shape and element type are known at compile time.

A fixed-size, multidimensional array of scalar elements, whose shape and type must be known at compile time, and every dimension of a Tile must be a power of 2. Tile has "value semantics", meaning copying it copies its elements. Is this basically deep copy? It is not a view. However, the compiler may optimize it.

For example, this declaration in our cuTile matmul:

```cpp
using f32_acc = ct::tile<float, ct::shape<32, 32>>;
```

declares a `<32, 32>` f32 2D tile.

Then there is a fundamental difference between tile and ordinary arrays: you cannot access individual elements inside it. It has no `operator[]`, and no interface for "take the i-th row and j-th column". You can only operate on the whole tile: `ct::full<f32_acc>(0.0f)` creates an all-zero tile, `ct::mma` multiplies two tiles and accumulates, and `store` writes the whole tile back.

My thought is that it forcefully batches.

But when the kernel receives parameters, they are raw pointers. How do these two worlds connect? It should be the three view lines at the beginning. From pointer to tile world, there are two steps.

The first step is using tile semantics to describe the raw pointer:

```cpp
ct::tensor_span{a, ct::extents{m, k}}
```

Treat matrix A as an `M * k` matrix.

Then do another thing:

```cpp
ct::partition_view{ct::tensor_span{...}, ct::shape{tm, tk}}
```

Split the matrix into tiles of `tm * tk`.

After processing the three matrices this way, we get three tile descriptions. Then it is very simple to look at.

We can use:

```cpp
auto [bx, by, bz] = ct::bid();
```

to get a three-dimensional id similar to a traditional kernel's thread id. After getting it, compute directly. The `(by, kk)` block of A multiplies the `(kk, bx)` block of B, and `mma` accumulates into `acc`:

```cpp
auto acc = ct::full<f32_acc>(0.0f);

std::size_t k_tiles = (k + tk - 1) / tk;
for (auto kk : ct::irange(std::size_t{0}, k_tiles)) {
    acc = ct::mma(a_view.load(by, kk),
                  b_view.load(kk, bx),
                  acc);
}

c_view.store(acc, by, bx);
```

Measured:

```text
kernel=matmul_cutile_naive
atype=fp16 btype=fp16 ctype=fp32
tile=32x32 k_tile=16 grid=32x32
avg_kernel_time_ms=0.301034
throughput_gflops=7133.69
verification=passed
```

Without tuning anything, 0.30ms, already 2.4x of the thread naive version (0.71ms). `mma` defaults to tensor core, and how data enters shared memory is also handled by the compiler. But it is still 7x away from WMMA's 0.041ms.

Then I started optimizing the cuTile version's performance.

The first cheap improvement is telling the compiler that the pointers are aligned:

```cpp
a = ct::assume_aligned(a, 16_ic);
b = ct::assume_aligned(b, 16_ic);
c = ct::assume_aligned(c, 16_ic);
```

0.117ms, 2.6x.

The second optimization point is writing extents directly as constants, because we know it serves a `1024 * 1024` matrix, letting it hack the shape. Something like this:

```cpp
auto a_view = ct::partition_view{
    ct::tensor_span{a, ct::extents{1024_ic, 1024_ic}},
    ct::shape{tm, tk}};

for (auto kk : ct::irange(0, 64)) {   // 1024 / 16, known at compile time
    ...
}
```

0.107ms, another 10% faster.

Finally, it is sweeping hyperparameters. Write a small script to run all combinations of tm/tn/tk. It can sweep to `64 x 256 x 32` at 0.0287ms, tying cuBLAS.

Final measured result:

```text
atype=fp16 btype=fp16 ctype=fp32
tile=64x256 k_tile=32
avg_kernel_time_ms=0.028724
throughput_gflops=74762.28
verification=passed
```

cuTile is more LLM friendly, meaning LLMs can write better code based on it, instead of the traditional CUDA set, because with the traditional set it always has to memorize many instructions and such, and the result after tuning is also average. Of course, maybe because in terms of design, matmul is tile's sweet spot. I have not explored more kernels yet. On the other hand, it is arch-independent. I feel that if there is a chance, I still hope Pegainfer can use DSLs well. But for developers, the required CUDA version may be relatively high, because it needs 13.3, though the compiled result should run on CUDA 13.

The era of DSLs is really coming, including more aggressive Megakernels.
