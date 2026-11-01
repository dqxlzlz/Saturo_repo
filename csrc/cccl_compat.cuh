#pragma once

// Compatibility shim for building vLLM's CUDA kernels with CUDA 13.0+.
//
// CCCL 3.0 (bundled with CUDA 13.0) removed the legacy reduction functors
// `cub::Sum` and `cub::Max` that several vLLM kernels still rely on
// (e.g. BlockReduce(...).Reduce(x, cub::Sum{}, n)). Re-provide minimal
// drop-in replacements so the existing call sites keep compiling.
//
// This only activates for CUDA >= 13 (non-ROCm); on CUDA <= 12.x the real
// cub::Sum / cub::Max are still available and this header is a no-op.
#if !defined(USE_ROCM) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)

#include <cub/cub.cuh>

namespace cub {

struct Sum {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                    const T& b) const {
    return a + b;
  }
};

struct Max {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                    const T& b) const {
    return a < b ? b : a;
  }
};

}  // namespace cub

#endif
