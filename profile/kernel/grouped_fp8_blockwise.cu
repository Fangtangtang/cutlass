/*
 * Standalone CUTLASS FP8 Grouped GEMM (SM100 Blackwell)
 *
 * Extracted from FlashInfer's group_gemm_fp8_groupwise_sm100.cuh
 * with all dependencies inlined for independent compilation.
 *
 * Build:
cd profile/kernel

/usr/local/cuda/bin/nvcc -std=c++17 \
    --generate-code=arch=compute_100a,code=[compute_100a,sm_100a] \
    --expt-relaxed-constexpr \
    -DGPU_TRACE_ENABLED \
    -I.. \
    -I../../include \
    -I../../tools/util/include \
    -I../../examples/common \
    -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
    grouped_fp8_blockwise.cu \
    -o grouped_fp8_blockwise

// run
./grouped_fp8_blockwise  --num_groups=256 --m_per_group=256 --n=1536 --k=3072
// trace
GPU_TRACE=0 ./grouped_fp8_blockwise  --num_groups=256 --m_per_group=256 --n=1536 --k=3072

 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../gpu_trace.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/packed_stride.hpp"

// ============================================================
// Inlined FlashInfer utilities
// ============================================================

#define FLASHINFER_ERROR(message)                                              \
  do {                                                                         \
    throw std::runtime_error(std::string(__FUNCTION__) + ": " + (message));    \
  } while (0)

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t e = (expr);                                                    \
    if (e != cudaSuccess) {                                                    \
      std::ostringstream oss;                                                  \
      oss << "CUDA error: " << cudaGetErrorString(e) << " at " << __FILE__    \
          << ":" << __LINE__;                                                  \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

#define FLASHINFER_CUDA_CALL(func) CUDA_CHECK(func)

#define CUTLASS_CHECK(status)                                                  \
  do {                                                                         \
    cutlass::Status error = (status);                                          \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::ostringstream oss;                                                  \
      oss << "CUTLASS error: " << cutlassGetStatusString(error) << " at "     \
          << __FILE__ << ":" << __LINE__;                                      \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

struct AlignedAllocator {
  void* base_ptr;
  void* cur_ptr;
  size_t remaining_space;
  AlignedAllocator(void* buf, size_t space)
      : base_ptr(buf), cur_ptr(buf), remaining_space(space) {}
  template <typename T>
  T* aligned_alloc(size_t size, size_t alignment, std::string name) {
    if (std::align(alignment, size, cur_ptr, remaining_space)) {
      T* result = reinterpret_cast<T*>(cur_ptr);
      cur_ptr = (char*)cur_ptr + size;
      remaining_space -= size;
      return result;
    } else {
      std::ostringstream oss;
      oss << "Buffer overflow when allocating memory for " << name
          << " with size " << size << " and alignment " << alignment
          << ", but only " << remaining_space << " bytes available.";
      FLASHINFER_ERROR(oss.str());
    }
    return nullptr;
  }
};

using namespace cute;

// ============================================================
// Concrete type configuration
// ============================================================

static constexpr int ScaleGranularityM = 128;
static constexpr int ScaleGranularityN = 128;
static constexpr int ScaleGranularityK = 128;
static constexpr bool ScaleMajorK = true;
static constexpr int MmaSM = 1;

using DTypeIn = cutlass::float_e4m3_t;
using DTypeOut = cutlass::bfloat16_t;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

using ElementA = DTypeIn;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = DTypeIn;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementD = DTypeOut;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementC = void;
using LayoutC = void;
constexpr int AlignmentC = 0;

using ElementAccumulator = float;
using ElementCompute = float;

using MmaTileShape_MNK = Shape<cute::Int<MmaSM * 128>, _128, _128>;
using ClusterShape_MNK = Shape<cute::Int<MmaSM>, _1, _1>;

using ScaleConfig = std::conditional_t<
    ScaleMajorK,
    cutlass::detail::Sm100BlockwiseScaleConfig<
        ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
        UMMA::Major::K, UMMA::Major::K>,
    cutlass::detail::Sm100BlockwiseScaleConfig<
        ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
        UMMA::Major::MN, UMMA::Major::MN>>;

using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using EpilogueSchedule =
    std::conditional_t<MmaSM == 1,
                       cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm,
                       cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK,
        ClusterShape_MNK, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute, ElementC, LayoutC*, AlignmentC,
        ElementD, LayoutD*, AlignmentD, EpilogueSchedule>::CollectiveOp;

using MainloopSchedule = std::conditional_t<
    MmaSM == 1,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100>;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
        cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA, ElementB,
        cute::tuple<LayoutB*, LayoutSFB*>, AlignmentB, ElementAccumulator,
        MmaTileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

static_assert(cute::is_same_v<
    typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA,
    LayoutSFA>);
static_assert(cute::is_same_v<
    typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB,
    LayoutSFB>);

// ============================================================
// GPU kernel: prepare per-group GEMM arguments
// ============================================================

__global__ void compute_sm100_cutlass_group_gemm_args(
    ElementA* A, ElementA* B, float* SFA, float* SFB, ElementD* D,
    int* m_indptr, const int* masked_m, int max_m, int n, int k,
    int num_groups, int scale_granularity_m, int scale_granularity_n,
    int scale_granularity_k,
    ProblemShape::UnderlyingProblemShape* problem_sizes,
    const ElementA** A_ptr, const ElementA** B_ptr,
    const float** SFA_ptr, const float** SFB_ptr,
    ElementD** D_ptr, StrideA* stride_A, StrideB* stride_B,
    StrideD* stride_D, LayoutSFA* layout_SFA, LayoutSFB* layout_SFB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_groups) return;

  int sf_n = n / scale_granularity_n;
  int sf_k = k / scale_granularity_k;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  int m_offset = m_indptr[i];
  int m_offset_next = m_indptr[i + 1];
  int stride_m = m_offset_next - m_offset;
  int actual_m = masked_m ? masked_m[i] : stride_m;
  int sf_m_offset = m_offset / scale_granularity_m;

  problem_sizes[i] =
      ProblemShape::UnderlyingProblemShape(actual_m, n, k);
  stride_A[i] =
      cutlass::make_cute_packed_stride(StrideA{}, {stride_m, k, 1});
  stride_B[i] =
      cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  stride_D[i] =
      cutlass::make_cute_packed_stride(StrideD{}, {stride_m, n, 1});

  A_ptr[i] = A + int64_t(m_offset) * int64_t(k);
  B_ptr[i] = B + int64_t(i) * int64_t(n) * int64_t(k);
  D_ptr[i] = D + int64_t(m_offset) * int64_t(n);

  layout_SFA[i] =
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(stride_m, n, k, 1));
  SFA_ptr[i] = SFA + int64_t(sf_m_offset) * int64_t(sf_k);
  layout_SFB[i] =
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(stride_m, n, k, 1));
  SFB_ptr[i] = SFB + int64_t(i) * int64_t(sf_n) * int64_t(sf_k);
}

// ============================================================
// Host launch function
// ============================================================

struct GemmState {
  Gemm gemm;
  typename Gemm::Arguments arguments;
  void* workspace_ptr;
};

GemmState prepare_grouped_gemm(void* int_buffer, size_t int_buffer_size_in_bytes,
                               void* float_buffer, size_t float_buffer_size_in_bytes,
                               ElementA* A, ElementA* B, float* SFA, float* SFB,
                               ElementD* D, int* m_indptr, const int* masked_m,
                               int max_m, int n, int k, int num_groups,
                               cudaStream_t stream) {
  AlignedAllocator allocator(int_buffer, int_buffer_size_in_bytes);

  auto problem_sizes =
      allocator.aligned_alloc<ProblemShape::UnderlyingProblemShape>(
          num_groups * sizeof(ProblemShape::UnderlyingProblemShape), 16,
          "problem_sizes");
  auto A_ptr = allocator.aligned_alloc<const ElementA*>(
      num_groups * sizeof(const ElementA*), 16, "A_ptr");
  auto B_ptr = allocator.aligned_alloc<const ElementA*>(
      num_groups * sizeof(const ElementA*), 16, "B_ptr");
  auto D_ptr = allocator.aligned_alloc<ElementD*>(
      num_groups * sizeof(ElementD*), 16, "D_ptr");
  auto SFA_ptr = allocator.aligned_alloc<const float*>(
      num_groups * sizeof(const float*), 16, "SFA_ptr");
  auto SFB_ptr = allocator.aligned_alloc<const float*>(
      num_groups * sizeof(const float*), 16, "SFB_ptr");
  auto stride_A = allocator.aligned_alloc<StrideA>(
      num_groups * sizeof(StrideA), 16, "stride_A");
  auto stride_B = allocator.aligned_alloc<StrideB>(
      num_groups * sizeof(StrideB), 16, "stride_B");
  auto stride_D = allocator.aligned_alloc<StrideD>(
      num_groups * sizeof(StrideD), 16, "stride_D");
  auto layout_SFA = allocator.aligned_alloc<LayoutSFA>(
      num_groups * sizeof(LayoutSFA), 16, "layout_SFA");
  auto layout_SFB = allocator.aligned_alloc<LayoutSFB>(
      num_groups * sizeof(LayoutSFB), 16, "layout_SFB");

  int num_threads = std::min(num_groups, 1024);
  int num_blocks = (num_groups + num_threads - 1) / num_threads;

  cudaLaunchConfig_t config;
  config.gridDim = num_blocks;
  config.blockDim = num_threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = true;
  config.numAttrs = 1;
  config.attrs = attrs;

  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
      &config, compute_sm100_cutlass_group_gemm_args, A, B, SFA, SFB, D,
      m_indptr, masked_m, max_m, n, k, num_groups, ScaleGranularityM,
      ScaleGranularityN, ScaleGranularityK, problem_sizes, A_ptr, B_ptr,
      SFA_ptr, SFB_ptr, D_ptr, stride_A, stride_B, stride_D, layout_SFA,
      layout_SFB));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  int const sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;

  GemmState state;
  state.arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_groups, problem_sizes, nullptr},
      {A_ptr, stride_A, B_ptr, stride_B, SFA_ptr, layout_SFA, SFB_ptr,
       layout_SFB},
      {{}, nullptr, nullptr, D_ptr, stride_D},
      hw_info};
  auto& fusion_args = state.arguments.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;

  size_t workspace_size = Gemm::get_workspace_size(state.arguments);
  AlignedAllocator float_allocator(float_buffer, float_buffer_size_in_bytes);
  state.workspace_ptr = float_allocator.aligned_alloc<void>(
      workspace_size, 16, "gemm_workspace");

  CUTLASS_CHECK(state.gemm.can_implement(state.arguments));
  CUTLASS_CHECK(state.gemm.initialize(state.arguments, state.workspace_ptr));

  return state;
}

void run_gemm_kernel(GemmState& state, cudaStream_t stream) {
  CUTLASS_CHECK(state.gemm.run(stream, /*cuda_adapter=*/nullptr, /*launch_with_pdl=*/true));
}

// ============================================================
// main() — test harness
// ============================================================

int main(int argc, const char** argv) {
  cutlass::CommandLine cmd(argc, argv);

  int num_groups = 4;
  int m_per_group = 256;
  int n = 2048;
  int k = 512;
  int iterations = 10;

  cmd.get_cmd_line_argument("num_groups", num_groups);
  cmd.get_cmd_line_argument("m_per_group", m_per_group);
  cmd.get_cmd_line_argument("n", n);
  cmd.get_cmd_line_argument("k", k);
  cmd.get_cmd_line_argument("iterations", iterations);

  int total_m = num_groups * m_per_group;
  int sf_m_total = total_m / ScaleGranularityM;
  int sf_n = n / ScaleGranularityN;
  int sf_k = k / ScaleGranularityK;

  std::cout << "=== FP8 Grouped GEMM (SM100) ===" << std::endl;
  std::cout << "num_groups: " << num_groups << std::endl;
  std::cout << "m_per_group: " << m_per_group << std::endl;
  std::cout << "n: " << n << std::endl;
  std::cout << "k: " << k << std::endl;
  std::cout << "total_m: " << total_m << std::endl;
  std::cout << "iterations: " << iterations << std::endl;

  // Build m_indptr on host: [0, m, 2m, ..., num_groups*m]
  std::vector<int> h_m_indptr(num_groups + 1);
  for (int i = 0; i <= num_groups; ++i) {
    h_m_indptr[i] = i * m_per_group;
  }

  // Host-side random data
  size_t A_bytes = (size_t)total_m * k * sizeof(__nv_fp8_e4m3);
  size_t B_bytes = (size_t)num_groups * n * k * sizeof(__nv_fp8_e4m3);
  size_t D_bytes = (size_t)total_m * n * sizeof(nv_bfloat16);
  size_t SFA_bytes = (size_t)sf_m_total * sf_k * sizeof(float);
  size_t SFB_bytes = (size_t)num_groups * sf_n * sf_k * sizeof(float);

  std::vector<uint8_t> h_A(A_bytes);
  std::vector<uint8_t> h_B(B_bytes);
  std::vector<float> h_SFA(sf_m_total * sf_k);
  std::vector<float> h_SFB(num_groups * sf_n * sf_k);

  srand(42);
  for (auto& v : h_A) v = static_cast<uint8_t>(rand() % 256);
  for (auto& v : h_B) v = static_cast<uint8_t>(rand() % 256);
  for (auto& v : h_SFA) v = 0.5f + static_cast<float>(rand()) / RAND_MAX;
  for (auto& v : h_SFB) v = 0.5f + static_cast<float>(rand()) / RAND_MAX;

  // Device allocations
  void *d_A, *d_B, *d_D, *d_SFA, *d_SFB;
  int* d_m_indptr;

  CUDA_CHECK(cudaMalloc(&d_A, A_bytes));
  CUDA_CHECK(cudaMalloc(&d_B, B_bytes));
  CUDA_CHECK(cudaMalloc(&d_D, D_bytes));
  CUDA_CHECK(cudaMalloc(&d_SFA, SFA_bytes));
  CUDA_CHECK(cudaMalloc(&d_SFB, SFB_bytes));
  CUDA_CHECK(
      cudaMalloc(&d_m_indptr, (num_groups + 1) * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), B_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, D_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_SFA, h_SFA.data(), SFA_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_SFB, h_SFB.data(), SFB_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_m_indptr, h_m_indptr.data(),
                         (num_groups + 1) * sizeof(int),
                         cudaMemcpyHostToDevice));

  // Workspace buffers
  size_t int_buf_size = 1 << 20;    // 1 MB
  size_t float_buf_size = 64 << 20; // 64 MB
  void *d_int_buf, *d_float_buf;
  CUDA_CHECK(cudaMalloc(&d_int_buf, int_buf_size));
  CUDA_CHECK(cudaMalloc(&d_float_buf, float_buf_size));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Prepare args once (args kernel + initialize run outside timing loop)
  auto state = prepare_grouped_gemm(
      d_int_buf, int_buf_size, d_float_buf, float_buf_size,
      (ElementA*)d_A, (ElementA*)d_B, (float*)d_SFA,
      (float*)d_SFB, (ElementD*)d_D, d_m_indptr,
      /*masked_m=*/nullptr, m_per_group, n, k, num_groups, stream);

  // Warmup
  for (int i = 0; i < 5; ++i) {
    run_gemm_kernel(state, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "Warmup done." << std::endl;

  // GPU Trace: setup, run one traced iteration, teardown
  GPUTraceParam gt_param;
  gpu_trace::setup_from_env(gt_param, stream);
  run_gemm_kernel(state, stream);
  gpu_trace::teardown(gt_param, stream);

  // Timing (GEMM kernel only)
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) {
    run_gemm_kernel(state, stream);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  float avg_ms = total_ms / iterations;

  double total_flops =
      2.0 * num_groups * m_per_group * n * k;
  double tflops = total_flops / (avg_ms * 1e-3) / 1e12;

  std::cout << "\n=== Results ===" << std::endl;
  std::cout << "Avg latency: " << avg_ms << " ms" << std::endl;
  std::cout << "Throughput:  " << tflops << " TFLOPS" << std::endl;

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_SFA));
  CUDA_CHECK(cudaFree(d_SFB));
  CUDA_CHECK(cudaFree(d_m_indptr));
  CUDA_CHECK(cudaFree(d_int_buf));
  CUDA_CHECK(cudaFree(d_float_buf));

  return 0;
}
