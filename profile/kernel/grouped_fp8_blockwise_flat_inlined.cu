/*
 * Flattened FP8 Grouped GEMM Kernel (SM100 Blackwell, 1SM)
 *
 * This file extracts and flattens the CUTLASS template chain for a specific
 * configuration: FP8 E4M3 grouped GEMM with 128x128x128 blockwise scaling,
 * 1SM warp specialization, RowMajor A, ColumnMajor B, BF16 output.
 *
 * Template chain (original):
 *   grouped_fp8_blockwise.cu
 *     -> GemmUniversalAdapter<GemmKernel>
 *     -> sm100_gemm_array_tma_warpspecialized_mma_transform.hpp  (kernel operator())
 *     -> sm100_mma_array_warpspecialized_blockwise_scaling.hpp   (mainloop collective)
 *     -> sm100_epilogue_array_tma_warpspecialized.hpp            (epilogue collective)
 *
 * Dead branches eliminated (all resolved at compile time):
 *   IsGroupedGemmKernel     = true   (grouped GEMM, not ptr-array)
 *   IsSchedDynamicPersistent= false  (static group scheduler)
 *   IsDynamicCluster        = false  (cluster = (1,1,1))
 *   has_mma_peer_cta        = false  (1SM, no 2SM peer CTA)
 *   IsOverlappingAccum      = false
 *   IsRuntimeDataType       = false  (static FP8 E4M3)
 *   ReuseSmemC              = false
 *   DelayTmaStore           = false
 *
 * Warp layout (12 warps = 384 threads, MaxThreadsPerBlock = round_up(288, 128)):
 *   Warp 0:    MMA            (UMMA compute)
 *   Warp 1:    Scheduler      (static group scheduler)
 *   Warp 2:    MainloopABLoad (TMA loads for A/B + tensormap updates)
 *   Warp 3:    EpilogueLoad   (conditionally active: fusion callback loads)
 *   Warp 4-7:  Epilogue       (accum: scale factor application + store: TMA D output)
 *   Warp 8:    MainloopSFLoad (scale factor loads from gmem to smem)
 *   Warp 9-11: Unused         (reg dealloc only)
 *
 * Build (from profile/kernel/):
    /usr/local/cuda/bin/nvcc -std=c++17 \
        --generate-code=arch=compute_100a,code=[compute_100a,sm_100a] \
        --expt-relaxed-constexpr \
        -I.. -I../../include -I../../tools/util/include -I../../examples/common \
        -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
        -DGPU_TRACE_ENABLED \
        grouped_fp8_blockwise_flat_inlined.cu -o grouped_fp8_blockwise_flat
 *
 * Run:
 *   ./grouped_fp8_blockwise_flat --num_groups=256 --m_per_group=256 --n=1536 --k=3072'
 Run template kernel (baseline):
  ./grouped_fp8_blockwise_flat --num_groups=256 --m_per_group=256 --n=1536 --k=3072 --iterations=10
      flat kernel:
  ./grouped_fp8_blockwise_flat --num_groups=256 --m_per_group=256 --n=1536 --k=3072 --iterations=10 --flat=1
      verification (compares both kernels element-by-element):
  ./grouped_fp8_blockwise_flat --num_groups=256 --m_per_group=256 --n=1536 --k=3072 --verify=1

Trace:
GPU_TRACE=0 ./grouped_fp8_blockwise_flat --num_groups=256 --m_per_group=256 --n=1536 --k=3072 --iterations=10 --flat=1
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../gpu_trace.h"

GPU_TRACE_SCOPE_DEC(gemm_outer);
GPU_TRACE_SCOPE_DEC(load_ab_consumed);
GPU_TRACE_SCOPE_DEC(load_ab_ready);
GPU_TRACE_SCOPE_DEC(start_load_sf);
GPU_TRACE_SCOPE_DEC(ready_to_gemm);
GPU_TRACE_SCOPE_DEC(wait_accumulator_ready);
GPU_TRACE_SCOPE_DEC(accumulator_released);
GPU_TRACE_SCOPE_DEC(accumulator_release);
GPU_TRACE_SCOPE_DEC(wait_sf);
// GPU_TRACE_SCOPE_DEC(epilogue_outer);
GPU_TRACE_SCOPE_DEC(epilogue_outer_end);

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
// Error-checking macros
// ============================================================

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

// ============================================================
// Aligned allocator for device workspace
// ============================================================

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
    }
    std::ostringstream oss;
    oss << "Buffer overflow allocating " << name << " (" << size << " bytes)";
    throw std::runtime_error(oss.str());
  }
};

using namespace cute;

// ============================================================
// Section 1: Concrete type configuration
// ============================================================

static constexpr int ScaleGranularityM = 128;
static constexpr int ScaleGranularityN = 128;
static constexpr int ScaleGranularityK = 128;
static constexpr bool ScaleMajorK [[maybe_unused]] = true;
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

using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
    ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
    UMMA::Major::K, UMMA::Major::K>;

using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK,
        ClusterShape_MNK, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute, ElementC, LayoutC*, AlignmentC,
        ElementD, LayoutD*, AlignmentD, EpilogueSchedule>::CollectiveOp;

using MainloopSchedule =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;

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

// ============================================================
// Section 2: Resolved kernel-level types (from GemmKernel)
// ============================================================

// These are the concrete types once all templates are resolved.
// The user can inspect these to understand exactly what types flow
// through the kernel.

using KernelParams       = typename GemmKernel::Params;
using KernelSharedStorage= typename GemmKernel::SharedStorage;
using TileScheduler      = typename GemmKernel::TileScheduler;
using TileShape          = typename GemmKernel::TileShape;
using TiledMma           = typename GemmKernel::TiledMma;
using CtaShape_MNK       = typename GemmKernel::CtaShape_MNK;
using AtomThrShapeMNK    = typename GemmKernel::AtomThrShapeMNK;
using EpilogueTile       = typename GemmKernel::EpilogueTile;

using MainloopABPipeline      = typename GemmKernel::MainloopABPipeline;
using MainloopABPipelineState  = typename GemmKernel::MainloopABPipelineState;
using MainloopSFPipeline       = typename GemmKernel::MainloopSFPipeline;
using MainloopSFPipelineState  = typename GemmKernel::MainloopSFPipelineState;
using AccumulatorPipeline      = typename GemmKernel::AccumulatorPipeline;
using AccumulatorPipelineState = typename GemmKernel::AccumulatorPipelineState;
using EpiLoadPipeline          = typename GemmKernel::EpiLoadPipeline;
using EpiLoadPipelineState     = typename GemmKernel::EpiLoadPipelineState;
using EpiStorePipeline         = typename GemmKernel::EpiStorePipeline;
using EpiStorePipelineState    = typename GemmKernel::EpiStorePipelineState;
using CLCPipeline              = typename GemmKernel::CLCPipeline;
using CLCPipelineState         = typename GemmKernel::CLCPipelineState;
using LoadOrderBarrier         = typename GemmKernel::LoadOrderBarrier;
using TmemAllocator            = typename GemmKernel::TmemAllocator;

// Resolved constants
static constexpr uint32_t MaxThreadsPerBlock   = GemmKernel::MaxThreadsPerBlock;
static constexpr uint32_t NumSchedThreads      = GemmKernel::NumSchedThreads;
static constexpr uint32_t NumMMAThreads        = GemmKernel::NumMMAThreads;
static constexpr uint32_t NumMainloopABLoadThreads = GemmKernel::NumMainloopABLoadThreads;
static constexpr uint32_t NumEpilogueLoadThreads   = GemmKernel::NumEpilogueLoadThreads;
static constexpr uint32_t NumEpilogueThreads       = GemmKernel::NumEpilogueThreads;
static constexpr uint32_t NumMainloopSFLoadThreads = GemmKernel::NumMainloopSFLoadThreads;
static constexpr uint32_t GenericRegisterRequirement = GemmKernel::GenericRegisterRequirement;
static constexpr uint32_t AccumRegisterRequirement   = GemmKernel::AccumRegisterRequirement;
static constexpr uint32_t AccumulatorPipelineStageCount = GemmKernel::AccumulatorPipelineStageCount; // 4 (to fit in TMEM)
static constexpr int SharedStorageSize = GemmKernel::SharedStorageSize;

// Compile-time verification of resolved constants
static_assert(!GemmKernel::IsSchedDynamicPersistent, "Expected static scheduler for grouped GEMM");
static_assert(cute::size(AtomThrShapeMNK{}) == 1, "Expected 1SM (no MMA peer CTA)");

// Mainloop internal types (for inlined mainloop code)
using ML_SmemLayoutA         = typename CollectiveMainloop::SmemLayoutA;
using ML_SmemLayoutB         = typename CollectiveMainloop::SmemLayoutB;
using ML_SmemLayoutScaleA    = typename CollectiveMainloop::SmemLayoutScaleA;
using ML_SmemLayoutScaleB    = typename CollectiveMainloop::SmemLayoutScaleB;
using ML_GmemTiledCopySFA    = typename CollectiveMainloop::GmemTiledCopySFA;
using ML_GmemTiledCopySFB    = typename CollectiveMainloop::GmemTiledCopySFB;
using ML_TmaInternalElementA = typename CollectiveMainloop::TmaInternalElementA;
using ML_TmaInternalElementB = typename CollectiveMainloop::TmaInternalElementB;
using ML_TMA_A               = typename CollectiveMainloop::Params::TMA_A;
using ML_TMA_B               = typename CollectiveMainloop::Params::TMA_B;
static constexpr uint32_t ML_TmaTransactionBytes              = CollectiveMainloop::TmaTransactionBytes;
static constexpr int      ML_K_BLOCK_MMAS_PER_SCALE_K         = CollectiveMainloop::K_BLOCK_MMAS_PER_SCALE_K;
static constexpr int      ML_ScaleKsPerTile                   = CollectiveMainloop::ScaleKsPerTile;
static constexpr int      ML_ScaleGranularityK                = CollectiveMainloop::ScaleGranularityK;
static constexpr int      ML_NumMainloopSFProducerThreadEvents = CollectiveMainloop::NumMainloopSFProducerThreadEvents;

// Epilogue internal types (for inlined epilogue code)
using EPI_CopyOpT2R      = typename CollectiveEpilogue::CopyOpT2R;
using EPI_CopyOpR2S      = typename CollectiveEpilogue::CopyOpR2S;
using EPI_CopyOpS2G      = typename CollectiveEpilogue::CopyOpS2G;
using EPI_CopyOpR2R      = typename CollectiveEpilogue::CopyOpR2R;
using EPI_CopyOpS2R      = typename CollectiveEpilogue::CopyOpS2R;
using EPI_SmemElementC    = typename CollectiveEpilogue::SmemElementC;
using EPI_SmemElementD    = typename CollectiveEpilogue::SmemElementD;
using EPI_SmemLayoutC     = typename CollectiveEpilogue::SmemLayoutC;
using EPI_SmemLayoutD     = typename CollectiveEpilogue::SmemLayoutD;
using EPI_FusionCallbacks = typename CollectiveEpilogue::FusionCallbacks;
using EPI_InternalStrideC = typename CollectiveEpilogue::InternalStrideC;
using EPI_InternalStrideD = typename CollectiveEpilogue::InternalStrideD;
static constexpr int  EPI_ThreadCount    = CollectiveEpilogue::ThreadCount;
static constexpr int  EPI_FragmentSize   = CollectiveEpilogue::DispatchPolicy::FragmentSize;
static constexpr bool EPI_UnrollEpiLoop  = CollectiveEpilogue::UnrollEpiLoop;
static constexpr bool EPI_is_source_supported = CollectiveEpilogue::is_source_supported;

// ============================================================
// Section 3: Flattened kernel
//
// This is the GemmKernel::operator() with all dead branches
// stripped for the 1SM grouped GEMM configuration.
//
// Warp assignment:
//   warp 0:    MMA
//   warp 1:    Scheduler
//   warp 2:    MainloopABLoad
//   warp 3:    EpilogueLoad (conditionally active based on fusion)
//   warp 4-7:  Epilogue (4 warps, 128 threads)
//   warp 8:    MainloopSFLoad
//   warp 9-11: Unused
// ============================================================

template <typename _CollectiveEpi = CollectiveEpilogue>
struct FlatGemmKernel {
CUTLASS_DEVICE void
operator()(KernelParams const& params, char* smem_buf) {
  using namespace cute;
  using X = Underscore;

  static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes);
  auto problem_shape = params.problem_shape;

  // --- Warp identification ---
  int warp_idx = cutlass::canonical_warp_idx_sync();
  enum class WarpCategory : int32_t {
    MMA = 0, Sched = 1, MainloopABLoad = 2, EpilogueLoad = 3,
    Epilogue = 4, MainloopSFLoad = 8, Unused = 9
  };
  WarpCategory warp_category = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
    if (warp_idx < 4)       return WarpCategory(warp_idx);
    else if (warp_idx < 8)  return WarpCategory::Epilogue;
    else if (warp_idx == 8) return WarpCategory::MainloopSFLoad;
    else                    return WarpCategory::Unused;
  }();

  uint32_t lane_predicate = cute::elect_one_sync();
  auto cluster_shape = Shape<_1,_1,_1>{};
  uint32_t cta_rank_in_cluster = 0;
  // --- Shared memory ---
  KernelSharedStorage& shared_storage = *reinterpret_cast<KernelSharedStorage*>(smem_buf);

  // --- Mainloop TMA descriptor pointers (inlined constructor, IsDynamicCluster=false) ---
  const ML_TMA_A* tma_load_a = &params.mainloop.tma_load_a;
  const ML_TMA_B* tma_load_b = &params.mainloop.tma_load_b;

  // --- Epilogue: fusion callbacks and load-needed check ---
  typename _CollectiveEpi::FusionCallbacks fusion_callbacks(params.epilogue.thread, shared_storage.tensors.epilogue.thread);
  bool is_epi_load_needed = fusion_callbacks.is_producer_load_needed();

  // --- Pipeline initialization ---

  // MainloopAB pipeline (producer: ABLoad warp, consumer: MMA warp)
  typename MainloopABPipeline::Params mainloop_ab_pipeline_params;
  if (warp_category == WarpCategory::MainloopABLoad) {
    mainloop_ab_pipeline_params.role = MainloopABPipeline::ThreadCategory::Producer;
  }
  if (warp_category == WarpCategory::MMA) {
    mainloop_ab_pipeline_params.role = MainloopABPipeline::ThreadCategory::Consumer;
  }
  mainloop_ab_pipeline_params.is_leader = lane_predicate && (warp_category == WarpCategory::MainloopABLoad);
  mainloop_ab_pipeline_params.transaction_bytes = ML_TmaTransactionBytes;
  mainloop_ab_pipeline_params.initializing_warp = 0;
  MainloopABPipeline mainloop_ab_pipeline(
      shared_storage.pipelines.mainloop.pipeline_ab,
      mainloop_ab_pipeline_params, cluster_shape,
      cute::true_type{}, cute::false_type{});

  // MainloopSF pipeline (producer: SFLoad warp, consumer: Epilogue warps)
  typename MainloopSFPipeline::Params mainloop_sf_pipeline_params;
  if (warp_category == WarpCategory::MainloopSFLoad) {
    mainloop_sf_pipeline_params.role = MainloopSFPipeline::ThreadCategory::Producer;
  }
  if (warp_category == WarpCategory::Epilogue) {
    mainloop_sf_pipeline_params.role = MainloopSFPipeline::ThreadCategory::Consumer;
  }
  mainloop_sf_pipeline_params.initializing_warp = 8;
  mainloop_sf_pipeline_params.producer_arv_count = ML_NumMainloopSFProducerThreadEvents;
  mainloop_sf_pipeline_params.consumer_arv_count = NumEpilogueThreads;
  MainloopSFPipeline mainloop_sf_pipeline(
      shared_storage.pipelines.mainloop.pipeline_sf,
      mainloop_sf_pipeline_params);

  // Epilogue Load pipeline (producer: EpiLoad warp, consumer: Epilogue warps)
  // Note: is_epi_load_needed=false so this pipeline is mostly unused
  typename EpiLoadPipeline::Params epi_load_pipeline_params;
  if (warp_category == WarpCategory::EpilogueLoad) {
    epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
  }
  if (warp_category == WarpCategory::Epilogue) {
    epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
  }
  epi_load_pipeline_params.dst_blockid = 0;
  epi_load_pipeline_params.producer_arv_count = NumEpilogueLoadThreads;
  epi_load_pipeline_params.consumer_arv_count = NumEpilogueThreads;
  epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
  epi_load_pipeline_params.initializing_warp = 4;
  EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

  // Epilogue Store pipeline (producer: Epilogue warps, consumer: TMA hardware)
  typename EpiStorePipeline::Params epi_store_pipeline_params;
  epi_store_pipeline_params.always_wait = true;
  EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

  // Load order barrier (MainloopABLoad arrives, EpilogueLoad waits)
  typename LoadOrderBarrier::Params load_order_barrier_params;
  load_order_barrier_params.group_id = (warp_category == WarpCategory::MainloopABLoad) ? 0 : 1;
  load_order_barrier_params.group_size = NumMainloopABLoadThreads;
  load_order_barrier_params.initializing_warp = 5;
  LoadOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, load_order_barrier_params);

  // CLC pipeline (static scheduler: Sched warp produces, all others consume)
  typename CLCPipeline::Params clc_pipeline_params;
  if (warp_category == WarpCategory::Sched) {
    clc_pipeline_params.role = CLCPipeline::ThreadCategory::Producer;
  } else {
    clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
  }
  clc_pipeline_params.initializing_warp = 1;
  clc_pipeline_params.producer_arv_count = 1;
  clc_pipeline_params.consumer_arv_count = NumMainloopABLoadThreads + NumEpilogueThreads +
                                           NumMMAThreads + NumMainloopSFLoadThreads;
  if (is_epi_load_needed) {
    clc_pipeline_params.consumer_arv_count += NumEpilogueLoadThreads;
  }
  CLCPipeline clc_pipeline(shared_storage.pipelines.clc, clc_pipeline_params);

  // Accumulator pipeline (producer: MMA warp, consumer: Epilogue warps)
  typename AccumulatorPipeline::Params accumulator_pipeline_params;
  if (warp_category == WarpCategory::MMA) {
    accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
  }
  if (warp_category == WarpCategory::Epilogue) {
    accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
  }
  accumulator_pipeline_params.producer_arv_count = 1;
  accumulator_pipeline_params.consumer_arv_count = NumEpilogueThreads; // size(AtomThrShapeMNK{})=1
  accumulator_pipeline_params.initializing_warp = 2;
  AccumulatorPipeline accumulator_pipeline(
      shared_storage.pipelines.mainloop.pipeline_accum,
      accumulator_pipeline_params, cluster_shape);

  // TMEM allocator
  TmemAllocator tmem_allocator{};

  // Sync allocation status between MMA and epilogue warps
  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      NumMMAThreads + NumEpilogueThreads,
      cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);

  // Epilogue throttle barrier: stall epilogue until prologue finishes
  cutlass::arch::ClusterBarrier& epilogue_throttle_barrier = shared_storage.pipelines.epilogue_throttle;
  if (warp_category == WarpCategory::MMA && lane_predicate) {
    epilogue_throttle_barrier.init(NumMMAThreads + NumSchedThreads + NumMainloopABLoadThreads +
                                   (is_epi_load_needed ? NumEpilogueLoadThreads : 0));
  }

  // --- Wait for all pipeline init to be visible ---
  cutlass::pipeline_init_arrive_relaxed(1 /* cluster_size */);

  // --- Pipeline state initialization ---
  MainloopABPipelineState mainloop_ab_pipe_consumer_state;
  MainloopABPipelineState mainloop_ab_pipe_producer_state = cutlass::make_producer_start_state<MainloopABPipeline>();
  EpiLoadPipelineState epi_load_pipe_consumer_state;
  EpiLoadPipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
  EpiStorePipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();
  CLCPipelineState clc_pipe_consumer_state;
  CLCPipelineState clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();
  AccumulatorPipelineState accumulator_pipe_consumer_state;
  AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();
  MainloopSFPipelineState mainloop_sf_pipe_consumer_state;
  MainloopSFPipelineState mainloop_sf_pipe_producer_state = cutlass::make_producer_start_state<MainloopSFPipeline>();

  // --- Accumulator allocation ---
  TiledMma tiled_mma;
  auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));
  Tensor accumulators = cutlass::detail::make_sm100_accumulator<
      AccumulatorPipelineStageCount, false /*IsOverlappingAccum*/>(
      tiled_mma, acc_shape, EpilogueTile{});

  cutlass::arch::wait_on_dependent_grids();

  // --- Tile scheduler ---
  TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler,
                          cute::block_id_in_cluster());
  auto work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
  auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

  cutlass::pipeline_init_wait(1 /* cluster_size */);

  // Early exit for grouped GEMM if no valid work
  if (!work_tile_info.is_valid()) return;
  int32_t sm_id = static_cast<int32_t>(cutlass::arch::SmId());

  auto problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(work_tile_info.L_idx), 1);

  // Calculate masks after cluster barrier
  dim3 block_id_in_cluster_dim = cute::block_id_in_cluster();
  mainloop_ab_pipeline.init_masks(cluster_shape, block_id_in_cluster_dim);
  accumulator_pipeline.init_masks(cluster_shape, block_id_in_cluster_dim);

  GPU_TRACE_INIT

  // ==========================================================
  // WARP 2: MainloopABLoad
  // Loads A and B tiles via TMA, handles tensormap updates between groups
  // ==========================================================
  if (warp_category == WarpCategory::MainloopABLoad) {
    cutlass::arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();
    cutlass::arch::wait_on_dependent_grids();

    // -- load_ab_init inlined (source: sm100_mma_array_warpspecialized_blockwise_scaling.hpp:580-640) --
    auto [M_init,N_init,K_init,L_init] = problem_shape_MNKL;
    const int32_t mock_L = 1;
    Tensor mA_mkl = tma_load_a->get_tma_tensor(make_shape(M_init,K_init,mock_L));
    Tensor mB_nkl = tma_load_b->get_tma_tensor(make_shape(N_init,K_init,mock_L));
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    auto cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));
    Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl);
    Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl);
    Tensor sA = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_A.begin()), ML_SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_B.begin()), ML_SmemLayoutB{});
    Layout cta_layout_mnk  = make_layout(cluster_shape);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster);
    auto [tAgA_mkl, tAsA] = tma_partition(*tma_load_a,
        get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
        group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl));
    auto [tBgB_nkl, tBsB] = tma_partition(*tma_load_b,
        get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
        group_modes<0,3>(sB), group_modes<0,3>(tCgB_nkl));
    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

    // -- tensormaps_init inlined (source: lines 1241-1265) --
    cute::TmaDescriptor* gmem_tensormap_ml = params.mainloop.tensormaps;
    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap_ml[sm_id];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap_ml[sm_id + params.hw_info.sm_count];
    if (cute::elect_one_sync()) {
      Tensor pA_tmap = make_tensor(tma_load_a->get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sA_tmap = make_tensor(make_smem_ptr(&shared_storage.tensormaps.mainloop.smem_tensormap_A), Int<1>{}, Int<1>{});
      Tensor pB_tmap = make_tensor(tma_load_b->get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sB_tmap = make_tensor(make_smem_ptr(&shared_storage.tensormaps.mainloop.smem_tensormap_B), Int<1>{}, Int<1>{});
      copy(recast<uint128_t>(pA_tmap), recast<uint128_t>(sA_tmap));
      copy(recast<uint128_t>(pB_tmap), recast<uint128_t>(sB_tmap));
    }
    __syncwarp();

    bool did_batch_change = true;
    bool do_load_order_arrive = is_epi_load_needed;

    epilogue_throttle_barrier.arrive();
    GET_GPU_TRACE(true);
    do {
      int32_t curr_batch = idx2crd(work_tile_info.L_idx, shape<4>(gA_mkl));
      problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);

      // -- tensormaps_perform_update inlined (source: lines 1331-1352) --
      if (did_batch_change) {
        if (cute::elect_one_sync()) {
          // update address
          cute::tma_descriptor_replace_addr_in_shared_mem(
              shared_storage.tensormaps.mainloop.smem_tensormap_A,
              params.mainloop.ptr_A[curr_batch]);
          cute::tma_descriptor_replace_addr_in_shared_mem(
              shared_storage.tensormaps.mainloop.smem_tensormap_B,
              params.mainloop.ptr_B[curr_batch]);
          auto group_shape_MNKL = problem_shape_MNKL;
          constexpr int MaxTensorRank = 5;
          {
            cute::array<uint32_t, MaxTensorRank> prob_shape_A  = {1,1,1,1,1};
            cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};
            ML_TmaInternalElementA const* ptr_A_null = nullptr;
            Tensor tensor_a = make_tensor(ptr_A_null,
                make_shape(get<0>(group_shape_MNKL), get<2>(group_shape_MNKL), Int<1>{}),
                params.mainloop.dA[curr_batch]);
            cute::detail::fill_tma_gmem_shape_stride(*tma_load_a, tensor_a, prob_shape_A, prob_stride_A);
            for (uint64_t& s : prob_stride_A) { s = (s * cute::sizeof_bits_v<ML_TmaInternalElementA>) / 8; }
            cute::tma_descriptor_replace_dims_strides_in_shared_mem(
                shared_storage.tensormaps.mainloop.smem_tensormap_A, prob_shape_A, prob_stride_A);
          }
          {
            cute::array<uint32_t, MaxTensorRank> prob_shape_B  = {1,1,1,1,1};
            cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};
            ML_TmaInternalElementB const* ptr_B_null = nullptr;
            Tensor tensor_b = make_tensor(ptr_B_null,
                make_shape(get<1>(group_shape_MNKL), get<2>(group_shape_MNKL), Int<1>{}),
                params.mainloop.dB[curr_batch]);
            cute::detail::fill_tma_gmem_shape_stride(*tma_load_b, tensor_b, prob_shape_B, prob_stride_B);
            for (uint64_t& s : prob_stride_B) { s = (s * cute::sizeof_bits_v<ML_TmaInternalElementB>) / 8; }
            cute::tma_descriptor_replace_dims_strides_in_shared_mem(
                shared_storage.tensormaps.mainloop.smem_tensormap_B, prob_shape_B, prob_stride_B);
          }
        }
        __syncwarp();
        // -- tensormaps_cp_fence_release inlined (source: lines 1354-1367) --
        if (cute::elect_one_sync()) {
          cute::tma_desc_commit_group();
          cute::tma_desc_wait_group();
        }
        tma_descriptor_cp_fence_release(tma_desc_a, shared_storage.tensormaps.mainloop.smem_tensormap_A);
        tma_descriptor_cp_fence_release(tma_desc_b, shared_storage.tensormaps.mainloop.smem_tensormap_B);
      }

      auto k_tile_iter = scheduler.get_k_tile_iterator(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, shape<3>(gA_mkl));
      auto k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
      auto k_tile_prologue = min(MainloopABPipeline::Stages, k_tile_count); // prologure stages: avaliable resource for 'prefetching'

      auto cta_coord_mnk = append<4>(
          make_coord(get<0>(cta_coord_mnkl), get<1>(cta_coord_mnkl), get<2>(cta_coord_mnkl)),
          Int<0>{});

      // -- load_ab inlined: prologue (source: lines 816-870) --
      {
        if (did_batch_change) {
          cute::tma_descriptor_fence_acquire(tma_desc_a);
          cute::tma_descriptor_fence_acquire(tma_desc_b);
        }
        Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnk) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnk));
        Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnk), _, get<3>(cta_coord_mnk));
        auto barrier_token = mainloop_ab_pipeline.producer_try_acquire(mainloop_ab_pipe_producer_state);
        int ab_count = k_tile_prologue; // fill in the pipeline with prologue iterations
        CUTLASS_PRAGMA_NO_UNROLL
        while (ab_count > 0) {
          mainloop_ab_pipeline.producer_acquire(mainloop_ab_pipe_producer_state, barrier_token);
          GPU_TRACE_MARK(load_ab_consumed);
          using BarrierType = typename MainloopABPipeline::ProducerBarrierType;
          BarrierType* tma_barrier = mainloop_ab_pipeline.producer_get_barrier(mainloop_ab_pipe_producer_state);
          int write_stage = mainloop_ab_pipe_producer_state.index();
          ++mainloop_ab_pipe_producer_state;
          barrier_token = mainloop_ab_pipeline.producer_try_acquire(mainloop_ab_pipe_producer_state);
          if (cute::elect_one_sync()) {
            copy(tma_load_a->with(tma_desc_a, *tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
            copy(tma_load_b->with(tma_desc_b, *tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));
          }
          --ab_count;
          ++k_tile_iter;
        }
      }

      if (do_load_order_arrive) {
        load_order_barrier.arrive();
        do_load_order_arrive = false;
      }

      // -- load_ab inlined: remaining (source: lines 816-870) --
      {
        Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnk) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnk));
        Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnk), _, get<3>(cta_coord_mnk));
        auto barrier_token = mainloop_ab_pipeline.producer_try_acquire(mainloop_ab_pipe_producer_state);
        int ab_count = k_tile_count - k_tile_prologue; // remaining iterations after prologue
        CUTLASS_PRAGMA_NO_UNROLL
        while (ab_count > 0) {
          // wait until the consumer has released the SMEM to be used
          mainloop_ab_pipeline.producer_acquire(mainloop_ab_pipe_producer_state, barrier_token);
          GPU_TRACE_MARK(load_ab_consumed);
          using BarrierType = typename MainloopABPipeline::ProducerBarrierType;
          BarrierType* tma_barrier = mainloop_ab_pipeline.producer_get_barrier(mainloop_ab_pipe_producer_state);
          int write_stage = mainloop_ab_pipe_producer_state.index();
          ++mainloop_ab_pipe_producer_state;
          // an overlapped 'prefetch'
          barrier_token = mainloop_ab_pipeline.producer_try_acquire(mainloop_ab_pipe_producer_state);
          if (cute::elect_one_sync()) {
            copy(tma_load_a->with(tma_desc_a, *tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
            copy(tma_load_b->with(tma_desc_b, *tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));
          }
          --ab_count;
          ++k_tile_iter;
        }
      }

      __syncwarp();

      auto [next_work, incr] = scheduler.fetch_next_work(
          work_tile_info, clc_pipeline, clc_pipe_consumer_state);
      work_tile_info = next_work;
      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      if (incr) ++clc_pipe_consumer_state;
      did_batch_change = curr_batch != idx2crd(work_tile_info.L_idx, shape<4>(gA_mkl));
    } while (work_tile_info.is_valid());

    RELEASE_GPU_TRACE;
    // -- load_ab_tail inlined --
    mainloop_ab_pipeline.producer_tail(mainloop_ab_pipe_producer_state);
  }

  // ==========================================================
  // WARP 8: MainloopSFLoad
  // Loads scale factors (SFA/SFB) from gmem to smem
  // ==========================================================
  else if (warp_category == WarpCategory::MainloopSFLoad) {
    cutlass::arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

    int32_t curr_batch = idx2crd(work_tile_info.L_idx, get<3>(problem_shape_MNKL));
    const ML_TMA_A* tma_load_a_sf = &params.mainloop.tma_load_a;
    const int32_t mock_L = 1;

    GET_GPU_TRACE(true);
    // -- Invariant SF setup (same for all groups) --
    ML_GmemTiledCopySFA scale_copy_a{};
    ML_GmemTiledCopySFB scale_copy_b{};
    auto thr_scale_copy_a = scale_copy_a.get_slice(threadIdx.x % size(scale_copy_a));
    auto thr_scale_copy_b = scale_copy_b.get_slice(threadIdx.x % size(scale_copy_b));
    Tensor sSFA = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_SFA.begin()),
        ML_SmemLayoutScaleA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_SFB.begin()),
        ML_SmemLayoutScaleB{});
    auto tSFAsSFA = thr_scale_copy_a.partition_D(sSFA);
    auto tSFBsSFB = thr_scale_copy_b.partition_D(sSFB);

    // -- load_sf_init inlined (source: lines 642-743) --
    auto [M0,N0,K0,L0] = problem_shape_MNKL;
    Tensor gA_mkl = local_tile(
        tma_load_a_sf->get_tma_tensor(make_shape(M0,K0,mock_L)),
        TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

    auto layout_SFA = params.mainloop.layout_SFA[curr_batch];
    auto layout_SFB = params.mainloop.layout_SFB[curr_batch];

    auto gSFA_mkl = local_tile(
        make_tensor(make_gmem_ptr(params.mainloop.ptr_SFA[curr_batch]), layout_SFA),
        CtaShape_MNK{}, make_coord(_,_,_), Step<_1, X,_1>{});
    auto gSFB_nkl = local_tile(
        make_tensor(make_gmem_ptr(params.mainloop.ptr_SFB[curr_batch]), layout_SFB),
        CtaShape_MNK{}, make_coord(_,_,_), Step< X,_1,_1>{});

    auto identSFA_mkl = local_tile(
        make_identity_tensor(shape(layout_SFA)),
        CtaShape_MNK{}, make_coord(_,_,_), Step<_1, X,_1>{});
    auto identSFB_nkl = local_tile(
        make_identity_tensor(shape(layout_SFB)),
        CtaShape_MNK{}, make_coord(_,_,_), Step< X,_1,_1>{});

    auto tSFAgSFA_mkl = thr_scale_copy_a.partition_S(gSFA_mkl);
    auto tSFAIdentSFA_mkl = thr_scale_copy_a.partition_S(identSFA_mkl);
    auto tSFBgSFB_nkl = thr_scale_copy_b.partition_S(gSFB_nkl);
    auto tSFBIdentSFB_nkl = thr_scale_copy_b.partition_S(identSFB_nkl);

    cutlass::arch::wait_on_dependent_grids();

    bool did_batch_change = true;
    do {
      int32_t curr_batch = idx2crd(work_tile_info.L_idx, size<4>(gA_mkl));
      problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);

      if (did_batch_change) {
        // -- load_sf_update inlined (source: lines 656-743) --
        auto [M,N,K,L] = problem_shape_MNKL;
        gA_mkl = local_tile(
            tma_load_a_sf->get_tma_tensor(make_shape(M,K,mock_L)),
            TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

        layout_SFA = params.mainloop.layout_SFA[curr_batch];
        layout_SFB = params.mainloop.layout_SFB[curr_batch];

        gSFA_mkl = local_tile(
            make_tensor(make_gmem_ptr(params.mainloop.ptr_SFA[curr_batch]), layout_SFA),
            CtaShape_MNK{}, make_coord(_,_,_), Step<_1, X,_1>{});
        gSFB_nkl = local_tile(
            make_tensor(make_gmem_ptr(params.mainloop.ptr_SFB[curr_batch]), layout_SFB),
            CtaShape_MNK{}, make_coord(_,_,_), Step< X,_1,_1>{});

        identSFA_mkl = local_tile(
            make_identity_tensor(shape(layout_SFA)),
            CtaShape_MNK{}, make_coord(_,_,_), Step<_1, X,_1>{});
        identSFB_nkl = local_tile(
            make_identity_tensor(shape(layout_SFB)),
            CtaShape_MNK{}, make_coord(_,_,_), Step< X,_1,_1>{});

        tSFAgSFA_mkl = thr_scale_copy_a.partition_S(gSFA_mkl);
        tSFAIdentSFA_mkl = thr_scale_copy_a.partition_S(identSFA_mkl);
        tSFBgSFB_nkl = thr_scale_copy_b.partition_S(gSFB_nkl);
        tSFBIdentSFB_nkl = thr_scale_copy_b.partition_S(identSFB_nkl);
      }

      auto k_tile_iter = scheduler.get_k_tile_iterator(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, shape<3>(gA_mkl));
      auto k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

      auto cta_coord_mnk = append<4>(
          make_coord(get<0>(cta_coord_mnkl), get<1>(cta_coord_mnkl), get<2>(cta_coord_mnkl)),
          Int<0>{});

      // -- load_sf inlined (source: lines 894-962) --
      {
        Tensor tSFAgSFA = tSFAgSFA_mkl(_, _, _, get<0>(cta_coord_mnk), _, get<3>(cta_coord_mnk));
        Tensor tSFBgSFB = tSFBgSFB_nkl(_, _, _, get<1>(cta_coord_mnk), _, get<3>(cta_coord_mnk));

        Tensor thr_tile_SFA_k = tSFAIdentSFA_mkl(_0{}, _, _, get<0>(cta_coord_mnk), _, get<3>(cta_coord_mnk));
        Tensor thr_tile_pSFA = make_tensor<bool>(shape(filter_zeros(thr_tile_SFA_k(_,_,_0{}), tSFAgSFA(_0{},_,_,_0{}).stride())));
        Tensor thr_tile_SFB_k = tSFBIdentSFB_nkl(_0{}, _, _, get<1>(cta_coord_mnk), _, get<3>(cta_coord_mnk));
        Tensor thr_tile_pSFB = make_tensor<bool>(shape(filter_zeros(thr_tile_SFB_k(_,_,_0{}), tSFBgSFB(_0{},_,_,_0{}).stride())));

        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > 0) {
          mainloop_sf_pipeline.producer_acquire(mainloop_sf_pipe_producer_state); // wait

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(thr_tile_pSFA); ++i) {
            Tensor thr_tile_SFA = filter_zeros(thr_tile_SFA_k(_,_,*k_tile_iter), tSFAgSFA(_0{},_,_,_0{}).stride());
            thr_tile_pSFA(i) = elem_less(thr_tile_SFA(i), shape(filter_zeros(layout_SFA))) && threadIdx.x % 32 < size(scale_copy_a);
          }

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(thr_tile_pSFB); ++i) {
            Tensor thr_tile_SFB = filter_zeros(thr_tile_SFB_k(_,_,*k_tile_iter), tSFBgSFB(_0{},_,_,_0{}).stride());
            thr_tile_pSFB(i) = elem_less(thr_tile_SFB(i), shape(filter_zeros(layout_SFB))) && threadIdx.x % 32 < size(scale_copy_b);
          }
          // small tensors, do not load with TMA (global to shared)
          copy_if(scale_copy_a, thr_tile_pSFA,
              filter_zeros(tSFAgSFA(_,_,_,*k_tile_iter)),
              filter_zeros(tSFAsSFA(_,_,_,mainloop_sf_pipe_producer_state.index())));
          copy_if(scale_copy_b, thr_tile_pSFB,
              filter_zeros(tSFBgSFB(_,_,_,*k_tile_iter)),
              filter_zeros(tSFBsSFB(_,_,_,mainloop_sf_pipe_producer_state.index())));
          mainloop_sf_pipeline.producer_commit(mainloop_sf_pipe_producer_state, cutlass::arch::cpasync_barrier_arrive_noinc);

          __syncwarp();

          ++mainloop_sf_pipe_producer_state;
          --k_tile_count;
          ++k_tile_iter;
        }
      }

      __syncwarp();

      auto [next_work, incr] = scheduler.fetch_next_work(
          work_tile_info, clc_pipeline, clc_pipe_consumer_state);
      work_tile_info = next_work;
      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      if (incr) ++clc_pipe_consumer_state;
      did_batch_change = curr_batch != idx2crd(work_tile_info.L_idx, size<4>(gA_mkl));
    } while (work_tile_info.is_valid());

    RELEASE_GPU_TRACE;
    // -- load_sf_tail inlined --
    mainloop_sf_pipeline.producer_tail(mainloop_sf_pipe_producer_state);
  }

  // ==========================================================
  // WARP 1: Scheduler (static group scheduler)
  // Produces work tile IDs for all other warps via CLC pipeline
  // ==========================================================
  else if (warp_category == WarpCategory::Sched) {
    cutlass::arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();
    epilogue_throttle_barrier.arrive();
    cutlass::arch::wait_on_dependent_grids();

    // Static scheduler loop (IsSchedDynamicPersistent = false)
    do {
      auto [next_work, incr] = scheduler.advance_to_next_work(
          clc_pipeline, clc_pipe_producer_state);
      work_tile_info = next_work;
      if (incr) ++clc_pipe_producer_state;
    } while (work_tile_info.is_valid());
    clc_pipeline.producer_tail(clc_pipe_producer_state);
  }

  // ==========================================================
  // WARP 0: MMA
  // Issues UMMA instructions, produces accumulator results in TMEM
  // ==========================================================
  else if (warp_category == WarpCategory::MMA) {
    cutlass::arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

    // TMEM allocation
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
    __syncwarp();
    tmem_allocation_result_barrier.arrive();
    uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
    accumulators.data() = tmem_base_ptr;
    int tmem_non_accumulator_base = tmem_base_ptr +
        cutlass::detail::find_tmem_tensor_col_offset(accumulators);

    // -- mma_init inlined (source: lines 758-803) --
    Tensor sA_mma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_A.begin()), ML_SmemLayoutA{});
    Tensor sB_mma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_B.begin()), ML_SmemLayoutB{});
    Tensor tCrA_ = TiledMma::make_fragment_A(sA_mma);
    Tensor tCrB_ = TiledMma::make_fragment_B(sB_mma);
    CUTE_STATIC_ASSERT_V(rank(tCrA_) == _4{});
    auto mma_tile_shape_A = make_shape(get<0>(shape(tCrA_.layout())),
                                       get<1>(shape(tCrA_.layout())),
                                       Int<ML_K_BLOCK_MMAS_PER_SCALE_K>{},
                                       _1{}); // tile on K dim
    auto mma_tile_shape_B = make_shape(get<0>(shape(tCrB_.layout())),
                                       get<1>(shape(tCrB_.layout())),
                                       Int<ML_K_BLOCK_MMAS_PER_SCALE_K>{},
                                       _1{});
    Tensor tCrA = flat_divide(tCrA_, mma_tile_shape_A)(_,_,_,_0{},_0{},_0{},_,_);
    Tensor tCrB = flat_divide(tCrB_, mma_tile_shape_B)(_,_,_,_0{},_0{},_0{},_,_);
    TiledMma tiled_mma;

    epilogue_throttle_barrier.arrive();

    GET_GPU_TRACE(true);

    do {
      auto [next_work, incr] = scheduler.fetch_next_work(
          work_tile_info, clc_pipeline, clc_pipe_consumer_state);
      if (incr) ++clc_pipe_consumer_state;

      problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
      auto k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

      // -- mma inlined (source: lines 985-1084) --
      {
        uint32_t skip_wait = k_tile_count <= 0;
        auto barrier_token = mainloop_ab_pipeline.consumer_try_wait(mainloop_ab_pipe_consumer_state, skip_wait);

        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        GPU_TRACE_MARK(gemm_outer);

        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > 0) {
          mainloop_ab_pipeline.consumer_wait(mainloop_ab_pipe_consumer_state);
          GPU_TRACE_MARK(load_ab_ready);
          int read_stage = mainloop_ab_pipe_consumer_state.index();
          auto curr_mainloop_pipe_consumer_state = mainloop_ab_pipe_consumer_state;

          ++mainloop_ab_pipe_consumer_state;
          --k_tile_count;
          skip_wait = k_tile_count <= 0;
          barrier_token = mainloop_ab_pipeline.consumer_try_wait(mainloop_ab_pipe_consumer_state, skip_wait);

          CUTLASS_PRAGMA_UNROLL
          for (int scale_k_iter = 0; scale_k_iter < size<3>(tCrA); ++scale_k_iter) { // 1
            GPU_TRACE_MARK(ready_to_gemm);
            // ! stall point
            accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
            GPU_TRACE_MARK(wait_accumulator_ready);
            auto acc = accumulators(_,_,_,accumulator_pipe_producer_state.index());

            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) { // 4
              cute::gemm(tiled_mma,
                         tCrA(_,_,k_block,scale_k_iter,read_stage),
                         tCrB(_,_,k_block,scale_k_iter,read_stage),
                         acc);
              tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            // wait until all 4 gemms complete
            accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
            
            ++accumulator_pipe_producer_state;
          }
          mainloop_ab_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
        }
      }

      work_tile_info = next_work;
      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
    } while (work_tile_info.is_valid());

    RELEASE_GPU_TRACE;

    cutlass::arch::launch_dependent_grids();
    tmem_allocator.release_allocation_lock();

    // Wait for epilogue to finish consuming accumulators
    accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);

    // No peer CTA in 1SM mode — skip tmem_deallocation_result_barrier

    // Free TMEM
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }

  // ==========================================================
  // WARP 4-7: Epilogue (128 threads)
  // Consumes accumulator via accum() (applies SFA*SFB scaling),
  // then stores D via TMA
  // ==========================================================
  else if (warp_category == WarpCategory::Epilogue) {
    cutlass::arch::warpgroup_reg_alloc<AccumRegisterRequirement>();

    // Throttle: wait for prologue to finish
    epilogue_throttle_barrier.wait(0);

    // Wait for TMEM allocation
    tmem_allocation_result_barrier.arrive_and_wait();
    uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
    accumulators.data() = tmem_base_ptr;

    // -- accum_init inlined (source: lines 746-755) --
    Tensor sSFA_epi = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_SFA.begin()),
        ML_SmemLayoutScaleA{});
    Tensor sSFB_epi = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_SFB.begin()),
        ML_SmemLayoutScaleB{});

    auto warp_idx_in_epi = cutlass::canonical_warp_idx_sync() - 4;
    bool do_tail_store = false;

    // -- store_init inlined (source: lines 600-621, tensormaps_init<false> lines 1372-1435) --
    cute::TmaDescriptor* epi_store_tensormap = nullptr;
    if (warp_idx_in_epi == 0) {
      // ElementC=void → offset_Ddesc=0
      epi_store_tensormap = &params.epilogue.tensormaps[sm_id * CollectiveEpilogue::NumTmaDescriptorsPerSm];
      if (cute::elect_one_sync()) {
        Tensor pD_tensormap = make_tensor(params.epilogue.tma_store_d.get_tma_descriptor(), Int<1>{}, Int<1>{});
        Tensor sD_tensormap = make_tensor(make_smem_ptr(&shared_storage.tensormaps.epilogue.smem_tensormap_D), Int<1>{}, Int<1>{});
        copy(recast<uint128_t>(pD_tensormap), recast<uint128_t>(sD_tensormap));
      }
      __syncwarp();
    }

    bool did_batch_change = true;

    GET_GPU_TRACE(true);

    do {
      int32_t curr_batch = work_tile_info.L_idx;

      if (did_batch_change && warp_idx_in_epi == 0) {
        // -- epilogue tensormaps_perform_update<false> inlined (source: lines 1524-1554) --
        __syncwarp();
        if (cute::elect_one_sync()) {
          cute::tma_descriptor_replace_addr_in_shared_mem(
              shared_storage.tensormaps.epilogue.smem_tensormap_D,
              params.epilogue.ptr_D[curr_batch]);
          // tensormaps_replace_global_tensor_properties<false> inlined (source: lines 1500-1521)
          auto epi_problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);
          const uint32_t epi_M = get<0>(epi_problem_shape_MNKL);
          const uint32_t epi_N = get<1>(epi_problem_shape_MNKL);
          constexpr int MaxTensorRank = 5;
          cute::array<uint32_t, MaxTensorRank> prob_shape_D  = {1,1,1,1,1};
          cute::array<uint64_t, MaxTensorRank> prob_stride_D = {0,0,0,0,0};
          ElementD const* ptr_D_null = nullptr;
          Tensor tensor_d = make_tensor(ptr_D_null, make_layout(make_shape(epi_M,epi_N,Int<1>{}), EPI_InternalStrideD{}));
          if (params.epilogue.dD != nullptr) {
            tensor_d = make_tensor(ptr_D_null, make_layout(make_shape(epi_M,epi_N,Int<1>{}), params.epilogue.dD[curr_batch]));
          } else {
            auto internal_shape_d = make_shape(static_cast<int>(epi_M), static_cast<int>(epi_N), 1);
            EPI_InternalStrideD stride_d = cutlass::make_internal_packed_stride(EPI_InternalStrideD{}, internal_shape_d);
            tensor_d = make_tensor(ptr_D_null, make_layout(make_shape(epi_M,epi_N,Int<1>{}), stride_d));
          }
          cute::detail::fill_tma_gmem_shape_stride(params.epilogue.tma_store_d, tensor_d, prob_shape_D, prob_stride_D);
          for (uint64_t& s : prob_stride_D) { s = (s * cute::sizeof_bits_v<ElementD>) / 8; }
          cute::tma_descriptor_replace_dims_strides_in_shared_mem(
              shared_storage.tensormaps.epilogue.smem_tensormap_D, prob_shape_D, prob_stride_D);
        }
        __syncwarp();
        // tensormaps_cp_fence_release<false> inlined (source: lines 1556-1588)
        if (cute::elect_one_sync()) {
          cute::tma_desc_commit_group();
          cute::tma_desc_wait_group();
        }
        tma_descriptor_cp_fence_release(epi_store_tensormap, shared_storage.tensormaps.epilogue.smem_tensormap_D);
      }

      auto [next_work, incr] = scheduler.fetch_next_work(
          work_tile_info, clc_pipeline, clc_pipe_consumer_state);
      if (incr) ++clc_pipe_consumer_state;

      problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);
      auto k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

      // -- accum inlined (source: lines 1096-1235) --
      // Setup: create epilogue-tiled accumulator views and SF broadcast tensors
      Tensor acc0 = accumulators(_,_,_,_0{});
      Tensor tAcc0 = acc0(make_coord(_,_),_0{},_0{});
      Tensor tAcc0_epi = flat_divide(tAcc0, EpilogueTile{});

      // Append N with stride 0 to SFA (broadcast SFA across N)
      Tensor sSFA_mn = make_tensor(sSFA_epi.data(), make_layout(
        make_shape(get<0>(sSFA_epi.shape()), get<1>(CtaShape_MNK{}), get<1>(sSFA_epi.shape()), get<2>(sSFA_epi.shape())),
        make_stride(get<0>(sSFA_epi.stride()), _0{}, get<1>(sSFA_epi.stride()), get<2>(sSFA_epi.stride()))
      ));
      Tensor sSFA_mn_epi = flat_divide(sSFA_mn, EpilogueTile{});

      // Append M with stride 0 to SFB (broadcast SFB across M)
      Tensor sSFB_mn = make_tensor(sSFB_epi.data(), make_layout(
        make_shape(get<0>(CtaShape_MNK{}), get<0>(sSFB_epi.shape()), get<1>(sSFB_epi.shape()), get<2>(sSFB_epi.shape())),
        make_stride(_0{}, get<0>(sSFB_epi.stride()), get<1>(sSFB_epi.stride()), get<2>(sSFB_epi.stride()))
      ));
      Tensor sSFB_mn_epi = flat_divide(sSFB_mn, EpilogueTile{});

      TiledCopy tiled_t2r = make_tmem_copy(EPI_CopyOpT2R{}, tAcc0_epi(_,_,_0{},_0{}));
      int thread_idx_accum = threadIdx.x % size(tiled_t2r);
      auto thread_t2r_epi = tiled_t2r.get_slice(thread_idx_accum);

      Tensor acc_ident_epi = make_identity_tensor(shape(tAcc0_epi));
      Tensor tTR_rAcc_epi = thread_t2r_epi.partition_D(acc_ident_epi);
      Tensor tTR_sSFA_epi_part = thread_t2r_epi.partition_D(sSFA_mn_epi);
      Tensor tTR_sSFB_epi_part = thread_t2r_epi.partition_D(sSFB_mn_epi);

      Tensor tTR_FullAcc = make_tensor<ElementAccumulator>(shape(tTR_rAcc_epi));
      Tensor tTR_PartAcc = make_tensor<ElementAccumulator>(shape(tTR_rAcc_epi(_,_,_,_0{},_0{})));

      Tensor tTR_rSFA_compact = make_fragment_like<ElementAccumulator>(filter_zeros(tTR_sSFA_epi_part(_,_,_,_,_,_,_0{})));
      Tensor tTR_rSFB_compact = make_fragment_like<ElementAccumulator>(filter_zeros(tTR_sSFB_epi_part(_,_,_,_,_,_,_0{})));

      Layout tTR_rSFA_layout = make_layout(tTR_sSFA_epi_part(_,_,_,_,_,_,_0{}).shape(), tTR_rSFA_compact.stride());
      Layout tTR_rSFB_layout = make_layout(tTR_sSFB_epi_part(_,_,_,_,_,_,_0{}).shape(), tTR_rSFB_compact.stride());

      clear(tTR_FullAcc);

      // GPU_TRACE_MARK(epilogue_outer);
      CUTLASS_PRAGMA_NO_UNROLL
      while (k_tile_count > 0) {

        GPU_TRACE_MARK(wait_sf);
        mainloop_sf_pipeline.consumer_wait(mainloop_sf_pipe_consumer_state);
        int read_idx = mainloop_sf_pipe_consumer_state.index();

        // synchronous copy from smem to register (computation happens in Cuda Core)
        copy(filter_zeros(tTR_sSFA_epi_part(_,_,_,_,_,_,read_idx)), tTR_rSFA_compact);
        copy(filter_zeros(tTR_sSFB_epi_part(_,_,_,_,_,_,read_idx)), tTR_rSFB_compact);

        Tensor tTR_rSFA = make_tensor(tTR_rSFA_compact.data(), tTR_rSFA_layout);
        Tensor tTR_rSFB = make_tensor(tTR_rSFB_compact.data(), tTR_rSFB_layout);

        mainloop_sf_pipeline.consumer_release(mainloop_sf_pipe_consumer_state);
        ++mainloop_sf_pipe_consumer_state;

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < ML_ScaleKsPerTile; ++k_block) { // 1

          accumulator_pipeline.consumer_wait(accumulator_pipe_consumer_state);
          GPU_TRACE_MARK(accumulator_released);
          Tensor acc_k = accumulators(_,_,_,accumulator_pipe_consumer_state.index());
          Tensor tAcc_k = acc_k(make_coord(_,_),_0{},_0{});
          Tensor tAcc_k_epi = flat_divide(tAcc_k, EpilogueTile{});
          Tensor tTR_tAcc = thread_t2r_epi.partition_S(tAcc_k_epi);

          CUTLASS_PRAGMA_UNROLL
          for (int epi_m = 0; epi_m < size<2>(tAcc_k_epi); ++epi_m) {
            CUTLASS_PRAGMA_UNROLL
            for (int epi_n = 0; epi_n < size<3>(tAcc_k_epi); ++epi_n) {

              auto scale_a = tTR_rSFA(_,_,_,epi_m,epi_n,k_block * ML_ScaleGranularityK);
              auto scale_b = tTR_rSFB(_,_,_,epi_m,epi_n,k_block * ML_ScaleGranularityK);

              Tensor full_acc = tTR_FullAcc(_,_,_,epi_m,epi_n);
              copy(tiled_t2r, tTR_tAcc(_,_,_,epi_m,epi_n), tTR_PartAcc);
              cutlass::arch::fence_view_async_tmem_load();

              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(full_acc); ++i) {
                ElementAccumulator scale = scale_a(i) * scale_b(i);
                full_acc(i) += scale * tTR_PartAcc(i);
              }
            }
          }
          cutlass::arch::fence_view_async_tmem_load();
          GPU_TRACE_MARK(accumulator_release);

          accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);
          ++accumulator_pipe_consumer_state;
        }

        --k_tile_count;
      }

      // -- tensormaps_fence_acquire<false> inlined --
      if (did_batch_change && warp_idx_in_epi == 0) {
        cute::tma_descriptor_fence_acquire(epi_store_tensormap);
      }

      // -- store (register-accumulator overload) inlined (source: lines 1049-1342) --
      {
        auto [M, N, K, L] = problem_shape_MNKL;
        auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
        int thread_idx_epi = threadIdx.x % EPI_ThreadCount;
        int warp_idx_epi = thread_idx_epi / cutlass::NumThreadsPerWarp;

        using ElementCompute_ = typename cutlass::epilogue::fusion::FusionCallbacksTraits<EPI_FusionCallbacks>::ElementCompute;
        using ElementCompute = cute::conditional_t<cute::is_void_v<ElementCompute_>,ElementAccumulator,ElementCompute_>;

        auto coord_shape = append<3>(make_shape(m_coord, n_coord), Int<0>{});

        Tensor mD_mn = params.epilogue.tma_store_d.get_tma_tensor(append<3>(make_shape(M,N), Int<1>{}));
        Tensor mD = coalesce(mD_mn, take<0,2>(CtaShape_MNK{}));
        Tensor gD = local_tile(mD, take<0,2>(CtaShape_MNK{}), coord_shape);

        Tensor gD_epi = flat_divide(gD, EpilogueTile{});

        auto ptr_sC = shared_storage.tensors.epilogue.collective.smem_C.begin();
        auto ptr_sD = shared_storage.tensors.epilogue.collective.smem_D.begin();
        Tensor sC_epi = cute::as_position_independent_swizzle_tensor(
                          make_tensor(make_smem_ptr(ptr_sC), EPI_SmemLayoutC{}));
        Tensor sD_epi = cute::as_position_independent_swizzle_tensor(
                          make_tensor(make_smem_ptr(ptr_sD), EPI_SmemLayoutD{}));

        auto thread_t2r_store = tiled_t2r.get_slice(thread_idx_epi);
        Tensor tTR_sD = thread_t2r_store.partition_D(sD_epi(_,_,_0{}));

        Tensor tTR_rD = make_tensor<EPI_SmemElementD>(shape(tTR_sD));
        constexpr int FragmentSize = EPI_FragmentSize;
        Tensor tTR_rD_frg = recast<cutlass::Array<EPI_SmemElementD, FragmentSize>>(coalesce(tTR_rD));

        TiledCopy tiled_s2r = make_tiled_copy_D(Copy_Atom<EPI_CopyOpS2R, EPI_SmemElementC>{}, tiled_t2r);
        auto thread_s2r = tiled_s2r.get_slice(thread_idx_epi);
        Tensor tSR_sC = thread_s2r.partition_S(sC_epi);
        Layout tSR_rC_layout = thread_s2r.retile_D(tTR_rD).layout();

        constexpr bool IsDirectS2R = cute::is_same_v<EPI_CopyOpS2R, AutoVectorizingCopyWithAssumedAlignment<128>>
                                    && decltype(max_common_vector(tSR_rC_layout, tSR_sC.layout()))::value <= 1;
        using RegisterElementC = cute::conditional_t<IsDirectS2R, ElementCompute, EPI_SmemElementC>;
        Tensor tTR_rC = make_tensor<RegisterElementC>(shape(tTR_sD));
        Tensor tSR_rC = thread_s2r.retile_D(tTR_rC);

        TiledCopy tiled_r2s = make_tiled_copy_D(Copy_Atom<EPI_CopyOpR2S, EPI_SmemElementD>{}, tiled_t2r);
        auto thread_r2s = tiled_r2s.get_slice(thread_idx_epi);
        Tensor tRS_rD = thread_r2s.retile_S(tTR_rD);
        Tensor tRS_sD = thread_r2s.partition_D(sD_epi);

        auto thrblk_s2g = params.epilogue.tma_store_d.get_slice(Int<0>{});
        Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);
        Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);

        // OOB predication
        Tensor mD_crd = make_identity_tensor(make_shape(M,N));
        Tensor cD_mn = local_tile(mD_crd, take<0,2>(CtaShape_MNK{}), make_coord(m_coord, n_coord));
        Tensor tTR_cD_mn = thread_t2r_store.partition_D(flat_divide(cD_mn, EpilogueTile{}));
        Tensor cD = make_coord_tensor(cD_mn.layout());
        Tensor tTR_cD = make_coord_tensor(tTR_cD_mn.layout());
        auto residue_cD = make_coord(M,N) - cD_mn(_0{});
        auto residue_tTR_cD = make_coord(M,N) - tTR_cD_mn(_0{});

        // Fusion callbacks
        constexpr bool RefSrc = false;
        auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                          problem_shape_MNKL,
                          CtaShape_MNK{},
                          cta_coord_mnkl,
                          TiledMma{},
                          EpilogueTile{},
                          tiled_t2r,
                          cD,
                          residue_cD,
                          tTR_cD,
                          residue_tTR_cD,
                          tTR_rC,
                          thread_idx_epi
                        };
        auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args);
        bool is_producer_load_needed_store = fusion_callbacks.is_producer_load_needed();
        bool is_C_load_needed = EPI_is_source_supported && fusion_callbacks.is_C_load_needed();

        auto synchronize = [] () CUTLASS_LAMBDA_FUNC_INLINE {
          cutlass::arch::NamedBarrier::sync(EPI_ThreadCount, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        };

        GPU_TRACE_MARK(epilogue_outer_end);
        bool issue_tma_store = warp_idx_epi == 0;

        // tma_store_fn lambda (DelayTmaStore=false, ReuseSmemC=false)
        auto tma_store_fn = [&] (int epi_m, int epi_n) {
          cutlass::arch::fence_view_async_shared();
          synchronize();
          if (issue_tma_store) {
            copy(params.epilogue.tma_store_d.with(epi_store_tensormap),
                bSG_sD(_,_,_,epi_store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
          }
          cst_callbacks.tma_store(epi_m, epi_n, epi_store_pipe_producer_state.count(), issue_tma_store);
          if (issue_tma_store) {
            epi_store_pipeline.producer_commit(epi_store_pipe_producer_state);
          }
          ++epi_store_pipe_producer_state;
          if (issue_tma_store) {
            epi_store_pipeline.producer_acquire(epi_store_pipe_producer_state);
          }
          synchronize();
        };

        // BEGIN EPILOGUE
        auto load_wait_state = epi_load_pipe_consumer_state;
        cutlass::ConsumerToken load_wait_token{cutlass::BarrierStatus::WaitDone};
        if (is_producer_load_needed_store) {
          load_wait_token = epi_load_pipeline.consumer_try_wait(load_wait_state);
        }

        cst_callbacks.begin();
        if (cst_callbacks.begin_sync_needed()) {
          synchronize();
        }

        constexpr int NumEpiSubtilesN = CUTE_STATIC_V(size<3>(decltype(gD_epi){}));
        constexpr int NumEpiSubtilesM = CUTE_STATIC_V(size<2>(decltype(gD_epi){}));
        #pragma unroll(EPI_UnrollEpiLoop ? NumEpiSubtilesN : 1)
        for (int iter_n = 0; iter_n < NumEpiSubtilesN; ++iter_n) {
          #pragma unroll(EPI_UnrollEpiLoop ? NumEpiSubtilesM : 1)
          for (int iter_m = 0; iter_m < NumEpiSubtilesM; ++iter_m) {
            int epi_m = iter_m, epi_n = iter_n;
            bool is_last_iteration = iter_m == NumEpiSubtilesM-1 && iter_n == NumEpiSubtilesN-1;

            cst_callbacks.begin_loop(epi_m, epi_n);

            if (is_producer_load_needed_store) {
              epi_load_pipeline.consumer_wait(load_wait_state, load_wait_token);
              if (is_C_load_needed) {
                copy(tiled_s2r, tSR_sC(_,_,_,load_wait_state.index()), tSR_rC);
              }
            }

            cst_callbacks.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed_store);

            if (is_producer_load_needed_store) {
              cutlass::arch::fence_view_async_shared();
              epi_load_pipeline.consumer_release(epi_load_pipe_consumer_state);
              ++epi_load_pipe_consumer_state;
              ++load_wait_state;
            }

            Tensor tTR_rAcc_epi_tile = tTR_FullAcc(_,_,_,epi_m,epi_n);
            Tensor tTR_rAcc_frg = recast<cutlass::Array<ElementAccumulator, FragmentSize>>(coalesce(tTR_rAcc_epi_tile));

            CUTLASS_PRAGMA_UNROLL
            for (int epi_v = 0; epi_v < size(tTR_rD_frg); ++epi_v) {
              tTR_rD_frg(epi_v) = cst_callbacks.visit(tTR_rAcc_frg(epi_v), epi_v, epi_m, epi_n);
            }

            Tensor reduction_buffer = make_tensor(raw_pointer_cast(sD_epi(_,_,epi_store_pipe_producer_state.index()).data()),
                                                  make_layout(stride<2>(get_nonswizzle_portion(EPI_SmemLayoutD{})), _1{}));
            cst_callbacks.reduce(reduction_buffer, synchronize, epi_m, epi_n, is_last_iteration, tTR_rD_frg);

            copy(tiled_r2s, tRS_rD, tRS_sD(_,_,_,epi_store_pipe_producer_state.index()));

            cst_callbacks.postreduce(epi_m, epi_n, epi_store_pipe_producer_state.count(), true /*issue_smem_store*/);

            tma_store_fn(epi_m, epi_n);

            cst_callbacks.end_loop(epi_m, epi_n);

            if (is_producer_load_needed_store) {
              load_wait_token = epi_load_pipeline.consumer_try_wait(load_wait_state, is_last_iteration);
            }
          }
        }

        cst_callbacks.end();
      }

      do_tail_store |= TileScheduler::compute_epilogue(work_tile_info, params.scheduler);

      GPU_TRACE_MARK(epilogue_outer_end);

      work_tile_info = next_work;
      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      did_batch_change = curr_batch != work_tile_info.L_idx;
    } while (work_tile_info.is_valid());

    RELEASE_GPU_TRACE;

    if (do_tail_store) {
      // store_tail is a no-op for ReuseSmemC=false
    }
  }

  // ==========================================================
  // WARP 3: EpilogueLoad (conditionally active)
  // Loads source C or aux inputs via TMA when fusion needs it
  // ==========================================================
  else if (warp_category == WarpCategory::EpilogueLoad && is_epi_load_needed) {
    cutlass::arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();
    cutlass::arch::wait_on_dependent_grids();

    bool do_load_order_wait = true;
    bool do_tail_load = false;

    // -- load_init inlined (source: lines 469-485, tensormaps_init<true> lines 1407-1417) --
    // is_source_supported=false → skip C TMA descriptor copy, tensormap=nullptr
    cute::TmaDescriptor* epi_load_tensormap = nullptr;

    bool did_batch_change = true;

    epilogue_throttle_barrier.arrive();

    do {
      int32_t curr_batch = work_tile_info.L_idx;
      if (did_batch_change) {
        // -- tensormaps_perform_update<true> inlined (source: lines 1524-1554) --
        // is_source_supported=false → no C addr/dims update, no fence_release for C
        // (entire update is a no-op for our config)
      }
      bool compute_epilogue = TileScheduler::compute_epilogue(work_tile_info, params.scheduler);

      auto [next_work, incr] = scheduler.fetch_next_work(
          work_tile_info, clc_pipeline, clc_pipe_consumer_state);
      work_tile_info = next_work;
      if (incr) ++clc_pipe_consumer_state;

      if (compute_epilogue) {
        if (do_load_order_wait) {
          load_order_barrier.wait();
          do_load_order_wait = false;
        }

        problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);

        // -- load inlined (source: lines 496-589) --
        // is_source_supported=false → no C TMA load, but fusion callbacks still called
        {
          // tensormaps_fence_acquire<true> → is_source_supported=false → no-op

          int lane_idx_load = cutlass::canonical_lane_idx();
          auto [M, N, K, L] = problem_shape_MNKL;
          auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
          auto coord_shape = append<3>(make_shape(m_coord, n_coord), Int<0>{});

          Tensor mC_mn = params.epilogue.tma_load_c.get_tma_tensor(append<3>(make_shape(M,N), Int<1>{}));
          Tensor mC = coalesce(mC_mn, take<0,2>(CtaShape_MNK{}));
          Tensor gC = local_tile(mC, take<0,2>(CtaShape_MNK{}), coord_shape);

          auto ptr_sC = shared_storage.tensors.epilogue.collective.smem_C.begin();
          Tensor gC_epi = flat_divide(gC, EpilogueTile{});
          Tensor sC_epi_load = make_tensor(make_smem_ptr(ptr_sC), EPI_SmemLayoutC{});

          auto thrblk_g2s = params.epilogue.tma_load_c.get_slice(Int<0>{});
          Tensor bGS_gC = thrblk_g2s.partition_S(gC_epi);
          Tensor bGS_sC = thrblk_g2s.partition_D(sC_epi_load);

          auto pld_args = cutlass::epilogue::fusion::detail::ProducerLoadArgs{
                            problem_shape_MNKL,
                            CtaShape_MNK{},
                            cta_coord_mnkl,
                            TiledMma{},
                            EpilogueTile{},
                            lane_idx_load
                          };
          auto pld_callbacks = fusion_callbacks.get_producer_load_callbacks(pld_args);
          bool is_C_load_needed = EPI_is_source_supported && fusion_callbacks.is_C_load_needed();

          bool issue_tma_load = cute::elect_one_sync();

          pld_callbacks.begin();

          CUTLASS_PRAGMA_UNROLL
          for (int iter_n = 0; iter_n < size<3>(gC_epi); ++iter_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int iter_m = 0; iter_m < size<2>(gC_epi); ++iter_m) {
              int epi_m = iter_m, epi_n = iter_n;

              constexpr uint16_t mcast_mask = 0;
              uint64_t* tma_barrier = epi_load_pipeline.producer_get_barrier(epi_load_pipe_producer_state);
              epi_load_pipeline.producer_acquire(epi_load_pipe_producer_state);

              if (issue_tma_load && is_C_load_needed) {
                copy(params.epilogue.tma_load_c.with(epi_load_tensormap, *tma_barrier, mcast_mask),
                    bGS_gC(_,_,_,epi_m,epi_n), bGS_sC(_,_,_,epi_load_pipe_producer_state.index()));
                epi_load_pipeline.producer_expect_transaction(epi_load_pipe_producer_state);
              }

              pld_callbacks.step(tma_barrier, epi_m, epi_n, epi_load_pipe_producer_state.count(), issue_tma_load);

              epi_load_pipeline.producer_commit(epi_load_pipe_producer_state);
              ++epi_load_pipe_producer_state;
            }
          }

          pld_callbacks.end();
        }
        do_tail_load = true;
      }

      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      did_batch_change = curr_batch != work_tile_info.L_idx;
    } while (work_tile_info.is_valid());

    if (do_tail_load) {
      // -- load_tail inlined --
      epi_load_pipeline.producer_tail(epi_load_pipe_producer_state);
    }
  }

  // ==========================================================
  // WARP 3 (when EpiLoad not needed), WARP 9-11 (Unused): just dealloc
  // ==========================================================
  else {
    cutlass::arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();
  }
}
}; // struct FlatGemmKernel

__global__ void __launch_bounds__(MaxThreadsPerBlock, 1)
grouped_fp8_gemm_kernel(CUTLASS_GRID_CONSTANT KernelParams const params) {
  extern __shared__ char smem_buf[];
  FlatGemmKernel<> op;
  op(params, smem_buf);
}


// ============================================================
// Section 4: Host-side functions
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

  problem_sizes[i] = ProblemShape::UnderlyingProblemShape(actual_m, n, k);
  stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {stride_m, k, 1});
  stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  stride_D[i] = cutlass::make_cute_packed_stride(StrideD{}, {stride_m, n, 1});

  A_ptr[i] = A + int64_t(m_offset) * int64_t(k);
  B_ptr[i] = B + int64_t(i) * int64_t(n) * int64_t(k);
  D_ptr[i] = D + int64_t(m_offset) * int64_t(n);

  layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(stride_m, n, k, 1));
  SFA_ptr[i] = SFA + int64_t(sf_m_offset) * int64_t(sf_k);
  layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(stride_m, n, k, 1));
  SFB_ptr[i] = SFB + int64_t(i) * int64_t(sf_n) * int64_t(sf_k);
}

// ============================================================
// Section 5: GemmState and launch wrappers
// ============================================================

struct GemmState {
  Gemm gemm;
  typename Gemm::Arguments arguments;
  void* workspace_ptr;
};

GemmState prepare_grouped_gemm(
    void* int_buffer, size_t int_buffer_size_in_bytes,
    void* float_buffer, size_t float_buffer_size_in_bytes,
    ElementA* A, ElementA* B, float* SFA, float* SFB,
    ElementD* D, int* m_indptr, const int* masked_m,
    int max_m, int n, int k, int num_groups, cudaStream_t stream) {

  AlignedAllocator allocator(int_buffer, int_buffer_size_in_bytes);
  auto problem_sizes = allocator.aligned_alloc<ProblemShape::UnderlyingProblemShape>(
      num_groups * sizeof(ProblemShape::UnderlyingProblemShape), 16, "problem_sizes");
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

  CUDA_CHECK(cudaLaunchKernelEx(
      &config, compute_sm100_cutlass_group_gemm_args, A, B, SFA, SFB, D,
      m_indptr, masked_m, max_m, n, k, num_groups, ScaleGranularityM,
      ScaleGranularityN, ScaleGranularityK, problem_sizes, A_ptr, B_ptr,
      SFA_ptr, SFB_ptr, D_ptr, stride_A, stride_B, stride_D, layout_SFA,
      layout_SFB));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int const sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;

  GemmState state;
  state.arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_groups, problem_sizes, nullptr},
      {A_ptr, stride_A, B_ptr, stride_B, SFA_ptr, layout_SFA, SFB_ptr, layout_SFB},
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
  // Use the original template-based launch for now.
  // This ensures correct grid dimensions, shared memory size, and cluster config.
  CUTLASS_CHECK(state.gemm.run(stream, nullptr, true));
}

// Alternatively, launch the flattened kernel directly:
void run_flat_gemm_kernel(GemmState& state, cudaStream_t stream) {
  // Use the stored params from the adapter (set by initialize())
  // rather than re-computing via to_underlying_arguments()
  auto const& params = state.gemm.params();
  auto grid = GemmKernel::get_grid_shape(params);
  auto block = GemmKernel::get_block_shape();

  int smem_size = SharedStorageSize;
  CUDA_CHECK(cudaFuncSetAttribute(
      grouped_fp8_gemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cudaLaunchConfig_t config;
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;

  CUDA_CHECK(cudaLaunchKernelEx(&config, grouped_fp8_gemm_kernel, params));
}

// ============================================================
// Section 6: Test harness
// ============================================================

int main(int argc, const char** argv) {
  cutlass::CommandLine cmd(argc, argv);

  int num_groups = 4;
  int m_per_group = 256;
  int n = 2048;
  int k = 512;
  int iterations = 1;
  bool use_flat = false;
  bool verify = false;

  cmd.get_cmd_line_argument("num_groups", num_groups);
  cmd.get_cmd_line_argument("m_per_group", m_per_group);
  cmd.get_cmd_line_argument("n", n);
  cmd.get_cmd_line_argument("k", k);
  cmd.get_cmd_line_argument("iterations", iterations);
  cmd.get_cmd_line_argument("flat", use_flat);
  cmd.get_cmd_line_argument("verify", verify);

  int total_m = num_groups * m_per_group;
  int sf_m_total = total_m / ScaleGranularityM;
  int sf_n = n / ScaleGranularityN;
  int sf_k = k / ScaleGranularityK;

  std::cout << "=== FP8 Grouped GEMM (SM100) — Flattened ===" << std::endl;
  std::cout << "num_groups: " << num_groups << std::endl;
  std::cout << "m_per_group: " << m_per_group << std::endl;
  std::cout << "n: " << n << std::endl;
  std::cout << "k: " << k << std::endl;
  std::cout << "total_m: " << total_m << std::endl;
  std::cout << "iterations: " << iterations << std::endl;
  std::cout << "kernel: " << (use_flat ? "flat" : "template") << std::endl;
  std::cout << "AccumulatorPipelineStageCount: " << AccumulatorPipelineStageCount << std::endl;

  // Build m_indptr
  std::vector<int> h_m_indptr(num_groups + 1);
  for (int i = 0; i <= num_groups; ++i) {
    h_m_indptr[i] = i * m_per_group;
  }

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

  void *d_A, *d_B, *d_D, *d_SFA, *d_SFB;
  int* d_m_indptr;

  CUDA_CHECK(cudaMalloc(&d_A, A_bytes));
  CUDA_CHECK(cudaMalloc(&d_B, B_bytes));
  CUDA_CHECK(cudaMalloc(&d_D, D_bytes));
  CUDA_CHECK(cudaMalloc(&d_SFA, SFA_bytes));
  CUDA_CHECK(cudaMalloc(&d_SFB, SFB_bytes));
  CUDA_CHECK(cudaMalloc(&d_m_indptr, (num_groups + 1) * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), B_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, D_bytes));
  CUDA_CHECK(cudaMemcpy(d_SFA, h_SFA.data(), SFA_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_SFB, h_SFB.data(), SFB_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_m_indptr, h_m_indptr.data(),
                         (num_groups + 1) * sizeof(int), cudaMemcpyHostToDevice));

  size_t int_buf_size = 1 << 20;
  size_t float_buf_size = 64 << 20;
  void *d_int_buf, *d_float_buf;
  CUDA_CHECK(cudaMalloc(&d_int_buf, int_buf_size));
  CUDA_CHECK(cudaMalloc(&d_float_buf, float_buf_size));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto state = prepare_grouped_gemm(
      d_int_buf, int_buf_size, d_float_buf, float_buf_size,
      (ElementA*)d_A, (ElementA*)d_B, (float*)d_SFA,
      (float*)d_SFB, (ElementD*)d_D, d_m_indptr,
      nullptr, m_per_group, n, k, num_groups, stream);

  GPUTraceParam gpu_trace_param;
  gpu_trace::setup_from_env(gpu_trace_param, stream);

  // Timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) {
    if (use_flat) {
      run_flat_gemm_kernel(state, stream);
    } else {
      run_gemm_kernel(state, stream);
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  gpu_trace::teardown(gpu_trace_param, stream);

  float total_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  float avg_ms = total_ms / iterations;
  double total_flops = 2.0 * num_groups * m_per_group * n * k;
  double tflops = total_flops / (avg_ms * 1e-3) / 1e12;

  std::cout << "\n=== Results ===" << std::endl;
  std::cout << "Avg latency: " << avg_ms << " ms" << std::endl;
  std::cout << "Throughput:  " << tflops << " TFLOPS" << std::endl;

  if (verify) {
    size_t D_elements = (size_t)total_m * n;

    void* d_D_ref;
    CUDA_CHECK(cudaMalloc(&d_D_ref, D_bytes));

    // Run template kernel → d_D_ref
    CUDA_CHECK(cudaMemset(d_D_ref, 0, D_bytes));
    {
      void *d_int_buf2, *d_float_buf2;
      CUDA_CHECK(cudaMalloc(&d_int_buf2, int_buf_size));
      CUDA_CHECK(cudaMalloc(&d_float_buf2, float_buf_size));
      auto state_ref = prepare_grouped_gemm(
          d_int_buf2, int_buf_size, d_float_buf2, float_buf_size,
          (ElementA*)d_A, (ElementA*)d_B, (float*)d_SFA,
          (float*)d_SFB, (ElementD*)d_D_ref, d_m_indptr,
          nullptr, m_per_group, n, k, num_groups, stream);
      run_gemm_kernel(state_ref, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(d_int_buf2));
      CUDA_CHECK(cudaFree(d_float_buf2));
    }

    // Run flat kernel → d_D
    CUDA_CHECK(cudaMemset(d_D, 0, D_bytes));
    {
      void *d_int_buf2, *d_float_buf2;
      CUDA_CHECK(cudaMalloc(&d_int_buf2, int_buf_size));
      CUDA_CHECK(cudaMalloc(&d_float_buf2, float_buf_size));
      auto state_flat = prepare_grouped_gemm(
          d_int_buf2, int_buf_size, d_float_buf2, float_buf_size,
          (ElementA*)d_A, (ElementA*)d_B, (float*)d_SFA,
          (float*)d_SFB, (ElementD*)d_D, d_m_indptr,
          nullptr, m_per_group, n, k, num_groups, stream);
      run_flat_gemm_kernel(state_flat, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(d_int_buf2));
      CUDA_CHECK(cudaFree(d_float_buf2));
    }

    // Compare
    std::vector<nv_bfloat16> h_D_ref(D_elements), h_D_flat(D_elements);
    CUDA_CHECK(cudaMemcpy(h_D_ref.data(), d_D_ref, D_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D_flat.data(), d_D, D_bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    float max_abs_err = 0.0f;
    for (size_t i = 0; i < D_elements; ++i) {
      float ref_val = __bfloat162float(h_D_ref[i]);
      float flat_val = __bfloat162float(h_D_flat[i]);
      float err = std::abs(ref_val - flat_val);
      if (err > max_abs_err) max_abs_err = err;
      if (err > 1e-3f) ++mismatches;
    }

    std::cout << "\n=== Verification ===" << std::endl;
    std::cout << "Total elements: " << D_elements << std::endl;
    std::cout << "Max abs error:  " << max_abs_err << std::endl;
    std::cout << "Mismatches (>1e-3): " << mismatches << std::endl;
    std::cout << "PASS: " << (mismatches == 0 ? "YES" : "NO") << std::endl;

    CUDA_CHECK(cudaFree(d_D_ref));
  }

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
