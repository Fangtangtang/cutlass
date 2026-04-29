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
 *   /usr/local/cuda/bin/nvcc -std=c++17 \
 *       --generate-code=arch=compute_100a,code=[compute_100a,sm_100a] \
 *       --expt-relaxed-constexpr \
 *       -I.. -I../../include -I../../tools/util/include -I../../examples/common \
 *       -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
 *       grouped_fp8_blockwise_flat.cu -o grouped_fp8_blockwise_flat
 *
 * Run:
 *   ./grouped_fp8_blockwise_flat --num_groups=256 --m_per_group=256 --n=1536 --k=3072'
 Run template kernel (baseline):
  ./grouped_fp8_blockwise_flat --num_groups=256 --m_per_group=256 --n=1536 --k=3072 --iterations=10
      flat kernel:
  ./grouped_fp8_blockwise_flat --num_groups=256 --m_per_group=256 --n=1536 --k=3072 --iterations=10 --flat=1
      verification (compares both kernels element-by-element):
  ./grouped_fp8_blockwise_flat --num_groups=256 --m_per_group=256 --n=1536 --k=3072 --verify=1

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
static constexpr uint32_t AccumulatorPipelineStageCount = GemmKernel::AccumulatorPipelineStageCount;
static constexpr uint32_t SchedulerPipelineStageCount   = GemmKernel::SchedulerPipelineStageCount;
static constexpr int SharedStorageSize = GemmKernel::SharedStorageSize;

// Compile-time verification of resolved constants
static_assert(!GemmKernel::IsSchedDynamicPersistent, "Expected static scheduler for grouped GEMM");
static_assert(cute::size(AtomThrShapeMNK{}) == 1, "Expected 1SM (no MMA peer CTA)");

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

template <typename CollectiveMainloop_ = CollectiveMainloop,
          typename CollectiveEpilogue_ = CollectiveEpilogue>
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

  // --- Collective objects (dependent types for NVCC execution space checking) ---
  CollectiveMainloop_ collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);
  CollectiveEpilogue_ collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

  // Runtime check: does the fusion callback need the EpiLoad warp to produce data?
  bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();

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
  mainloop_ab_pipeline_params.transaction_bytes = CollectiveMainloop_::TmaTransactionBytes;
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
  mainloop_sf_pipeline_params.producer_arv_count = CollectiveMainloop_::NumMainloopSFProducerThreadEvents;
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
  epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue_::TmaTransactionBytes;
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
  auto acc_shape = collective_mainloop.partition_accumulator_shape();
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

  // ==========================================================
  // WARP 2: MainloopABLoad
  // Loads A and B tiles via TMA, handles tensormap updates between groups
  // ==========================================================
  if (warp_category == WarpCategory::MainloopABLoad) {
    cutlass::arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();
    cutlass::arch::wait_on_dependent_grids();

    auto load_inputs = collective_mainloop.load_ab_init(
        problem_shape_MNKL, params.mainloop,
        shared_storage.tensors.mainloop, shared_storage.tensormaps.mainloop,
        params.hw_info.sm_count, sm_id, work_tile_info.L_idx);
    Tensor gA_mkl = get<0>(load_inputs);
    auto input_tensormaps = get<rank(load_inputs) - 1>(load_inputs);
    bool did_batch_change = true;
    bool do_load_order_arrive = is_epi_load_needed;

    epilogue_throttle_barrier.arrive();

    do {
      int32_t curr_batch = idx2crd(work_tile_info.L_idx, shape<4>(gA_mkl));
      problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);

      if (did_batch_change) {
        collective_mainloop.tensormaps_perform_update(
            shared_storage.tensormaps.mainloop, params.mainloop,
            input_tensormaps, problem_shape, curr_batch);
      }

      auto k_tile_iter = scheduler.get_k_tile_iterator(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, shape<3>(gA_mkl));
      auto k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
      auto k_tile_prologue = min(MainloopABPipeline::Stages, k_tile_count);

      auto cta_coord_mnk = append<4>(
          make_coord(get<0>(cta_coord_mnkl), get<1>(cta_coord_mnkl), get<2>(cta_coord_mnkl)),
          Int<0>{});

      // Prologue loads
      auto [ab_state_1, k_iter_1] = collective_mainloop.load_ab(
          params.mainloop, mainloop_ab_pipeline, mainloop_ab_pipe_producer_state,
          load_inputs, cta_coord_mnk, k_tile_iter, k_tile_prologue, did_batch_change);
      mainloop_ab_pipe_producer_state = ab_state_1;

      if (do_load_order_arrive) {
        load_order_barrier.arrive();
        do_load_order_arrive = false;
      }

      // Remaining loads
      auto [ab_state_2, k_iter_2] = collective_mainloop.load_ab(
          params.mainloop, mainloop_ab_pipeline, mainloop_ab_pipe_producer_state,
          load_inputs, cta_coord_mnk, k_iter_1, k_tile_count - k_tile_prologue, false);
      mainloop_ab_pipe_producer_state = ab_state_2;

      __syncwarp();

      auto [next_work, incr] = scheduler.fetch_next_work(
          work_tile_info, clc_pipeline, clc_pipe_consumer_state);
      work_tile_info = next_work;
      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      if (incr) ++clc_pipe_consumer_state;
      did_batch_change = curr_batch != idx2crd(work_tile_info.L_idx, shape<4>(gA_mkl));
    } while (work_tile_info.is_valid());

    collective_mainloop.load_ab_tail(mainloop_ab_pipeline, mainloop_ab_pipe_producer_state);
  }

  // ==========================================================
  // WARP 8: MainloopSFLoad
  // Loads scale factors (SFA/SFB) from gmem to smem
  // ==========================================================
  else if (warp_category == WarpCategory::MainloopSFLoad) {
    cutlass::arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

    int32_t curr_batch = idx2crd(work_tile_info.L_idx, get<3>(problem_shape_MNKL));
    auto mainloop_sf_inputs = collective_mainloop.load_sf_init(
        problem_shape_MNKL, params.mainloop, shared_storage.tensors.mainloop, curr_batch);
    Tensor gA_mkl = get<0>(mainloop_sf_inputs);

    cutlass::arch::wait_on_dependent_grids();

    bool did_batch_change = true;
    do {
      int32_t curr_batch = idx2crd(work_tile_info.L_idx, size<4>(gA_mkl));
      problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);

      if (did_batch_change) {
        mainloop_sf_inputs = collective_mainloop.load_sf_update(
            problem_shape_MNKL, params.mainloop, shared_storage.tensors.mainloop, curr_batch);
      }

      auto k_tile_iter = scheduler.get_k_tile_iterator(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, shape<3>(gA_mkl));
      auto k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

      auto cta_coord_mnk = append<4>(
          make_coord(get<0>(cta_coord_mnkl), get<1>(cta_coord_mnkl), get<2>(cta_coord_mnkl)),
          Int<0>{});

      auto [sf_state, k_iter] = collective_mainloop.load_sf(
          mainloop_sf_pipeline, mainloop_sf_pipe_producer_state,
          mainloop_sf_inputs, cta_coord_mnk, k_tile_iter, k_tile_count);
      mainloop_sf_pipe_producer_state = sf_state;

      __syncwarp();

      auto [next_work, incr] = scheduler.fetch_next_work(
          work_tile_info, clc_pipeline, clc_pipe_consumer_state);
      work_tile_info = next_work;
      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      if (incr) ++clc_pipe_consumer_state;
      did_batch_change = curr_batch != idx2crd(work_tile_info.L_idx, size<4>(gA_mkl));
    } while (work_tile_info.is_valid());

    collective_mainloop.load_sf_tail(mainloop_sf_pipeline, mainloop_sf_pipe_producer_state);
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

    auto mma_inputs = collective_mainloop.mma_init(
        params.mainloop,
        collective_mainloop.slice_accumulator(accumulators, 0),
        shared_storage.tensors.mainloop, tmem_non_accumulator_base);

    epilogue_throttle_barrier.arrive();

    do {
      auto [next_work, incr] = scheduler.fetch_next_work(
          work_tile_info, clc_pipeline, clc_pipe_consumer_state);
      if (incr) ++clc_pipe_consumer_state;

      problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
      auto k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

      // MMA compute (leader CTA only, which is always us in 1SM mode)
      auto [ab_consumer_next, acc_producer_next] = collective_mainloop.mma(
          cute::make_tuple(mainloop_ab_pipeline, accumulator_pipeline),
          cute::make_tuple(mainloop_ab_pipe_consumer_state, accumulator_pipe_producer_state),
          accumulators, mma_inputs, cta_coord_mnkl, k_tile_count);
      mainloop_ab_pipe_consumer_state = ab_consumer_next;
      accumulator_pipe_producer_state = acc_producer_next;

      work_tile_info = next_work;
      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
    } while (work_tile_info.is_valid());

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

    auto accum_inputs = collective_mainloop.accum_init(shared_storage.tensors.mainloop);

    auto warp_idx_in_epi = cutlass::canonical_warp_idx_sync() - 4;
    bool do_tail_store = false;

    auto epi_store_tensormap = get<0>(collective_epilogue.store_init(
        params.epilogue, shared_storage.tensormaps.epilogue,
        params.hw_info.sm_count, sm_id));

    bool did_batch_change = true;

    auto pipelines = cute::make_tuple(accumulator_pipeline, mainloop_sf_pipeline);
    auto states = cute::make_tuple(accumulator_pipe_consumer_state, mainloop_sf_pipe_consumer_state);

    do {
      int32_t curr_batch = work_tile_info.L_idx;

      if (did_batch_change && warp_idx_in_epi == 0) {
        collective_epilogue.template tensormaps_perform_update<false /*IsEpiLoad*/>(
            shared_storage.tensormaps.epilogue, params.epilogue,
            epi_store_tensormap, problem_shape, curr_batch);
      }

      auto [next_work, incr] = scheduler.fetch_next_work(
          work_tile_info, clc_pipeline, clc_pipe_consumer_state);
      if (incr) ++clc_pipe_consumer_state;

      problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);
      auto k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

      // Apply blockwise scale factors: accum reads TMEM partials + smem SFA/SFB,
      // produces register-resident fully-scaled accumulators
      auto [accum, tiled_t2r, next_state] = collective_mainloop.accum(
          pipelines, states, accumulators, accum_inputs, cta_coord_mnkl,
          typename CollectiveEpilogue_::CopyOpT2R{},
          typename CollectiveEpilogue_::EpilogueTile{},
          k_tile_count);
      states = next_state;

      // Epilogue store: apply fusion (alpha scaling) + TMA store D
      if (did_batch_change && warp_idx_in_epi == 0) {
        collective_epilogue.template tensormaps_fence_acquire<false>(epi_store_tensormap);
      }
      auto [load_state_next, store_state_next] = collective_epilogue.store(
          epi_load_pipeline, epi_load_pipe_consumer_state,
          epi_store_pipeline, epi_store_pipe_producer_state,
          problem_shape_MNKL, CtaShape_MNK{}, cta_coord_mnkl,
          TileShape{}, TiledMma{},
          accum, shared_storage.tensors.epilogue,
          epi_store_tensormap, tiled_t2r);

      do_tail_store |= TileScheduler::compute_epilogue(work_tile_info, params.scheduler);
      epi_load_pipe_consumer_state = load_state_next;
      epi_store_pipe_producer_state = store_state_next;

      work_tile_info = next_work;
      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      did_batch_change = curr_batch != work_tile_info.L_idx;
    } while (work_tile_info.is_valid());

    if (do_tail_store) {
      collective_epilogue.store_tail(
          epi_load_pipeline, epi_load_pipe_consumer_state,
          epi_store_pipeline, epi_store_pipe_producer_state,
          CtaShape_MNK{});
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

    auto epi_load_tensormap = get<0>(collective_epilogue.load_init(
        params.epilogue, shared_storage.tensormaps.epilogue, params.hw_info.sm_count, sm_id));
    bool did_batch_change = true;
    constexpr bool IsEpiLoad = true;

    epilogue_throttle_barrier.arrive();

    do {
      int32_t curr_batch = work_tile_info.L_idx;
      if (did_batch_change) {
        collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
            shared_storage.tensormaps.epilogue, params.epilogue,
            epi_load_tensormap, problem_shape, curr_batch);
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
        epi_load_pipe_producer_state = collective_epilogue.template load<false /*IsOverlappingAccum*/>(
            epi_load_pipeline, epi_load_pipe_producer_state,
            problem_shape_MNKL, CtaShape_MNK{}, cta_coord_mnkl,
            TileShape{}, TiledMma{},
            shared_storage.tensors.epilogue,
            cute::make_tuple(epi_load_tensormap, did_batch_change),
            false /*reverse_epi_n*/);
        do_tail_load = true;
      }

      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      did_batch_change = curr_batch != work_tile_info.L_idx;
    } while (work_tile_info.is_valid());

    if (do_tail_load) {
      collective_epilogue.load_tail(
          epi_load_pipeline, epi_load_pipe_producer_state,
          epi_store_pipeline, epi_store_pipe_producer_state);
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
