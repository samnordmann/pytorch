#pragma once
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/evaluator_common.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAFunctions.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Similar to segmented fusion but simplified to be
//!  purely manually controlled.
class TORCH_CUDA_CU_API MultiGroupFusion : NonCopyable {
 public:
  using ProcessRankType = int;

  // Print out the fusion in std::cout
  void print();

  // Returns list of all groups from the fusion.
  const auto& fusionGroups() {
    return groups_;
  }

  // Returns a fusion graph that contains all the expressions
  //  from all the groups, flattened on the same graph.
  Fusion* completeFusion() {
    return original_fusion_.get();
  }

  // Returns true if the given group is specified by user
  //  to be auto scheduled.
  bool shouldAutoSchedule(SegmentedGroup* group) {
    return auto_scheduled_groups_.count(group);
  }

  // Returns the process rank running the group:
  auto getProcessRank(SegmentedGroup* group) {
    return group_to_rank_map_.at(group);
  }

  // Returns the device running the group:
  auto getDeviceFor(SegmentedGroup* group) {
    return group_to_device_map_.at(group);
  }

 private:
  // Note: while not directly using `SegmentedFusion`,
  //  still using the layer of segmented groups so we can
  //  re-use all the auto-scheduling and fusion segment
  //  guarding logic if we need.
  using GroupPtr = std::unique_ptr<SegmentedGroup>;
  friend class MultiGroupFusionBuilder;

  // Segmented groups, each becoming a separate kernel
  //  at compile time.
  std::vector<GroupPtr> groups_;

  // The complete fusion having all the expressions on
  //  the same graph.
  std::unique_ptr<Fusion> original_fusion_ = nullptr;

  // Keeps track of which group to auto schedule
  std::unordered_set<SegmentedGroup*> auto_scheduled_groups_;

  // Temporary parameters below to enable multi-process and
  //  mult-device fusion.
  // This currently seem rather restricted.
  //
  // Should really build out some flexibility along this line.

  // Maps from group to the device where the group will run.
  std::unordered_map<SegmentedGroup*, c10::Device> group_to_device_map_;

  // Maps from group to the process rank that will run this group.
  std::unordered_map<SegmentedGroup*, ProcessRankType> group_to_rank_map_;
};

//! Record object keeping track of the group construction
//!  at compute definition stage.
struct GroupRecord {
  // Unique identifier for the group.
  int unique_id = 0;

  // Tracks if this group is meant to be auto-scheduled.
  bool auto_schedule = false;

  // Tracks which process rank will run this kernel
  MultiGroupFusion::ProcessRankType process_rank = -1;

  // Tracks which device this group will run on.
  c10::Device device =
      c10::Device(DeviceType::CUDA, at::cuda::current_device());

  // Tracks list of expressions that go into this group
  std::vector<Expr*> exprs;

  // Internal states that build up as the user definition
  //  of the fusion graph runs.
  //
  // Note:
  // Goal is to track tensor values only, and all scalar
  //  values should be handled on CPU (eventually).

  // All available tensors within the group's context,
  //  including both group inputs and values produced
  //  within the group.
  VectorOfUniqueEntries<TensorView*> context_tensors;

  // All tensors that were computed within the group.
  VectorOfUniqueEntries<TensorView*> internal_tensors;

  // All inputs to the group.
  VectorOfUniqueEntries<TensorView*> group_inputs;

  // All outputs to the group.
  VectorOfUniqueEntries<TensorView*> group_outputs;
};

//! User interface for building multi-group fusion in
//!  scheduling time.
class TORCH_CUDA_CU_API MultiGroupFusionBuilder : NonCopyable {
 public:
  MultiGroupFusionBuilder();

  // User interface for triggering lowering.
  std::unique_ptr<MultiGroupFusion> build();

  // Query function for the fusion containing all
  //  the expressions.
  auto completeFusion() const {
    return original_fusion_.get();
  }

  // Mark starting point of a new group, i.e kernel
  void newGroup(
      bool auto_schedule = false,
      MultiGroupFusion::ProcessRankType process_rank = -1,
      c10::Device device =
          c10::Device(DeviceType::CUDA, at::cuda::current_device()));

  // Make the given tensor a group output
  void addGroupOutput(TensorView* tv);

  // Make the given tensor a global output
  void addFusionOutput(TensorView* tv);

  // Make the given tensor a global input
  void addFusionInput(TensorView* tv);

  // Interface to call from ir builder which will
  //  notify any creation of new nodes, so that
  //  the group placement can be controlled by
  //  scheduler fully manually.
  void newStmt(IrBuilderPasskey, Statement* stmt);

 private:
  auto& currentGroup() {
    TORCH_INTERNAL_ASSERT(
        !group_creation_records_.empty(), "call newGroup first.");
    return group_creation_records_.back();
  }

  std::unique_ptr<SegmentedGroup> buildGroup(const GroupRecord& group_record);

 private:
  //! Keeps track of user decided group segmentation
  //!  in scheduling time.
  std::vector<GroupRecord> group_creation_records_;

  //! Keeps track of currently available tensorviews to
  //!  avoid re-computation.
  std::unordered_map<TensorView*, GroupRecord*> context_tensor_map_;

  //! Original fusion this builder will be
  //!  spliting. Using an owning pointer to
  //!  avoid redundant copying.
  std::unique_ptr<Fusion> original_fusion_ = nullptr;

  //! Running counter to generate unique group id.
  int running_group_counter_ = 0;

  //! Keep track of validity of the current builder.
  //!  each builder can only produce multi-group fusion once.
  bool valid_ = true;
};

//! Runtime to support running multi-group fusion on
//!  multiple devices.
class TORCH_CUDA_CU_API MultiDeviceRuntime {
  using CompiledKernelPtr = std::unique_ptr<FusionExecutor>;

 public:
  explicit MultiDeviceRuntime(
      std::unique_ptr<MultiGroupFusion> multi_group_fusion,
      MultiGroupFusion::ProcessRankType process_rank = -1)
      : multi_group_fusion_(std::move(multi_group_fusion)),
        process_rank_(process_rank) {
    // Initialize some rank dependency info
    buildValueToRankMap();
  }

  // Run kernels with the given global inputs, compile if needed.
  std::vector<at::Tensor> runWithInput(std::vector<IValue> inputs);

  // Interface to querry underlying fusion.
  auto multiGroupFusion() const {
    return multi_group_fusion_.get();
  }

  // Interface short-cut to querry the flattened fusion
  //  containing all the expressions in all groups.
  auto flattenedFusion() const {
    return multi_group_fusion_->completeFusion();
  }

  void buildValueToRankMap();

 private:
  // Get list of global inputs to the complete fusion.
  const auto& globalInputs() const {
    return multi_group_fusion_->completeFusion()->inputs();
  }

  // Get list of global outputs from the complete fusion.
  const auto& globalOutputs() const {
    return multi_group_fusion_->completeFusion()->outputs();
  }

  // Generate and compile cuda kernel corresponding to
  //  the given segmented group.
  CompiledKernelPtr compileGroup(
      SegmentedGroup* group,
      std::vector<IValue> group_input);

  // Retrieve all inputs corresponding to the given group from
  //  the current context.
  std::vector<IValue> getGroupIValueInputs(SegmentedGroup*);

  // Retrieve computed values from the current context,
  //  throws an error if the given val hasn't been computed.
  inline IValue getIValueFromFusionVal(Val* val);

  // Run the kernel corresponding to the given index, with the given
  //  pytorch tensor inputs.
  void runKernel(int group_idx, std::vector<IValue>& group_inputs);

  // Build actual fusion graph from segmented group.
  std::unique_ptr<Fusion> getFusionCopyFromGroup(SegmentedGroup* group);

 private:
  // Workspace when running multiple kernels, keeps track
  //  of intermediate tensors produced by each kernel.
  std::unordered_map<Val*, IValue> context_values_;

  // Keep track of which rank to receive the value from, if it
  //  is not to be computed from the current rank.
  std::unordered_map<Val*, MultiGroupFusion::ProcessRankType>
      context_source_rank_;

  // Keep track of which rank will use which value to determine where
  //  to send data.
  using RankVector = VectorOfUniqueEntries<MultiGroupFusion::ProcessRankType>;
  std::unordered_map<Val*, RankVector> value_to_user_rank_;

  // Keeps track of if compilation has run.
  bool compiled_ = false;

  // Underlying multigroup fusion.
  std::unique_ptr<MultiGroupFusion> multi_group_fusion_ = nullptr;

  // Compiled kernels from multi_group_fusion_
  std::vector<CompiledKernelPtr> compiled_kernels_;

  // Keeps track of heuristics that are used to schedule
  //  the auto-scheduled kernels.
  std::unordered_map<SegmentedGroup*, std::unique_ptr<SchedulerEntry>>
      auto_scheduler_registry_;

  // Keeps track of process rank owning this runtime,
  //  not sure if this will ever change throughout the runtime's lifetime.
  MultiGroupFusion::ProcessRankType process_rank_ = -1;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
