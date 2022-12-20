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
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupBuilder.hpp>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

using ProcessRankType = int;

class GroupedExpr;

// Maybe we should wrap Group in a class
// SegmentedMultiGroupFusion that inherits from SegmentFusion
class TORCH_CUDA_CU_API Group final : public SegmentedGroup {
public:

  Group(
        MultiGroupFusionBuilder* multi_group_fusion,
        bool auto_sch,
        ProcessRankType prank,
        c10::Device dev
      );

  // Tracks if this group is meant to be auto-scheduled.
  bool auto_schedule;

  // Tracks which device this group will run on.
  c10::Device device;

  //Track the MultiGroupFusion which it belongs to
  MultiGroupFusionBuilder* multi_group_fusion_;

  // Tracks which process rank will run this kernel
  ProcessRankType process_rank;

  // Unique identifier for the group.
  int unique_id;

  MultiGroupFusionBuilder* getMultiGroupFusion() {
    return multi_group_fusion_;
  }

  void addInput(TensorView* input){
    group_inputs.pushBack(input);
    input_vals.push_back(input);
  }

  void addOutput(TensorView* output){
    group_outputs.pushBack(output);
    output_vals.push_back(output);
  }

  // // Tracks list of expressions that go into this group
  // std::vector<Expr*> exprs;

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

// TODO: unify group_inputs with SegmentedGroup::input_vals
  // All inputs to the group.
  VectorOfUniqueEntries<TensorView*> group_inputs;

// TODO: unify group_outputs with SegmentedGroup::output_vals
  // All outputs to the group.
  VectorOfUniqueEntries<TensorView*> group_outputs;

  // GroupedExpr* groupedExpr(){
  //   if (gexpr_){
  //     buildGroupedExpr_();
  //   }
  //   return gexpr_;
  // }
private:
// // Stores the aggregated expr
// // WARN:since assignement implies registering to the MultiDeviceFusion,
// // this can be called only when the inputs/outputs of the group have been set.
  // GroupedExpr* gexpr_ = nullptr; 

  // void buildGroupedExpr_();
};


//! 1) Definition inheriting from Expr.
//!      - Members must be private or protected -- why ?
//!      - Accessor functions for members
//!      - Constructors need to register with the Fusion after inputs/outputs
//!         are defined
//!      - Implementation of bool sameAs(...)
//!  2) dispatch.h/.cpp must be updated to include dispatch of the new Val
//!  3) Default mutator function should be added to mutator.h/.cpp
//!  4) Printing functions should be added to ir_iostream.h/.cpp
//!  5) Lower case convenience functions should be added to arith.h/.cpp (If
//!     user facing)
//!  6) An enum value must be added to ExprType in type.h
//!  7) A string entry must be added in expr_type_string_map
//!  8) Entry added to ir_graphviz .cpp/.h

class TORCH_CUDA_CU_API GroupedExpr final : public Expr {
public:

  GroupedExpr(IrBuilderPasskey, Group* group);

  GroupedExpr(const GroupedExpr* src, IrCloner* ir_cloner);

  Group* getGroup() const{
    return group_;
  }

private:
  Group* group_ = nullptr;
};



class TORCH_CUDA_CU_API SendRecv final : public Expr {
public:

  SendRecv(IrBuilderPasskey, Group* src, Group* dst, TensorView* tv);

  SendRecv(const SendRecv* src, IrCloner* ir_cloner);

private:
  Group* src_;

  Group* dst_;

  TensorView* tv_;
};



//! Similar to segmented fusion but simplified to be
//!  purely manually controlled.
class TORCH_CUDA_CU_API MultiGroupFusion final : public Fusion {

   public:

  // Print out the fusion in std::cout
  void print();

  // Returns list of all groups from the fusion.
  const auto& fusionGroups() {
    return groups_;
  }

  // Returns a fusion graph that contains all the expressions
  //  from all the groups, flattened on the same graph.
  MultiGroupFusionBuilder* completeFusion() {
    return original_fusion_;
  }

 private:
  friend class MultiGroupFusionBuilder;
  // Segmented groups, each becoming a separate kernel
  //  at compile time.
  std::vector<std::unique_ptr<Group>> groups_;
  // std::vector<Group> groups_;

  // The complete fusion having all the expressions on
  //  the same graph.
  MultiGroupFusionBuilder* original_fusion_ = nullptr;
  // std::unique_ptr<Fusion> original_fusion_ = nullptr;
};

//! User interface for building multi-group fusion in
//!  scheduling time.
class TORCH_CUDA_CU_API MultiGroupFusionBuilder : public Fusion {
 public:
  MultiGroupFusionBuilder();

  // Print out the fusion in std::cout
  void print();

  // User interface for triggering lowering.
  std::unique_ptr<MultiGroupFusion> build();

  // Returns list of all groups from the fusion.
  const auto& fusionGroups() {
    return groups_;
  }

  MultiGroupFusionBuilder* completeFusion() {
    return this;
  }


  // Mark starting point of a new group, i.e kernel
  void newGroup(
      bool auto_schedule = false,
      ProcessRankType process_rank = -1,
      c10::Device device =
          c10::Device(DeviceType::CUDA, at::cuda::current_device()));

  // Make the given tensor a group output
  void addGroupOutput(TensorView* tv);

  // Make the given tensor a global output
  void addFusionOutput(TensorView* tv);

  // Make the given tensor a global input
  void addFusionInput(TensorView* tv);

  // bool shouldAutoSchedule(SegmentedGroup* group) {
  //   return group->shouldAutoSchedule();
  // }

  // Interface to call from ir builder which will
  //  notify any creation of new nodes, so that
  //  the group placement can be controlled by
  //  scheduler fully manually.
  void newStmt(IrBuilderPasskey, Statement* stmt);

  auto& getCurrentGroup() {
    TORCH_INTERNAL_ASSERT(
        !groups_.empty(), "call newGroup first.");
    return *current_group_;
  }

  void setCurrentGroup(Group* group) {
    current_group_ = group;
  }

 private:
  // std::unique_ptr<SegmentedGroup> buildGroup(const Group& group_record);

  //! Keeps track of user decided group segmentation
  //!  in scheduling time.
  using GroupPtr = std::unique_ptr<Group>;
  std::vector<GroupPtr> groups_;

public:
  //! Keeps track of currently available tensorviews to
  //!  avoid re-computation.
  std::unordered_map<TensorView*, Group*> context_tensor_map_;

  //! Running counter to generate unique group id.
  int running_group_counter_ = 0;

private:
  //! Keep track of validity of the current builder.
  //!  each builder can only produce multi-group fusion once.
  bool valid_ = true;

  //! Keep track of the current group, which either has been manually set
  //! through setCurrentGroup method or is the latest group created
  Group* current_group_;
};

//! Runtime to support running multi-group fusion on
//!  multiple devices.
class TORCH_CUDA_CU_API MultiDeviceRuntime {
  using CompiledKernelPtr = std::unique_ptr<FusionExecutor>;

 public:
  explicit MultiDeviceRuntime(
      MultiGroupFusionBuilder* multi_group_fusion,
      c10::intrusive_ptr<c10d::ProcessGroup> process_group,
      ProcessRankType process_rank = -1)
      : multi_group_fusion_(multi_group_fusion),
        process_group_(process_group), process_rank_(process_rank) {
    // Initialize some rank dependency info
     std::cout << "buildValueToRankMap on rank " << process_rank << std::endl;
    buildValueToRankMap();
  }

  // Run kernels with the given global inputs, compile if needed.
  std::vector<at::Tensor> runWithInput(std::vector<IValue> inputs);

  // Interface to querry underlying fusion.
  auto multiGroupFusion() const {
    return multi_group_fusion_;
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
      Group* group,
      std::vector<IValue> group_input);

  // Retrieve all inputs corresponding to the given group from
  //  the current context.
  std::vector<IValue> getGroupIValueInputs(Group*);

  // Retrieve computed values from the current context,
  //  throws an error if the given val hasn't been computed.
  inline IValue getIValueFromFusionVal(Val* val);

  // Run the kernel corresponding to the given index, with the given
  //  pytorch tensor inputs.
  void runKernel(int group_idx, std::vector<IValue>& group_inputs);

  // Build actual fusion graph from segmented group.
  std::unique_ptr<Fusion> getFusionCopyFromGroup(Group* group);

 private:
  // Workspace when running multiple kernels, keeps track
  //  of intermediate tensors produced by each kernel.
  std::unordered_map<Val*, IValue> context_values_;

  // Keep track of which rank to receive the value from, if it
  //  is not to be computed from the current rank.
  std::unordered_map<Val*, ProcessRankType>
      context_source_rank_;

  // Keep track of which rank will use which value to determine where
  //  to send data.
  using RankVector = VectorOfUniqueEntries<ProcessRankType>;
  std::unordered_map<Val*, RankVector> value_to_user_rank_;

  // Keeps track of if compilation has run.
  bool compiled_ = false;

  // Underlying multigroup fusion.
  MultiGroupFusionBuilder* multi_group_fusion_ = nullptr;

  // Compiled kernels from multi_group_fusion_
  std::vector<CompiledKernelPtr> compiled_kernels_;

  // Keeps track of heuristics that are used to schedule
  //  the auto-scheduled kernels.
  std::unordered_map<Group*, std::unique_ptr<SchedulerEntry>>
      auto_scheduler_registry_;

  // Process group. Interface for inter-process collectives
  c10::intrusive_ptr<c10d::ProcessGroup> process_group_;

  // Keeps track of process rank owning this runtime,
  //  not sure if this will ever change throughout the runtime's lifetime.
  ProcessRankType process_rank_ = -1;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
