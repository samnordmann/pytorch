#pragma once
#include <torch/csrc/jit/codegen/cuda/aggregate_dag.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAFunctions.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

using ProcessRankType = int;

class TORCH_CUDA_CU_API Group final : public SegmentedGroup {
public:

  Group(
        MultiGroupFusion* multi_group_fusion,
        bool auto_sch,
        ProcessRankType prank,
        c10::Device dev
      );

  // Tracks if this group is meant to be auto-scheduled.
  bool auto_schedule;

  // Tracks which device this group will run on. Should belong to runtime? Not necessarily bc optimization can depend on it. Same for rank
  c10::Device device;

  //Track the MultiGroupFusion which it belongs to
  MultiGroupFusion* multi_group_fusion_; //TODO: put in private

  // Tracks which process rank will run this kernel
  ProcessRankType process_rank; //Should belong to runtime? I think no bc optimization can depend on it. Same for rank

  // Unique identifier for the group.
  int unique_id;

  // Copy the complete fusion and then change the inputs and outputs.
  // TODO: can probably be optimized this to simplify
  std::unique_ptr<Fusion> makeFusionCopy();

  MultiGroupFusion* getMultiGroupFusion() {
    return multi_group_fusion_;
  }

  void addInput(Val* input){
    TORCH_INTERNAL_ASSERT(!std::count (input_vals.begin(), input_vals.end(), input),
      "added twice the same val as input of the current group");
    input_vals.push_back(input);
  }

  void addOutput(Val* output){
    output_vals.push_back(output);
  }

  // All available tensors within the group's context,
  //  including both group inputs and values produced
  //  within the group.
  VectorOfUniqueEntries<TensorView*> context_tensors; //TODO: not useful. remove, and also maybe remove internal_tensors.

  // All tensors that were computed within the group.
  VectorOfUniqueEntries<TensorView*> internal_tensors;
};


//! User interface for building multi-group fusion
class TORCH_CUDA_CU_API MultiGroupFusion : public Fusion {
 public:
  MultiGroupFusion();

  // Print out the fusion in std::cout
  void print();

  // Returns list of all groups from the fusion.
  const auto& groups() {
    return groups_;
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

  // Interface that is called insided IrBuilder each time a new ir is created
  void newStmt(IrBuilderPasskey, Statement* stmt);

  auto& getCurrentGroup() {
    TORCH_INTERNAL_ASSERT(
        !groups_.empty(), "call newGroup first.");
    return *current_group_;
  }

  void setCurrentGroup(Group* group) {
    current_group_ = group;
  }

  AggregateDag aggregateDag(){
    return aggregate_dag_;
  }

  void buildAggregateDag(){
    aggregate_dag_.build(this);
  }

 private:
  //! Stores groups of the fusion
  std::vector<std::shared_ptr<Group>> groups_;

public:
  //! Keeps track of currently available tensorviews to
  //!  avoid re-computation.
  std::unordered_map<TensorView*, Group*> context_tensor_map_;

  //! Running counter to generate unique group id.
  int running_group_counter_ = 0;

private:
  //! Holds the dag with AggregateVals, AggregateExpr and collective communications
  AggregateDag aggregate_dag_;

  //! Keep track of the current group, which either has been manually set
  //! through setCurrentGroup method or is the latest group created
  Group* current_group_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
