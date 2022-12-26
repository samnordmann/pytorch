#pragma once
#include <torch/csrc/jit/codegen/cuda/aggregate_dag.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/evaluator_common.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/multigroup_fusion.h>
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

enum class StatusType {
  not_ready,
  in_progress,
  ready
};

// We only need to iterate over the aggregateDag.
// At initialization, we take all the statements of the dags and we check if the rank has to deal with this statment
// This should be implemented in a handle function
// the rank deals with a group if it runs it
// the rank deals with AggregateValue iff it belongs to a group ran by it
// the rank deals with send/recv if its receiver or sender

// for all AggregateStatement that the rank deals with, it initialize an entry in status with key StatusType::not_ready
// then we will only be iterating over the status. Run exits only when all status are "ready"
// We choose for now an eager approach: we do smth as soon as we can.

// AggregateVals are put "ready" when an IValue is ready. So for global input this can be done at init
// For the other it is when the "definition" of the AggregateVal is ready

// An AggregateExpr can be launched iff all its AggregateVals input are "ready"
// When its launched it is marked as in_progress
// When it is "ready" we can tag its output as ready

// A send can be posted iff its input is ready. It can be marked "ready" when send is complete?

// A receive can be posted as soon as the space for receiveing as been allocated.

//! Runtime to support running multi-group fusion on
//!  multiple devices.

class TORCH_CUDA_CU_API MultiDeviceRuntime : public OptOutDispatch {
  using CompiledKernelPtr = std::unique_ptr<FusionExecutor>;

 public:
  explicit MultiDeviceRuntime(
      MultiGroupFusion* multi_group_fusion,
      c10::intrusive_ptr<c10d::ProcessGroup> process_group,
      ProcessRankType process_rank = -1)
      : multi_group_fusion_(multi_group_fusion),
        process_group_(process_group), process_rank_((ProcessRankType)process_group->getRank()),
          a_dag_(multi_group_fusion->aggregateDag()) {
    // Initialize some rank dependency info
    buildValueToRankMap();

    for (auto val: a_dag_.vals()){
      status[val] = StatusType::not_ready;
    }
    for (auto expr: a_dag_.unordered_exprs()){
      status[expr] = StatusType::not_ready;
    }
  }

  // Run kernels with the given global inputs, compile if needed.
  std::vector<at::Tensor> runWithInput(std::vector<IValue> inputs);

  // Interface to querry underlying fusion.
  auto multiGroupFusion() const {
    return multi_group_fusion_;
  }

  void buildValueToRankMap();

  std::unordered_map<Statement*, StatusType> status;

 private:
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
  MultiGroupFusion* multi_group_fusion_ = nullptr;

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

  AggregateDag a_dag_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
