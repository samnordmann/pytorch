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

// enum class StatusType {
//   not_ready,
//   in_progress,
//   ready
// };

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

class TORCH_CUDA_CU_API MultiDeviceRuntime : public IterVisitor {
 public:
  using CompiledKernelPtr = std::unique_ptr<FusionExecutor>;

  explicit MultiDeviceRuntime(
      MultiGroupFusion* multi_group_fusion,
      c10::intrusive_ptr<c10d::ProcessGroup> process_group,
      ProcessRankType process_rank = -1)
      : IterVisitor(),process_group_(process_group),
      process_rank_((ProcessRankType)process_group->getRank()),
       multi_group_fusion_(multi_group_fusion), a_dag_(multi_group_fusion->aggregateDag()) {}


  void handle(AggregateExpr* aExpr);
  void handle(SendRecv* sr);

  // Run kernels with the given global inputs, compile if needed.
  std::vector<at::Tensor> runWithInput(std::vector<IValue> inputs);

  bool shouldRun(GroupPtr group){
    return group->process_rank == process_rank_;
  }

 private:
  // Generate and compile cuda kernel corresponding to
  //  the given segmented group.
  CompiledKernelPtr compileGroup(
      GroupPtr group,
      std::vector<IValue> group_input);

  // Retrieve all inputs corresponding to the given group from
  //  the current context.
  std::vector<IValue> getGroupIValueInputs(GroupPtr);

  // Retrieve computed values from the current context,
  //  throws an error if the given val hasn't been computed.
  inline IValue getIValueFromFusionVal(Val* val);

 private:
  // Workspace when running multiple kernels, keeps track
  //  of intermediate tensors produced by each kernel.
  std::unordered_map<Val*, IValue> context_values_;
  // Compiled kernels from multi_group_fusion_
  std::unordered_map<GroupPtr, CompiledKernelPtr> compiled_kernels_;

  // Keeps track of heuristics that are used to schedule
  //  the auto-scheduled kernels.
  std::unordered_map<GroupPtr, std::unique_ptr<SchedulerEntry>>
      auto_scheduler_registry_;

  // Process group. Interface for inter-process collectives
  c10::intrusive_ptr<c10d::ProcessGroup> process_group_;

  // Keeps track of process rank owning this runtime,
  //  not sure if this will ever change throughout the runtime's lifetime.
  ProcessRankType process_rank_;

  MultiGroupFusion* multi_group_fusion_ = nullptr;

  AggregateDag* a_dag_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
