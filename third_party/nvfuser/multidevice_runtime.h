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

//! Runtime for multi_group_fusion.
//! This class inherits from IterVisitor because the runtime executor
//! is ordered by the traversal of the aggregate dag
class TORCH_CUDA_CU_API MultiDeviceRuntime : public IterVisitor {
 public:
  using CompiledKernelPtr = std::unique_ptr<FusionExecutor>;

  MultiDeviceRuntime(
      MultiGroupFusion* multi_group_fusion,
      c10::intrusive_ptr<c10d::ProcessGroup> process_group)
      : IterVisitor(),process_group_(process_group),
      process_rank_((ProcessRankType)process_group->getRank()),
       multi_group_fusion_(multi_group_fusion), a_dag_(multi_group_fusion->aggregateDag()) {}

  // Implement the execution of exprs of the AggregateDag. Called inside "runWithInput"
  void handle(AggregateExpr* aExpr);
  void handle(SendRecv* sr);

  // Run the multidevice fusion with the given global inputs, compile if needed.
  std::vector<at::Tensor> runWithInput(std::vector<IValue> inputs);

  // Check if the current process should run a Group
  bool shouldRun(GroupPtr group){
    return group->process_rank == process_rank_;
  }

 private:
  // Generate and compile cuda kernel corresponding to
  //  the given Group
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

  // the MultiGroupFusion to execute
  MultiGroupFusion* multi_group_fusion_ = nullptr;

  // AggregateDag built from the MultiGroupFusion whose traversal
  // defines the runtime execution.
  AggregateDag* a_dag_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
