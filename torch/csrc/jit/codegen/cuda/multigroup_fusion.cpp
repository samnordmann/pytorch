#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
// #include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/multigroup_fusion.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

Group::Group(MultiGroupFusion* multi_group_fusion, bool auto_sch, 
                    ProcessRankType prank, c10::Device dev)
    : SegmentedGroup((Fusion*)multi_group_fusion), auto_schedule(auto_sch),
      device(dev), multi_group_fusion_(multi_group_fusion), process_rank(prank){

    unique_id = multi_group_fusion->running_group_counter_++;

    // Copy currently availabe tensors in the context
    //  into the group's context. This includes all
    //  global inputs and all other groups' outputs
    //  defined so far.
    for (auto& it : multi_group_fusion->context_tensor_map_) {
      context_tensors.pushBack(it.first);
    }
}

MultiGroupFusion::MultiGroupFusion() {

  // Register owning builder to this fusion.
  setActiveMultiGroupFusionBuilder(this);
}

void MultiGroupFusion::newGroup(bool auto_schedule, ProcessRankType process_rank,
                                                            c10::Device device)
{
  groups_.push_back(std::make_shared<Group>(this, auto_schedule,
                                                    process_rank, device));

  //Set the newly created group as the current group
  setCurrentGroup(groups_.back().get());
}


void MultiGroupFusion::newStmt(IrBuilderPasskey, Statement* stmt)
{
  if (auto expr = dynamic_cast<Expr*>(stmt)) {
    auto& current_group = getCurrentGroup();

    for (auto input_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      // Check that all inputs required by this new expression
      //  are defined under the current group's context.
      TORCH_INTERNAL_ASSERT(
          current_group.context_tensors.has(input_tv),
          "tensor input ",
          input_tv->toString(),
          " not in context");

      // If we are pulling inputs from other groups, we need
      //  to mark that as a group input.
      if (!current_group.internal_tensors.has(input_tv)
          && !std::count (current_group.input_vals.begin(), current_group.input_vals.end(), input_tv)
          ){
        current_group.addInput(input_tv);
      }
    }

    for (auto output_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
      // Track defined values under current context
      current_group.context_tensors.pushBack(output_tv);

      // Track internally defined tensors, i.e. not from context.
      current_group.internal_tensors.pushBack(output_tv);
    }

    // Created this way, the expression list is
    //  guaranteed to be in topological order.
    current_group.exprs_.push_back(expr);
  }
}

void MultiGroupFusion::addGroupOutput(TensorView* tv) {
  auto& group = getCurrentGroup();

  // Check that the given tensor is defined internally
  //  within the group's context.
  TORCH_INTERNAL_ASSERT(
      group.internal_tensors.has(tv), tv->toString(), "not in group");

  // Add the tv to the group outputs.
  group.addOutput(tv);

  // Add the tv to the global context, since
  //  it is a group output.
  context_tensor_map_[tv] = &group;
}

void MultiGroupFusion::addFusionOutput(TensorView* tv) {
  auto& group = getCurrentGroup();

  TORCH_INTERNAL_ASSERT(
      group.internal_tensors.has(tv),
      "can only add tensors from current group to fusion output.");

  // Register tv as a global ouputput.
  addOutput(tv);

  // Register tv as a group output.
  group.addOutput(tv);
}

void MultiGroupFusion::addFusionInput(TensorView* tv) {
  // Register tv as a global input.
  addInput(tv);

  // Add this tv to the global context.
  context_tensor_map_[tv] = nullptr;
}


} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
