

#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/multidevice_runtime.h>

#include <fstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

MultiGroupFusionBuilder::MultiGroupFusionBuilder() {
  // Create a new instance of fusion graph to hold
  //  the flattened fusion.
  original_fusion_ = std::make_unique<Fusion>();

  // Register owning builder to this fusion.
  original_fusion_->setActiveMultiGroupFusionBuilder(this);
}

void MultiGroupFusionBuilder::newGroup(bool auto_schedule) {
  // Create a new record.
  GroupRecord new_group_record;
  new_group_record.unique_id = running_group_counter_++;
  new_group_record.auto_schedule = auto_schedule;

  // Copy currently avaialbe tenors in the context
  //  into the group's context. This includes all
  //  global inputs and all other groups' outputs
  //  defined so far.
  for (auto& it : context_tensor_map_) {
    // Copy tensorview available globally.
    new_group_record.context_tensors.pushBack(it.first);
  }

  // Save the newly created group record.
  group_creation_records_.push_back(new_group_record);
}

void MultiGroupFusionBuilder::newStmt(IrBuilderPasskey, Statement* stmt) {
  // Only analyze expressions for now
  if (auto expr = dynamic_cast<Expr*>(stmt)) {
    // Get the current group.
    auto& current_group = currentGroup();

    for (auto input_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      // Check that all inputs required by this new expression
      //  are defined under the current group's context.
      //
      // TODO (Shiming):
      //   we should at some point support re-computation, i.e.
      // pull expressions out of other groups and re-materialize them.
      // Not a huge priority for now.
      TORCH_INTERNAL_ASSERT(
          current_group.context_tensors.has(input_tv),
          "tensor input ",
          input_tv->toString(),
          " not in context");

      // If we are pulling inputs from other groups, we need
      //  to mark that as a group input.
      if (!current_group.internal_tensors.has(input_tv)) {
        current_group.group_inputs.pushBack(input_tv);
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
    current_group.exprs.push_back(expr);
  }
}

void MultiGroupFusionBuilder::addGroupOutput(TensorView* tv) {
  auto& group = currentGroup();

  // Check that the given tensor is defined internally
  //  within the group's context.
  TORCH_INTERNAL_ASSERT(
      group.internal_tensors.has(tv), tv->toString(), "not in group");

  // Add the tv to the group outputs.
  group.group_outputs.pushBack(tv);

  // Add the tv to the global context, since
  //  it is a group output.
  context_tensor_map_[tv] = &group;
}

void MultiGroupFusionBuilder::addFusionOutput(TensorView* tv) {
  auto& group = currentGroup();

  TORCH_INTERNAL_ASSERT(
      group.internal_tensors.has(tv),
      "can only add tensors from current group to fusion output.");

  // Register tv as a global ouputput.
  original_fusion_->addOutput(tv);

  // Register tv as a group output.
  group.group_outputs.pushBack(tv);
}

void MultiGroupFusionBuilder::addFusionInput(TensorView* tv) {
  // Register tv as a global input.
  original_fusion_->addInput(tv);

  // Add this tv to the global context.
  context_tensor_map_[tv] = nullptr;
}

// Realize the group record into an actual group
std::unique_ptr<SegmentedGroup> MultiGroupFusionBuilder::buildGroup(
    const GroupRecord& group_record) {
  // Create a new instance of segmented group.
  auto new_group = std::make_unique<SegmentedGroup>(original_fusion_.get());

  // Copy exprs into segmented group
  new_group->exprs_ = group_record.exprs;

  // TODO:
  //  just using input and output vals for now skipping building
  // group connection edges for now.

  // Copy inputs into segmented group
  new_group->input_vals = {
      group_record.group_inputs.vector().begin(),
      group_record.group_inputs.vector().end()};

  // Copy outputs into segmented group
  new_group->output_vals = {
      group_record.group_outputs.vector().begin(),
      group_record.group_outputs.vector().end()};

  return new_group;
}

std::unique_ptr<MultiGroupFusion> MultiGroupFusionBuilder::build() {
  TORCH_INTERNAL_ASSERT(valid_, "builder is one time composition");

  auto multigroup_fusion_ptr = std::make_unique<MultiGroupFusion>();
  auto& multigroup_fusion = *multigroup_fusion_ptr;

  // Build all the groups according to the record entries:
  for (auto& record : group_creation_records_) {
    // Build the segmented group
    multigroup_fusion.groups_.push_back(buildGroup(record));

    // track auto scheduled groups:
    if (record.auto_schedule) {
      multigroup_fusion.auto_scheduled_groups_.insert(
          multigroup_fusion.groups_.back().get());
    }
  }

  // Invalidate this builder within original_fusion_
  original_fusion_->invalidateMultiGroupFusionBuilder();

  // Transfer ownership of original fusion.
  multigroup_fusion.original_fusion_ = std::move(original_fusion_);

  return multigroup_fusion_ptr;
}

inline IValue MultiDeviceRuntime::getIValueFromFusionVal(Val* val) {
  // Try to find value from the dynamic context.
  auto val_it = context_values_.find(val);
  TORCH_INTERNAL_ASSERT(val_it != context_values_.end());
  return val_it->second;
}

std::vector<IValue> MultiDeviceRuntime::getGroupIValueInputs(
    SegmentedGroup* group) {
  std::vector<IValue> group_input;
  std::transform(
      group->input_vals.begin(),
      group->input_vals.end(),
      std::back_inserter(group_input),
      [this](auto input_val) { return getIValueFromFusionVal(input_val); });
  return group_input;
}

std::unique_ptr<Fusion> MultiDeviceRuntime::getFusionCopyFromGroup(
    SegmentedGroup* group) {
  std::unique_ptr<Fusion> fusion_copy = std::make_unique<Fusion>();
  // WAR: copy the complete fusion and then change the inputs and outputs.
  //  to simplify the process of creating a sub-graph of original fusion.
  auto original_to_copy_map =
      Fusion::copy(group->completeFusion(), fusion_copy.get());

  // Remove original inputs
  std::vector<Val*> input_list(
      fusion_copy->inputs().begin(), fusion_copy->inputs().end());
  for (auto inp : input_list) {
    fusion_copy->removeInput(inp);
  }

  // Remove original outputs
  std::vector<Val*> output_list(
      fusion_copy->outputs().begin(), fusion_copy->outputs().end());
  for (auto out : output_list) {
    fusion_copy->removeOutput(out);
  }

  // Add group inputs
  for (auto input : group->input_vals) {
    fusion_copy->addInput(original_to_copy_map.clone(input));
  }

  // Add group outputs
  for (auto output : group->output_vals) {
    fusion_copy->addOutput(original_to_copy_map.clone(output));
  }

  return fusion_copy;
}

namespace {

// Check device of TensorType in all inputs ensure all tensors are on cuda
// devices.
// return common device index (or -1 if device differs).
// TODO:
//  Copy pasted from kernel_cache.cpp, need to unify eventually.
int getCommonDeviceCUDA(const at::ArrayRef<IValue>& inputs) {
  int index = -1;
  for (const auto& input : inputs) {
    if (!input.isTensor()) {
      continue;
    }
    const auto& device = input.toTensor().device();
    // skip cpu scalar tensor as they'll be promoted to scalar later
    if (device.is_cpu() && is_cpu_scalar(input.toTensor())) {
      continue;
    }
    TORCH_CHECK(device.is_cuda(), "nvfuser only supports cuda device");
    auto cur_index = device.index();
    if (index != -1 && index != cur_index) {
      return -1;
    }
    index = (int)cur_index; // NOLINT
  }
  return index;
}

// Update launch parameters if scheduler needs to set the launch params.
void updateLaunchParamsFromScheduler(
    SchedulerEntry* scheduler,
    LaunchParams& lparams) {
  // Set launch parameters form scheduler.
  if (scheduler->hasReductionParam()) {
    lparams = scheduler->reductionParams().lparams;
  } else {
    lparams = scheduler->pointwiseParams().lparams;
  }
}

} // namespace

MultiDeviceRuntime::CompiledKernelPtr MultiDeviceRuntime::compileGroup(
    SegmentedGroup* group,
    std::vector<IValue> group_inputs) {
  // Make a copy of the fusion graph we want to generate
  //  CUDA kernel and compile.
  auto fusion_from_group = getFusionCopyFromGroup(group);

  // Placeholder for auto schedule parameters if any.
  c10::optional<SchedulerEntry*> maybe_scheduler_entry = c10::nullopt;

  // Auto schedule if requested
  if (multi_group_fusion_->shouldAutoSchedule(group)) {
    // Get runtime info from fusion graph and concrete tensor inputs.
    SchedulerRuntimeInfo runtime_info(
        fusion_from_group.get(), group_inputs, true);

    // Get heuristic tag that applies to the given fusion and input info.
    auto heuristic = SchedulerEntry::proposeHeuristics(
        fusion_from_group.get(), runtime_info);
    TORCH_INTERNAL_ASSERT(heuristic.has_value(), "cannot auto schedule fusion");

    // Generate scheduler parameters from tag.
    auto scheduler = SchedulerEntry::makeEntry(
        heuristic.value(), fusion_from_group.get(), runtime_info);

    // Apply schedule to fusion graph.
    scheduler->schedule(fusion_from_group.get());

    maybe_scheduler_entry = scheduler.get();

    // Cache scheduler in registry to retrieve launch parameters.
    auto_scheduler_registry_[group] = std::move(scheduler);
  }

  auto executor_ptr = std::make_unique<FusionExecutor>();

  // Infer which device this fusion runs from input device ids.
  // TODO: fix should bind device with group?
  const int device_index = getCommonDeviceCUDA(group_inputs);
  TORCH_CHECK(device_index >= 0, "device is not coherent for fusion inputs");

  // Set launch parameters
  LaunchParams launch_params;

  // Set compile options
  CompileOptions options;
  options.device = c10::Device(DeviceType::CUDA, device_index);

  // Set parameters inferred by auto scheduler.
  if (maybe_scheduler_entry.has_value()) {
    auto scheduler_entry = maybe_scheduler_entry.value();

    // Set index mode from scheduler.
    options.index_mode = scheduler_entry->indexMode();

    // Set launch parameters with auto scheduler.
    updateLaunchParamsFromScheduler(scheduler_entry, launch_params);
  }

  // Lower the fusion and compile the generated kernel.
  executor_ptr->compileFusion(
      fusion_from_group.get(), group_inputs, launch_params, options);

  return executor_ptr;
}

// TODO:
//  As we build out the logic we probably want to separate
//  multifusion logic from actual runtime.

// Launch kernel and record the kernel output into current context
void MultiDeviceRuntime::runKernel(
    int group_idx,
    std::vector<IValue>& group_input) {
  // Segmented group to run:
  auto group = multi_group_fusion_->fusionGroups().at(group_idx).get();

  // Compiled kernel:
  auto& executor = compiled_kernels_.at(group_idx);

  // Use default launch parameters.
  LaunchParams launch_params;

  // If the kernel was auto-scheduled, we need to
  //  pull the launch parameters from the scheduler.
  auto scheduler_it = auto_scheduler_registry_.find(group);
  if (scheduler_it != auto_scheduler_registry_.end()) {
    updateLaunchParamsFromScheduler(scheduler_it->second.get(), launch_params);
  }

  // Run the kernels and pull the output
  auto outputs = executor->runFusion(
      group_input,
      launch_params,
      // WAR: using constant group id for simplicity, so would not
      //  be able to do dynamic shape yet.
      0);

  // Bind context tensors to the actual kernel outputs:
  int number_of_outputs = group->output_vals.size();

  TORCH_INTERNAL_ASSERT(outputs.size() == number_of_outputs);

  for (auto output_idx : c10::irange(number_of_outputs)) {
    context_values_[group->output_vals.at(output_idx)] = outputs.at(output_idx);
  }
}

std::vector<at::Tensor> MultiDeviceRuntime::runWithInput(
    std::vector<IValue> inputs) {
  // Make sure inputs align at global boundary.
  TORCH_INTERNAL_ASSERT(inputs.size() == globalInputs().size());

  // Make initial context with input values:
  for (auto input_idx : c10::irange(inputs.size())) {
    context_values_[globalInputs().at(input_idx)] = inputs.at(input_idx);
  }

  // Run through the groups to launch kernel
  for (auto group_idx :
       c10::irange(multi_group_fusion_->fusionGroups().size())) {
    auto group = multi_group_fusion_->fusionGroups().at(group_idx).get();

    // Convert group inputs from fusion value to IValue.
    auto group_input = getGroupIValueInputs(group);

    // Run the lowering and compilation step if we haven't compiled yet.
    if (!compiled_) {
      compiled_kernels_.push_back(compileGroup(group, group_input));
    }

    // Launch kernel and record the kernel output into current context
    runKernel(group_idx, group_input);
  }

  // Collect global outputs from context
  std::vector<at::Tensor> outputs;

  std::transform(
      globalOutputs().begin(),
      globalOutputs().end(),
      std::back_inserter(outputs),
      [this](auto output_val) {
        return getIValueFromFusionVal(output_val).toTensor();
      });

  // Clear life time of intermediate tensors.
  context_values_.clear();

  return outputs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
