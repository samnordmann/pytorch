#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/multidevice_runtime.h>
#include <torch/csrc/jit/codegen/cuda/multigroup_fusion.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

inline IValue MultiDeviceRuntime::getIValueFromFusionVal(Val* val) {
  // Try to find value from the dynamic context.
  auto val_it = context_values_.find(val);
  TORCH_INTERNAL_ASSERT(val_it != context_values_.end());
  return val_it->second;
}

std::vector<IValue> MultiDeviceRuntime::getGroupIValueInputs(
    GroupPtr group) {
  std::vector<IValue> group_input;
  std::transform(
      group->input_vals.begin(),
      group->input_vals.end(),
      std::back_inserter(group_input),
      [this](auto input_val) { return getIValueFromFusionVal(input_val); });
  return group_input;
}

namespace {
// Update launch parameters if scheduler needs to set the launch params.
void updateLaunchParamsFromScheduler(
    SchedulerEntry* scheduler,
    LaunchParams& lparams) {
  // Set launch parameters form scheduler.
  if (scheduler->params()->isA<ReductionParams>()) {
    lparams = scheduler->reductionParams().lparams;
  } else {
    TORCH_INTERNAL_ASSERT(scheduler->params()->isA<PointwiseParams>());
    lparams = scheduler->pointwiseParams().lparams;
  }
}

} // namespace

void MultiDeviceRuntime::buildValueToRankMap() {
  for (auto group : multi_group_fusion_->groups()) {
    auto group_rank = group->process_rank;

    // Fill the rank which will define the output values
    for (auto output_val : group->output_vals) {
      context_source_rank_[output_val] = group_rank;
    }

    // Fill the rank which will consume the input values
    for (auto input_val : group->input_vals) {
      value_to_user_rank_[input_val].pushBack(group_rank);
    }
  }
}

MultiDeviceRuntime::CompiledKernelPtr MultiDeviceRuntime::compileGroup(
    GroupPtr group,
    std::vector<IValue> group_inputs) {
  // Make a copy of the fusion graph we want to generate
  //  CUDA kernel and compile.
  auto fusion_from_group = group->makeFusionCopy();
  // Placeholder for auto schedule parameters if any.
  c10::optional<SchedulerEntry*> maybe_scheduler_entry = c10::nullopt;

  // Auto schedule if requested
  if (group->auto_schedule) {

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

  auto args = KernelArgumentHolder::createKernelArgumentHolder(group_inputs);
  // Set parameters inferred by auto scheduler.
  if (maybe_scheduler_entry.has_value()) {
    auto scheduler_entry = maybe_scheduler_entry.value();
    args.setIndexMode(scheduler_entry->indexMode());
    // Set launch parameters with auto scheduler.
    updateLaunchParamsFromScheduler(scheduler_entry, launch_params);
  }

  args.setDeviceIndex(device_index);
  // Lower the fusion and compile the generated kernel.
  executor_ptr->compileFusion(fusion_from_group.get(), args, launch_params);

  return executor_ptr;
}


// Launch kernel and record the kernel output into current context
void MultiDeviceRuntime::runKernel(
    GroupPtr group,
    std::vector<IValue>& group_input) {
  // Segmented group to run:

  // Compiled kernel:
  auto& executor = compiled_kernels_[group];



  // Container for the resulting tensors
  std::vector<at::Tensor> outputs;


  if (shouldRun(group)) {
    // Use default launch parameters.
    LaunchParams launch_params;
    // If the kernel was auto-scheduled, we need to
    //  pull the launch parameters from the scheduler.
    auto scheduler_it = auto_scheduler_registry_.find(group);
    if (scheduler_it != auto_scheduler_registry_.end()) {
      updateLaunchParamsFromScheduler(scheduler_it->second.get(), launch_params);
    }
    outputs = executor->runFusion(
        group_input,
        launch_params,
        // WAR: using constant group id for simplicity, so would not
        //  be able to do dynamic shape yet.
        0);
  } else {
    // Allocate space for kernel outputs.
    // TODO only allocate if we are gonna indeed use this Ivalue in a future kernel
    outputs = executor->allocOutputSpace(group_input);
  }

  // Store the outputs or place holders in the context
  // Bind context tensors to the actual kernel outputs:
  int number_of_outputs = group->output_vals.size();
  TORCH_INTERNAL_ASSERT(outputs.size() == number_of_outputs);

  for (auto output_idx : c10::irange(number_of_outputs)) {
  //retrieves the Val that corresponds to this output IValue
    auto output_val = group->output_vals.at(output_idx);

    // Fill tensor data or placeholder to context.
    context_values_[output_val] = outputs.at(output_idx);
  }
}

void MultiDeviceRuntime::handle(SendRecv* sr){
  auto sender_group = sr->in()->getGroup();
  auto receiver_group = sr->out()->getGroup();

  bool is_sender = shouldRun(sender_group);
  bool is_receiver = shouldRun(receiver_group);
  int sender_rank = sender_group->process_rank;
  int receiver_rank = receiver_group->process_rank;
  auto val = sr->in()->getOriginalVal();

  std::vector<at::Tensor> tensor = {context_values_.at(val).toTensor()};
  if (is_sender){
      process_group_->send(tensor, receiver_rank, 0);
  }
  if (is_receiver){
        auto work = process_group_->recv(tensor, sender_rank, 0); // receive the tensor
        while (!work->isCompleted()); // wait for completion
        context_values_[val] = (IValue)(tensor[0]); // store the received tensor
  }
}

void MultiDeviceRuntime::handle(AggregateExpr* aExpr){
  auto group = aExpr->getGroup();
  // Convert group inputs from fusion value to IValue.
  auto group_input = getGroupIValueInputs(group);

  // Run the lowering and compilation step if we haven't compiled yet.
  if (!compiled_) {
    compiled_kernels_[group] = compileGroup(group, group_input);
  }

  // Launch kernel and record the kernel output into current context
  runKernel(group, group_input);
}

std::vector<at::Tensor> MultiDeviceRuntime::runWithInput(
    std::vector<IValue> inputs) {

  // Make sure inputs align at global boundary.
  TORCH_INTERNAL_ASSERT(inputs.size() == multi_group_fusion_->inputs().size());

  // Make initial context with input values:
  for (auto input_idx : c10::irange(inputs.size())) {
    context_values_[multi_group_fusion_->inputs().at(input_idx)] = inputs.at(input_idx);
  }

  // Run through the groups to launch kernel
  traverseTo(a_dag_, a_dag_->outputs());

  // Collect global outputs from context
  std::vector<at::Tensor> outputs;
//TODO: could be written in an auxiliary function as in getGroupIValueInputs
  std::transform(
      multi_group_fusion_->outputs().begin(),
      multi_group_fusion_->outputs().end(),
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
