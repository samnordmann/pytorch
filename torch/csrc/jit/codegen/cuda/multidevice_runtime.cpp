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

// TODO:
//  As we build out the logic we probably want to separate
//  multifusion logic from actual runtime.

// Launch kernel and record the kernel output into current context
// TODO: this name should probably be runGroup now, since it doesn't
//  necessarily launch a kernel.
void MultiDeviceRuntime::runKernel(
    GroupPtr group,
    std::vector<IValue>& group_input) {
  // Segmented group to run:

  // Compiled kernel:
  auto& executor = compiled_kernels_[group];


  // Device and rank info from this group
  auto group_rank = group->process_rank;
  auto group_device = group->device;

  // Container for the resulting tensors
  std::vector<at::Tensor> outputs;

  // In the case where either rank is -1, we default to always
  //  run this group.
  bool always_run = group_rank == -1 || process_rank_ == -1;
  bool running_kernel = always_run || group_rank == process_rank_;
  // ========================================================================
  //  Section for receiving data from other rank:
  auto input_n = group_input.size();
  TORCH_INTERNAL_ASSERT(group->input_vals.size() == input_n);

  if (running_kernel) {
    for (auto input_idx : c10::irange(input_n)) {
      auto input_source_rank_it =
          context_source_rank_.find(group->input_vals.at(input_idx));
      if (input_source_rank_it == context_source_rank_.end()) {
        std::unordered_set<Val*> global_inputs(
            multi_group_fusion_->inputs().begin(),
            multi_group_fusion_->inputs().end());

        // Check that if there's no source rank definition then this
        //  value is a global input.
        // TODO:
        // At the very beginning of this run the global inputs could be
        //  on any device so there'd need to be an initial processing
        //  to make sure all the running groups have them available.
        TORCH_INTERNAL_ASSERT(
            global_inputs.count(group->input_vals.at(input_idx)));
        continue;
      }

      auto input_source_rank = input_source_rank_it->second;
      if (input_source_rank != -1 && input_source_rank != process_rank_) {
        // Receive this value here
        std::vector<at::Tensor> tensor_to_receive = {group_input.at(input_idx).toTensor()}; // holder for the received tensor
        auto work = process_group_->recv(tensor_to_receive, input_source_rank, 0); // receive the tensor
        while (!work->isCompleted()); // wait for completion
        group_input.at(input_idx) = (IValue)(tensor_to_receive[0]); // store the received tensor
      }
    }
  }
  // ========================================================================
  //  Section for running the kernel:

  if (running_kernel) {
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
    // For simplicity, just allocate space for all the potential
    //  kernel outputs. Optimization possible but quite open ended
    //  for now.
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

  // ========================================================================
  //  Section for sending data to other rank:

  // Running_kernel would mean that current rank has valid data,
  //  !always_run would mean that some ranks do not have it so we need to
  //  send it.
  if (running_kernel && !always_run) {
    for (auto output_val : group->output_vals) {
      auto destination_it = value_to_user_rank_.find(output_val);
      if (destination_it == value_to_user_rank_.end()) {
        // Not an intermediate value.
        continue;
      }

      std::vector<at::Tensor> tensor_to_send = {context_values_.at(output_val).toTensor()};
      auto& rank_vector = destination_it->second;
      if (rank_vector.has(-1)) {
        // TODO: send to all ranks
      } else {
        for (auto rank : rank_vector.vector()) {
          // Send to rank
          process_group_->send(tensor_to_send, rank, 0);
        }
      }
    }
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
