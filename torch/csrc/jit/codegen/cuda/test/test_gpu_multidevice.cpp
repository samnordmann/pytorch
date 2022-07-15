#if defined(USE_CUDA)
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/mma_type.h>
#include <torch/csrc/jit/codegen/cuda/multidevice_runtime.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

// fuser and IR parser
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>

#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupBuilder.hpp>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

namespace {

// Make a tensor that is known to be fully contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeContigTensor(size_t ndims, DataType dtype = DataType::Float) {
  return TensorViewBuilder()
      .ndims(ndims)
      .dtype(dtype)
      .contiguity(std::vector<bool>(ndims, true))
      .build();
}

// Make a tensor that is known to be non-contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeSymbolicTensor(size_t ndims, DataType dtype = DataType::Float) {
  return TensorViewBuilder().ndims(ndims).dtype(dtype).build();
}

// Make a non-contiguous tensor of compile-time known sizes
TensorView* makeConcreteTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float) {
  return TensorViewBuilder().shape(shape).dtype(dtype).build();
}

void checkIntValue(
    ExpressionEvaluator& evaluator,
    Val* val,
    Int::ScalarType expected_value) {
  TORCH_CHECK(val->isAnInt());
  const auto actual_value = evaluator.evaluate(val);
  TORCH_CHECK(actual_value.has_value());
  TORCH_CHECK(actual_value.value() == expected_value);
}

void checkIntValue(
    kir::ExpressionEvaluator& evaluator,
    const Val* val,
    Int::ScalarType expected_value) {
  const auto actual_value = evaluator.evaluate(val);
  TORCH_CHECK(actual_value.has_value());
  TORCH_CHECK(actual_value.value() == expected_value);
}

bool cudaArchGuardShouldSkip(int required_major, int required_minor) {
  int capability_major = at::cuda::getCurrentDeviceProperties()->major;
  int capability_minor = at::cuda::getCurrentDeviceProperties()->minor;

  if (capability_major < required_major ||
      (capability_major == required_major &&
       capability_minor < required_minor)) {
    return true;
  }
  return false;
}

#define NVFUSER_TEST_CUDA_ARCH_GUARD(REQUIRED_MAJOR, REQUIRED_MINOR)          \
  if (cudaArchGuardShouldSkip(REQUIRED_MAJOR, REQUIRED_MINOR)) {              \
    GTEST_SKIP() << "Requires GPU capability above " << REQUIRED_MAJOR << "." \
                 << REQUIRED_MINOR << " to run.\n";                           \
  }

#define NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(                                \
    REQUIRED_MAJOR, REQUIRED_MINOR, COMPILE_FUSION)                          \
  if (cudaArchGuardShouldSkip(REQUIRED_MAJOR, REQUIRED_MINOR)) {             \
    ASSERT_ANY_THROW(COMPILE_FUSION);                                        \
    GTEST_SKIP() << "(Lowered Only) Requires GPU capability above "          \
                 << REQUIRED_MAJOR << "." << REQUIRED_MINOR << " to run.\n"; \
  } else {                                                                   \
    COMPILE_FUSION;                                                          \
  }

} // namespace

// To see the two kernels generated:
// PYTORCH_NVFUSER_DUMP=cuda_kernel ./test_jit --gtest_filter=*DoubleReduction*

// To see what ops are on each kernel:
// PYTORCH_NVFUSER_DUMP=segmented_fusion ./test_jit
// --gtest_filter=*DoubleReduction*

TEST_F(NVFuserTest, FusionDoubleReduction_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(3);

  fusion.addInput(tv0);

  TensorView* tv1 = sum(tv0, {0});
  TensorView* tv2 = sum(tv1, {0});

  fusion.addOutput(tv2);

  FusionExecutorCache fec(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({8, 8, 8}, options);
  auto cg_outputs = fec.runFusionWithInputs({input});

  auto ref = input.sum(0).sum(0);

  testValidate(&fusion, cg_outputs, {input}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionMutiGroupDoubleReduction_CUDA) {
  // Using the new interface to build multi-group fusion
  MultiGroupFusionBuilder fusion_builder;

  // Fusion guard is on the fusion managed within builder.
  FusionGuard fg(fusion_builder.completeFusion());

  TensorView* tv0 = makeContigTensor(3);

  fusion_builder.addFusionInput(tv0);

  // Each expression has to belong to some group,
  //  and each group will become one cuda kernel
  //  after lowering time.

  // Create the first group.
  //  The builder now points to the first created group,
  // all operations following this line will make changes
  // to the first group.
  fusion_builder.newGroup(
      // auto-schedule
      true);

  TensorView* tv1 = sum(tv0, {0});

  fusion_builder.addGroupOutput(tv1);

  // Create the second group.
  //  The builder now points to the second created group,
  // all operations following this line will make changes
  // to the second group.
  fusion_builder.newGroup(
      // auto-schedule
      true);

  TensorView* tv2 = sum(tv1, {0});

  fusion_builder.addFusionOutput(tv2);

  // Build actual fusion graphs and pass it to a
  //  multi-device runtime.
  MultiDeviceRuntime runtime(fusion_builder.build());

  // Create at input tensors.
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({8, 8, 8}, options);

  // See group partitions:
  runtime.multiGroupFusion()->print();

  // Run the multiple kernels created.
  // To see the two kernels generated:
  // PYTORCH_NVFUSER_DUMP=cuda_kernel ./test_jit
  // --gtest_filter=*GroupDoubleReduction*
  auto cg_outputs = runtime.runWithInput({input});

  // Validate result
  auto ref = input.sum(0).sum(0);
  testValidate(
      runtime.flattenedFusion(),
      cg_outputs,
      {input},
      {ref},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionMultiRankReduction_CUDA) {
  // Using the new interface to build multi-group fusion
  MultiGroupFusionBuilder fusion_builder;

  // Fusion guard is on the fusion managed within builder.
  FusionGuard fg(fusion_builder.completeFusion());

  TensorView* tv0 = makeContigTensor(3);

  fusion_builder.addFusionInput(tv0);

  // Each expression has to belong to some group,
  //  and each group will become one cuda kernel
  //  after lowering time.

  // Create the first group.
  //  The builder now points to the first created group,
  // all operations following this line will make changes
  // to the first group.
  fusion_builder.newGroup(
      // auto-schedule
      true,
      // Process rank that runs this group:
      // -1 means all group runs.
      -1,
      // Cuda device that runs this group:
      c10::Device(DeviceType::CUDA, at::cuda::current_device()));

  TensorView* tv1 = sum(tv0, {0});

  fusion_builder.addGroupOutput(tv1);

  // Create the second group.
  //  The builder now points to the second created group,
  // all operations following this line will make changes
  // to the second group.
  fusion_builder.newGroup(
      // auto-schedule
      true,
      // Process rank that runs this group:
      // -1 means all group runs.
      -1,
      // Cuda device that runs this group:
      c10::Device(DeviceType::CUDA, at::cuda::current_device()));

  TensorView* tv2 = sum(tv1, {0});

  fusion_builder.addFusionOutput(tv2);

  // Build actual fusion graphs and pass it to a
  //  multi-device runtime.
  MultiDeviceRuntime runtime(
      fusion_builder.build(),
      // Process rank that should come from ENV:
      -1);

  // Create at input tensors.
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({8, 8, 8}, options);

  // See group partitions:
  runtime.multiGroupFusion()->print();

  // Run the multiple kernels created.
  // To see the two kernels generated:
  // PYTORCH_NVFUSER_DUMP=cuda_kernel ./test_jit
  // --gtest_filter=*GroupDoubleReduction*
  auto cg_outputs = runtime.runWithInput({input});

  // Validate result

  // Only the rank holding the output should check the
  //  result here:
  // if(rank = ...?)
  auto ref = input.sum(0).sum(0);
  testValidate(
      runtime.flattenedFusion(),
      cg_outputs,
      {input},
      {ref},
      __LINE__,
      __FILE__);
}

int parse_env(int &grank, int &gsize) {
    char *env;

    env = std::getenv("OMPI_COMM_WORLD_RANK");
    if (!env) {
      env = std::getenv("WORLD_RANK");
      if (!env) {
        return 1;
      }
    }
    grank = std::atoi(env);

    env = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (!env) {
      env = std::getenv("WORLD_SIZE");
      if (!env) {
        return 1;
      }
    }
    gsize = std::atoi(env);
    return 0;
}

TEST_F(NVFuserTest, FusionMutiGroupProcessGroup) {
  int grank, gsize;

  if (parse_env(grank, gsize)) {
    GTEST_SKIP() << "distributed config is not provided";
  }

  c10d::TCPStoreOptions store_opts;
  store_opts.isServer = (grank == 0) ? true : false;
  auto store = c10::make_intrusive<c10d::TCPStore>("localhost", store_opts);

  c10d::ProcessGroupBuilder pgBuilder;
  auto pg = pgBuilder.getProcessGroup("nccl", store, grank, gsize);
  pg->barrier();

}

#undef NVFUSER_TEST_CUDA_ARCH_GUARD

} // namespace jit
} // namespace torch

#endif
