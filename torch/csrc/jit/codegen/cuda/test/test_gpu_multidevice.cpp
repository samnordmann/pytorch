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
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>
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

// TEST_F(NVFuserTest, FusionMutiGroupDoubleReduction_CUDA) {
//   // Using the new interface to build multi-group fusion
//   MultiGroupFusionBuilder fusion_builder;

//   // Fusion guard is on the fusion managed within builder.
//   FusionGuard fg(fusion_builder.completeFusion());

//   TensorView* tv0 = makeContigTensor(3);

//   fusion_builder.addFusionInput(tv0);

//   // Each expression has to belong to some group,
//   //  and each group will become one cuda kernel
//   //  after lowering time.

//   // Create the first group.
//   //  The builder now points to the first created group,
//   // all operations following this line will make changes
//   // to the first group.
//   fusion_builder.newGroup(
//       // auto-schedule
//       true);

//   TensorView* tv1 = sum(tv0, {0});

//   fusion_builder.addGroupOutput(tv1);

//   // Create the second group.
//   //  The builder now points to the second created group,
//   // all operations following this line will make changes
//   // to the second group.
//   fusion_builder.newGroup(
//       // auto-schedule
//       true);

//   TensorView* tv2 = sum(tv1, {0});

//   fusion_builder.addFusionOutput(tv2);

//   // Build actual fusion graphs and pass it to a
//   //  multi-device runtime.
//   MultiDeviceRuntime runtime(fusion_builder.build());

//   // Create at input tensors.
//   auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
//   at::Tensor input = at::randn({8, 8, 8}, options);

//   // See group partitions:
//   runtime.multiGroupFusion()->print();

//   // Run the multiple kernels created.
//   // To see the two kernels generated:
//   // PYTORCH_NVFUSER_DUMP=cuda_kernel ./test_jit
//   // --gtest_filter=*GroupDoubleReduction*
//   auto cg_outputs = runtime.runWithInput({input});

//   // Validate result
//   auto ref = input.sum(0).sum(0);
//   testValidate(
//       runtime.flattenedFusion(),
//       cg_outputs,
//       {input},
//       {ref},
//       __LINE__,
//       __FILE__);
// }

// TEST_F(NVFuserTest, FusionMultiRankReduction_CUDA) {
//   // Using the new interface to build multi-group fusion
//   MultiGroupFusionBuilder fusion_builder;

//   // Fusion guard is on the fusion managed within builder.
//   FusionGuard fg(fusion_builder.completeFusion());

//   TensorView* tv0 = makeContigTensor(3);

//   fusion_builder.addFusionInput(tv0);

//   // Each expression has to belong to some group,
//   //  and each group will become one cuda kernel
//   //  after lowering time.

//   // Create the first group.
//   //  The builder now points to the first created group,
//   // all operations following this line will make changes
//   // to the first group.
//   fusion_builder.newGroup(
//       // auto-schedule
//       true,
//       // Process rank that runs this group:
//       // -1 means all group runs.
//       -1,
//       // Cuda device that runs this group:
//       c10::Device(DeviceType::CUDA, at::cuda::current_device()));

//   TensorView* tv1 = sum(tv0, {0});

//   fusion_builder.addGroupOutput(tv1);

//   // Create the second group.
//   //  The builder now points to the second created group,
//   // all operations following this line will make changes
//   // to the second group.
//   fusion_builder.newGroup(
//       // auto-schedule
//       true,
//       // Process rank that runs this group:
//       // -1 means all group runs.
//       -1,
//       // Cuda device that runs this group:
//       c10::Device(DeviceType::CUDA, at::cuda::current_device()));

//   TensorView* tv2 = sum(tv1, {0});

//   fusion_builder.addFusionOutput(tv2);

//   // Build actual fusion graphs and pass it to a
//   //  multi-device runtime.
//   MultiDeviceRuntime runtime(
//       fusion_builder.build(),
//       // Process rank that should come from ENV:
//       -1);

//   // Create at input tensors.
//   auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
//   at::Tensor input = at::randn({8, 8, 8}, options);

//   // See group partitions:
//   runtime.multiGroupFusion()->print();

//   // Run the multiple kernels created.
//   // To see the two kernels generated:
//   // PYTORCH_NVFUSER_DUMP=cuda_kernel ./test_jit
//   // --gtest_filter=*GroupDoubleReduction*
//   auto cg_outputs = runtime.runWithInput({input});

//   // Validate result

//   // Only the rank holding the output should check the
//   //  result here:
//   // if(rank = ...?)
//   auto ref = input.sum(0).sum(0);
//   testValidate(
//       runtime.flattenedFusion(),
//       cg_outputs,
//       {input},
//       {ref},
//       __LINE__,
//       __FILE__);
// }

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

TEST_F(NVFuserTest, SendRecvTest) {
  // Using the new interface to build multi-group fusion
  MultiGroupFusionBuilder fusion_builder;
  int grank, gsize;

  if (parse_env(grank, gsize)) {
    GTEST_SKIP() << "distributed config is not provided";
  }

  c10d::TCPStoreOptions store_opts;
  store_opts.isServer = (grank == 0) ? true : false;
  auto store = c10::make_intrusive<c10d::TCPStore>("localhost", store_opts);

  c10d::ProcessGroupBuilder pgBuilder;
  auto pg = pgBuilder.getProcessGroup("nccl", store, grank, gsize);


  if (grank==0){
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::Device("cuda:0"));
      at::Tensor input = at::randn({8}, options);
      std::vector<at::Tensor> tensor_to_send = {input};
      pg->send(tensor_to_send, 1, 0);
      std::cout << "sent tensor:\n" << tensor_to_send[0] << std::endl;
  } else{
        auto options = at::TensorOptions().dtype(at::kFloat).device(at::Device("cuda:1"));
        std::vector<at::Tensor> tensor_to_receive= {at::empty({8}, options)};
        auto work = pg->recv(tensor_to_receive, 0, 0);
        while (!work->isCompleted()); // wait for completion
        std::cout << "received tensor:\n" << tensor_to_receive[0] << std::endl;
  } 
}


TEST_F(NVFuserTest, FusionMultiGPU) {
  // Using the new interface to build multi-group fusion
  MultiGroupFusionBuilder fusion_builder;
  int grank, gsize;

  // defining the process group
  if (parse_env(grank, gsize)) {
    GTEST_SKIP() << "distributed config is not provided";
  }

  c10d::TCPStoreOptions store_opts;
  store_opts.isServer = (grank == 0) ? true : false;
  auto store = c10::make_intrusive<c10d::TCPStore>("localhost", store_opts);

  c10d::ProcessGroupBuilder pgBuilder;
  auto pg = pgBuilder.getProcessGroup("nccl", store, grank, gsize);
  // process group defined

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
      0,
      // Cuda device that runs this group:
      at::Device("cuda:0")
      );

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
      1,
      // Cuda device that runs this group:
      at::Device("cuda:1")
    );

  TensorView* tv2 = sum(tv1, {0});

  fusion_builder.addFusionOutput(tv2);

  // Build actual fusion graphs and pass it to a
  //  multi-device runtime.
  MultiDeviceRuntime runtime(
      fusion_builder.build(),
      pg, grank);

  if (grank == 0) {
    // See group partitions:
    runtime.multiGroupFusion()->print();
  }


  // Create at input tensors.
  if (grank == 0){
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::Device("cuda:0"));
    at::Tensor input = at::randn({8, 8, 8}, options);
    auto cg_outputs = runtime.runWithInput({input});// Run the multiple kernels created.

    std::cout << "Expected result:\n" << input.sum({0}).sum({0}) << std::endl;
  } else{
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::Device("cuda:1"));
    at::Tensor input = at::empty({8, 8, 8}, options); //creates an empty placeholder
    auto cg_outputs = runtime.runWithInput({input});// Run the multiple kernels created.

    std::cout << "Obtained result:\n" << cg_outputs << std::endl;
  }

}

TEST_F(NVFuserTest, FusionMultiGPU_Reduce) {
  MultiGroupFusionBuilder fusion_builder;
  int grank, gsize;

// Processgroup setup
  if (parse_env(grank, gsize)) {
    GTEST_SKIP() << "distributed config is not provided";
  }

  c10d::TCPStoreOptions store_opts;
  store_opts.isServer = (grank == 0) ? true : false;
  auto store = c10::make_intrusive<c10d::TCPStore>("localhost", store_opts);

  c10d::ProcessGroupBuilder pgBuilder;
  auto pg = pgBuilder.getProcessGroup("nccl", store, grank, gsize);

// Fusion definition
  FusionGuard fg(fusion_builder.completeFusion());

  TensorView* tva = makeContigTensor(2);
  TensorView* tvb = makeContigTensor(2);

  fusion_builder.addFusionInput(tva);
  fusion_builder.addFusionInput(tvb);

  fusion_builder.newGroup(true, 0, at::Device("cuda:0"));
  TensorView* tva1 = sum(tva, {0});
  fusion_builder.addGroupOutput(tva1);

  fusion_builder.newGroup(true, 1, at::Device("cuda:1"));
  TensorView* tvb1 = sum(tvb, {0});
  fusion_builder.addGroupOutput(tvb1);

  fusion_builder.newGroup(true, 2, at::Device("cuda:2"));
  TensorView* tv2 = add(tva1, tvb1);
  fusion_builder.addFusionOutput(tv2);

  // print the fusion
  if (grank == 0) {
    runtime.multiGroupFusion()->print();
  }

  // create runtime
  MultiDeviceRuntime runtime(fusion_builder.build(), pg, grank);

  // Create at input tensors.
    TensorOptions  options;
    if (grank == 0){
        options = at::TensorOptions().dtype(at::kFloat).device(at::Device("cuda:0"));
    } else if (grank == 1) {
        options = at::TensorOptions().dtype(at::kFloat).device(at::Device("cuda:1"));
    } else if (grank == 2) {
        options = at::TensorOptions().dtype(at::kFloat).device(at::Device("cuda:2"));
    }
    at::Tensor input_tva = at::randn({8, 8}, options);
    at::Tensor input_tvb = at::randn({8, 8}, options);

// run
    auto cg_outputs = runtime.runWithInput({input_tva, input_tvb});// Run the multiple kernels created.

// print results
  if (grank == 0){
    std::vector<at::Tensor> sent_tva = {input_tva};
    pg->send(sent_tva, 2, 0);
  } else if (grank == 1){
    std::vector<at::Tensor> sent_tvb = {input_tvb};
    pg->send(sent_tvb, 2, 0);
  } else if (grank == 2){
    std::vector<at::Tensor> received_tva = {input_tva};
    std::vector<at::Tensor> received_tvb = {input_tvb};
    auto work0 = pg->recv(received_tva, 0, 0);
    auto work1 = pg->recv(received_tvb, 1, 0);
    while (!work0->isCompleted() || !work1->isCompleted());
    auto ref = input_tva.sum({0}) + input_tvb.sum({0});
    std::cout << "Expected output:\n" << ref << std::endl;
    std::cout << "Obtained output:\n" << cg_outputs << std::endl;
  //   TORCH_INTERNAL_ASSERT(
  //       ref.equal(cg_outputs[0]), "Obtained output is not the one expected");
  // }


  pg->barrier();
}

#undef NVFUSER_TEST_CUDA_ARCH_GUARD

} // namespace jit
} // namespace torch

#endif
