#include <torch/csrc/distributed/c10d/ProcessGroupBuilder.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

namespace c10d {

c10::intrusive_ptr<c10d::ProcessGroup> ProcessGroupBuilder::getProcessGroup(
    std::string backend,
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
{
#ifdef USE_C10D_GLOO
  if (backend == "nccl") {
    auto pg_opts = c10::make_intrusive<::c10d::ProcessGroupNCCL::Options>();
    return c10::make_intrusive<::c10d::ProcessGroupNCCL>(
        store, rank, size, pg_opts);

  }
#endif

#ifdef USE_C10D_GLOO
  if (backend == "gloo") {
    auto pg_opts = c10d::ProcessGroupGloo::Options::create();
    return c10::make_intrusive<::c10d::ProcessGroupGloo>(
        store, rank, size, pg_opts);
  }
#endif
  TORCH_CHECK(false, "no dist backend available");
  // return nullptr;
}

} // namespace c10d
