#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {

class TORCH_API ProcessGroupBuilder : public torch::CustomClassHolder {
 public:

  c10::intrusive_ptr<c10d::ProcessGroup> getProcessGroup(
      std::string backend,
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size);
};

} // namespace c10d
