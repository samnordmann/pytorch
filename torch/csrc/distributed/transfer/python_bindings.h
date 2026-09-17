#pragma once

#include <torch/csrc/utils/python_stub.h>

namespace torch::distributed::transfer {

void initPythonBindings(PyObject* module);

} // namespace torch::distributed::transfer
