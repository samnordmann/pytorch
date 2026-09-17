#include <torch/csrc/distributed/transfer/python_bindings.h>

#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace torch::distributed::transfer {
namespace {

constexpr std::size_t kTinyDuplicateThreshold = 32;
constexpr std::uint64_t kMaximumBitmapValueCount = 1ULL << 24;
constexpr std::uint64_t kMaximumBitmapValuesPerIndex = 16;

class BufferView final {
 public:
  BufferView(PyObject* object, const char* name) : name_(name) {
    if (PyObject_GetBuffer(object, &view_, PyBUF_FORMAT | PyBUF_STRIDES) != 0) {
      throw py::error_already_set();
    }
    acquired_ = true;
    try {
      validate();
    } catch (...) {
      release();
      throw;
    }
  }

  BufferView(const BufferView&) = delete;
  BufferView& operator=(const BufferView&) = delete;

  ~BufferView() {
    release();
  }

  const void* data() const {
    return view_.buf;
  }

  Py_ssize_t nbytes() const {
    return view_.len;
  }

  std::size_t count() const {
    return static_cast<std::size_t>(view_.len) / sizeof(std::int32_t);
  }

 private:
  static char nativeEndianPrefix() {
    const std::uint16_t value = 1;
    std::uint8_t first_byte = 0;
    std::memcpy(&first_byte, &value, sizeof(first_byte));
    return first_byte == 1 ? '<' : '>';
  }

  static bool isNativeInt32Format(const char* format) {
    if (format == nullptr) {
      return false;
    }
    if (std::strcmp(format, "i") == 0 || std::strcmp(format, "@i") == 0 ||
        std::strcmp(format, "=i") == 0) {
      return true;
    }
    return format[0] == nativeEndianPrefix() &&
        std::strcmp(format + 1, "i") == 0;
  }

  void validate() const {
    if (view_.ndim != 1 || view_.itemsize != sizeof(std::int32_t) ||
        view_.len < 0 || view_.len % sizeof(std::int32_t) != 0 ||
        !isNativeInt32Format(view_.format)) {
      throw py::value_error(
          std::string(name_) +
          " must be a one-dimensional C-contiguous native-int32 buffer");
    }
    const auto count = view_.len / sizeof(std::int32_t);
    if (view_.buf == nullptr || view_.shape == nullptr ||
        view_.strides == nullptr || view_.shape[0] != count ||
        (view_.suboffsets != nullptr && view_.suboffsets[0] >= 0)) {
      throw py::value_error(
          std::string(name_) + " has an inconsistent PEP 3118 shape");
    }
    if (PyBuffer_IsContiguous(&view_, 'C') == 0) {
      throw py::value_error(
          std::string(name_) +
          " must be a one-dimensional C-contiguous native-int32 buffer");
    }
    if (count == 0) {
      throw py::value_error(std::string(name_) + " must be non-empty");
    }
    if (static_cast<std::uintmax_t>(view_.len) >
        std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error(
          std::string(name_) + " is too large for this platform");
    }
  }

  void release() {
    if (acquired_) {
      PyBuffer_Release(&view_);
      acquired_ = false;
    }
  }

  const char* name_;
  Py_buffer view_{};
  bool acquired_{false};
};

struct IndexAnalysis final {
  std::int32_t local_minimum;
  std::int32_t local_maximum;
  std::int32_t remote_minimum;
  std::int32_t remote_maximum;
  bool destination_has_duplicate;
};

struct BoundsAnalysis final {
  std::int32_t minimum;
  std::int32_t maximum;
};

struct DestinationAnalysis final {
  BoundsAnalysis bounds;
  bool strictly_monotonic;
};

std::int32_t loadInt32(const char* data, std::size_t index) {
  std::int32_t value = 0;
  std::memcpy(
      &value, data + index * sizeof(std::int32_t), sizeof(std::int32_t));
  return value;
}

std::uint64_t mixKey(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

bool hasTinyDuplicate(const char* data, std::size_t count) {
  for (std::size_t index = 1; index < count; ++index) {
    const auto value = loadInt32(data, index);
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (loadInt32(data, prior) == value) {
        return true;
      }
    }
  }
  return false;
}

bool hasBitmapDuplicate(
    const char* data,
    std::size_t count,
    std::int32_t minimum,
    std::uint64_t value_count) {
  std::vector<std::uint64_t> bitmap(
      static_cast<std::size_t>((value_count + 63) / 64), 0);
  for (std::size_t index = 0; index < count; ++index) {
    const auto offset = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(loadInt32(data, index)) - minimum);
    const auto word = static_cast<std::size_t>(offset / 64);
    const auto mask = std::uint64_t{1} << (offset % 64);
    if ((bitmap[word] & mask) != 0) {
      return true;
    }
    bitmap[word] |= mask;
  }
  return false;
}

bool hasFlatHashDuplicate(const char* data, std::size_t count) {
  if (count > std::numeric_limits<std::size_t>::max() / 2) {
    throw std::overflow_error("int32 index count exceeds hash-table capacity");
  }
  const auto requested_capacity = count * 2;
  std::size_t capacity = 1;
  while (capacity < requested_capacity) {
    if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error("int32 index hash-table capacity overflow");
    }
    capacity *= 2;
  }
  std::vector<std::uint64_t> table(capacity, 0);
  const auto mask = capacity - 1;
  for (std::size_t index = 0; index < count; ++index) {
    const auto key = static_cast<std::uint64_t>(
                         static_cast<std::uint32_t>(loadInt32(data, index))) +
        1;
    auto slot = static_cast<std::size_t>(mixKey(key)) & mask;
    while (table[slot] != 0) {
      if (table[slot] == key) {
        return true;
      }
      slot = (slot + 1) & mask;
    }
    table[slot] = key;
  }
  return false;
}

bool hasDuplicate(
    const char* data,
    std::size_t count,
    std::int32_t minimum,
    std::int32_t maximum,
    bool strictly_monotonic) {
  if (count < 2 || strictly_monotonic) {
    return false;
  }
  const auto value_count =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(maximum) - minimum) +
      1;
  if (value_count < count) {
    return true;
  }
  if (count <= kTinyDuplicateThreshold) {
    return hasTinyDuplicate(data, count);
  }
  const auto dense_enough =
      count >= (value_count + kMaximumBitmapValuesPerIndex - 1) /
          kMaximumBitmapValuesPerIndex;
  if (value_count <= kMaximumBitmapValueCount && dense_enough) {
    return hasBitmapDuplicate(data, count, minimum, value_count);
  }
  return hasFlatHashDuplicate(data, count);
}

BoundsAnalysis analyzeBounds(const char* data, std::size_t count) {
  const auto first = loadInt32(data, 0);
  BoundsAnalysis result{first, first};
  for (std::size_t index = 1; index < count; ++index) {
    const auto value = loadInt32(data, index);
    result.minimum = std::min(result.minimum, value);
    result.maximum = std::max(result.maximum, value);
  }
  return result;
}

DestinationAnalysis finishNonmonotonicDestination(
    const char* data,
    std::size_t count,
    std::size_t suffix_index,
    BoundsAnalysis bounds) {
  if (suffix_index < count) {
    const auto suffix = analyzeBounds(
        data + suffix_index * sizeof(std::int32_t), count - suffix_index);
    bounds.minimum = std::min(bounds.minimum, suffix.minimum);
    bounds.maximum = std::max(bounds.maximum, suffix.maximum);
  }
  return {bounds, false};
}

DestinationAnalysis analyzeDestination(const char* data, std::size_t count) {
  // Real framework selections are commonly ordered. Prove their direction
  // with a short scalar probe, then let the compiler vectorize the long tail's
  // order and bounds reductions. Non-monotonic inputs leave the probe quickly
  // and retain the same exact duplicate-analysis fallbacks.
  constexpr std::size_t kMonotonicProbeElementCount = 8;
  const auto first = loadInt32(data, 0);
  DestinationAnalysis result{{first, first}, true};
  if (count == 1) {
    return result;
  }

  auto previous = loadInt32(data, 1);
  if (previous == first) {
    return finishNonmonotonicDestination(data, count, 2, result.bounds);
  }

  const auto probe_end = std::min(count, kMonotonicProbeElementCount);
  std::size_t index = 2;
  if (previous > first) {
    result.bounds.maximum = previous;
    for (; index < probe_end; ++index) {
      const auto value = loadInt32(data, index);
      if (value <= previous) {
        result.bounds.minimum = std::min(result.bounds.minimum, value);
        return finishNonmonotonicDestination(
            data, count, index + 1, result.bounds);
      }
      result.bounds.maximum = value;
      previous = value;
    }

    auto minimum = result.bounds.minimum;
    auto maximum = result.bounds.maximum;
    bool strictly_monotonic = true;
    for (; index < count; ++index) {
      const auto value = loadInt32(data, index);
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
      strictly_monotonic &= loadInt32(data, index - 1) < value;
    }
    return {{minimum, maximum}, strictly_monotonic};
  }

  result.bounds.minimum = previous;
  for (; index < probe_end; ++index) {
    const auto value = loadInt32(data, index);
    if (value >= previous) {
      result.bounds.maximum = std::max(result.bounds.maximum, value);
      return finishNonmonotonicDestination(
          data, count, index + 1, result.bounds);
    }
    result.bounds.minimum = value;
    previous = value;
  }

  auto minimum = result.bounds.minimum;
  auto maximum = result.bounds.maximum;
  bool strictly_monotonic = true;
  for (; index < count; ++index) {
    const auto value = loadInt32(data, index);
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
    strictly_monotonic &= loadInt32(data, index - 1) > value;
  }
  return {{minimum, maximum}, strictly_monotonic};
}

IndexAnalysis analyzeSnapshot(
    const char* data,
    std::size_t local_count,
    std::size_t remote_count,
    bool destination_is_remote) {
  const auto* remote = data + local_count * sizeof(std::int32_t);
  BoundsAnalysis local_bounds{};
  BoundsAnalysis remote_bounds{};
  DestinationAnalysis destination{};
  if (destination_is_remote) {
    local_bounds = analyzeBounds(data, local_count);
    destination = analyzeDestination(remote, remote_count);
    remote_bounds = destination.bounds;
  } else {
    destination = analyzeDestination(data, local_count);
    local_bounds = destination.bounds;
    remote_bounds = analyzeBounds(remote, remote_count);
  }
  IndexAnalysis result{
      local_bounds.minimum,
      local_bounds.maximum,
      remote_bounds.minimum,
      remote_bounds.maximum,
      false};
  const auto* destination_data = destination_is_remote ? remote : data;
  const auto destination_count =
      destination_is_remote ? remote_count : local_count;
  result.destination_has_duplicate = hasDuplicate(
      destination_data,
      destination_count,
      destination.bounds.minimum,
      destination.bounds.maximum,
      destination.strictly_monotonic);
  return result;
}

py::tuple snapshotAndAnalyzeInt32Indices(
    py::object local_indices,
    py::object remote_indices,
    bool destination_is_remote) {
  py::bytes snapshot;
  std::size_t local_count = 0;
  std::size_t remote_count = 0;
  {
    BufferView local(local_indices.ptr(), "local_indices");
    BufferView remote(remote_indices.ptr(), "remote_indices");
    if (local.nbytes() > PY_SSIZE_T_MAX - remote.nbytes()) {
      throw std::overflow_error("combined int32 index snapshot is too large");
    }
    const auto snapshot_nbytes = local.nbytes() + remote.nbytes();
    PyObject* raw_snapshot =
        PyBytes_FromStringAndSize(nullptr, snapshot_nbytes);
    if (raw_snapshot == nullptr) {
      throw py::error_already_set();
    }
    snapshot = py::reinterpret_steal<py::bytes>(raw_snapshot);
    auto* snapshot_data = PyBytes_AS_STRING(snapshot.ptr());
    std::memcpy(snapshot_data, local.data(), local.nbytes());
    std::memcpy(snapshot_data + local.nbytes(), remote.data(), remote.nbytes());
    local_count = local.count();
    remote_count = remote.count();
  }

  const auto* snapshot_data = PyBytes_AS_STRING(snapshot.ptr());
  IndexAnalysis analysis{};
  {
    py::gil_scoped_release release;
    analysis = analyzeSnapshot(
        snapshot_data, local_count, remote_count, destination_is_remote);
  }
  return py::make_tuple(
      std::move(snapshot),
      local_count,
      remote_count,
      analysis.local_minimum,
      analysis.local_maximum,
      analysis.remote_minimum,
      analysis.remote_maximum,
      analysis.destination_has_duplicate);
}

} // namespace

void initPythonBindings(PyObject* module) {
  auto root = py::handle(module).cast<py::module>();
  auto transfer = root.def_submodule(
      "_distributed_transfer", "Private distributed transfer helpers.");
  transfer.def(
      "_snapshot_and_analyze_int32_indices",
      &snapshotAndAnalyzeInt32Indices,
      py::arg("local_indices"),
      py::arg("remote_indices"),
      py::arg("destination_is_remote"),
      R"doc(
Create one immutable packed snapshot of two native-int32 buffers and analyze it.

This is a private implementation helper for ``torch.distributed._transfer``.
)doc");
}

} // namespace torch::distributed::transfer
