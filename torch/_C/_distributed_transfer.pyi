# This module is defined in torch/csrc/distributed/transfer/python_bindings.cpp

def _snapshot_and_analyze_int32_indices(
    local_indices: object,
    remote_indices: object,
    destination_is_remote: bool,
) -> tuple[bytes, int, int, int, int, int, int, bool]: ...
