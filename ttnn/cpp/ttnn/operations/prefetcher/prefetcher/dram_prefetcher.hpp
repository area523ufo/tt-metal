// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/types.hpp"

namespace ttnn {

// Worker-sender prefetcher op: `global_cb` is the GlobalCircularBuffer it fills. The DRAM-core
// counterpart is ttnn.experimental.start_tensor_prefetcher, which delivers into PrefetcherPipes.
ttnn::Tensor dram_prefetcher(
    std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    bool enable_performance_mode = false);

}  // namespace ttnn
