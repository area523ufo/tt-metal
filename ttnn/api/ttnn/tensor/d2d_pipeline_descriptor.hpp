// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Description of every D2DStreamService endpoint in a multi-rank pipeline, gathered onto
// one rank and written to a file another process can read without a MeshDevice. Written by
// D2DStreamService::export_pipeline_descriptor.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn {

enum class D2DEndpointRole : uint8_t { Sender = 0, Receiver = 1 };

// One coord of a D2D endpoint.
struct D2DEndpointCoordDescriptor {
    tt::tt_metal::distributed::MeshCoordinate coord{0, 0};
    tt::tt_fabric::FabricNodeId fabric_node_id{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_fabric::FabricNodeId peer_fabric_node_id{tt::tt_fabric::MeshId{0}, 0};
    uint32_t pcie_device_id = 0;               // PCIe device number on the owning host; UINT32_MAX if none
    tt::tt_metal::CoreCoord service_core_noc;  // virtual NoC coords
    tt::tt_metal::DeviceAddr backing_tensor_addr = 0;
    tt::tt_metal::DeviceAddr stage_gate_base_addr = 0;  // gate i is the uint32 at +4 * i; 0 without a gate

    bool operator==(const D2DEndpointCoordDescriptor&) const = default;
};

struct D2DEndpointDescriptor {
    D2DEndpointRole role = D2DEndpointRole::Sender;
    int peer_rank = 0;
    uint32_t metadata_size_bytes = 0;
    uint32_t num_stage_gates = 0;  // receivers only
    uint32_t stage_gate_slot_offset_bytes = 0;
    uint32_t stage_gate_flags_offset_bytes = 0;
    std::vector<D2DEndpointCoordDescriptor> coords;

    bool operator==(const D2DEndpointDescriptor&) const = default;
};

struct D2DRankDescriptor {
    int rank = 0;
    std::string host_name;
    std::vector<D2DEndpointDescriptor> endpoints;

    bool operator==(const D2DRankDescriptor&) const = default;
};

struct D2DPipelineDescriptor {
    static constexpr uint32_t kVersion = 1;

    std::vector<D2DRankDescriptor> ranks;  // ordered by rank

    bool operator==(const D2DPipelineDescriptor&) const = default;

    std::vector<uint8_t> serialize() const;
    static D2DPipelineDescriptor deserialize(const std::vector<uint8_t>& bytes);

    // Atomic write (tmp + rename).
    void write_to_file(const std::string& path) const;

    // Wait for the file to appear, then read it. Throws on timeout or version mismatch.
    static D2DPipelineDescriptor wait_and_read(const std::string& path, uint32_t timeout_ms = 10000);
};

// /dev/shm/tt_d2d_pipeline_<service_id>.bin
std::string d2d_pipeline_descriptor_path(const std::string& service_id);

}  // namespace ttnn
