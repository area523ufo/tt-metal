// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/d2d_pipeline_descriptor.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

#include "d2d_pipeline_descriptor_generated.h"

namespace ttnn {

namespace fb = d2d_flatbuffer;

std::vector<uint8_t> D2DPipelineDescriptor::serialize() const {
    flatbuffers::FlatBufferBuilder builder(1024);
    std::vector<flatbuffers::Offset<fb::D2DRank>> rank_offsets;
    for (const auto& r : ranks) {
        std::vector<flatbuffers::Offset<fb::D2DEndpoint>> endpoint_offsets;
        for (const auto& ep : r.endpoints) {
            std::vector<flatbuffers::Offset<fb::D2DEndpointCoord>> coord_offsets;
            for (const auto& c : ep.coords) {
                std::vector<uint32_t> coord(c.coord.coords().begin(), c.coord.coords().end());
                coord_offsets.push_back(fb::CreateD2DEndpointCoord(
                    builder,
                    builder.CreateVector(coord),
                    *c.fabric_node_id.mesh_id,
                    c.fabric_node_id.chip_id,
                    *c.peer_fabric_node_id.mesh_id,
                    c.peer_fabric_node_id.chip_id,
                    c.pcie_device_id,
                    static_cast<uint32_t>(c.service_core_noc.x),
                    static_cast<uint32_t>(c.service_core_noc.y),
                    c.backing_tensor_addr,
                    c.stage_gate_base_addr));
            }
            endpoint_offsets.push_back(fb::CreateD2DEndpoint(
                builder,
                static_cast<fb::D2DEndpointRole>(ep.role),
                ep.peer_rank,
                ep.metadata_size_bytes,
                ep.num_stage_gates,
                ep.stage_gate_slot_offset_bytes,
                ep.stage_gate_flags_offset_bytes,
                builder.CreateVector(coord_offsets)));
        }
        rank_offsets.push_back(fb::CreateD2DRank(
            builder, r.rank, builder.CreateString(r.host_name), builder.CreateVector(endpoint_offsets)));
    }
    builder.Finish(fb::CreateD2DPipelineDescriptor(builder, kVersion, builder.CreateVector(rank_offsets)));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

D2DPipelineDescriptor D2DPipelineDescriptor::deserialize(const std::vector<uint8_t>& bytes) {
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    TT_FATAL(fb::VerifyD2DPipelineDescriptorBuffer(verifier), "D2DPipelineDescriptor: malformed buffer");
    const auto* root = fb::GetD2DPipelineDescriptor(bytes.data());
    TT_FATAL(
        root->version() == kVersion,
        "D2DPipelineDescriptor version mismatch: got {}, expected {}",
        root->version(),
        kVersion);

    D2DPipelineDescriptor desc;
    if (root->ranks() == nullptr) {
        return desc;
    }
    for (const auto* fr : *root->ranks()) {
        D2DRankDescriptor r{.rank = fr->rank(), .host_name = fr->host_name() ? fr->host_name()->str() : ""};
        if (fr->endpoints() != nullptr) {
            for (const auto* fe : *fr->endpoints()) {
                D2DEndpointDescriptor ep{
                    .role = static_cast<D2DEndpointRole>(fe->role()),
                    .peer_rank = fe->peer_rank(),
                    .metadata_size_bytes = fe->metadata_size_bytes(),
                    .num_stage_gates = fe->num_stage_gates(),
                    .stage_gate_slot_offset_bytes = fe->stage_gate_slot_offset_bytes(),
                    .stage_gate_flags_offset_bytes = fe->stage_gate_flags_offset_bytes()};
                if (fe->coords() != nullptr) {
                    for (const auto* fc : *fe->coords()) {
                        TT_FATAL(fc->coord() != nullptr, "D2DPipelineDescriptor: coord entry missing coord");
                        ep.coords.push_back(D2DEndpointCoordDescriptor{
                            .coord = tt::tt_metal::distributed::MeshCoordinate(
                                ttsl::Span<const uint32_t>(fc->coord()->data(), fc->coord()->size())),
                            .fabric_node_id =
                                tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{fc->mesh_id()}, fc->chip_id()),
                            .peer_fabric_node_id = tt::tt_fabric::FabricNodeId(
                                tt::tt_fabric::MeshId{fc->peer_mesh_id()}, fc->peer_chip_id()),
                            .pcie_device_id = fc->pcie_device_id(),
                            .service_core_noc =
                                tt::tt_metal::CoreCoord(fc->service_core_noc_x(), fc->service_core_noc_y()),
                            .backing_tensor_addr = fc->backing_tensor_addr(),
                            .stage_gate_base_addr = fc->stage_gate_base_addr()});
                    }
                }
                r.endpoints.push_back(std::move(ep));
            }
        }
        desc.ranks.push_back(std::move(r));
    }
    return desc;
}

void D2DPipelineDescriptor::write_to_file(const std::string& path) const {
    const auto bytes = serialize();
    const std::string tmp_path = path + ".tmp";
    std::ofstream ofs(tmp_path, std::ios::binary);
    TT_FATAL(ofs.is_open(), "Failed to open D2D pipeline descriptor for writing: {}", tmp_path);
    ofs.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    ofs.close();
    TT_FATAL(!ofs.fail(), "Failed to write D2D pipeline descriptor: {}", tmp_path);
    TT_FATAL(
        std::rename(tmp_path.c_str(), path.c_str()) == 0,
        "Failed to rename D2D pipeline descriptor from {} to {}: {}",
        tmp_path,
        path,
        std::strerror(errno));
}

D2DPipelineDescriptor D2DPipelineDescriptor::wait_and_read(const std::string& path, uint32_t timeout_ms) {
    const auto start = std::chrono::steady_clock::now();
    while (!std::filesystem::exists(path)) {
        TT_FATAL(
            std::chrono::steady_clock::now() - start <= std::chrono::milliseconds(timeout_ms),
            "Timeout waiting for D2D pipeline descriptor: {}",
            path);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::ifstream ifs(path, std::ios::binary);
    TT_FATAL(ifs.is_open(), "Failed to open D2D pipeline descriptor: {}", path);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return deserialize(bytes);
}

std::string d2d_pipeline_descriptor_path(const std::string& service_id) {
    return fmt::format("/dev/shm/tt_d2d_pipeline_{}.bin", service_id);
}

}  // namespace ttnn
