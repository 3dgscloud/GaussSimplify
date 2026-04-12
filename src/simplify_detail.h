// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Internal shared types and function declarations for simplify pipeline.
// NOT a public header — used only by simplify_*.cpp internal translation units.

#pragma once

#include "gs/simplify_types.h"

#include "gf/core/gauss_ir.h"
#include "simplify_math.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace gs::detail {

using math::kEpsCov;
using math::kMinScale;
using math::kMinQuatNorm;
using math::kTwoPiPow1p5;

// --- Data types ---

struct ExtraInfo {
    std::string name;
    int dim = 0;
    std::vector<float> data;
};

struct ActivatedCloud {
    int32_t count = 0;
    std::vector<float> positions;   // 3*N
    std::vector<float> scales;      // 3*N, activated (exp'd)
    std::vector<float> rotations;   // 4*N, normalized quaternions [w,x,y,z]
    std::vector<float> alphas;      // N, activated (sigmoid'd)
    std::vector<float> colors;      // 3*N, passthrough
    std::vector<float> sh;          // sh_coeffs_per_point*3 * N, passthrough
    int sh_coeffs_per_point = 0;

    std::vector<ExtraInfo> extras;
};

struct CacheEntry {
    std::array<float, 9> R{};
    float mass = 0.0f;
};

// --- Utilities ---

bool report_progress(const ProgressCallback& progress, float value, const std::string& stage);
float median_of(const std::vector<float>& values);

// --- Activation / Deactivation ---

int sh_degree_from_coeffs_per_channel(int coeffs_per_channel);
int sh_coeffs_per_channel_for_degree(int degree);
void reduce_sh_degree(ActivatedCloud& cloud, int target_degree);

ActivatedCloud activate_from_ir(const gf::GaussianCloudIR& ir);
gf::GaussianCloudIR deactivate_to_ir(const ActivatedCloud& cloud, const gf::GaussMetadata& meta);

// --- Copy utilities ---

void copy_point(const ActivatedCloud& src, int src_idx, ActivatedCloud& dst, int dst_idx);
void ensure_cloud_capacity(ActivatedCloud& cloud, int32_t count, int sh_dim,
                           bool has_colors, bool has_sh,
                           const std::vector<ExtraInfo>& src_extras);

// --- Pruning ---

ActivatedCloud prune_by_opacity(const ActivatedCloud& input, float requested_threshold,
                                std::vector<int32_t>& survivor_ids);
ActivatedCloud prune_by_statistical_outlier(const ActivatedCloud& input,
                                            int nb_neighbors, float std_ratio,
                                            std::vector<int32_t>& survivor_ids);

// --- Merge ---

CacheEntry build_cache_entry(const ActivatedCloud& cloud, int i);
void build_cache(const ActivatedCloud& cloud, std::vector<CacheEntry>& cache);

float compute_edge_cost_euclidean(const ActivatedCloud& cloud, int i, int j);
void compute_edge_costs(const ActivatedCloud& cloud,
                        const std::vector<std::pair<int, int>>& edges,
                        const std::vector<float>& point_weights,
                        std::vector<float>& costs);

void greedy_pairs_from_edges(const std::vector<std::pair<int, int>>& edges,
                             const std::vector<float>& costs,
                             int count, int max_pairs,
                             std::vector<size_t>& order,
                             std::vector<uint8_t>& used,
                             std::vector<std::pair<int, int>>& pairs);

void merge_pairs(const ActivatedCloud& input,
                 const std::vector<std::pair<int, int>>& pairs,
                 std::vector<uint8_t>& used,
                 std::vector<int>& keep_idx,
                 ActivatedCloud& out);

} // namespace gs::detail
