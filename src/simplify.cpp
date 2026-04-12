// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gs/simplify.h"

#include "gf/core/gauss_ir.h"
#include "gf/core/validate.h"
#include "simplify_knn.h"
#include "simplify_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gs {

namespace {

using math::kEpsCov;
using math::kMinEval;
using math::kMinProb;
using math::kMinQuatNorm;
using math::kMinScale;
using math::kTwoPiPow1p5;

// --- Internal working representation with activated values ---

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

// --- Progress helper ---

bool report_progress(const ProgressCallback& progress, const float value, const std::string& stage) {
    if (!progress)
        return true;
    return progress(std::clamp(value, 0.0f, 1.0f), stage);
}

// --- Median ---

float median_of(const std::vector<float>& values) {
    if (values.empty())
        return 0.0f;
    std::vector<float> copy(values);
    const size_t mid = copy.size() / 2;
    std::nth_element(copy.begin(), copy.begin() + static_cast<ptrdiff_t>(mid), copy.end());
    if ((copy.size() & 1U) != 0U)
        return copy[mid];
    const float lo = *std::max_element(copy.begin(), copy.begin() + static_cast<ptrdiff_t>(mid));
    return 0.5f * (lo + copy[mid]);
}

// --- IR <-> ActivatedCloud conversion ---

ActivatedCloud activate_from_ir(const gf::GaussianCloudIR& ir) {
    ActivatedCloud cloud;
    const int32_t n = ir.numPoints;
    cloud.count = n;

    // Positions: copy as-is
    cloud.positions = ir.positions;

    // Scales: apply exp (IR stores log-scale)
    cloud.scales.resize(static_cast<size_t>(n) * 3);
    for (int32_t i = 0; i < n * 3; ++i)
        cloud.scales[static_cast<size_t>(i)] = math::activated_scale(ir.scales[static_cast<size_t>(i)]);

    // Rotations: normalize quaternions
    cloud.rotations = ir.rotations;
    for (int32_t i = 0; i < n; ++i) {
        const size_t i4 = static_cast<size_t>(i) * 4;
        float qw = cloud.rotations[i4 + 0];
        float qx = cloud.rotations[i4 + 1];
        float qy = cloud.rotations[i4 + 2];
        float qz = cloud.rotations[i4 + 3];
        const float inv_q = 1.0f / std::max(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), kMinQuatNorm);
        cloud.rotations[i4 + 0] = qw * inv_q;
        cloud.rotations[i4 + 1] = qx * inv_q;
        cloud.rotations[i4 + 2] = qy * inv_q;
        cloud.rotations[i4 + 3] = qz * inv_q;
    }

    // Alphas: apply sigmoid (IR stores pre-sigmoid)
    cloud.alphas.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i)
        cloud.alphas[static_cast<size_t>(i)] = math::sigmoid(ir.alphas[static_cast<size_t>(i)]);

    // Colors and SH: copy as-is
    cloud.colors = ir.colors;
    cloud.sh = ir.sh;

    // Compute sh_coeffs_per_point
    if (n > 0 && !ir.sh.empty()) {
        const int total_sh = static_cast<int>(ir.sh.size()) / n;
        cloud.sh_coeffs_per_point = total_sh / 3;
    }

    // Extras: copy as-is, convert map -> flat vector
    cloud.extras.reserve(ir.extras.size());
    for (const auto& [name, data] : ir.extras) {
        ExtraInfo ei;
        ei.name = name;
        ei.data = data;
        if (n > 0 && !data.empty())
            ei.dim = static_cast<int>(data.size()) / n;
        cloud.extras.push_back(std::move(ei));
    }

    return cloud;
}

// --- SH degree helpers ---

int sh_degree_from_coeffs_per_channel(const int coeffs_per_channel) {
    // per_channel = (d+1)^2 - 1, so d = sqrt(per_channel + 1) - 1
    if (coeffs_per_channel <= 0) return 0;
    const int d = static_cast<int>(std::sqrt(static_cast<double>(coeffs_per_channel) + 1.0) + 0.5) - 1;
    // Verify: (d+1)^2 - 1 should equal coeffs_per_channel
    if ((d + 1) * (d + 1) - 1 != coeffs_per_channel) return -1;  // Invalid
    return d;
}

int sh_coeffs_per_channel_for_degree(const int degree) {
    if (degree <= 0) return 0;
    return (degree + 1) * (degree + 1) - 1;
}

void reduce_sh_degree(ActivatedCloud& cloud, const int target_degree) {
    if (target_degree < 0) return;  // Don't change
    if (cloud.sh.empty() || cloud.sh_coeffs_per_point <= 0) return;  // No SH to reduce

    const int current_degree = sh_degree_from_coeffs_per_channel(cloud.sh_coeffs_per_point);
    if (current_degree < 0 || target_degree >= current_degree) return;  // Can't upgrade or invalid

    if (target_degree == 0) {
        // Drop all SH data
        cloud.sh.clear();
        cloud.sh.shrink_to_fit();
        cloud.sh_coeffs_per_point = 0;
        return;
    }

    const int target_coeffs = sh_coeffs_per_channel_for_degree(target_degree);
    const int target_dim = target_coeffs * 3;  // floats per point
    const int current_dim = cloud.sh_coeffs_per_point * 3;

    // Truncate each point's SH data in-place
    const int32_t n = cloud.count;
    std::vector<float> reduced(static_cast<size_t>(n) * static_cast<size_t>(target_dim));
    for (int32_t i = 0; i < n; ++i) {
        const size_t src_offset = static_cast<size_t>(i) * static_cast<size_t>(current_dim);
        const size_t dst_offset = static_cast<size_t>(i) * static_cast<size_t>(target_dim);
        std::copy_n(cloud.sh.begin() + static_cast<ptrdiff_t>(src_offset),
                    target_dim,
                    reduced.begin() + static_cast<ptrdiff_t>(dst_offset));
    }
    cloud.sh = std::move(reduced);
    cloud.sh_coeffs_per_point = target_coeffs;
}

gf::GaussianCloudIR deactivate_to_ir(const ActivatedCloud& cloud, const gf::GaussMetadata& meta) {
    gf::GaussianCloudIR ir;
    const int32_t n = cloud.count;
    ir.numPoints = n;

    ir.positions = cloud.positions;

    // Scales: apply log (back to log-scale)
    ir.scales.resize(static_cast<size_t>(n) * 3);
    for (int32_t i = 0; i < n * 3; ++i)
        ir.scales[static_cast<size_t>(i)] = std::log(std::max(cloud.scales[static_cast<size_t>(i)], kMinScale));

    ir.rotations = cloud.rotations;

    // Alphas: apply logit (back to pre-sigmoid)
    ir.alphas.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i)
        ir.alphas[static_cast<size_t>(i)] = math::logit_from_alpha(cloud.alphas[static_cast<size_t>(i)]);

    ir.colors = cloud.colors;
    ir.sh = cloud.sh;
    // Convert flat extras vector back to map
    ir.extras.clear();
    for (const auto& ei : cloud.extras)
        ir.extras[ei.name] = ei.data;
    ir.meta = meta;
    // Update shDegree to reflect actual SH content
    ir.meta.shDegree = sh_degree_from_coeffs_per_channel(cloud.sh_coeffs_per_point);
    if (ir.meta.shDegree < 0) ir.meta.shDegree = 0;

    return ir;
}

// --- Row copy helper ---

void copy_point(const ActivatedCloud& src, const int src_idx,
                ActivatedCloud& dst, const int dst_idx) {
    const size_t s3 = static_cast<size_t>(src_idx) * 3;
    const size_t d3 = static_cast<size_t>(dst_idx) * 3;
    const size_t s4 = static_cast<size_t>(src_idx) * 4;
    const size_t d4 = static_cast<size_t>(dst_idx) * 4;

    std::copy_n(src.positions.begin() + static_cast<ptrdiff_t>(s3), 3,
                dst.positions.begin() + static_cast<ptrdiff_t>(d3));
    std::copy_n(src.scales.begin() + static_cast<ptrdiff_t>(s3), 3,
                dst.scales.begin() + static_cast<ptrdiff_t>(d3));
    std::copy_n(src.rotations.begin() + static_cast<ptrdiff_t>(s4), 4,
                dst.rotations.begin() + static_cast<ptrdiff_t>(d4));
    dst.alphas[static_cast<size_t>(dst_idx)] = src.alphas[static_cast<size_t>(src_idx)];

    // Colors (3 per point)
    if (!src.colors.empty()) {
        std::copy_n(src.colors.begin() + static_cast<ptrdiff_t>(s3), 3,
                    dst.colors.begin() + static_cast<ptrdiff_t>(d3));
    }

    // SH
    const int sh_dim = src.sh_coeffs_per_point * 3;
    if (sh_dim > 0 && !src.sh.empty()) {
        const size_t ss = static_cast<size_t>(src_idx) * static_cast<size_t>(sh_dim);
        const size_t ds = static_cast<size_t>(dst_idx) * static_cast<size_t>(sh_dim);
        std::copy_n(src.sh.begin() + static_cast<ptrdiff_t>(ss), sh_dim,
                    dst.sh.begin() + static_cast<ptrdiff_t>(ds));
    }

    // Extras
    for (size_t ei = 0; ei < src.extras.size(); ++ei) {
        const int dim = src.extras[ei].dim;
        if (dim <= 0) continue;
        const auto& src_data = src.extras[ei].data;
        auto& dst_data = dst.extras[ei].data;
        const size_t so = static_cast<size_t>(src_idx) * static_cast<size_t>(dim);
        const size_t do_ = static_cast<size_t>(dst_idx) * static_cast<size_t>(dim);
        std::copy_n(src_data.begin() + static_cast<ptrdiff_t>(so), dim,
                    dst_data.begin() + static_cast<ptrdiff_t>(do_));
    }
}

// --- Opacity pruning ---

ActivatedCloud prune_by_opacity(const ActivatedCloud& input, const float requested_threshold,
                                std::vector<int32_t>& survivor_ids) {
    if (input.count == 0)
        return input;

    const float median_alpha = median_of(input.alphas);
    const float threshold = std::min(requested_threshold, median_alpha);

    survivor_ids.clear();
    survivor_ids.reserve(static_cast<size_t>(input.count));
    for (int32_t i = 0; i < input.count; ++i) {
        if (input.alphas[static_cast<size_t>(i)] >= threshold)
            survivor_ids.push_back(i);
    }

    const int32_t out_count = static_cast<int32_t>(survivor_ids.size());
    const int sh_dim = input.sh_coeffs_per_point * 3;

    ActivatedCloud out;
    out.count = out_count;
    out.sh_coeffs_per_point = input.sh_coeffs_per_point;
    out.positions.resize(static_cast<size_t>(out_count) * 3);
    out.scales.resize(static_cast<size_t>(out_count) * 3);
    out.rotations.resize(static_cast<size_t>(out_count) * 4);
    out.alphas.resize(static_cast<size_t>(out_count));
    if (!input.colors.empty())
        out.colors.resize(static_cast<size_t>(out_count) * 3);
    if (sh_dim > 0)
        out.sh.resize(static_cast<size_t>(out_count) * static_cast<size_t>(sh_dim));

    // Setup extras output
    out.extras.resize(input.extras.size());
    for (size_t ei = 0; ei < input.extras.size(); ++ei) {
        out.extras[ei].name = input.extras[ei].name;
        out.extras[ei].dim = input.extras[ei].dim;
        if (input.extras[ei].dim > 0)
            out.extras[ei].data.resize(static_cast<size_t>(out_count) * static_cast<size_t>(input.extras[ei].dim));
    }

    for (int32_t dst_row = 0; dst_row < out_count; ++dst_row)
        copy_point(input, survivor_ids[static_cast<size_t>(dst_row)], out, dst_row);

    return out;
}

// --- Statistical Outlier Removal ---

ActivatedCloud prune_by_statistical_outlier(const ActivatedCloud& input,
                                            const int nb_neighbors,
                                            const float std_ratio,
                                            std::vector<int32_t>& survivor_ids) {
    if (input.count == 0)
        return input;

    const int k_eff = std::min(std::max(1, nb_neighbors), std::max(1, input.count - 1));

    // Build KD-tree
    knn::PointCloudAdaptor cloud{input.positions.data(), static_cast<size_t>(input.count)};
    knn::KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    // Compute per-point mean distance to k nearest neighbors
    std::vector<float> mean_dists(static_cast<size_t>(input.count));
    const size_t query_count = static_cast<size_t>(std::min(input.count, k_eff + 1));

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < input.count; ++i) {
        size_t ret_indices[64];
        double out_dists_sqr[64];
        nanoflann::KNNResultSet<double> result_set(query_count);
        result_set.init(ret_indices, out_dists_sqr);
        const double query[3] = {
            static_cast<double>(input.positions[static_cast<size_t>(i) * 3 + 0]),
            static_cast<double>(input.positions[static_cast<size_t>(i) * 3 + 1]),
            static_cast<double>(input.positions[static_cast<size_t>(i) * 3 + 2]),
        };
        index.findNeighbors(result_set, query, nanoflann::SearchParameters(0.0, true));

        // Skip self (index 0), take next k_eff neighbors
        double sum = 0.0;
        const size_t take = std::min(static_cast<size_t>(k_eff),
            result_set.size() > 0 ? result_set.size() - 1 : size_t{0});
        for (size_t j = 0; j < take; ++j)
            sum += std::sqrt(out_dists_sqr[j + 1]);
        mean_dists[static_cast<size_t>(i)] = take > 0
            ? static_cast<float>(sum / static_cast<double>(take)) : 0.0f;
    }

    // Compute global mean and std
    double global_sum = 0.0;
    for (const float d : mean_dists)
        global_sum += static_cast<double>(d);
    const double global_mean = global_sum / static_cast<double>(input.count);

    double var_sum = 0.0;
    for (const float d : mean_dists) {
        const double diff = static_cast<double>(d) - global_mean;
        var_sum += diff * diff;
    }
    const double global_std = std::sqrt(var_sum / static_cast<double>(input.count));

    const float threshold = static_cast<float>(global_mean + static_cast<double>(std_ratio) * global_std);

    // Collect survivors
    survivor_ids.clear();
    survivor_ids.reserve(static_cast<size_t>(input.count));
    for (int32_t i = 0; i < input.count; ++i) {
        if (mean_dists[static_cast<size_t>(i)] <= threshold)
            survivor_ids.push_back(i);
    }

    const int32_t out_count = static_cast<int32_t>(survivor_ids.size());
    if (out_count == input.count)
        return input;  // Nothing removed

    const int sh_dim = input.sh_coeffs_per_point * 3;

    ActivatedCloud out;
    out.count = out_count;
    out.sh_coeffs_per_point = input.sh_coeffs_per_point;
    out.positions.resize(static_cast<size_t>(out_count) * 3);
    out.scales.resize(static_cast<size_t>(out_count) * 3);
    out.rotations.resize(static_cast<size_t>(out_count) * 4);
    out.alphas.resize(static_cast<size_t>(out_count));
    if (!input.colors.empty())
        out.colors.resize(static_cast<size_t>(out_count) * 3);
    if (sh_dim > 0)
        out.sh.resize(static_cast<size_t>(out_count) * static_cast<size_t>(sh_dim));

    out.extras.resize(input.extras.size());
    for (size_t ei = 0; ei < input.extras.size(); ++ei) {
        out.extras[ei].name = input.extras[ei].name;
        out.extras[ei].dim = input.extras[ei].dim;
        if (input.extras[ei].dim > 0)
            out.extras[ei].data.resize(static_cast<size_t>(out_count) * static_cast<size_t>(input.extras[ei].dim));
    }

    for (int32_t dst_row = 0; dst_row < out_count; ++dst_row)
        copy_point(input, survivor_ids[static_cast<size_t>(dst_row)], out, dst_row);

    return out;
}

// --- Build rotation/mass cache ---

CacheEntry build_cache_entry(const ActivatedCloud& cloud, const int i) {
    CacheEntry entry;
    const size_t i3 = static_cast<size_t>(i) * 3;
    const size_t i4 = static_cast<size_t>(i) * 4;

    const float sx = std::max(cloud.scales[i3 + 0], kMinScale);
    const float sy = std::max(cloud.scales[i3 + 1], kMinScale);
    const float sz = std::max(cloud.scales[i3 + 2], kMinScale);

    math::quat_to_rotmat(
        cloud.rotations[i4 + 0], cloud.rotations[i4 + 1],
        cloud.rotations[i4 + 2], cloud.rotations[i4 + 3],
        entry.R);

    const float alpha = cloud.alphas[static_cast<size_t>(i)];
    const float scale_prod = math::strict_prod3(sx, sy, sz);
    entry.mass = math::strict_add(math::strict_mul(math::strict_mul(kTwoPiPow1p5, alpha), scale_prod), 1e-12f);
    return entry;
}

void build_cache(const ActivatedCloud& cloud, std::vector<CacheEntry>& cache) {
    cache.resize(static_cast<size_t>(cloud.count));

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < cloud.count; ++i)
        cache[static_cast<size_t>(i)] = build_cache_entry(cloud, i);
}

// --- Edge cost (Euclidean distance) ---

float compute_edge_cost_euclidean(const ActivatedCloud& cloud, const int i, const int j) {
    const size_t i3 = static_cast<size_t>(i) * 3;
    const size_t j3 = static_cast<size_t>(j) * 3;
    const float dx = cloud.positions[i3 + 0] - cloud.positions[j3 + 0];
    const float dy = cloud.positions[i3 + 1] - cloud.positions[j3 + 1];
    const float dz = cloud.positions[i3 + 2] - cloud.positions[j3 + 2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void compute_edge_costs(const ActivatedCloud& cloud,
                        const std::vector<std::pair<int, int>>& edges,
                        std::vector<float>& costs) {
    costs.assign(edges.size(), std::numeric_limits<float>::infinity());

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < edges.size(); ++i) {
        const auto [u, v] = edges[i];
        costs[i] = compute_edge_cost_euclidean(cloud, u, v);
    }
}

// --- Greedy pair selection ---

void greedy_pairs_from_edges(const std::vector<std::pair<int, int>>& edges,
                             const std::vector<float>& costs,
                             const int count,
                             const int max_pairs,
                             std::vector<size_t>& order,
                             std::vector<uint8_t>& used,
                             std::vector<std::pair<int, int>>& pairs) {
    order.clear();
    order.reserve(edges.size());
    for (size_t i = 0; i < costs.size(); ++i) {
        if (std::isfinite(costs[i]))
            order.push_back(i);
    }
    const auto cmp = [&](const size_t lhs, const size_t rhs) {
        return costs[lhs] < costs[rhs];
    };
    // Only need the top max_pairs edges; partial_sort is O(N * K) vs O(N log N)
    if (max_pairs > 0 && static_cast<size_t>(max_pairs) < order.size()) {
        std::partial_sort(order.begin(),
                          order.begin() + static_cast<ptrdiff_t>(max_pairs),
                          order.end(), cmp);
    } else {
        std::stable_sort(order.begin(), order.end(), cmp);
    }

    used.assign(static_cast<size_t>(count), uint8_t{0});
    pairs.clear();
    pairs.reserve(static_cast<size_t>(std::max(0, max_pairs)));
    for (const size_t edge_idx : order) {
        const auto [u, v] = edges[edge_idx];
        if (used[static_cast<size_t>(u)] || used[static_cast<size_t>(v)])
            continue;
        used[static_cast<size_t>(u)] = 1;
        used[static_cast<size_t>(v)] = 1;
        pairs.emplace_back(u, v);
        if (max_pairs > 0 && static_cast<int>(pairs.size()) >= max_pairs)
            break;
    }
}

// --- Resize helpers (only grow, never shrink) ---

void ensure_cloud_capacity(ActivatedCloud& cloud, const int32_t count, const int sh_dim,
                           const bool has_colors, const bool has_sh,
                           const std::vector<ExtraInfo>& src_extras) {
    cloud.count = count;
    const auto n3 = static_cast<size_t>(count) * 3;
    const auto n4 = static_cast<size_t>(count) * 4;
    const auto n1 = static_cast<size_t>(count);
    cloud.positions.resize(n3);
    cloud.scales.resize(n3);
    cloud.rotations.resize(n4);
    cloud.alphas.resize(n1);
    if (has_colors)
        cloud.colors.resize(n3);
    if (has_sh && sh_dim > 0)
        cloud.sh.resize(static_cast<size_t>(count) * static_cast<size_t>(sh_dim));
    cloud.extras.resize(src_extras.size());
    for (size_t ei = 0; ei < src_extras.size(); ++ei) {
        cloud.extras[ei].name = src_extras[ei].name;
        cloud.extras[ei].dim = src_extras[ei].dim;
        if (src_extras[ei].dim > 0)
            cloud.extras[ei].data.resize(static_cast<size_t>(count) * static_cast<size_t>(src_extras[ei].dim));
    }
}

// --- Merge pairs via moment matching ---

void merge_pairs(const ActivatedCloud& input,
                 const std::vector<std::pair<int, int>>& pairs,
                 std::vector<uint8_t>& used,
                 std::vector<int>& keep_idx,
                 ActivatedCloud& out) {
    if (pairs.empty()) {
        out = input;  // NOLINT(bugprone-unhandled-self-assignment)
        return;
    }

    used.assign(static_cast<size_t>(input.count), uint8_t{0});
    for (const auto [u, v] : pairs) {
        used[static_cast<size_t>(u)] = 1;
        used[static_cast<size_t>(v)] = 1;
    }

    keep_idx.clear();
    keep_idx.reserve(static_cast<size_t>(input.count));
    for (int i = 0; i < input.count; ++i) {
        if (!used[static_cast<size_t>(i)])
            keep_idx.push_back(i);
    }

    const int32_t out_count = static_cast<int32_t>(keep_idx.size() + pairs.size());
    const int sh_dim = input.sh_coeffs_per_point * 3;

    out.sh_coeffs_per_point = input.sh_coeffs_per_point;
    ensure_cloud_capacity(out, out_count, sh_dim,
                          !input.colors.empty(), !input.sh.empty(),
                          input.extras);

    // Copy unmerged points
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < static_cast<int>(keep_idx.size()); ++k)
        copy_point(input, keep_idx[k], out, k);

    // Merge pairs
    #pragma omp parallel for schedule(dynamic, 16)
    for (int pair_idx = 0; pair_idx < static_cast<int>(pairs.size()); ++pair_idx) {
        const auto [i, j] = pairs[static_cast<size_t>(pair_idx)];
        const auto cache_i = build_cache_entry(input, i);
        const auto cache_j = build_cache_entry(input, j);
        const size_t i3 = static_cast<size_t>(i) * 3;
        const size_t j3 = static_cast<size_t>(j) * 3;

        const float sxi = std::max(input.scales[i3 + 0], kMinScale);
        const float syi = std::max(input.scales[i3 + 1], kMinScale);
        const float szi = std::max(input.scales[i3 + 2], kMinScale);
        const float sxj = std::max(input.scales[j3 + 0], kMinScale);
        const float syj = std::max(input.scales[j3 + 1], kMinScale);
        const float szj = std::max(input.scales[j3 + 2], kMinScale);

        const float alpha_i = input.alphas[static_cast<size_t>(i)];
        const float alpha_j = input.alphas[static_cast<size_t>(j)];
        const float wi = cache_i.mass;
        const float wj = cache_j.mass;
        const float W = std::max(wi + wj, 1e-12f);

        const int out_row = static_cast<int>(keep_idx.size()) + pair_idx;
        const size_t o3 = static_cast<size_t>(out_row) * 3;
        const size_t o4 = static_cast<size_t>(out_row) * 4;

        // Weighted mean position
        out.positions[o3 + 0] = (wi * input.positions[i3 + 0] + wj * input.positions[j3 + 0]) / W;
        out.positions[o3 + 1] = (wi * input.positions[i3 + 1] + wj * input.positions[j3 + 1]) / W;
        out.positions[o3 + 2] = (wi * input.positions[i3 + 2] + wj * input.positions[j3 + 2]) / W;

        // Weighted covariance
        std::array<float, 9> sig_i{}, sig_j{};
        math::sigma_from_rot_var(cache_i.R, sxi * sxi, syi * syi, szi * szi, sig_i);
        math::sigma_from_rot_var(cache_j.R, sxj * sxj, syj * syj, szj * szj, sig_j);

        // Add displacement outer products
        const float dix = input.positions[i3 + 0] - out.positions[o3 + 0];
        const float diy = input.positions[i3 + 1] - out.positions[o3 + 1];
        const float diz = input.positions[i3 + 2] - out.positions[o3 + 2];
        const float djx = input.positions[j3 + 0] - out.positions[o3 + 0];
        const float djy = input.positions[j3 + 1] - out.positions[o3 + 1];
        const float djz = input.positions[j3 + 2] - out.positions[o3 + 2];

        sig_i[0] += dix * dix; sig_i[1] += dix * diy; sig_i[2] += dix * diz;
        sig_i[3] += diy * dix; sig_i[4] += diy * diy; sig_i[5] += diy * diz;
        sig_i[6] += diz * dix; sig_i[7] += diz * diy; sig_i[8] += diz * diz;
        sig_j[0] += djx * djx; sig_j[1] += djx * djy; sig_j[2] += djx * djz;
        sig_j[3] += djy * djx; sig_j[4] += djy * djy; sig_j[5] += djy * djz;
        sig_j[6] += djz * djx; sig_j[7] += djz * djy; sig_j[8] += djz * djz;

        std::array<float, 9> sigma{};
        for (int a = 0; a < 9; ++a)
            sigma[static_cast<size_t>(a)] = (wi * sig_i[static_cast<size_t>(a)] + wj * sig_j[static_cast<size_t>(a)]) / W;
        sigma[1] = sigma[3] = 0.5f * (sigma[1] + sigma[3]);
        sigma[2] = sigma[6] = 0.5f * (sigma[2] + sigma[6]);
        sigma[5] = sigma[7] = 0.5f * (sigma[5] + sigma[7]);
        sigma[0] += kEpsCov;
        sigma[4] += kEpsCov;
        sigma[8] += kEpsCov;

        // Decompose covariance -> scale + rotation
        std::array<float, 3> scaling_raw{};
        std::array<float, 4> rotation{};
        math::decompose_sigma_to_raw_scale_quat(sigma, scaling_raw, rotation);

        out.scales[o3 + 0] = math::activated_scale(scaling_raw[0]);
        out.scales[o3 + 1] = math::activated_scale(scaling_raw[1]);
        out.scales[o3 + 2] = math::activated_scale(scaling_raw[2]);
        out.rotations[o4 + 0] = rotation[0];
        out.rotations[o4 + 1] = rotation[1];
        out.rotations[o4 + 2] = rotation[2];
        out.rotations[o4 + 3] = rotation[3];

        // Opacity: probabilistic OR
        out.alphas[static_cast<size_t>(out_row)] = alpha_i + alpha_j - alpha_i * alpha_j;

        // Colors: mass-weighted average
        if (!input.colors.empty()) {
            const size_t ci = i3, cj = j3, co = o3;
            for (int k = 0; k < 3; ++k)
                out.colors[co + static_cast<size_t>(k)] =
                    (wi * input.colors[ci + static_cast<size_t>(k)] +
                     wj * input.colors[cj + static_cast<size_t>(k)]) / W;
        }

        // SH: mass-weighted average
        if (sh_dim > 0) {
            const size_t si = static_cast<size_t>(i) * static_cast<size_t>(sh_dim);
            const size_t sj = static_cast<size_t>(j) * static_cast<size_t>(sh_dim);
            const size_t so = static_cast<size_t>(out_row) * static_cast<size_t>(sh_dim);
            for (int k = 0; k < sh_dim; ++k)
                out.sh[so + static_cast<size_t>(k)] =
                    (wi * input.sh[si + static_cast<size_t>(k)] +
                     wj * input.sh[sj + static_cast<size_t>(k)]) / W;
        }

        // Extras: mass-weighted average
        for (size_t ei = 0; ei < out.extras.size(); ++ei) {
            const int dim = out.extras[ei].dim;
            if (dim <= 0) continue;
            const auto& src_data = input.extras[ei].data;
            auto& dst_data = out.extras[ei].data;
            const size_t e_i = static_cast<size_t>(i) * static_cast<size_t>(dim);
            const size_t e_j = static_cast<size_t>(j) * static_cast<size_t>(dim);
            const size_t e_o = static_cast<size_t>(out_row) * static_cast<size_t>(dim);
            for (int k = 0; k < dim; ++k)
                dst_data[e_o + static_cast<size_t>(k)] =
                    (wi * src_data[e_i + static_cast<size_t>(k)] +
                     wj * src_data[e_j + static_cast<size_t>(k)]) / W;
        }
    }
}

// --- Count helpers ---

int32_t target_count_for(const int32_t input_count, const double ratio) {
    const double clamped_ratio = std::clamp(ratio, 0.0, 1.0);
    return std::clamp(
        static_cast<int32_t>(std::ceil(static_cast<double>(input_count) * clamped_ratio)),
        int32_t{1},
        std::max(int32_t{1}, input_count));
}

int32_t pass_merge_cap_for(const int32_t input_count, const double merge_cap) {
    const double clamped_merge_cap = std::clamp(merge_cap, 0.01, 0.5);
    return std::max(int32_t{1}, static_cast<int32_t>(clamped_merge_cap * static_cast<double>(input_count)));
}

float progress_for_count(const int32_t input_count, const int32_t target_count, const int32_t current_count) {
    if (input_count <= target_count)
        return 0.95f;
    const float denom = static_cast<float>(std::max(1, input_count - target_count));
    const float numer = static_cast<float>(std::clamp(input_count - current_count, int32_t{0}, input_count - target_count));
    return 0.10f + 0.85f * (numer / denom);
}

// --- Main simplify implementation ---

gf::Expected<gf::GaussianCloudIR> simplify_impl(
    const gf::GaussianCloudIR& input,
    SimplifyAuditTrail* audit,
    const SimplifyOptions& options,
    ProgressCallback progress) {
    try {
        if (input.numPoints <= 0 || input.positions.empty())
            return gf::MakeError("Splat simplify: input is empty");

        if (!report_progress(progress, 0.0f, "Activating"))
            return gf::MakeError("Cancelled");

        const int32_t input_count = input.numPoints;

        // Activate
        ActivatedCloud current = activate_from_ir(input);

        if (audit) {
            audit->original_count = input_count;
        }

        // SH degree reduction (early, to save merge work)
        reduce_sh_degree(current, options.target_sh_degree);

        // Opacity pruning
        if (!report_progress(progress, 0.05f, "Pruning opacity"))
            return gf::MakeError("Cancelled");

        std::vector<int32_t> prune_survivors;
        current = prune_by_opacity(current, options.opacity_prune_threshold, prune_survivors);

        if (audit) {
            audit->post_prune_count = current.count;
            audit->prune_survivor_ids = std::move(prune_survivors);
        }

        if (current.count == 0)
            return gf::MakeError("Splat simplify: no visible gaussians after pruning");

        // Statistical outlier removal
        if (options.sor_nb_neighbors > 0 && current.count > 1) {
            if (!report_progress(progress, 0.08f, "Statistical outlier removal"))
                return gf::MakeError("Cancelled");

            std::vector<int32_t> sor_survivors;
            current = prune_by_statistical_outlier(
                current, options.sor_nb_neighbors, options.sor_std_ratio, sor_survivors);

            if (audit) {
                audit->sor_removed = audit->post_prune_count - current.count;
                audit->post_sor_count = current.count;
            }

            if (current.count == 0)
                return gf::MakeError("Splat simplify: no gaussians left after statistical outlier removal");
        }

        const int32_t target_count = target_count_for(input_count, options.ratio);
        if (current.count <= target_count) {
            report_progress(progress, 1.0f, "Complete");
            auto ir = deactivate_to_ir(current, input.meta);
            if (audit) audit->final_count = ir.numPoints;
            return ir;
        }

        const int32_t pass_merge_cap = pass_merge_cap_for(input_count, options.merge_cap);

        // Scratch buffers (reused across passes)
        std::vector<CacheEntry> cache;
        std::vector<float> costs;
        std::vector<size_t> order;
        std::vector<uint8_t> used_rows;
        std::vector<std::pair<int, int>> pairs;
        std::vector<int> keep_idx;
        ActivatedCloud scratch;  // Reused across passes to avoid realloc

        int pass = 0;
        while (current.count > target_count) {
            const float pass_progress = progress_for_count(input_count, target_count, current.count);
            const std::string pass_prefix = "Pass " + std::to_string(pass + 1) + ": ";

            // Build kNN graph
            if (!report_progress(progress, pass_progress, pass_prefix + "building kNN graph"))
                return gf::MakeError("Cancelled");
            const auto edges = knn::build_knn_union_edges(
                current.positions.data(), current.count, options.knn_k);
            if (edges.empty())
                return gf::MakeError(
                    "Splat simplify stalled at " + std::to_string(current.count) +
                    " gaussians (target " + std::to_string(target_count) + ")");

            // Compute edge costs
            if (!report_progress(progress, pass_progress + 0.01f, pass_prefix + "computing edge costs"))
                return gf::MakeError("Cancelled");
            compute_edge_costs(current, edges, costs);

            // Select pairs
            if (!report_progress(progress, pass_progress + 0.02f, pass_prefix + "selecting pairs"))
                return gf::MakeError("Cancelled");
            const int32_t merges_needed = current.count - target_count;
            const int32_t max_pairs_this_pass = merges_needed > 0 ? std::min(merges_needed, pass_merge_cap) : 0;
            greedy_pairs_from_edges(edges, costs, current.count, max_pairs_this_pass,
                                    order, used_rows, pairs);
            if (pairs.empty())
                return gf::MakeError(
                    "Splat simplify stalled at " + std::to_string(current.count) +
                    " gaussians (target " + std::to_string(target_count) + ")");

            // Merge
            if (!report_progress(progress, pass_progress + 0.03f,
                                 pass_prefix + "merging " + std::to_string(pairs.size()) + " pairs"))
                return gf::MakeError("Cancelled");
            merge_pairs(current, pairs, used_rows, keep_idx, scratch);
            std::swap(current, scratch);

            // Record merges in audit trail
            if (audit) {
                for (const auto [u, v] : pairs)
                    audit->merges.push_back({u, v, pass});
            }

            ++pass;
        }

        report_progress(progress, 1.0f, "Complete");
        auto ir = deactivate_to_ir(current, input.meta);
        if (audit) audit->final_count = ir.numPoints;
        return ir;
    } catch (const std::exception& e) {
        return gf::MakeError(std::string("Splat simplify failed: ") + e.what());
    }
}

} // anonymous namespace

gf::Expected<gf::GaussianCloudIR> simplify(
    const gf::GaussianCloudIR& input,
    const SimplifyOptions& options,
    ProgressCallback progress) {
    return simplify_impl(input, nullptr, options, std::move(progress));
}

gf::Expected<gf::GaussianCloudIR> simplify_with_audit(
    const gf::GaussianCloudIR& input,
    SimplifyAuditTrail& audit,
    const SimplifyOptions& options,
    ProgressCallback progress) {
    return simplify_impl(input, &audit, options, std::move(progress));
}

} // namespace gs
