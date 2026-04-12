// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Merge-related functions for Gaussian splat simplification.

#include "simplify_detail.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gs::detail {

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
                        const std::vector<float>& point_weights,
                        std::vector<float>& costs) {
    costs.assign(edges.size(), std::numeric_limits<float>::infinity());
    const bool has_weights = !point_weights.empty();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < edges.size(); ++i) {
        const auto [u, v] = edges[i];
        float cost = compute_edge_cost_euclidean(cloud, u, v);
        if (has_weights) {
            cost *= 0.5f * (point_weights[static_cast<size_t>(u)] +
                            point_weights[static_cast<size_t>(v)]);
        }
        costs[i] = cost;
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

} // namespace gs::detail
