// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// kNN graph builder for Gaussian splat simplification.
// Uses nanoflann KD-tree with OpenMP parallelism.

#pragma once

#include "nanoflann.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gs::knn {

struct PointCloudAdaptor {
    const float* points = nullptr;
    size_t num_points = 0;

    [[nodiscard]] inline size_t kdtree_get_point_count() const { return num_points; }
    [[nodiscard]] inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return static_cast<double>(points[idx * 3 + dim]);
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>,
    PointCloudAdaptor,
    3>;

[[nodiscard]] inline std::vector<std::pair<int, int>> build_knn_union_edges(
    const float* positions, const int count, const int knn_k) {
    if (count <= 1 || knn_k <= 0)
        return {};

    const int k_eff = std::min(std::max(1, knn_k), std::max(1, count - 1));
    PointCloudAdaptor cloud{positions, static_cast<size_t>(count)};
    KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

#ifdef _OPENMP
    const int n_threads = omp_get_max_threads();
#else
    const int n_threads = 1;
#endif

    const size_t estimated_edges = static_cast<size_t>(count) * static_cast<size_t>(k_eff);
    std::vector<std::vector<std::uint64_t>> thread_keys(static_cast<size_t>(n_threads));
    for (auto& tk : thread_keys)
        tk.reserve(estimated_edges / static_cast<size_t>(n_threads) + 64);

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        auto& edge_keys = thread_keys[static_cast<size_t>(tid)];

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < count; ++i) {
            const size_t query_count = static_cast<size_t>(std::min(count, k_eff + 1));
            std::vector<size_t> ret_indices(query_count, 0);
            std::vector<double> out_dists_sqr(query_count, 0.0);
            nanoflann::KNNResultSet<double> result_set(query_count);
            result_set.init(ret_indices.data(), out_dists_sqr.data());
            const double query[3] = {
                static_cast<double>(positions[static_cast<size_t>(i) * 3 + 0]),
                static_cast<double>(positions[static_cast<size_t>(i) * 3 + 1]),
                static_cast<double>(positions[static_cast<size_t>(i) * 3 + 2]),
            };
            index.findNeighbors(result_set, query, nanoflann::SearchParameters(0.0f, true));

            const size_t take = std::min(static_cast<size_t>(k_eff),
                result_set.size() > 0 ? result_set.size() - 1 : size_t{0});
            for (size_t j = 0; j < take; ++j) {
                const int neighbor = static_cast<int>(ret_indices[j + 1]);
                if (neighbor < 0 || neighbor == i)
                    continue;
                const int u = std::min(i, neighbor);
                const int v = std::max(i, neighbor);
                edge_keys.push_back(
                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
                    static_cast<std::uint32_t>(v));
            }
        }
    }

    // Merge and deduplicate
    size_t total = 0;
    for (const auto& tk : thread_keys)
        total += tk.size();

    std::vector<std::uint64_t> edge_keys;
    edge_keys.reserve(total);
    for (const auto& tk : thread_keys)
        edge_keys.insert(edge_keys.end(), tk.begin(), tk.end());

    std::sort(edge_keys.begin(), edge_keys.end());
    edge_keys.erase(std::unique(edge_keys.begin(), edge_keys.end()), edge_keys.end());

    std::vector<std::pair<int, int>> edges;
    edges.reserve(edge_keys.size());
    for (const std::uint64_t key : edge_keys) {
        const int u = static_cast<int>(key >> 32);
        const int v = static_cast<int>(key & 0xffffffffU);
        edges.emplace_back(u, v);
    }
    return edges;
}

} // namespace gs::knn
