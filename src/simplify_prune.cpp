// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Pruning/filtering functions for Gaussian splat point clouds.

#include "simplify_detail.h"
#include "simplify_knn.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gs::detail {

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

} // namespace gs::detail
