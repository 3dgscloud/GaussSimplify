// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gs/simplify.h"

#include "gf/core/errors.h"
#include "gf/core/gauss_ir.h"
#include "simplify_detail.h"
#include "simplify_knn.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace gs {

using namespace detail;

namespace {

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
