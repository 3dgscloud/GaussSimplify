// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <string>
#include <vector>
#include <cstdint>

namespace gs {

struct AABBRegion {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
};

struct SimplifyOptions {
    double ratio = 0.1;                    // Target count as fraction of input (0.0-1.0)
    int knn_k = 16;                        // kNN neighbors for merge candidate graph
    double merge_cap = 0.5;                // Max fraction of points merged per pass (0.01-0.5)
    float opacity_prune_threshold = 0.1f;  // Opacity pruning threshold
    int target_sh_degree = -1;             // Target SH degree (-1=keep, 0-3=reduce)
    int sor_nb_neighbors = 0;              // Statistical outlier removal: kNN neighbors (0=disabled)
    float sor_std_ratio = 2.0f;            // Statistical outlier removal: std multiplier threshold
    float keep_weight = 3.0f;              // Region cost multiplier: higher = points inside keep_regions are less likely to merge.
                                          //   1.0 = no bias (inside and outside treated equally)
                                          //   >1.0 = protect points inside boxes (they merge last)
                                          //   <1.0 = preferentially merge points inside boxes
                                          //   Typical range: 2.0-10.0 for visible effect.
                                          //   The total point count is still controlled by ratio; weight only biases merge priority.
    std::vector<AABBRegion> keep_regions;  // AABB regions to preserve. Empty = no region bias.
};

using ProgressCallback = std::function<bool(float progress, const std::string& stage)>;

struct MergeRecord {
    int32_t left;
    int32_t right;
    int32_t pass;
};

struct SimplifyAuditTrail {
    int32_t original_count = 0;
    int32_t post_prune_count = 0;
    int32_t post_sor_count = 0;
    int32_t sor_removed = 0;
    int32_t final_count = 0;
    std::vector<int32_t> prune_survivor_ids;
    std::vector<MergeRecord> merges;
};

} // namespace gs
