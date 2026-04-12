// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gs/simplify.h"

#include "gf/core/gauss_ir.h"
#include "gf/core/validate.h"
#include "simplify_math.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float kFloatTol = 1e-4f;

struct PointSpec {
    std::array<float, 3> position = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> log_scale = {0.0f, 0.0f, 0.0f};
    std::array<float, 4> rotation = {1.0f, 0.0f, 0.0f, 0.0f};
    float alpha = 0.5f; // Activated opacity, not logit.
    std::array<float, 3> color = {0.0f, 0.0f, 0.0f};
};

float dot_quat(const std::array<float, 4>& lhs, const std::array<float, 4>& rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2] + lhs[3] * rhs[3];
}

float activated_alpha_from_ir(const gf::GaussianCloudIR& ir, const int index) {
    return gs::math::sigmoid(ir.alphas[static_cast<size_t>(index)]);
}

float activated_scale_from_ir(const gf::GaussianCloudIR& ir, const int index, const int dim) {
    return gs::math::activated_scale(ir.scales[static_cast<size_t>(index) * 3 + static_cast<size_t>(dim)]);
}

gf::GaussianCloudIR make_cloud(const std::vector<PointSpec>& points,
                               const std::vector<float>& extra_scalar = {}) {
    gf::GaussianCloudIR ir;
    ir.numPoints = static_cast<int32_t>(points.size());
    ir.meta.shDegree = 0;
    ir.positions.resize(points.size() * 3);
    ir.scales.resize(points.size() * 3);
    ir.rotations.resize(points.size() * 4);
    ir.alphas.resize(points.size());
    ir.colors.resize(points.size() * 3);

    for (size_t i = 0; i < points.size(); ++i) {
        const size_t i3 = i * 3;
        const size_t i4 = i * 4;
        const auto& point = points[i];

        ir.positions[i3 + 0] = point.position[0];
        ir.positions[i3 + 1] = point.position[1];
        ir.positions[i3 + 2] = point.position[2];

        ir.scales[i3 + 0] = point.log_scale[0];
        ir.scales[i3 + 1] = point.log_scale[1];
        ir.scales[i3 + 2] = point.log_scale[2];

        ir.rotations[i4 + 0] = point.rotation[0];
        ir.rotations[i4 + 1] = point.rotation[1];
        ir.rotations[i4 + 2] = point.rotation[2];
        ir.rotations[i4 + 3] = point.rotation[3];

        ir.alphas[i] = gs::math::logit_from_alpha(point.alpha);

        ir.colors[i3 + 0] = point.color[0];
        ir.colors[i3 + 1] = point.color[1];
        ir.colors[i3 + 2] = point.color[2];
    }

    if (!extra_scalar.empty())
        ir.extras.emplace("feature", extra_scalar);

    const auto validation = gf::ValidateBasic(ir, true);
    EXPECT_TRUE(validation.message.empty()) << validation.message;
    return ir;
}

gf::GaussianCloudIR expect_ok(gf::Expected<gf::GaussianCloudIR> result) {
    EXPECT_TRUE(result.ok()) << result.error().message;
    return std::move(result.value());
}

std::vector<std::pair<int32_t, int32_t>> merge_pairs_by_pass(const gs::SimplifyAuditTrail& audit,
                                                             const int32_t pass) {
    std::vector<std::pair<int32_t, int32_t>> pairs;
    for (const auto& merge : audit.merges) {
        if (merge.pass == pass)
            pairs.emplace_back(std::min(merge.left, merge.right), std::max(merge.left, merge.right));
    }
    std::sort(pairs.begin(), pairs.end());
    return pairs;
}

std::map<int32_t, int> merge_count_per_pass(const gs::SimplifyAuditTrail& audit) {
    std::map<int32_t, int> counts;
    for (const auto& merge : audit.merges)
        ++counts[merge.pass];
    return counts;
}

} // namespace

// --- Math function tests ---

TEST(SimplifyMath, SigmoidLogitRoundTrip) {
    const float probabilities[] = {1e-8f, 1e-4f, 0.25f, 0.5f, 0.9f, 0.9999999f};
    for (const float p : probabilities) {
        const float expected = gs::math::clamp_prob(p);
        const float actual = gs::math::sigmoid(gs::math::logit_from_alpha(p));
        EXPECT_NEAR(actual, expected, 1e-6f);
    }
}

TEST(SimplifyMath, QuaternionMatrixRoundTrip) {
    std::array<float, 4> q = {0.8253356f, 0.1509070f, 0.3018140f, 0.4527210f};
    const float norm = std::sqrt(dot_quat(q, q));
    for (float& v : q)
        v /= norm;

    std::array<float, 9> rotation{};
    std::array<float, 4> reconstructed{};
    gs::math::quat_to_rotmat(q[0], q[1], q[2], q[3], rotation);
    gs::math::rotmat_to_quat(rotation, reconstructed);

    const float alignment = std::abs(dot_quat(q, reconstructed));
    EXPECT_NEAR(alignment, 1.0f, 1e-5f);
}

TEST(SimplifyMath, EigenDecompositionIdentity) {
    // Identity matrix should have eigenvalues [1, 1, 1]
    std::array<float, 9> identity = {1,0,0, 0,1,0, 0,0,1};
    const auto eig = gs::math::eigen_symmetric_3x3_jacobi(identity);
    EXPECT_NEAR(eig.values[0], 1.0f, 1e-5f);
    EXPECT_NEAR(eig.values[1], 1.0f, 1e-5f);
    EXPECT_NEAR(eig.values[2], 1.0f, 1e-5f);
}

TEST(SimplifyMath, EigenDecompositionDiagonal) {
    // Diagonal [4, 2, 1] sorted descending
    std::array<float, 9> A = {4,0,0, 0,2,0, 0,0,1};
    const auto eig = gs::math::eigen_symmetric_3x3_jacobi(A);
    EXPECT_NEAR(eig.values[0], 4.0f, 1e-4f);
    EXPECT_NEAR(eig.values[1], 2.0f, 1e-4f);
    EXPECT_NEAR(eig.values[2], 1.0f, 1e-4f);
}

// --- Simplify algorithm tests ---

TEST(Simplify, PrunesByOpacityWithoutMergingWhenTargetAlreadyMet) {
    const auto input = make_cloud({
        {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f, .color = {1.0f, 0.0f, 0.0f}},
        {.position = {1.0f, 0.0f, 0.0f}, .alpha = 0.2f, .color = {0.0f, 1.0f, 0.0f}},
        {.position = {2.0f, 0.0f, 0.0f}, .alpha = 0.8f, .color = {0.0f, 0.0f, 1.0f}},
    });

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = 1.0;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.5f;

    const auto& output = expect_ok(gs::simplify_with_audit(input, audit, options));

    EXPECT_EQ(audit.original_count, 3);
    EXPECT_EQ(audit.post_prune_count, 2);
    EXPECT_EQ(audit.final_count, 2);
    EXPECT_EQ(audit.prune_survivor_ids, (std::vector<int32_t>{0, 2}));
    EXPECT_TRUE(audit.merges.empty());

    const auto validation = gf::ValidateBasic(output, true);
    EXPECT_TRUE(validation.message.empty()) << validation.message;
    EXPECT_EQ(output.numPoints, 2);
    EXPECT_NEAR(output.positions[0], 0.0f, kFloatTol);
    EXPECT_NEAR(output.positions[3], 2.0f, kFloatTol);
    EXPECT_NEAR(activated_alpha_from_ir(output, 0), 0.9f, kFloatTol);
    EXPECT_NEAR(activated_alpha_from_ir(output, 1), 0.8f, kFloatTol);
}

TEST(Simplify, MergesTwoPointsWithCorrectPositionColorOpacityAndScale) {
    const auto input = make_cloud(
        {
            {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.5f, .color = {0.0f, 0.2f, 0.4f}},
            {.position = {2.0f, 0.0f, 0.0f}, .alpha = 0.5f, .color = {1.0f, 0.6f, 0.8f}},
        },
        {10.0f, 20.0f});

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = 0.5;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;

    const auto& output = expect_ok(gs::simplify_with_audit(input, audit, options));

    ASSERT_EQ(audit.merges.size(), 1u);
    EXPECT_EQ(audit.merges[0].left, 0);
    EXPECT_EQ(audit.merges[0].right, 1);
    EXPECT_EQ(audit.merges[0].pass, 0);
    EXPECT_EQ(audit.final_count, 1);

    const auto validation = gf::ValidateBasic(output, true);
    EXPECT_TRUE(validation.message.empty()) << validation.message;
    EXPECT_EQ(output.numPoints, 1);

    // Position: mass-weighted center, equal mass → midpoint
    EXPECT_NEAR(output.positions[0], 1.0f, kFloatTol);
    EXPECT_NEAR(output.positions[1], 0.0f, kFloatTol);
    EXPECT_NEAR(output.positions[2], 0.0f, kFloatTol);

    // Color: mass-weighted average, equal mass → midpoint
    EXPECT_NEAR(output.colors[0], 0.5f, kFloatTol);
    EXPECT_NEAR(output.colors[1], 0.4f, kFloatTol);
    EXPECT_NEAR(output.colors[2], 0.6f, kFloatTol);

    // Extras: mass-weighted average → 15.0
    const auto feature_it = output.extras.find("feature");
    ASSERT_NE(feature_it, output.extras.end());
    EXPECT_EQ(feature_it->second.size(), 1u);
    EXPECT_NEAR(feature_it->second[0], 15.0f, kFloatTol);

    // Opacity: probabilistic OR  0.5 + 0.5 - 0.25 = 0.75
    EXPECT_NEAR(activated_alpha_from_ir(output, 0), 0.75f, kFloatTol);

    // Scale: Two identical points (scale=1, rotation=I) at x=0 and x=2,
    // merged center at x=1. Covariance per point:
    //   sigma_i = R*diag(s^2)*R^T + displacement*displacement^T
    //   = diag(1,1,1) + [1,0,0; 0,0,0; 0,0,0]   (displacement from center)
    // Weighted merge: sigma = diag(1,1,1) + [1,0,0; 0,0,0; 0,0,0] + eps
    // Eigenvalues: 2+eps, 1+eps, 1+eps  →  sqrt ≈ √2, 1, 1
    EXPECT_NEAR(activated_scale_from_ir(output, 0, 0), std::sqrt(2.0f), kFloatTol);
    EXPECT_NEAR(activated_scale_from_ir(output, 0, 1), 1.0f, kFloatTol);
    EXPECT_NEAR(activated_scale_from_ir(output, 0, 2), 1.0f, kFloatTol);
}

TEST(Simplify, SelectsNearestDisjointPairsInSinglePass) {
    const auto input = make_cloud({
        {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.8f},
        {.position = {0.1f, 0.0f, 0.0f}, .alpha = 0.8f},
        {.position = {10.0f, 0.0f, 0.0f}, .alpha = 0.8f},
        {.position = {10.3f, 0.0f, 0.0f}, .alpha = 0.8f},
    });

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = 0.5;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;

    expect_ok(gs::simplify_with_audit(input, audit, options));

    EXPECT_EQ(audit.original_count, 4);
    EXPECT_EQ(audit.post_prune_count, 4);
    EXPECT_EQ(audit.final_count, 2);
    const auto pass_zero_pairs = merge_pairs_by_pass(audit, 0);
    EXPECT_EQ(pass_zero_pairs,
              (std::vector<std::pair<int32_t, int32_t>>{{0, 1}, {2, 3}}));
}

TEST(Simplify, PruningCanReachTargetWithoutAnyMergePasses) {
    const auto input = make_cloud({
        {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {1.0f, 0.0f, 0.0f}, .alpha = 0.8f},
        {.position = {2.0f, 0.0f, 0.0f}, .alpha = 0.2f},
        {.position = {3.0f, 0.0f, 0.0f}, .alpha = 0.1f},
    });

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = 0.5;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.5f;

    expect_ok(gs::simplify_with_audit(input, audit, options));

    EXPECT_EQ(audit.post_prune_count, 2);
    EXPECT_EQ(audit.final_count, 2);
    EXPECT_EQ(audit.prune_survivor_ids, (std::vector<int32_t>{0, 1}));
    EXPECT_TRUE(audit.merges.empty());
}

TEST(Simplify, RatioIsClampedToKeepAtLeastOnePoint) {
    const auto input = make_cloud({
        {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {1.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {2.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {3.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {4.0f, 0.0f, 0.0f}, .alpha = 0.9f},
    });

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = -1.0;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;

    const auto& output = expect_ok(gs::simplify_with_audit(input, audit, options));

    EXPECT_EQ(audit.final_count, 1);
    EXPECT_EQ(output.numPoints, 1);
}

TEST(Simplify, RatioAboveOneKeepsAllSurvivors) {
    const auto input = make_cloud({
        {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {1.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {2.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {3.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {4.0f, 0.0f, 0.0f}, .alpha = 0.9f},
    });

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = 2.0;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;

    const auto& output = expect_ok(gs::simplify_with_audit(input, audit, options));

    EXPECT_EQ(audit.post_prune_count, 5);
    EXPECT_EQ(audit.final_count, 5);
    EXPECT_TRUE(audit.merges.empty());
    EXPECT_EQ(output.numPoints, 5);
}

TEST(Simplify, ZeroMergeCapFallsBackToOneMergePerPass) {
    const auto input = make_cloud({
        {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {1.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {2.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {3.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {4.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {5.0f, 0.0f, 0.0f}, .alpha = 0.9f},
    });

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = 0.0;
    options.knn_k = 1;
    options.merge_cap = 0.0;
    options.opacity_prune_threshold = 0.0f;

    expect_ok(gs::simplify_with_audit(input, audit, options));

    EXPECT_EQ(audit.final_count, 1);
    ASSERT_EQ(audit.merges.size(), 5u);

    const auto pass_counts = merge_count_per_pass(audit);
    ASSERT_EQ(pass_counts.size(), 5u);
    for (const auto& [pass, count] : pass_counts) {
        EXPECT_EQ(count, 1) << "pass=" << pass << " should have exactly 1 merge";
    }
}

// --- New rigorous tests ---

TEST(Simplify, AsymmetricMergeCorrectlyWeightedByMass) {
    // Two points with different alphas -> different masses.
    // mass ratio ~ 0.3 : 0.8 = 3 : 8
    // w0/W ~ 3/11, w1/W ~ 8/11
    const auto input = make_cloud(
        {
            {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.3f, .color = {1.0f, 0.0f, 0.0f}},
            {.position = {2.0f, 0.0f, 0.0f}, .alpha = 0.8f, .color = {0.0f, 0.0f, 1.0f}},
        },
        {10.0f, 20.0f});

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = 0.5;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;

    const auto output = expect_ok(gs::simplify_with_audit(input, audit, options));

    ASSERT_EQ(audit.merges.size(), 1u);
    EXPECT_EQ(audit.final_count, 1);
    EXPECT_EQ(output.numPoints, 1);

    // Position: mass-weighted center at 16/11
    EXPECT_NEAR(output.positions[0], 16.0f / 11.0f, kFloatTol);
    EXPECT_NEAR(output.positions[1], 0.0f, kFloatTol);
    EXPECT_NEAR(output.positions[2], 0.0f, kFloatTol);

    // Color: mass-weighted
    EXPECT_NEAR(output.colors[0], 3.0f / 11.0f, kFloatTol);
    EXPECT_NEAR(output.colors[1], 0.0f, kFloatTol);
    EXPECT_NEAR(output.colors[2], 8.0f / 11.0f, kFloatTol);

    // Extras: mass-weighted -> (3/11)*10 + (8/11)*20 = 190/11
    const auto feature_it = output.extras.find("feature");
    ASSERT_NE(feature_it, output.extras.end());
    EXPECT_NEAR(feature_it->second[0], 190.0f / 11.0f, kFloatTol);

    // Opacity: probabilistic OR  0.3 + 0.8 - 0.24 = 0.86
    EXPECT_NEAR(activated_alpha_from_ir(output, 0), 0.86f, kFloatTol);

    // Scale: weighted covariance gives sigma_xx = 2387/1331 + eps
    // sigma_yy = sigma_zz = 1 + eps (displacement along x only)
    EXPECT_NEAR(activated_scale_from_ir(output, 0, 0),
                std::sqrt(2387.0f / 1331.0f), kFloatTol);
    EXPECT_NEAR(activated_scale_from_ir(output, 0, 1), 1.0f, kFloatTol);
    EXPECT_NEAR(activated_scale_from_ir(output, 0, 2), 1.0f, kFloatTol);
}

TEST(Simplify, MergesWithDifferentScalesAtSamePosition) {
    // Two points at same position, different scales -> unequal mass.
    // Point 0: scale=(4,1,1), mass ~ 0.5*4=2.0
    // Point 1: scale=(1,1,1), mass ~ 0.5*1=0.5
    // w0/W ~ 4/5, w1/W ~ 1/5
    // sigma = 0.8*diag(16,1,1) + 0.2*diag(1,1,1) + eps*I = diag(13+eps, 1+eps, 1+eps)
    const auto input = make_cloud({
        {.position = {0.0f, 0.0f, 0.0f},
         .log_scale = {std::log(4.0f), 0.0f, 0.0f},
         .alpha = 0.5f,
         .color = {1.0f, 0.0f, 0.0f}},
        {.position = {0.0f, 0.0f, 0.0f},
         .log_scale = {0.0f, 0.0f, 0.0f},
         .alpha = 0.5f,
         .color = {0.0f, 0.0f, 1.0f}},
    });

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = 0.5;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;

    const auto output = expect_ok(gs::simplify_with_audit(input, audit, options));

    ASSERT_EQ(audit.merges.size(), 1u);
    EXPECT_EQ(output.numPoints, 1);

    const auto validation = gf::ValidateBasic(output, true);
    EXPECT_TRUE(validation.message.empty()) << validation.message;

    // Position: same position -> no displacement -> center unchanged
    EXPECT_NEAR(output.positions[0], 0.0f, kFloatTol);
    EXPECT_NEAR(output.positions[1], 0.0f, kFloatTol);
    EXPECT_NEAR(output.positions[2], 0.0f, kFloatTol);

    // Color: mass-weighted -> (0.8*1+0.2*0, 0, 0.8*0+0.2*1) = (0.8, 0, 0.2)
    EXPECT_NEAR(output.colors[0], 0.8f, kFloatTol);
    EXPECT_NEAR(output.colors[1], 0.0f, kFloatTol);
    EXPECT_NEAR(output.colors[2], 0.2f, kFloatTol);

    // Opacity: 0.5 + 0.5 - 0.25 = 0.75
    EXPECT_NEAR(activated_alpha_from_ir(output, 0), 0.75f, kFloatTol);

    // Scale: sqrt(13+eps) ~ 3.6056, 1, 1
    EXPECT_NEAR(activated_scale_from_ir(output, 0, 0), std::sqrt(13.0f), kFloatTol);
    EXPECT_NEAR(activated_scale_from_ir(output, 0, 1), 1.0f, kFloatTol);
    EXPECT_NEAR(activated_scale_from_ir(output, 0, 2), 1.0f, kFloatTol);
}

TEST(Simplify, MultiPassMergeWithMergeCap) {
    // 8 points in 4 clusters of 2 with non-uniform inter-cluster spacing.
    // This avoids kNN tie-breaking issues after merges shift positions.
    // ratio=0.25 -> target=2. merge_cap=0.3 -> cap=max(1,int(0.3*8))=2.
    // Pass 0: 8 -> 6 (2 merges of closest clusters)
    // Pass 1: 6 -> 4 (2 merges of next closest)
    // Pass 2: 4 -> 2 (2 merges of remaining)
    const auto input = make_cloud({
        {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.8f},
        {.position = {0.01f, 0.0f, 0.0f}, .alpha = 0.8f},
        {.position = {10.0f, 0.0f, 0.0f}, .alpha = 0.8f},
        {.position = {10.01f, 0.0f, 0.0f}, .alpha = 0.8f},
        {.position = {100.0f, 0.0f, 0.0f}, .alpha = 0.8f},
        {.position = {100.01f, 0.0f, 0.0f}, .alpha = 0.8f},
        {.position = {1000.0f, 0.0f, 0.0f}, .alpha = 0.8f},
        {.position = {1000.01f, 0.0f, 0.0f}, .alpha = 0.8f},
    });

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = 0.25;
    options.knn_k = 1;
    options.merge_cap = 0.3;
    options.opacity_prune_threshold = 0.0f;

    expect_ok(gs::simplify_with_audit(input, audit, options));

    EXPECT_EQ(audit.original_count, 8);
    EXPECT_EQ(audit.final_count, 2);
    EXPECT_EQ(audit.merges.size(), 6u);

    const auto pass_counts = merge_count_per_pass(audit);
    EXPECT_EQ(pass_counts.size(), 3u) << "expected exactly 3 passes";
    for (const auto& [pass, count] : pass_counts) {
        EXPECT_EQ(count, 2) << "pass=" << pass << " should have exactly 2 merges";
    }
}

TEST(Simplify, SinglePointPassesThroughUnchanged) {
    const auto input = make_cloud({
        {.position = {1.0f, 2.0f, 3.0f},
         .log_scale = {std::log(0.5f), 0.0f, 0.0f},
         .rotation = {1.0f, 0.0f, 0.0f, 0.0f},
         .alpha = 0.9f,
         .color = {0.1f, 0.2f, 0.3f}},
    });

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = 0.5;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;

    const auto output = expect_ok(gs::simplify_with_audit(input, audit, options));

    EXPECT_EQ(audit.original_count, 1);
    EXPECT_EQ(audit.post_prune_count, 1);
    EXPECT_EQ(audit.final_count, 1);
    EXPECT_TRUE(audit.merges.empty());

    EXPECT_EQ(output.numPoints, 1);
    EXPECT_NEAR(output.positions[0], 1.0f, kFloatTol);
    EXPECT_NEAR(output.positions[1], 2.0f, kFloatTol);
    EXPECT_NEAR(output.positions[2], 3.0f, kFloatTol);
    EXPECT_NEAR(activated_alpha_from_ir(output, 0), 0.9f, kFloatTol);
    EXPECT_NEAR(output.colors[0], 0.1f, kFloatTol);
    EXPECT_NEAR(output.colors[1], 0.2f, kFloatTol);
    EXPECT_NEAR(output.colors[2], 0.3f, kFloatTol);
}

TEST(Simplify, NonIdentityRotationMergeProducesValidOutput) {
    // Two points with non-trivial rotations and different scales.
    // We verify structural validity rather than exact eigenvalues,
    // since the quaternion sign/orientation convention is subtle.
    const float cos22 = std::cos(3.14159265f / 8.0f); // cos(22.5 deg)
    const float sin22 = std::sin(3.14159265f / 8.0f); // sin(22.5 deg)

    const auto input = make_cloud({
        {.position = {0.0f, 0.0f, 0.0f},
         .log_scale = {std::log(4.0f), 0.0f, 0.0f},
         .rotation = {cos22, 0.0f, 0.0f, sin22},
         .alpha = 0.8f},
        {.position = {1.0f, 0.0f, 0.0f},
         .log_scale = {0.0f, std::log(2.0f), 0.0f},
         .rotation = {1.0f, 0.0f, 0.0f, 0.0f},
         .alpha = 0.8f},
    });

    gs::SimplifyAuditTrail audit;
    gs::SimplifyOptions options;
    options.ratio = 0.5;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;

    const auto output = expect_ok(gs::simplify_with_audit(input, audit, options));

    const auto validation = gf::ValidateBasic(output, true);
    EXPECT_TRUE(validation.message.empty()) << validation.message;

    EXPECT_EQ(output.numPoints, 1);

    // Quaternion should be normalized
    const float qw = output.rotations[0];
    const float qx = output.rotations[1];
    const float qy = output.rotations[2];
    const float qz = output.rotations[3];
    EXPECT_NEAR(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), 1.0f, 1e-5f);

    // All scales positive and finite
    for (int d = 0; d < 3; ++d) {
        const float s = activated_scale_from_ir(output, 0, d);
        EXPECT_GT(s, 0.0f);
        EXPECT_TRUE(std::isfinite(s));
    }

    // No NaN in output
    for (int d = 0; d < 3; ++d) {
        EXPECT_FALSE(std::isnan(output.positions[d]));
        EXPECT_FALSE(std::isnan(output.colors[d]));
    }
}

// --- SH reduction tests ---

gf::GaussianCloudIR make_cloud_with_sh(const std::vector<PointSpec>& points, int sh_degree) {
    auto ir = make_cloud(points);
    ir.meta.shDegree = sh_degree;
    if (sh_degree > 0) {
        const int sh_per_point = gf::ShCoeffsPerPoint(sh_degree);
        ir.sh.resize(static_cast<size_t>(points.size()) * static_cast<size_t>(sh_per_point));
        for (size_t i = 0; i < points.size(); ++i) {
            for (int k = 0; k < sh_per_point; ++k) {
                ir.sh[i * static_cast<size_t>(sh_per_point) + static_cast<size_t>(k)] =
                    static_cast<float>(i * 100 + k);  // Predictable values
            }
        }
    }
    const auto validation = gf::ValidateBasic(ir, true);
    EXPECT_TRUE(validation.message.empty()) << validation.message;
    return ir;
}

TEST(Simplify, ReducesSH3ToSH0) {
    const auto input = make_cloud_with_sh(
        {{.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f},
         {.position = {1.0f, 0.0f, 0.0f}, .alpha = 0.9f}},
        3);

    gs::SimplifyOptions options;
    options.ratio = 1.0;
    options.target_sh_degree = 0;
    options.opacity_prune_threshold = 0.0f;

    const auto output = expect_ok(gs::simplify(input, options));

    EXPECT_EQ(output.meta.shDegree, 0);
    EXPECT_TRUE(output.sh.empty());
    EXPECT_EQ(output.numPoints, 2);
    // Colors (DC) should still be present
    EXPECT_FALSE(output.colors.empty());
}

TEST(Simplify, ReducesSH3ToSH1) {
    const auto input = make_cloud_with_sh(
        {{.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f},
         {.position = {1.0f, 0.0f, 0.0f}, .alpha = 0.9f}},
        3);

    gs::SimplifyOptions options;
    options.ratio = 1.0;
    options.target_sh_degree = 1;
    options.opacity_prune_threshold = 0.0f;

    const auto output = expect_ok(gs::simplify(input, options));

    const int sh1_per_point = gf::ShCoeffsPerPoint(1);  // 9
    EXPECT_EQ(output.meta.shDegree, 1);
    ASSERT_EQ(output.sh.size(), static_cast<size_t>(2 * sh1_per_point));

    // First point's SH should be the first 9 values from the original SH3 data
    for (int k = 0; k < sh1_per_point; ++k) {
        EXPECT_NEAR(output.sh[static_cast<size_t>(k)], static_cast<float>(k), 1e-4f);
    }
    // Second point's SH should be the first 9 values from original point 1's data
    for (int k = 0; k < sh1_per_point; ++k) {
        EXPECT_NEAR(output.sh[static_cast<size_t>(sh1_per_point + k)],
                    static_cast<float>(100 + k), 1e-4f);
    }
}

TEST(Simplify, TargetSHDegreeHigherThanCurrentIsNoOp) {
    const auto input = make_cloud_with_sh(
        {{.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f}},
        1);

    const int sh1_per_point = gf::ShCoeffsPerPoint(1);  // 9
    ASSERT_EQ(input.sh.size(), static_cast<size_t>(sh1_per_point));

    gs::SimplifyOptions options;
    options.ratio = 1.0;
    options.target_sh_degree = 3;  // Can't upgrade
    options.opacity_prune_threshold = 0.0f;

    const auto output = expect_ok(gs::simplify(input, options));

    // Should remain SH1
    EXPECT_EQ(output.meta.shDegree, 1);
    ASSERT_EQ(output.sh.size(), static_cast<size_t>(sh1_per_point));
}

TEST(Simplify, DefaultSHDegreeKeepsOriginal) {
    const auto input = make_cloud_with_sh(
        {{.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f}},
        3);

    const int sh3_per_point = gf::ShCoeffsPerPoint(3);  // 45
    gs::SimplifyOptions options;
    options.ratio = 1.0;
    // target_sh_degree defaults to -1
    options.opacity_prune_threshold = 0.0f;

    const auto output = expect_ok(gs::simplify(input, options));

    EXPECT_EQ(output.meta.shDegree, 3);
    EXPECT_EQ(output.sh.size(), static_cast<size_t>(sh3_per_point));
}

TEST(Simplify, SHReductionWithMergePreservesCorrectDegree) {
    // Two points with SH3, merge them, target SH1
    const auto input = make_cloud_with_sh(
        {{.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.5f},
         {.position = {1.0f, 0.0f, 0.0f}, .alpha = 0.5f}},
        3);

    gs::SimplifyOptions options;
    options.ratio = 0.5;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;
    options.target_sh_degree = 1;

    const auto output = expect_ok(gs::simplify(input, options));

    EXPECT_EQ(output.numPoints, 1);
    EXPECT_EQ(output.meta.shDegree, 1);
    const int sh1_per_point = gf::ShCoeffsPerPoint(1);  // 9
    ASSERT_EQ(output.sh.size(), static_cast<size_t>(sh1_per_point));
}

// --- Statistical Outlier Removal tests ---

TEST(Simplify, SORRemovesIsolatedPoints) {
    // Create 50 dense points in a tight cluster, plus 3 isolated outliers far away
    std::vector<PointSpec> points;

    // Dense cluster: 50 points in a grid
    for (int i = 0; i < 50; ++i) {
        const float x = static_cast<float>(i % 5) * 0.1f;
        const float y = static_cast<float>((i / 5) % 5) * 0.1f;
        const float z = static_cast<float>(i / 25) * 0.1f;
        points.push_back({.position = {x, y, z}, .alpha = 0.9f});
    }

    // 3 isolated outliers very far from the cluster
    points.push_back({.position = {500.0f, 500.0f, 500.0f}, .alpha = 0.9f});
    points.push_back({.position = {-500.0f, -500.0f, -500.0f}, .alpha = 0.9f});
    points.push_back({.position = {0.0f, 0.0f, 500.0f}, .alpha = 0.9f});

    auto input = make_cloud(points);

    gs::SimplifyOptions options;
    options.ratio = 1.0;  // No simplification, just SOR
    options.opacity_prune_threshold = 0.0f;
    options.sor_nb_neighbors = 5;
    options.sor_std_ratio = 2.0f;

    auto output = expect_ok(gs::simplify(input, options));

    // Should have removed the 3 outliers, keeping 50 dense points
    EXPECT_EQ(output.numPoints, 50);
}

TEST(Simplify, SORDisabledByDefault) {
    // Same data as above, but without SOR enabled
    std::vector<PointSpec> points;
    for (int i = 0; i < 10; ++i) {
        points.push_back({.position = {static_cast<float>(i) * 0.1f, 0.0f, 0.0f}, .alpha = 0.9f});
    }
    points.push_back({.position = {100.0f, 100.0f, 100.0f}, .alpha = 0.9f});

    auto input = make_cloud(points);

    gs::SimplifyOptions options;
    options.ratio = 1.0;
    options.opacity_prune_threshold = 0.0f;
    // sor_nb_neighbors = 0 by default (disabled)

    auto output = expect_ok(gs::simplify(input, options));

    // Outlier should NOT be removed
    EXPECT_EQ(output.numPoints, 11);
}

TEST(Simplify, SORPreservesAllAttributes) {
    // Create points with extras, verify SOR preserves them
    std::vector<PointSpec> points;
    for (int i = 0; i < 10; ++i) {
        points.push_back({
            .position = {static_cast<float>(i) * 0.1f, 0.0f, 0.0f},
            .color = {static_cast<float>(i) / 10.0f, 0.5f, 0.2f},
            .alpha = 0.8f
        });
    }
    // One outlier
    points.push_back({.position = {50.0f, 50.0f, 50.0f}, .color = {1.0f, 1.0f, 1.0f}, .alpha = 0.8f});

    // Extra scalar per point
    std::vector<float> extras(points.size());
    for (size_t i = 0; i < extras.size(); ++i)
        extras[i] = static_cast<float>(i) * 1.5f;

    auto input = make_cloud(points, extras);

    gs::SimplifyOptions options;
    options.ratio = 1.0;
    options.opacity_prune_threshold = 0.0f;
    options.sor_nb_neighbors = 5;
    options.sor_std_ratio = 2.0f;

    auto output = expect_ok(gs::simplify(input, options));

    EXPECT_EQ(output.numPoints, 10);
    // Verify colors preserved
    EXPECT_FALSE(output.colors.empty());
    EXPECT_EQ(output.colors.size(), static_cast<size_t>(10 * 3));
    // Verify extras preserved
    ASSERT_TRUE(output.extras.count("feature") > 0);
    EXPECT_EQ(output.extras["feature"].size(), static_cast<size_t>(10));
}

TEST(Simplify, SORWithAuditTrail) {
    std::vector<PointSpec> points;
    for (int i = 0; i < 30; ++i) {
        points.push_back({.position = {static_cast<float>(i % 5) * 0.1f,
                                        static_cast<float>(i / 5) * 0.1f, 0.0f}, .alpha = 0.9f});
    }
    points.push_back({.position = {500.0f, 0.0f, 0.0f}, .alpha = 0.9f});

    auto input = make_cloud(points);

    gs::SimplifyOptions options;
    options.ratio = 1.0;
    options.opacity_prune_threshold = 0.0f;
    options.sor_nb_neighbors = 5;
    options.sor_std_ratio = 1.5f;

    gs::SimplifyAuditTrail audit;
    auto output = expect_ok(gs::simplify_with_audit(input, audit, options));

    EXPECT_EQ(audit.original_count, 31);
    EXPECT_GT(audit.sor_removed, 0);
    EXPECT_EQ(audit.post_sor_count, output.numPoints);
    EXPECT_EQ(audit.final_count, output.numPoints);
}

// --- Keep Region tests ---

TEST(Simplify, KeepRegionPreservesPointsInsideBox) {
    // 6 points: 4 in a cluster at origin, 2 in a cluster at x=5
    // Define keep_region around the origin cluster.
    // Without keep_region, points in both clusters are equally likely to merge.
    // With keep_region (high weight), origin cluster points are less likely to merge
    // -> more points survive near origin.
    const auto input = make_cloud({
        // Origin cluster (4 points, tightly packed)
        {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {0.1f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {0.0f, 0.1f, 0.0f}, .alpha = 0.9f},
        {.position = {0.1f, 0.1f, 0.0f}, .alpha = 0.9f},
        // Far cluster (2 points, tightly packed)
        {.position = {5.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {5.1f, 0.0f, 0.0f}, .alpha = 0.9f},
    });

    gs::SimplifyOptions options;
    options.ratio = 0.5; // target 3 points
    options.knn_k = 2;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;
    options.keep_weight = 100.0f; // Very high: origin cluster almost never merges
    options.keep_regions.push_back({-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f});

    const auto output = expect_ok(gs::simplify(input, options));

    // With very high keep_weight, the 2 points at x=5 should merge (cost *1.0),
    // while origin cluster points should survive (cost *100.0).
    // Result: 4 origin + 1 merged far = 5... but target is 3, so more merges needed.
    // The far cluster merges first, then some origin points merge too.
    // But the origin cluster should still have more survivors than the far cluster.
    ASSERT_EQ(output.numPoints, 3);

    // Count survivors near origin vs far
    int near_origin = 0;
    int near_far = 0;
    for (int i = 0; i < output.numPoints; ++i) {
        const float x = output.positions[static_cast<size_t>(i) * 3];
        if (x < 2.0f)
            ++near_origin;
        else
            ++near_far;
    }
    // With high keep_weight, more survivors should be near origin than far
    EXPECT_GT(near_origin, near_far);
}

TEST(Simplify, KeepRegionEmptyIsNoOp) {
    // Same data, no keep_regions — should behave as before
    const auto input = make_cloud({
        {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {0.1f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {5.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {5.1f, 0.0f, 0.0f}, .alpha = 0.9f},
    });

    gs::SimplifyOptions options;
    options.ratio = 0.5;
    options.knn_k = 1;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;
    // keep_regions is empty by default

    gs::SimplifyAuditTrail audit;
    const auto output = expect_ok(gs::simplify_with_audit(input, audit, options));

    EXPECT_EQ(output.numPoints, 2);
    EXPECT_EQ(audit.final_count, 2);
}

TEST(Simplify, KeepRegionWeightOneIsNoOp) {
    // keep_weight = 1.0 means no bias, same as no regions
    const auto input = make_cloud({
        {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {0.1f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {5.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {5.1f, 0.0f, 0.0f}, .alpha = 0.9f},
    });

    gs::SimplifyOptions options_no_region;
    options_no_region.ratio = 0.5;
    options_no_region.knn_k = 1;
    options_no_region.merge_cap = 0.5;
    options_no_region.opacity_prune_threshold = 0.0f;

    gs::SimplifyOptions options_with_region;
    options_with_region.ratio = 0.5;
    options_with_region.knn_k = 1;
    options_with_region.merge_cap = 0.5;
    options_with_region.opacity_prune_threshold = 0.0f;
    options_with_region.keep_weight = 1.0f;
    options_with_region.keep_regions.push_back({-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f});

    const auto out_no = expect_ok(gs::simplify(input, options_no_region));
    const auto out_yes = expect_ok(gs::simplify(input, options_with_region));

    // Both should produce same point count
    EXPECT_EQ(out_no.numPoints, out_yes.numPoints);
}

TEST(Simplify, KeepRegionMultipleBoxes) {
    // 8 points: 2 clusters, each protected by a separate box
    const auto input = make_cloud({
        // Cluster A: 2 points
        {.position = {0.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {0.1f, 0.0f, 0.0f}, .alpha = 0.9f},
        // Cluster B: 2 points
        {.position = {10.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {10.1f, 0.0f, 0.0f}, .alpha = 0.9f},
        // Unprotected: 4 points in middle
        {.position = {5.0f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {5.1f, 0.0f, 0.0f}, .alpha = 0.9f},
        {.position = {5.0f, 0.1f, 0.0f}, .alpha = 0.9f},
        {.position = {5.1f, 0.1f, 0.0f}, .alpha = 0.9f},
    });

    gs::SimplifyOptions options;
    options.ratio = 0.5; // target 4
    options.knn_k = 2;
    options.merge_cap = 0.5;
    options.opacity_prune_threshold = 0.0f;
    options.keep_weight = 100.0f;
    // Protect both clusters
    options.keep_regions.push_back({-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f});
    options.keep_regions.push_back({9.0f, -1.0f, -1.0f, 11.0f, 1.0f, 1.0f});

    const auto output = expect_ok(gs::simplify(input, options));
    ASSERT_EQ(output.numPoints, 4);

    // The 4 unprotected middle points should be preferentially merged.
    // Count survivors in each region
    int cluster_a = 0, cluster_b = 0, middle = 0;
    for (int i = 0; i < output.numPoints; ++i) {
        const float x = output.positions[static_cast<size_t>(i) * 3];
        if (x < 2.0f) ++cluster_a;
        else if (x > 8.0f) ++cluster_b;
        else ++middle;
    }
    // Protected clusters should have more survivors than middle
    EXPECT_GE(cluster_a + cluster_b, middle);
}
