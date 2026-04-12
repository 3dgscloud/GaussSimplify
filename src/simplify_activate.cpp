// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Activation, deactivation, SH helpers, and copy utilities.

#include "simplify_detail.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace gs::detail {

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

// --- SH degree helpers ---

int sh_degree_from_coeffs_per_channel(const int coeffs_per_channel) {
    if (coeffs_per_channel <= 0) return 0;
    const int d = static_cast<int>(std::sqrt(static_cast<double>(coeffs_per_channel) + 1.0) + 0.5) - 1;
    if ((d + 1) * (d + 1) - 1 != coeffs_per_channel) return -1;
    return d;
}

int sh_coeffs_per_channel_for_degree(const int degree) {
    if (degree <= 0) return 0;
    return (degree + 1) * (degree + 1) - 1;
}

void reduce_sh_degree(ActivatedCloud& cloud, const int target_degree) {
    if (target_degree < 0) return;
    if (cloud.sh.empty() || cloud.sh_coeffs_per_point <= 0) return;

    const int current_degree = sh_degree_from_coeffs_per_channel(cloud.sh_coeffs_per_point);
    if (current_degree < 0 || target_degree >= current_degree) return;

    if (target_degree == 0) {
        cloud.sh.clear();
        cloud.sh.shrink_to_fit();
        cloud.sh_coeffs_per_point = 0;
        return;
    }

    const int target_coeffs = sh_coeffs_per_channel_for_degree(target_degree);
    const int target_dim = target_coeffs * 3;
    const int current_dim = cloud.sh_coeffs_per_point * 3;

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

// --- IR <-> ActivatedCloud conversion ---

ActivatedCloud activate_from_ir(const gf::GaussianCloudIR& ir) {
    ActivatedCloud cloud;
    const int32_t n = ir.numPoints;
    cloud.count = n;

    cloud.positions = ir.positions;

    cloud.scales.resize(static_cast<size_t>(n) * 3);
    for (int32_t i = 0; i < n * 3; ++i)
        cloud.scales[static_cast<size_t>(i)] = math::activated_scale(ir.scales[static_cast<size_t>(i)]);

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

    cloud.alphas.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i)
        cloud.alphas[static_cast<size_t>(i)] = math::sigmoid(ir.alphas[static_cast<size_t>(i)]);

    cloud.colors = ir.colors;
    cloud.sh = ir.sh;

    if (n > 0 && !ir.sh.empty()) {
        const int total_sh = static_cast<int>(ir.sh.size()) / n;
        cloud.sh_coeffs_per_point = total_sh / 3;
    }

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

gf::GaussianCloudIR deactivate_to_ir(const ActivatedCloud& cloud, const gf::GaussMetadata& meta) {
    gf::GaussianCloudIR ir;
    const int32_t n = cloud.count;
    ir.numPoints = n;

    ir.positions = cloud.positions;

    ir.scales.resize(static_cast<size_t>(n) * 3);
    for (int32_t i = 0; i < n * 3; ++i)
        ir.scales[static_cast<size_t>(i)] = std::log(std::max(cloud.scales[static_cast<size_t>(i)], kMinScale));

    ir.rotations = cloud.rotations;

    ir.alphas.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i)
        ir.alphas[static_cast<size_t>(i)] = math::logit_from_alpha(cloud.alphas[static_cast<size_t>(i)]);

    ir.colors = cloud.colors;
    ir.sh = cloud.sh;
    ir.extras.clear();
    for (const auto& ei : cloud.extras)
        ir.extras[ei.name] = ei.data;
    ir.meta = meta;
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

    if (!src.colors.empty()) {
        std::copy_n(src.colors.begin() + static_cast<ptrdiff_t>(s3), 3,
                    dst.colors.begin() + static_cast<ptrdiff_t>(d3));
    }

    const int sh_dim = src.sh_coeffs_per_point * 3;
    if (sh_dim > 0 && !src.sh.empty()) {
        const size_t ss = static_cast<size_t>(src_idx) * static_cast<size_t>(sh_dim);
        const size_t ds = static_cast<size_t>(dst_idx) * static_cast<size_t>(sh_dim);
        std::copy_n(src.sh.begin() + static_cast<ptrdiff_t>(ss), sh_dim,
                    dst.sh.begin() + static_cast<ptrdiff_t>(ds));
    }

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

// --- Resize helper ---

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

} // namespace gs::detail
