// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "gs/simplify_types.h"

#include "gf/core/errors.h"
#include "gf/core/gauss_ir.h"

namespace gs {

// Simplify a Gaussian cloud to a target ratio of the original count.
// Returns the simplified GaussianCloudIR on success, or an error on failure/cancellation.
gf::Expected<gf::GaussianCloudIR> simplify(
    const gf::GaussianCloudIR& input,
    const SimplifyOptions& options = {},
    ProgressCallback progress = {});

// Same as simplify(), but also records the merge audit trail.
gf::Expected<gf::GaussianCloudIR> simplify_with_audit(
    const gf::GaussianCloudIR& input,
    SimplifyAuditTrail& audit,
    const SimplifyOptions& options = {},
    ProgressCallback progress = {});

} // namespace gs
