// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#include <emscripten/val.h>
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gs/simplify.h"
#include "gs/version.h"

#include "gf/core/gauss_ir.h"
#include "gf/core/model_info.h"
#include "gf/core/validate.h"
#include "gf/io/registry.h"

#ifdef __EMSCRIPTEN__
using namespace emscripten;

namespace {

// --- Typed array conversion helpers ---

val vectorToUint8Array(const std::vector<uint8_t>& vec) {
    if (vec.empty())
        return val::global("Uint8Array").new_(0);
    return val(typed_memory_view(vec.size(), vec.data())).call<val>("slice");
}

val vectorToFloat32Array(const std::vector<float>& vec) {
    if (vec.empty())
        return val::global("Float32Array").new_(0);
    return val(typed_memory_view(vec.size(), vec.data())).call<val>("slice");
}

// --- IR <-> JS marshaling ---

val gaussIRToJS(const gf::GaussianCloudIR& ir) {
    val result = val::object();
    result.set("numPoints", ir.numPoints);

    result.set("positions", vectorToFloat32Array(ir.positions));
    result.set("scales", vectorToFloat32Array(ir.scales));
    result.set("rotations", vectorToFloat32Array(ir.rotations));
    result.set("alphas", vectorToFloat32Array(ir.alphas));
    result.set("colors", vectorToFloat32Array(ir.colors));
    result.set("sh", vectorToFloat32Array(ir.sh));

    // Extras
    val extras = val::object();
    for (const auto& pair : ir.extras)
        extras.set(pair.first, vectorToFloat32Array(pair.second));
    result.set("extras", extras);

    // Metadata
    val meta = val::object();
    meta.set("shDegree", ir.meta.shDegree);
    meta.set("sourceFormat", ir.meta.sourceFormat);
    result.set("meta", meta);

    return result;
}

gf::GaussianCloudIR jsToGaussIR(val jsIR) {
    gf::GaussianCloudIR ir;
    ir.numPoints = jsIR["numPoints"].as<int32_t>();

    auto fill = [&](val source, std::vector<float>& dest) {
        if (!source.isUndefined() && !source.isNull())
            dest = convertJSArrayToNumberVector<float>(source);
    };

    fill(jsIR["positions"], ir.positions);
    fill(jsIR["scales"], ir.scales);
    fill(jsIR["rotations"], ir.rotations);
    fill(jsIR["alphas"], ir.alphas);
    fill(jsIR["colors"], ir.colors);
    fill(jsIR["sh"], ir.sh);

    val ex = jsIR["extras"];
    if (!ex.isUndefined() && !ex.isNull()) {
        val keys = val::global("Object").call<val>("keys", ex);
        unsigned int len = keys["length"].as<unsigned int>();
        for (unsigned int i = 0; i < len; ++i) {
            std::string key = keys[i].as<std::string>();
            ir.extras[key] = convertJSArrayToNumberVector<float>(ex[key]);
        }
    }

    val m = jsIR["meta"];
    if (!m.isUndefined() && !m.isNull()) {
        ir.meta.shDegree = m["shDegree"].as<int32_t>();
        if (m.hasOwnProperty("sourceFormat"))
            ir.meta.sourceFormat = m["sourceFormat"].as<std::string>();
    }
    return ir;
}

// --- ModelInfo -> JS ---

val modelInfoToJS(const gf::ModelInfo& info) {
    val result = val::object();

    // Basic info
    val basic = val::object();
    basic.set("numPoints", info.numPoints);
    if (info.fileSize > 0)
        basic.set("fileSize", static_cast<double>(info.fileSize));
    if (!info.sourceFormat.empty())
        basic.set("sourceFormat", info.sourceFormat);
    result.set("basic", basic);

    // Rendering properties
    val rendering = val::object();
    rendering.set("shDegree", info.shDegree);
    rendering.set("antialiased", info.antialiased);
    result.set("rendering", rendering);

    // Metadata
    val meta = val::object();
    meta.set("handedness", gf::HandednessToString(info.handedness));
    meta.set("upAxis", gf::UpAxisToString(info.upAxis));
    meta.set("unit", gf::LengthUnitToString(info.unit));
    meta.set("colorSpace", gf::ColorSpaceToString(info.colorSpace));
    result.set("meta", meta);

    // Geometry statistics
    if (info.numPoints > 0) {
        val bounds = val::object();
        val x = val::array();
        x.call<void>("push", info.bounds.minX);
        x.call<void>("push", info.bounds.maxX);
        bounds.set("x", x);

        val y = val::array();
        y.call<void>("push", info.bounds.minY);
        y.call<void>("push", info.bounds.maxY);
        bounds.set("y", y);

        val z = val::array();
        z.call<void>("push", info.bounds.minZ);
        z.call<void>("push", info.bounds.maxZ);
        bounds.set("z", z);
        result.set("bounds", bounds);
    }

    // Scale statistics
    if (info.scaleStats.count > 0) {
        val scaleStats = val::object();
        scaleStats.set("min", info.scaleStats.min);
        scaleStats.set("max", info.scaleStats.max);
        scaleStats.set("avg", info.scaleStats.avg);
        result.set("scaleStats", scaleStats);
    }

    // Alpha statistics
    if (info.alphaStats.count > 0) {
        val alphaStats = val::object();
        alphaStats.set("min", info.alphaStats.min);
        alphaStats.set("max", info.alphaStats.max);
        alphaStats.set("avg", info.alphaStats.avg);
        result.set("alphaStats", alphaStats);
    }

    // Data size breakdown
    val sizes = val::object();
    sizes.set("positions", gf::FormatBytes(info.positionsSize));
    sizes.set("scales", gf::FormatBytes(info.scalesSize));
    sizes.set("rotations", gf::FormatBytes(info.rotationsSize));
    sizes.set("alphas", gf::FormatBytes(info.alphasSize));
    sizes.set("colors", gf::FormatBytes(info.colorsSize));
    sizes.set("sh", gf::FormatBytes(info.shSize));
    sizes.set("total", gf::FormatBytes(info.totalSize));
    result.set("sizes", sizes);

    // Extra attributes
    if (!info.extraAttrs.empty()) {
        val extraAttrs = val::object();
        for (const auto& [name, size] : info.extraAttrs)
            extraAttrs.set(name, gf::FormatBytes(size));
        result.set("extraAttrs", extraAttrs);
    }

    return result;
}

} // anonymous namespace

// --- Exported WASM class ---

class GaussSimplifyWASM {
public:
    GaussSimplifyWASM() : registry_(std::make_unique<gf::IORegistry>()) {}

    val read(val jsData, const std::string& format, bool strict = false) {
        try {
            std::vector<uint8_t> data = convertJSArrayToNumberVector<uint8_t>(jsData);
            auto* reader = registry_->ReaderForExt(format);
            if (!reader)
                return err("No reader for " + format);

            auto ir_or = reader->Read(data.data(), data.size(), {strict});
            if (!ir_or)
                return err(ir_or.error().message);

            auto validation = gf::ValidateBasic(ir_or.value(), strict);
            if (!validation.message.empty() && strict)
                return err(validation.message);

            val res = val::object();
            res.set("data", gaussIRToJS(ir_or.value()));
            if (!validation.message.empty())
                res.set("warning", validation.message);
            return res;
        } catch (const std::exception& e) {
            return err(e.what());
        }
    }

    val simplify(val jsIR,
                 double ratio, int knn_k, double merge_cap,
                 float opacity_prune_threshold, int target_sh_degree,
                 int sor_nb_neighbors, float sor_std_ratio) {
        try {
            gf::GaussianCloudIR ir = jsToGaussIR(jsIR);

            gs::SimplifyOptions opts;
            opts.ratio = ratio;
            opts.knn_k = knn_k;
            opts.merge_cap = merge_cap;
            opts.opacity_prune_threshold = opacity_prune_threshold;
            opts.target_sh_degree = target_sh_degree;
            opts.sor_nb_neighbors = sor_nb_neighbors;
            opts.sor_std_ratio = sor_std_ratio;

            // No progress callback -- runs synchronously in WASM
            auto result = gs::simplify(ir, opts, {});
            if (!result)
                return err(result.error().message);

            val res = val::object();
            res.set("data", gaussIRToJS(result.value()));
            return res;
        } catch (const std::exception& e) {
            return err(e.what());
        }
    }

    val write(val jsIR, const std::string& format, bool strict = false) {
        try {
            auto* writer = registry_->WriterForExt(format);
            if (!writer)
                return err("No writer for " + format);

            auto data_or = writer->Write(jsToGaussIR(jsIR), {strict});
            if (!data_or)
                return err(data_or.error().message);

            val res = val::object();
            res.set("data", vectorToUint8Array(data_or.value()));
            return res;
        } catch (const std::exception& e) {
            return err(e.what());
        }
    }

    val getModelInfo(val jsIR) {
        try {
            gf::GaussianCloudIR ir = jsToGaussIR(jsIR);
            gf::ModelInfo info = gf::GetModelInfo(ir, 0);

            val res = val::object();
            res.set("data", modelInfoToJS(info));
            return res;
        } catch (const std::exception& e) {
            return err(e.what());
        }
    }

    val getSupportedFormats() {
        val f = val::array();
        for (auto& s : {"ply", "compressed.ply", "splat", "ksplat", "spz", "sog"})
            f.call<void>("push", val(s));
        return f;
    }

    std::string getVersion() {
        return GAUSS_SIMPLIFY_VERSION_STRING;
    }

private:
    std::unique_ptr<gf::IORegistry> registry_;

    val err(const std::string& m) {
        val e = val::object();
        e.set("error", m);
        return e;
    }
};

EMSCRIPTEN_BINDINGS(gauss_simplify) {
    class_<GaussSimplifyWASM>("GaussSimplifyWASM")
        .constructor<>()
        .function("read", &GaussSimplifyWASM::read)
        .function("simplify", &GaussSimplifyWASM::simplify)
        .function("write", &GaussSimplifyWASM::write)
        .function("getModelInfo", &GaussSimplifyWASM::getModelInfo)
        .function("getSupportedFormats", &GaussSimplifyWASM::getSupportedFormats)
        .function("getVersion", &GaussSimplifyWASM::getVersion);
}

#endif
