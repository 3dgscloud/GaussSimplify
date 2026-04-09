#include "gs/simplify.h"

#include "gf/core/gauss_ir.h"
#include "gf/io/reader.h"
#include "gf/io/registry.h"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool ReadFile(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) return false;
    const auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(size));
    return f.good();
}

gf::GaussianCloudIR LoadSogFile(const std::string& path) {
    std::vector<uint8_t> data;
    if (!ReadFile(path, data)) {
        std::cerr << "Failed to read: " << path << "\n";
        return {};
    }
    gf::IORegistry registry;
    auto* reader = registry.ReaderForExt("sog");
    if (!reader) {
        std::cerr << "No SOG reader\n";
        return {};
    }
    auto result = reader->Read(data.data(), data.size(), {});
    if (!result) {
        std::cerr << "Read failed: " << result.error().message << "\n";
        return {};
    }
    return std::move(result.value());
}

gf::GaussianCloudIR CreateSyntheticData(int numPoints) {
    gf::GaussianCloudIR ir;
    ir.numPoints = numPoints;
    ir.meta.shDegree = 3;

    const int shCoeffsPerPoint = gf::ShCoeffsPerPoint(3);
    ir.positions.resize(static_cast<size_t>(numPoints) * 3);
    ir.scales.resize(static_cast<size_t>(numPoints) * 3);
    ir.rotations.resize(static_cast<size_t>(numPoints) * 4);
    ir.alphas.resize(static_cast<size_t>(numPoints));
    ir.colors.resize(static_cast<size_t>(numPoints) * 3);
    ir.sh.resize(static_cast<size_t>(numPoints) * static_cast<size_t>(shCoeffsPerPoint));

    for (int i = 0; i < numPoints; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(numPoints);
        const size_t i3 = static_cast<size_t>(i) * 3;
        const size_t i4 = static_cast<size_t>(i) * 4;

        ir.positions[i3 + 0] = std::sin(t * 6.28f) * 5.0f;
        ir.positions[i3 + 1] = std::cos(t * 6.28f) * 5.0f;
        ir.positions[i3 + 2] = t * 10.0f - 5.0f;

        ir.scales[i3 + 0] = -2.0f + t;
        ir.scales[i3 + 1] = -2.0f + t * 0.5f;
        ir.scales[i3 + 2] = -2.0f + t * 0.3f;

        ir.rotations[i4 + 0] = 1.0f;
        ir.rotations[i4 + 1] = 0.0f;
        ir.rotations[i4 + 2] = 0.0f;
        ir.rotations[i4 + 3] = 0.0f;

        ir.alphas[static_cast<size_t>(i)] = 2.0f - t * 3.0f;

        ir.colors[i3 + 0] = t;
        ir.colors[i3 + 1] = 1.0f - t;
        ir.colors[i3 + 2] = 0.5f;
    }
    return ir;
}

// --- Benchmark: real SOG file at different ratios ---

void BM_Simplify_RealSog(benchmark::State& state) {
    static gf::GaussianCloudIR ir;
    static bool loaded = false;
    if (!loaded) {
        ir = LoadSogFile("diaosu.sog");
        loaded = true;
        if (ir.numPoints == 0) {
            state.SkipWithError("Failed to load diaosu.sog");
            return;
        }
    }

    const double ratio = static_cast<double>(state.range(0)) / 100.0;

    for (auto _ : state) {
        gs::SimplifyOptions opts;
        opts.ratio = ratio;
        opts.knn_k = 16;
        opts.merge_cap = 0.5;
        opts.opacity_prune_threshold = 0.005f;

        auto result = gs::simplify(ir, opts);
        if (!result) {
            state.SkipWithError(result.error().message.c_str());
            return;
        }
        benchmark::DoNotOptimize(result.value());
    }

    state.SetItemsProcessed(static_cast<int64_t>(ir.numPoints) * state.iterations());
    state.SetLabel(std::to_string(ir.numPoints) + " pts, ratio=" +
                   std::to_string(static_cast<int>(ratio * 100)) + "%");
}

// --- Benchmark: synthetic data at various sizes ---

void BM_Simplify_Synthetic(benchmark::State& state) {
    const int numPoints = state.range(0);
    auto ir = CreateSyntheticData(numPoints);

    for (auto _ : state) {
        gs::SimplifyOptions opts;
        opts.ratio = 0.5;
        opts.knn_k = 16;
        opts.merge_cap = 0.5;
        opts.opacity_prune_threshold = 0.005f;

        auto result = gs::simplify(ir, opts);
        if (!result) {
            state.SkipWithError(result.error().message.c_str());
            return;
        }
        benchmark::DoNotOptimize(result.value());
    }

    state.SetItemsProcessed(static_cast<int64_t>(numPoints) * state.iterations());
    state.SetLabel(std::to_string(numPoints) + " pts, ratio=50%");
}

} // namespace

BENCHMARK(BM_Simplify_RealSog)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50)
    ->Arg(75)
    ->MinTime(3.0)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Simplify_Synthetic)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(50000)
    ->Arg(100000)
    ->MinTime(2.0)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
