// SPDX-FileCopyrightText: 2026 GaussSimplify Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gs/simplify.h"
#include "gs/version.h"

#include "gf/core/gauss_ir.h"
#include "gf/core/validate.h"
#include "gf/io/reader.h"
#include "gf/io/registry.h"
#include "gf/io/writer.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string GetExt(const std::string& path) {
    constexpr std::string_view kCompressed = ".compressed.ply";
    if (path.size() >= kCompressed.size() &&
        path.compare(path.size() - kCompressed.size(), kCompressed.size(), kCompressed) == 0) {
        return std::string{kCompressed};
    }
    const auto pos = path.find_last_of('.');
    if (pos == std::string::npos)
        return "";
    return path.substr(pos + 1);
}

bool FileToBytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) {
        std::cerr << "Failed to open: " << path << "\n";
        return false;
    }
    const auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(size));
    return f.good();
}

bool BytesToFile(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f.good()) {
        std::cerr << "Failed to open for write: " << path << "\n";
        return false;
    }
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return f.good();
}

void PrintUsage() {
    std::cerr << "Usage: gauss_simplify <input> <output> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --ratio <float>           Target ratio (default: 0.1)\n"
              << "  --target <int>            Target count (overrides --ratio)\n"
              << "  --knn <int>               kNN neighbors (default: 16)\n"
              << "  --merge-cap <float>       Merge cap per pass (default: 0.5)\n"
              << "  --prune-threshold <float> Opacity prune threshold (default: 0.1)\n"
              << "  --version                Show version\n"
              << "  --in-format <ext>         Override input format\n"
              << "  --out-format <ext>        Override output format\n"
              << "  --verbose                 Print progress\n"
              << "  --help                    Show this help\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        PrintUsage();
        return 1;
    }

    const std::string arg1 = argv[1];
    if (arg1 == "--version") {
        std::cout << "gauss_simplify version " << GAUSS_SIMPLIFY_VERSION_STRING << "\n";
        return 0;
    }
    if (arg1 == "--help" || arg1 == "-h") {
        PrintUsage();
        return 0;
    }

    if (argc < 3) {
        std::cerr << "Error: both input and output paths are required.\n\n";
        PrintUsage();
        return 1;
    }

    const std::string in_path = argv[1];
    const std::string out_path = argv[2];
    std::string in_ext = GetExt(in_path);
    std::string out_ext = GetExt(out_path);

    gs::SimplifyOptions options;
    bool use_target_count = false;
    int target_count = 0;
    bool verbose = false;

    for (int i = 3; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--verbose") {
            verbose = true;
        } else if (flag == "--help" || flag == "-h") {
            PrintUsage();
            return 0;
        } else if (i + 1 < argc) {
            const std::string val = argv[++i];
            if (flag == "--ratio") {
                options.ratio = std::stod(val);
            } else if (flag == "--target") {
                use_target_count = true;
                target_count = std::stoi(val);
            } else if (flag == "--knn") {
                options.knn_k = std::stoi(val);
            } else if (flag == "--merge-cap") {
                options.merge_cap = std::stod(val);
            } else if (flag == "--prune-threshold") {
                options.opacity_prune_threshold = std::stof(val);
            } else if (flag == "--in-format") {
                in_ext = val;
            } else if (flag == "--out-format") {
                out_ext = val;
            } else {
                std::cerr << "Unknown option: " << flag << "\n";
                return 1;
            }
        } else {
            std::cerr << "Option '" << flag << "' requires a value.\n";
            return 1;
        }
    }

    // Read input
    gf::IORegistry registry;
    auto* reader = registry.ReaderForExt(in_ext);
    if (!reader) {
        std::cerr << "No reader for format: " << in_ext << "\n";
        return 1;
    }

    std::vector<uint8_t> in_data;
    if (!FileToBytes(in_path, in_data)) return 1;

    gf::ReadOptions read_opt;
    auto ir_or = reader->Read(in_data.data(), in_data.size(), read_opt);
    if (!ir_or) {
        std::cerr << "Read failed: " << ir_or.error().message << "\n";
        return 1;
    }
    auto ir = std::move(ir_or.value());

    std::cerr << "Input: " << ir.numPoints << " points\n";

    // Override ratio with explicit target count
    if (use_target_count && ir.numPoints > 0) {
        options.ratio = static_cast<double>(target_count) / static_cast<double>(ir.numPoints);
    }

    // Progress callback
    gs::ProgressCallback progress;
    if (verbose) {
        progress = [](const float p, const std::string& stage) -> bool {
            std::cerr << "[" << static_cast<int>(p * 100) << "%] " << stage << "\n";
            return true;
        };
    }

    // Simplify
    auto result = gs::simplify(ir, options, progress);
    if (!result) {
        std::cerr << "Simplify failed: " << result.error().message << "\n";
        return 1;
    }
    auto simplified = std::move(result.value());

    std::cerr << "Output: " << simplified.numPoints << " points\n";

    // Write output
    auto* writer = registry.WriterForExt(out_ext);
    if (!writer) {
        std::cerr << "No writer for format: " << out_ext << "\n";
        return 1;
    }

    gf::WriteOptions write_opt;
    auto out_or = writer->Write(simplified, write_opt);
    if (!out_or) {
        std::cerr << "Write failed: " << out_or.error().message << "\n";
        return 1;
    }

    if (!BytesToFile(out_path, out_or.value())) return 1;

    std::cerr << "Done: " << in_path << " -> " << out_path << "\n";
    return 0;
}
