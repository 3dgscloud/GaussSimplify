# GaussSimplify

<div align="center">

![License](https://img.shields.io/badge/license-GPL%203.0-blue.svg)
![C++](https://img.shields.io/badge/C++-17-blue.svg)
![CMake](https://img.shields.io/badge/CMake-3.26+-green.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)
![npm](https://img.shields.io/npm/v/@gausssimplify/wasm?label=npm)
[![GitHub Release](https://img.shields.io/github/v/release/3dgscloud/GaussSimplify)](https://github.com/3dgscloud/GaussSimplify/releases)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/3dgscloud/GaussSimplify)](https://github.com/3dgscloud/GaussSimplify/commits)

</div>

A high-performance 3D Gaussian Splat simplification library. Reduces point count while preserving visual quality using kNN-based merge candidate selection and moment-matching pair merging.

## Features

- **Opacity Pruning** — Removes low-opacity gaussians before merging
- **Statistical Outlier Removal (SOR)** — Removes spatially isolated "flyer" gaussians that opacity pruning cannot catch
- **Region-Aware Simplification** — Preserve quality in user-specified AABB regions via cost weighting
- **kNN Merge Graph** — Builds a k-nearest-neighbor graph to find optimal merge candidates
- **Moment Matching** — Merges pairs by matching mean and covariance (position, scale, rotation)
- **SH Degree Reduction** — Optionally reduce spherical harmonics degree to save memory
- **Multi-pass Iterative** — Iteratively merges until target count is reached
- **OpenMP Parallel** — Multi-threaded kNN, merge, and cache construction
- **Multi-format I/O** — Read/write PLY, Compressed PLY, Splat, KSplat, SPZ, SOG via [GaussForge](https://github.com/3dgscloud/GaussForge)
- **WebAssembly Support** — Run in the browser via `@gausssimplify/wasm`

## Quick Start

### Building

```bash
# Configure Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build
```

Dependencies (auto-fetched via FetchContent):
- [GaussForge](https://github.com/3dgscloud/GaussForge) — format I/O
- [nanoflann](https://github.com/jlblancoc/nanoflann) — KD-tree for kNN
- OpenMP — parallel computation (optional, desktop only)

### CLI Usage

```bash
# Basic: simplify to 10% of original
./build/gauss_simplify input.ply output.ply --ratio 0.1

# Target a specific point count
./build/gauss_simplify input.ply output.ply --target 50000

# With SH degree reduction and verbose output
./build/gauss_simplify input.ply output.ply --ratio 0.2 --sh-degree 1 --verbose

# Convert format while simplifying
./build/gauss_simplify input.ply output.splat --ratio 0.1
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--ratio <float>` | 0.1 | Target count as fraction of input |
| `--target <int>` | — | Target count (overrides --ratio) |
| `--knn <int>` | 16 | kNN neighbors for merge graph |
| `--merge-cap <float>` | 0.5 | Max fraction merged per pass |
| `--prune-threshold <float>` | 0.1 | Opacity pruning threshold |
| `--sor-nb <int>` | 0 | SOR: kNN neighbors (0=disabled) |
| `--sor-std <float>` | 2.0 | SOR: std multiplier threshold |
| `--keep-region <x>,<y>,<z>,<x2>,<y2>,<z2>` | — | AABB region to preserve (repeatable) |
| `--keep-weight <float>` | 3.0 | Region cost multiplier (1.0 = no bias) |
| `--sh-degree <int>` | keep | Target SH degree (0-3) |
| `--in-format <ext>` | auto | Override input format |
| `--out-format <ext>` | auto | Override output format |
| `--verbose` | off | Print progress |

### C++ API

```cpp
#include "gs/simplify.h"

// Load your GaussianCloudIR (via GaussForge readers)
gf::GaussianCloudIR ir = /* ... */;

// Simplify
gs::SimplifyOptions opts;
opts.ratio = 0.1;
opts.target_sh_degree = 1;

auto result = gs::simplify(ir, opts);
if (result) {
    gf::GaussianCloudIR simplified = result.value();
    // Use simplified cloud...
}
```

### WebAssembly (WASM)

GaussSimplify provides a WebAssembly build for use in browsers and Node.js.

**Installation:**
```bash
npm install @gausssimplify/wasm
```

**Quick Example:**
```typescript
import { createGaussSimplify } from '@gausssimplify/wasm';

const api = await createGaussSimplify();

// Read -> Simplify -> Write
const { data: ir } = await api.read(fileData, 'ply');
const { data: simplified } = await api.simplify(ir, { ratio: 0.1 });
const { data: output } = await api.write(simplified, 'splat');
```

For detailed WASM usage and API documentation, see the [WASM README](wasm/README.md).

## Supported Formats

Powered by [GaussForge](https://github.com/3dgscloud/GaussForge).

| Format | Extension | Read | Write | Description |
|--------|-----------|------|-------|-------------|
| PLY | `.ply` | ✅ | ✅ | Standard PLY format |
| Compressed PLY | `.compressed.ply` | ✅ | ✅ | Compressed PLY format |
| SPLAT | `.splat` | ✅ | ✅ | Splat format |
| KSPLAT | `.ksplat` | ✅ | ✅ | K-Splat format |
| SPZ | `.spz` | ✅ | ✅ | SPZ compressed format |
| SOG | `.sog` | ✅ | ✅ | SOG format |

## Project Structure

```
GaussSimplify/
├── include/gs/              # Public headers (simplify.h, simplify_types.h)
├── src/
│   ├── simplify.cpp         # Main pipeline + public API
│   ├── simplify_activate.cpp # Activation, deactivation, SH, copy utilities
│   ├── simplify_prune.cpp   # Opacity pruning + SOR
│   ├── simplify_merge.cpp   # Merge algorithm (moment matching)
│   ├── simplify_detail.h    # Shared internal types and declarations
│   ├── simplify_knn.h       # kNN graph construction (header-only)
│   ├── simplify_math.h      # Math utilities (header-only)
│   ├── main.cpp             # CLI tool
│   └── wasm/                # WebAssembly bindings
├── external/          # Third-party (nanoflann)
├── wasm/              # npm package (TypeScript wrapper)
├── cmakes/            # CMake config files
├── tests/             # Unit tests
└── benchmarks/        # Performance benchmarks
```

## Testing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build
cd build && ctest
```

## Benchmark

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target simplify_benchmark
./build/simplify_benchmark
```

## Related Resources

- [GaussForge](https://github.com/3dgscloud/GaussForge) — Gaussian Splatting format conversion library
- [LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio)

## License

This project is licensed under the [GPLv3](LICENSE). See the [LICENSE](LICENSE) file for details.
