# GaussSimplify

A high-performance 3D Gaussian Splat simplification library. Reduces point count while preserving visual quality using kNN-based merge candidate selection and moment-matching pair merging.

## Features

- **Opacity pruning** — removes low-opacity gaussians before merging
- **kNN merge graph** — builds a k-nearest-neighbor graph to find optimal merge candidates
- **Moment matching** — merges pairs by matching mean and covariance (position, scale, rotation)
- **SH degree reduction** — optionally reduce spherical harmonics degree to save memory
- **Multi-pass iterative** — iteratively merges until target count is reached
- **OpenMP parallel** — multi-threaded kNN, merge, and cache construction
- **WASM support** — run in the browser via `@gausssimplify/wasm`

## Build

### Native (CMake)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Dependencies (auto-fetched via FetchContent):
- [GaussForge](https://github.com/3dgscloud/GaussForge) — format I/O (PLY, Splat, KSplat, SPZ, SOG)
- [nanoflann](https://github.com/jlblancoc/nanoflann) — KD-tree for kNN
- OpenMP — parallel computation (optional, desktop only)

### CLI Usage

```bash
./build/gauss_simplify input.ply output.ply --ratio 0.1
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--ratio <float>` | 0.1 | Target count as fraction of input |
| `--target <int>` | — | Target count (overrides --ratio) |
| `--knn <int>` | 16 | kNN neighbors |
| `--merge-cap <float>` | 0.5 | Max fraction merged per pass |
| `--prune-threshold <float>` | 0.1 | Opacity pruning threshold |
| `--sh-degree <int>` | keep | Target SH degree (0-3) |
| `--verbose` | off | Print progress |

### C++ API

```cpp
#include "gs/simplify.h"

gf::GaussianCloudIR ir = /* load from file */;
gs::SimplifyOptions opts;
opts.ratio = 0.1;
opts.target_sh_degree = 1;
auto result = gs::simplify(ir, opts);
if (result) {
    gf::GaussianCloudIR simplified = result.value();
}
```

### WASM

See [wasm/README.md](wasm/README.md).

## Test

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

## License

GPL-3.0-or-later
