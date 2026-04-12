# @gausssimplify/wasm

WebAssembly version of GaussSimplify, providing TypeScript wrapper library for 3D Gaussian Splat simplification in browsers and Node.js.

## Features

- Read/write multiple Gaussian splatting formats: PLY, Compressed PLY, Splat, KSplat, SPZ, SOG
- High-performance simplification via kNN + moment matching
- Model info retrieval for display
- TypeScript type support
- Browser and Node.js compatible

## Installation

```bash
npm install @gausssimplify/wasm
```

## Usage

### Browser

```typescript
import { createGaussSimplify } from '@gausssimplify/wasm';

async function simplifyFile(file: File) {
    // Initialize
    const api = await createGaussSimplify();

    // Read file
    const fileData = await file.arrayBuffer();
    const { data: ir } = await api.read(new Uint8Array(fileData), 'ply');
    console.log(`Loaded ${ir.numPoints} points`);

    // Get model info for display
    const { data: info } = await api.getModelInfo(ir);
    console.log(`Bounds: ${JSON.stringify(info.bounds)}`);

    // Simplify to 10%
    const { data: simplified } = await api.simplify(ir, { ratio: 0.1 });
    console.log(`Simplified to ${simplified.numPoints} points`);

    // Export as .splat
    const { data: output } = await api.write(simplified, 'splat');

    // Download
    const blob = new Blob([output], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'output.splat';
    a.click();
    URL.revokeObjectURL(url);
}
```

### Node.js

```typescript
import { createGaussSimplify } from '@gausssimplify/wasm';
import fs from 'fs';

async function simplify() {
    const api = await createGaussSimplify();

    // Read file
    const inputData = fs.readFileSync('input.ply');
    const { data: ir } = await api.read(inputData, 'ply');
    console.log(`Loaded ${ir.numPoints} points`);

    // Simplify
    const { data: simplified } = await api.simplify(ir, {
        ratio: 0.1,
        target_sh_degree: 1,
    });

    // Write output
    const { data: output } = await api.write(simplified, 'splat');
    fs.writeFileSync('output.splat', output);
}

simplify().catch(console.error);
```

## API

### `createGaussSimplify(moduleFactory?)`

Create and initialize a GaussSimplify WASM instance (singleton).

### `destroyGaussSimplify()`

Dispose the WASM instance and free memory.

### `api.read(data, format, options?)` → `Promise<ReadResult>`

Read file bytes into `GaussianCloudIR`.

| Param | Type | Description |
|-------|------|-------------|
| `data` | `ArrayBuffer \| Uint8Array` | File content |
| `format` | `string` | `'ply'`, `'splat'`, `'ksplat'`, `'spz'`, `'sog'` |
| `options.strict` | `boolean` | Strict validation (default: `false`) |

### `api.simplify(ir, options?)` → `Promise<SimplifyResult>`

Simplify a `GaussianCloudIR`.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ratio` | `number` | 0.1 | Target fraction of points to keep |
| `knn_k` | `number` | 16 | kNN neighbors for merge graph |
| `merge_cap` | `number` | 0.5 | Max fraction merged per pass |
| `opacity_prune_threshold` | `number` | 0.1 | Remove gaussians below this opacity |
| `target_sh_degree` | `number` | -1 | SH degree (-1 = keep original) |
| `sor_nb_neighbors` | `number` | 0 | SOR: kNN neighbors, 0 = disabled |
| `sor_std_ratio` | `number` | 2.0 | SOR: std multiplier threshold |
| `keep_weight` | `number` | 3.0 | Region cost multiplier (1.0 = no bias, >1 protects regions) |
| `keep_regions` | `AABBRegion[]` | [] | Regions to preserve |

#### `AABBRegion`

```typescript
interface AABBRegion {
    min_x: number; min_y: number; min_z: number;
    max_x: number; max_y: number; max_z: number;
}
```

### `api.write(ir, format, options?)` → `Promise<WriteResult>`

Write `GaussianCloudIR` to file bytes.

| Param | Type | Description |
|-------|------|-------------|
| `ir` | `GaussianCloudIR` | Gaussian cloud data |
| `format` | `string` | Output format |
| `options.strict` | `boolean` | Strict mode (default: `false`) |

### `api.getModelInfo(ir)` → `Promise<ModelInfoResult>`

Get bounding box, point stats, and size breakdown from an IR.

### `api.getSupportedFormats()` → `string[]`

Returns: `['ply', 'compressed.ply', 'splat', 'ksplat', 'spz', 'sog']`

### `api.getVersion()` → `string`

Library version string.

## Supported Formats

- `ply` — Standard PLY format
- `compressed.ply` — Compressed PLY format
- `splat` — Splat format
- `ksplat` — KSplat format
- `spz` — SPZ compressed format
- `sog` — SOG format

Powered by [GaussForge](https://github.com/3dgscloud/GaussSimplify).

## Development

### Build from Source

Prerequisites: [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html), Node.js 18+

```bash
cd wasm
npm install
npm run build       # Build WASM + TypeScript
npm run build:wasm  # Build WASM only
npm run build:ts    # Build TypeScript only
```

### Build Output

- `gauss_simplify.node.js` — Node.js WASM module
- `gauss_simplify.web.js` — Browser/Worker WASM module
- `dist/index.node.js` — Node.js entry with types
- `dist/index.web.js` — Browser entry with types

## Error Handling

All methods may throw errors. Use try-catch for robust handling:

```typescript
try {
    const { data: simplified } = await api.simplify(ir, { ratio: 0.1 });
} catch (error) {
    console.error('Simplify failed:', error.message);
}
```

## Requirements

- Emscripten SDK (for building WASM)
- Node.js 18+ (for development)
- TypeScript 5+ (for development)
