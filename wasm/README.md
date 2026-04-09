# @gausssimplify/wasm

WASM build of GaussSimplify — run 3D Gaussian Splat simplification in the browser or Node.js.

## Install

```bash
npm install @gausssimplify/wasm
```

## Usage

```typescript
import { createGaussSimplify } from '@gausssimplify/wasm';

const api = await createGaussSimplify();

// 1. Read file
const response = await fetch('model.ply');
const { data: ir } = await api.read(new Uint8Array(await response.arrayBuffer()), 'ply');

// 2. Get model info (for display)
const { data: info } = await api.getModelInfo(ir);
console.log(`Points: ${info.basic.numPoints}, Bounds:`, info.bounds);

// 3. Simplify to 10%
const { data: simplified } = await api.simplify(ir, {
    ratio: 0.1,
    target_sh_degree: 1,
});

// 4. Export
const { data: bytes } = await api.write(simplified, 'splat');
```

## API

### `createGaussSimplify(factory?)` → `Promise<GaussSimplify>`

Initialize the WASM module (singleton). Optionally pass a custom module factory.

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

### `api.write(ir, format, options?)` → `Promise<WriteResult>`

Write `GaussianCloudIR` to file bytes.

### `api.getModelInfo(ir)` → `Promise<ModelInfoResult>`

Get bounding box, stats, and size breakdown from an IR.

### `api.getSupportedFormats()` → `string[]`

List supported formats: `['ply', 'compressed.ply', 'splat', 'ksplat', 'spz', 'sog']`.

### `api.getVersion()` → `string`

Library version string.

## Build from Source

Prerequisites: [Emscripten](https://emscripten.org/docs/getting_started/downloads.html)

```bash
cd wasm
npm install
npm run build
```

This produces:
- `gauss_simplify.node.js` — Node.js WASM module
- `gauss_simplify.web.js` — Browser/Worker WASM module
- `dist/index.node.js` — Node.js entry with types
- `dist/index.web.js` — Browser entry with types

## Supported Formats

Read/Write: PLY, Compressed PLY, Splat, KSplat, SPZ, SOG

Powered by [GaussForge](https://github.com/3dgscloud/GaussForge).

## License

GPL-3.0-or-later
