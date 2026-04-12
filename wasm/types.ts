/**
 * TypeScript type definitions for @gausssimplify/wasm
 */

export interface GaussMetadata {
    shDegree: number;
    sourceFormat?: string;
}

export interface GaussianCloudIR {
    numPoints: number;
    positions: Float32Array;
    scales: Float32Array;
    rotations: Float32Array;
    alphas: Float32Array;
    colors: Float32Array;
    sh: Float32Array;
    extras: Record<string, Float32Array>;
    meta: GaussMetadata;
}

export interface SimplifyOptions {
    ratio?: number;                    // Target ratio (default: 0.1)
    knn_k?: number;                    // kNN neighbors (default: 16)
    merge_cap?: number;                // Max fraction merged per pass (default: 0.5)
    opacity_prune_threshold?: number;  // Opacity pruning threshold (default: 0.1)
    target_sh_degree?: number;         // Target SH degree, -1 = keep original (default: -1)
    sor_nb_neighbors?: number;         // Statistical outlier removal: kNN neighbors, 0 = disabled (default: 0)
    sor_std_ratio?: number;            // Statistical outlier removal: std multiplier threshold (default: 2.0)
}

export interface ReadResult {
    data: GaussianCloudIR;
    warning?: string;
    error?: string;
}

export interface SimplifyResult {
    data: GaussianCloudIR;
    error?: string;
}

export interface WriteResult {
    data: Uint8Array;
    error?: string;
}

export type SupportedFormat = 'ply' | 'compressed.ply' | 'splat' | 'ksplat' | 'spz' | 'sog';

export interface ReadOptions {
    strict?: boolean;
}

export interface WriteOptions {
    strict?: boolean;
}

// Model Info types
export interface FloatStats {
    min: number;
    max: number;
    avg: number;
}

export interface BoundingBox {
    x: [number, number];
    y: [number, number];
    z: [number, number];
}

export interface ModelInfoBasic {
    numPoints: number;
    fileSize?: number;
    sourceFormat?: string;
}

export interface ModelInfoRendering {
    shDegree: number;
    antialiased: boolean;
}

export interface ModelInfoMeta {
    handedness: string;
    upAxis: string;
    unit: string;
    colorSpace: string;
}

export interface ModelInfoSizes {
    positions: string;
    scales: string;
    rotations: string;
    alphas: string;
    colors: string;
    sh: string;
    total: string;
}

export interface ModelInfo {
    basic: ModelInfoBasic;
    rendering: ModelInfoRendering;
    meta: ModelInfoMeta;
    bounds?: BoundingBox;
    scaleStats?: FloatStats;
    alphaStats?: FloatStats;
    sizes: ModelInfoSizes;
    extraAttrs?: Record<string, string>;
}

export interface ModelInfoResult {
    data: ModelInfo;
    error?: string;
}
