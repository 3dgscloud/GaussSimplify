/**
 * GaussSimplify WASM Base Class
 */
import type {
    GaussianCloudIR,
    SimplifyOptions,
    ReadResult,
    WriteResult,
    SupportedFormat,
    ReadOptions,
    WriteOptions,
    ModelInfoResult,
} from './types';

export interface EmscriptenModule {
    GaussSimplifyWASM: new () => GaussSimplifyWASMInstance;
    [key: string]: any;
}

export interface GaussSimplifyWASMInstance {
    read(data: Uint8Array, format: string, strict: boolean): any;
    simplify(ir: any, ratio: number, knn_k: number, merge_cap: number,
             opacity_prune_threshold: number, target_sh_degree: number,
             sor_nb_neighbors: number, sor_std_ratio: number): any;
    write(ir: any, format: string, strict: boolean): any;
    getModelInfo(ir: any): any;
    getSupportedFormats(): string[];
    getVersion(): string;
    delete(): void;
}

export abstract class GaussSimplifyBase {
    protected module: EmscriptenModule | null = null;
    protected instance: GaussSimplifyWASMInstance | null = null;
    protected initPromise: Promise<void> | null = null;

    protected abstract importWasmModule(): Promise<any>;

    async init(moduleFactory?: (overrides?: any) => EmscriptenModule): Promise<void> {
        if (this.initPromise) return this.initPromise;

        this.initPromise = (async () => {
            try {
                let moduleInstance: any;
                if (moduleFactory) {
                    moduleInstance = moduleFactory();
                } else {
                    const createModule = await this.importWasmModule();
                    const factory = createModule.default;
                    moduleInstance = typeof factory === 'function' ? factory() : factory;
                }

                if (moduleInstance && typeof moduleInstance.then === 'function') {
                    moduleInstance = await moduleInstance;
                }

                this.module = moduleInstance as EmscriptenModule;
                this.instance = new this.module.GaussSimplifyWASM();
            } catch (error) {
                this.initPromise = null;
                throw new Error(
                    `GaussSimplify Init Failed: ${error instanceof Error ? error.message : String(error)}`
                );
            }
        })();
        return this.initPromise;
    }

    protected ensureInitialized(): void {
        if (!this.instance) throw new Error('GaussSimplify not initialized. Call init() first.');
    }

    async read(data: ArrayBuffer | Uint8Array, format: string, options: ReadOptions = {}): Promise<ReadResult> {
        this.ensureInitialized();
        const input = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
        const result = this.instance!.read(input, format, options.strict || false);
        if (result.error) throw new Error(result.error);
        return {
            data: result.data,
            ...(result.warning && { warning: result.warning }),
        } as ReadResult;
    }

    async simplify(ir: GaussianCloudIR, options: SimplifyOptions = {}): Promise<ReadResult> {
        this.ensureInitialized();
        const result = this.instance!.simplify(
            ir,
            options.ratio ?? 0.1,
            options.knn_k ?? 16,
            options.merge_cap ?? 0.5,
            options.opacity_prune_threshold ?? 0.1,
            options.target_sh_degree ?? -1,
            options.sor_nb_neighbors ?? 0,
            options.sor_std_ratio ?? 2.0
        );
        if (result.error) throw new Error(result.error);
        return { data: result.data } as ReadResult;
    }

    async write(ir: GaussianCloudIR, format: string, options: WriteOptions = {}): Promise<WriteResult> {
        this.ensureInitialized();
        const result = this.instance!.write(ir, format, options.strict || false);
        if (result.error) throw new Error(result.error);
        return { data: result.data } as WriteResult;
    }

    async getModelInfo(ir: GaussianCloudIR): Promise<ModelInfoResult> {
        this.ensureInitialized();
        const result = this.instance!.getModelInfo(ir);
        if (result.error) throw new Error(result.error);
        return { data: result.data } as ModelInfoResult;
    }

    getSupportedFormats(): SupportedFormat[] {
        this.ensureInitialized();
        return this.instance!.getSupportedFormats() as SupportedFormat[];
    }

    getVersion(): string {
        this.ensureInitialized();
        return this.instance!.getVersion();
    }

    isFormatSupported(format: string): boolean {
        this.ensureInitialized();
        const formats = this.getSupportedFormats();
        return formats.includes(format as SupportedFormat);
    }

    dispose(): void {
        if (this.instance) {
            this.instance.delete();
            this.instance = null;
        }
        this.module = null;
        this.initPromise = null;
    }
}
