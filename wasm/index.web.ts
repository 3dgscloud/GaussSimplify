import { GaussSimplifyBase } from './base';
export * from './types';

export class GaussSimplify extends GaussSimplifyBase {
    protected async importWasmModule() {
        // @ts-ignore
        return import('./gauss_simplify.web.js');
    }
}

let _instance: GaussSimplify | null = null;
export async function createGaussSimplify(factory?: any): Promise<GaussSimplify> {
    if (!_instance) {
        _instance = new GaussSimplify();
        await _instance.init(factory);
    }
    return _instance;
}

export function destroyGaussSimplify(): void {
    if (_instance) {
        _instance.dispose();
        _instance = null;
    }
}
