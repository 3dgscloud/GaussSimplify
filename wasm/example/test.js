#!/usr/bin/env node

/**
 * Local test script - Test @gausssimplify/wasm package using tiny_gauss.ply
 *
 * Usage:
 *   1. Install local package: npm install
 *   2. Run test: npm test
 */

import { createGaussSimplify } from '@gausssimplify/wasm';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TEST_FILE = path.join(__dirname, 'tiny_gauss.ply');

async function test() {
    console.log('Testing @gausssimplify/wasm local package\n');
    console.log(`Test file: ${TEST_FILE}\n`);

    try {
        // 1. Check test file
        if (!fs.existsSync(TEST_FILE)) {
            console.error(`Error: Test file does not exist: ${TEST_FILE}`);
            process.exit(1);
        }

        // 2. Initialize
        console.log('1. Initializing GaussSimplify...');
        const api = await createGaussSimplify();
        console.log('   OK\n');

        // 3. Version
        console.log('2. Getting version...');
        const version = api.getVersion();
        console.log(`   Version: ${version}\n`);

        // 4. Supported formats
        console.log('3. Getting supported formats...');
        const formats = api.getSupportedFormats();
        console.log(`   Formats: ${formats.join(', ')}\n`);

        // 5. Read test file
        console.log('4. Reading test file...');
        const inputData = fs.readFileSync(TEST_FILE);
        console.log(`   File size: ${inputData.length} bytes`);

        const readResult = await api.read(inputData, 'ply');
        const ir = readResult.data;
        console.log(`   Points: ${ir.numPoints}`);
        console.log(`   SH degree: ${ir.meta.shDegree}`);
        console.log(`   Source format: ${ir.meta.sourceFormat || 'Unknown'}\n`);

        // 6. Get model info
        console.log('5. Getting model info...');
        const infoResult = await api.getModelInfo(ir);
        if (infoResult.data) {
            const info = infoResult.data;
            console.log(`   Points: ${info.basic.numPoints}`);
            console.log(`   SH degree: ${info.rendering.shDegree}`);
            console.log(`   Antialiased: ${info.rendering.antialiased}`);
            if (info.bounds) {
                console.log(`   Bounds X: [${info.bounds.x[0]}, ${info.bounds.x[1]}]`);
                console.log(`   Bounds Y: [${info.bounds.y[0]}, ${info.bounds.y[1]}]`);
                console.log(`   Bounds Z: [${info.bounds.z[0]}, ${info.bounds.z[1]}]`);
            }
            if (info.scaleStats) {
                console.log(`   Scale: min=${info.scaleStats.min} max=${info.scaleStats.max} avg=${info.scaleStats.avg}`);
            }
            if (info.alphaStats) {
                console.log(`   Alpha: min=${info.alphaStats.min} max=${info.alphaStats.max} avg=${info.alphaStats.avg}`);
            }
            console.log(`   Total size: ${info.sizes.total}`);
        }
        console.log('');

        // 7. Simplify
        console.log('6. Simplifying (ratio=0.5)...');
        const simplifyResult = await api.simplify(ir, { ratio: 0.5 });
        const simplified = simplifyResult.data;
        console.log(`   Input points: ${ir.numPoints}`);
        console.log(`   Output points: ${simplified.numPoints}`);
        console.log(`   Reduction: ${((1 - simplified.numPoints / ir.numPoints) * 100).toFixed(1)}%\n`);

        // 8. Write to various formats
        console.log('7. Writing simplified result...');
        const outputFormats = ['splat', 'ply', 'spz'];

        for (const outFormat of outputFormats) {
            try {
                const writeResult = await api.write(simplified, outFormat);
                const outputFile = path.join(__dirname, `output_simplified.${outFormat}`);
                fs.writeFileSync(outputFile, writeResult.data);
                console.log(`   ${outFormat}: ${writeResult.data.length} bytes -> ${outputFile}`);
            } catch (error) {
                console.log(`   ${outFormat} failed: ${error.message}`);
            }
        }

        // 9. Test with different options
        console.log('\n8. Testing with different simplify options...');
        const opts = [
            { ratio: 0.8, label: 'ratio=0.8 (keep 80%)' },
            { ratio: 0.3, target_sh_degree: 0, label: 'ratio=0.3, sh_degree=0' },
            { ratio: 0.1, knn_k: 8, label: 'ratio=0.1, knn_k=8' },
        ];

        for (const opt of opts) {
            try {
                const res = await api.simplify(ir, opt);
                console.log(`   ${opt.label}: ${ir.numPoints} -> ${res.data.numPoints} points`);
            } catch (error) {
                console.log(`   ${opt.label} failed: ${error.message}`);
            }
        }

        console.log('\nAll tests completed!');

    } catch (error) {
        console.error('\nTest failed:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

test();
