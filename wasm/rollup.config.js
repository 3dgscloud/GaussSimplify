import typescript from '@rollup/plugin-typescript';

const webExternal = ['./gauss_simplify.web.js'];
const nodeExternal = ['./gauss_simplify.node.js', 'path', 'fs', 'module', 'crypto', 'url'];

export default [
  {
    input: 'index.web.ts',
    output: {
      dir: 'dist',
      entryFileNames: 'index.web.js',
      format: 'es',
      sourcemap: true,
    },
    plugins: [
      typescript({ tsconfig: './tsconfig.json', declaration: true, declarationDir: 'dist', declarationMap: true }),
    ],
    external: webExternal,
  },
  {
    input: 'index.node.ts',
    output: [
      {
        dir: 'dist',
        entryFileNames: 'index.node.js',
        format: 'es',
        sourcemap: true,
      }
    ],
    plugins: [
      typescript({ tsconfig: './tsconfig.json', declaration: true, declarationDir: 'dist', declarationMap: true }),
    ],
    external: nodeExternal,
  },
];
