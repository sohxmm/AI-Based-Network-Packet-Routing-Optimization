/**
 * ESLint configuration.
 *
 * package.json defined a `lint` script and installed four eslint plugins, but
 * no configuration file existed anywhere in the repository, so `npm run lint`
 * failed immediately with "no configuration found". A lint script that cannot
 * run is worse than no lint script: it looks like the code is checked.
 */
module.exports = {
  root: true,
  env: { browser: true, es2022: true, node: true },
  extends: [
    "eslint:recommended",
    "plugin:react/recommended",
    "plugin:react/jsx-runtime",
    "plugin:react-hooks/recommended",
  ],
  ignorePatterns: ["dist", "node_modules", ".eslintrc.cjs"],
  parserOptions: { ecmaVersion: "latest", sourceType: "module" },
  settings: { react: { version: "18.3" } },
  plugins: ["react-refresh"],
  rules: {
    "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
    "react/prop-types": "off",
    "no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
  },
  overrides: [
    {
      files: ["**/__tests__/**", "**/*.test.jsx", "**/*.test.js"],
      env: { node: true },
      globals: {
        describe: "readonly",
        it: "readonly",
        expect: "readonly",
        vi: "readonly",
        beforeEach: "readonly",
        afterEach: "readonly",
      },
    },
  ],
};
