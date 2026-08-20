import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

/**
 * Dev proxies mirror the real route prefixes.
 *
 * The old config proxied `/api`, which no code has ever called — the frontend
 * talked to `http://localhost:8000` directly. That meant three networking
 * configurations existed (hardcoded URL, this proxy, and the nginx proxy) and
 * two of them were decoration. Everything now goes through the same relative
 * paths in dev and in Docker.
 */
const BACKEND = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  // Emit the automatic JSX runtime everywhere, including test files, so a
  // component file does not need a React import just to render under Vitest.
  esbuild: { jsx: "automatic" },
  server: {
    proxy: {
      "/network": { target: BACKEND, changeOrigin: true },
      "/sim": { target: BACKEND, changeOrigin: true },
      "/metrics": { target: BACKEND, changeOrigin: true },
      "/benchmark": { target: BACKEND, changeOrigin: true },
      "/experiments": { target: BACKEND, changeOrigin: true },
      "/health": { target: BACKEND, changeOrigin: true },
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.js"],
  },
});
