import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/network":     { target: "http://localhost:8000", changeOrigin: true },
      "/sim":         { target: "http://localhost:8000", changeOrigin: true },
      "/metrics":     { target: "http://localhost:8000", changeOrigin: true },
      "/benchmark":   { target: "http://localhost:8000", changeOrigin: true },
      "/experiments": { target: "http://localhost:8000", changeOrigin: true },
      "/ws":          { target: "ws://localhost:8000", ws: true },
    }
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: [],
  },
});
