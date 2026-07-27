import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Vite config for the Layer 9 dashboard.
 *
 * base: "./"  — the built bundle is served by the FastAPI app (Layer 8) from
 *               a static mount, so asset URLs must be relative.
 * proxy      — `npm run dev` runs on :5173 with hot reload while the real
 *               Python API runs on :8000; /api and /ws are proxied across so
 *               dev and production hit identical paths.
 *
 * Everything is bundled locally (including three.js) — no CDN, no runtime
 * downloads. The dashboard must work with the network cable unplugged, same
 * as the rest of the pipeline.
 */
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // No sourcemaps: dist/ is committed so the dashboard works without Node,
    // and maps would triple its size for no benefit — debugging happens in
    // `npm run dev`, which has full sourcemaps.
    sourcemap: false,
    rollupOptions: {
      output: {
        // three.js is most of the bundle and never changes between edits —
        // splitting it keeps the app chunk small and cacheable.
        manualChunks: { three: ["three"], react: ["react", "react-dom"] },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
