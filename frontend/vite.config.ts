import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API + vendored libs to the Python backend so the
// React app can run with `vite dev` while the backend stays on :8000:
//   1. python -m src.serve --port 8000        (or --live)
//   2. npm run dev                             (http://127.0.0.1:5173)
// Production (`npm run build`) emits frontend/dist/, which src/serve.py
// serves directly at http://127.0.0.1:8000/ when present.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // BACKEND_URL is set by run.py when the backend port != 8000.
      "/v1": process.env.BACKEND_URL ?? "http://127.0.0.1:8000",
      "/lib": process.env.BACKEND_URL ?? "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
