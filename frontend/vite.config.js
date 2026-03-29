import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Do not run `npx vite` from a path containing `%` in a folder name — Vite's dev server crashes.
// `npm run dev` uses `vite build --watch` plus static `serve` (no Vite HTML middleware at runtime).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
});
