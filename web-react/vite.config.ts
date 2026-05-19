import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../web-dist",
    emptyOutDir: true,
    assetsInlineLimit: 0,
  },
  server: {
    fs: {
      // allow Vite to follow our symlinks into the project root
      allow: [".."],
    },
  },
});
