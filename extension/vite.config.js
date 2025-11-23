import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        // Copy manifest
        { src: "public/manifest.json", dest: "" },
        // Copy background script
        { src: "src/background.js", dest: "" },
        // Copy popup
        { src: "src/popup.html", dest: "" },
        // Optional: copy popup.js if not bundled
        { src: "src/popup.js", dest: "" }
      ],
    }),
  ],
  build: {
    outDir: "dist",
  },
});
