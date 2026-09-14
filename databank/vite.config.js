import vue from "@vitejs/plugin-vue";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  const configuredBase = env.VITE_PUBLIC_PATH || "/";
  const base = configuredBase.endsWith("/") ? configuredBase : `${configuredBase}/`;
  return {
    base,
    plugins: [vue()],
    build: { outDir: "dist", assetsDir: "assets" },
    server: { host: "127.0.0.1", port: 8080 },
  };
});
