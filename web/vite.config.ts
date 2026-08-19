import { readFile } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const fixtureDirectory = fileURLToPath(
  new URL("../contracts/examples/", import.meta.url),
);
const fixtureNames = new Set([
  "melody.request.json",
  "harmonize.response.json",
]);

function contractFixtures(): Plugin {
  const middleware = async (
    request: IncomingMessage,
    response: ServerResponse,
    next: (error?: unknown) => void,
  ) => {
    const filename = request.url?.split("?")[0].replace(/^\/+/, "");
    if (!filename || !fixtureNames.has(filename)) {
      next();
      return;
    }
    try {
      const content = await readFile(`${fixtureDirectory}${filename}`);
      response.statusCode = 200;
      response.setHeader("Content-Type", "application/json");
      response.setHeader("Cache-Control", "no-store");
      response.end(content);
    } catch (error) {
      next(error);
    }
  };

  return {
    name: "harmonaizer-contract-fixtures",
    configureServer(server) {
      server.middlewares.use("/__contracts/examples", middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/__contracts/examples", middleware);
    },
  };
}

export default defineConfig({
  plugins: [react(), contractFixtures()],
  server: {
    host: "127.0.0.1",
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
