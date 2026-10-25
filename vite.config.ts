import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5180, watch: { ignored: ['**/backend/**'] } },
});
