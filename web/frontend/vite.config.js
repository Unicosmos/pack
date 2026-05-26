import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@pages': resolve(__dirname, 'src/components/pages'),
      '@ui': resolve(__dirname, 'src/components/ui'),
      '@task': resolve(__dirname, 'src/components/task'),
      '@sku': resolve(__dirname, 'src/components/sku'),
      '@upload': resolve(__dirname, 'src/components/upload'),
      '@hooks': resolve(__dirname, 'src/components/hooks'),
      '@layout': resolve(__dirname, 'src/components/layout'),
      '@api': resolve(__dirname, 'src/api'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@stores': resolve(__dirname, 'src/stores')
    }
  },
  build: {
    outDir: resolve(__dirname, '../backend/static'),
    emptyOutDir: true
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/static': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})