<template>
  <div class="image-preview">
    <div class="preview-container">
      <img v-if="src" :src="src" :alt="alt" class="preview-image" />
      <div v-else class="preview-placeholder">
        <el-icon class="placeholder-icon"><Picture /></el-icon>
        <span>暂无图片</span>
      </div>
    </div>
    <div v-if="filename" class="preview-filename">{{ filename }}</div>
  </div>
</template>

<script setup>
import { Picture } from '@element-plus/icons-vue'

defineProps({
  src: {
    type: String,
    default: ''
  },
  alt: {
    type: String,
    default: 'Preview'
  },
  filename: {
    type: String,
    default: ''
  }
})
</script>

<style scoped>
.image-preview {
  display: inline-block;
}

.preview-container {
  position: relative;
  max-width: 100%;
  min-height: 150px;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  overflow: hidden;
  background: #f5f5f5;
}

.preview-image {
  display: block;
  max-width: 100%;
  max-height: 400px;
  margin: 0 auto;
}

.preview-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  color: #999;
}

.placeholder-icon {
  font-size: 48px;
  margin-bottom: 8px;
}

.preview-filename {
  margin-top: 8px;
  font-size: 12px;
  color: #666;
  text-align: center;
}
</style>