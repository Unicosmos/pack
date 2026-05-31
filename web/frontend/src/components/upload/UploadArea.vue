<template>
  <div class="upload-area" :class="{ dragover: isDragover }" @click="triggerUpload" @dragover.prevent="isDragover = true" @dragleave.prevent="isDragover = false" @drop.prevent="handleDrop">
    <input type="file" ref="fileInput" @change="handleFileSelect" accept="image/*" multiple directory webkitdirectory style="display: none">
    <div class="upload-content">
      <span class="upload-icon">📁</span>
      <span class="upload-text">点击或拖拽上传图片/文件夹</span>
    </div>
    <div class="upload-hint">支持 JPG、PNG，可多选</div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'

const emit = defineEmits(['files-selected'])

const fileInput = ref(null)
const isDragover = ref(false)

const triggerUpload = () => {
  fileInput.value?.click()
}

const handleFileSelect = (e) => {
  const files = Array.from(e.target.files)
  const imageFiles = filterAndWrapFiles(files)
  if (imageFiles.length > 0) {
    emit('files-selected', imageFiles)
  } else {
    ElMessage.warning('未找到图片文件')
  }
}

const handleDrop = (e) => {
  isDragover.value = false
  const files = Array.from(e.dataTransfer.files)
  const imageFiles = filterAndWrapFiles(files)
  if (imageFiles.length > 0) {
    emit('files-selected', imageFiles)
  } else {
    ElMessage.warning('未找到图片文件')
  }
}

const filterAndWrapFiles = (files) => {
  return files
    .filter(f => f.type.startsWith('image/') || f.type === '')
    .filter(f => {
      if (f.type.startsWith('image/')) return true
      const ext = f.name.toLowerCase().split('.').pop()
      return ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(ext)
    })
    .map(file => ({
      file: file,
      preview: URL.createObjectURL(file),
      name: file.name,
      size: file.size,
      type: file.type || 'image/' + file.name.toLowerCase().split('.').pop()
    }))
}

defineExpose({
  clearInput: () => {
    if (fileInput.value) {
      fileInput.value.value = ''
    }
  }
})
</script>

<style scoped>
.upload-area {
  border: 2px dashed var(--color-border);
  border-radius: var(--radius-md);
  padding: 36px 24px;
  text-align: center;
  cursor: pointer;
  transition: border-color var(--transition-fast);
}

.upload-area:hover {
  border-color: var(--color-primary);
}

.upload-area.dragover {
  border-color: var(--color-primary);
  border-style: solid;
  background: rgba(59, 159, 232, 0.08);
}

.upload-content {
  margin-bottom: 6px;
}

.upload-icon {
  font-size: 16px;
  margin-right: 6px;
}

.upload-text {
  color: var(--color-primary);
  font-size: 14px;
}

.upload-hint {
  color: var(--color-text-tertiary);
  font-size: 12px;
}
</style>