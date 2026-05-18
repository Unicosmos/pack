<template>
  <div class="upload-area" :class="{ dragover: isDragover }" @click="triggerUpload" @dragover.prevent="isDragover = true" @dragleave.prevent="isDragover = false" @drop.prevent="handleDrop">
    <input type="file" ref="fileInput" @change="handleFileSelect" accept="image/*" multiple directory webkitdirectory style="display: none">
    <div class="upload-icon">📤</div>
    <div class="upload-text">点击或拖拽上传图片/文件夹</div>
    <div class="upload-hint">支持 JPG、PNG 格式，可多选或选择文件夹</div>
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
  border: 2px dashed #667eea;
  border-radius: 10px;
  padding: 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
}

.upload-area:hover {
  background: #f0f5ff;
  border-color: #764ba2;
}

.upload-area.dragover {
  background: #e8f0fe;
  border-color: #667eea;
  border-style: solid;
}

.upload-icon {
  font-size: 48px;
  color: #667eea;
  margin-bottom: 15px;
}

.upload-text {
  color: #667eea;
  font-size: 18px;
  margin-bottom: 8px;
}

.upload-hint {
  color: #999;
  font-size: 14px;
}
</style>
