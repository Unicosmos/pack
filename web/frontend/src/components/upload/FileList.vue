<template>
  <div v-if="files.length > 0" class="preview-container show">
    <div class="files-summary">
      <span>已选择 {{ files.length }} 个文件</span>
      <button class="btn-clear" @click="handleClear">清空</button>
    </div>
    <div class="files-list">
      <div v-for="(file, idx) in files" :key="idx" class="file-item">
        <img :src="file.preview" class="file-thumb" :alt="file.name">
        <div class="file-info">
          <div class="file-name">{{ file.name }}</div>
          <div class="file-size">{{ formatFileSize(file.size) }}</div>
        </div>
        <button class="file-remove" @click="handleRemove(idx)">×</button>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  files: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['remove', 'clear'])

const handleRemove = (idx) => {
  emit('remove', idx)
}

const handleClear = () => {
  emit('clear')
}

const formatFileSize = (bytes) => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}
</script>

<style scoped>
.preview-container {
  margin-top: 20px;
  display: none;
}

.preview-container.show {
  display: block;
}

.files-summary {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 1px solid #f0f0f0;
}

.files-summary span {
  font-size: 14px;
  color: #666;
}

.btn-clear {
  padding: 6px 12px;
  background: #f5f5f5;
  color: #666;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

.btn-clear:hover {
  background: #e0e0e0;
}

.files-list {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  max-height: 300px;
  overflow-y: auto;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px;
  background: #fafafa;
  border-radius: 8px;
  border: 1px solid #f0f0f0;
  width: calc(50% - 6px);
  min-width: 280px;
}

.file-thumb {
  width: 50px;
  height: 50px;
  object-fit: cover;
  border-radius: 4px;
  border: 1px solid #e0e0e0;
}

.file-info {
  flex: 1;
  min-width: 0;
}

.file-name {
  font-size: 13px;
  color: #333;
  margin-bottom: 3px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-size {
  font-size: 11px;
  color: #999;
}

.file-remove {
  padding: 4px 8px;
  background: #fef0f0;
  color: #f56c6c;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
}

.file-remove:hover {
  background: #fde2e2;
}
</style>
