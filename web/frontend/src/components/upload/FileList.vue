<template>
  <div v-if="files.length > 0" class="file-list">
    <div class="file-list-header">
      <span>已选择 <b>{{ files.length }}</b> 个文件</span>
      <span class="btn-clear" @click="handleClear">清空</span>
    </div>
    <div v-for="(file, idx) in files" :key="idx" class="file-item">
      <span>
        <span class="file-icon">🖼️</span>
        <span class="file-name">{{ file.name }}</span>
        <span class="file-size">{{ formatFileSize(file.size) }}</span>
      </span>
      <span class="file-remove" @click="handleRemove(idx)">删除</span>
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
.file-list {
  margin-top: 16px;
}

.file-list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  color: var(--color-text-secondary);
  font-size: 13px;
}

.btn-clear {
  color: var(--color-danger);
  cursor: pointer;
  font-size: 12px;
}

.btn-clear:hover {
  opacity: 0.8;
}

.file-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 10px 14px;
  margin-bottom: 8px;
}

.file-item:last-child {
  margin-bottom: 0;
}

.file-name {
  color: var(--color-text-primary);
  margin-left: 6px;
}

.file-size {
  color: var(--color-text-tertiary);
  font-size: 12px;
  margin-left: 10px;
}

.file-remove {
  color: var(--color-danger);
  cursor: pointer;
  font-size: 12px;
}

.file-remove:hover {
  opacity: 0.8;
}
</style>