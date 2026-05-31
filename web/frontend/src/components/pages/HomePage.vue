<template>
  <div class="home-page">
    <div class="home-content">
      <SysStatusBar @navigate="handleNavigate" />

      <section class="upload-section">
        <UploadArea @files-selected="handleFilesSelected" />
        <FileList :files="store.selectedFiles" @remove="handleFileRemove" @clear="handleClearFiles" />

        <div v-if="store.selectedFiles.length > 0" class="btn-group" style="margin-top: var(--spacing-lg);">
          <button class="btn btn-success" :disabled="isProcessing" @click="handleUpload">
            {{ isProcessing ? '上传中...' : (store.selectedFiles.length > 1 ? '📦 批量上传' : '📷 上传图片') }}
          </button>
          <button class="btn btn-secondary" @click="handleReset">🔄 重置</button>
        </div>
      </section>

      <section v-if="store.error" class="error-state">
        <div class="error-icon">❌</div>
        <div class="error-text">{{ store.error }}</div>
      </section>

      <section v-if="store.batchResults.length > 0" class="result-section">
        <div class="result-header">
          <h3>📊 检测结果</h3>
          <span class="result-count">共 {{ store.batchResults.length }} 张图片</span>
        </div>
        <div class="result-body">
          <DetectionList :results="store.batchResults" :mode="store.currentMode" />

          <div class="btn-group" style="margin-top: var(--spacing-lg);">
            <button class="btn btn-primary" @click="goToTasks">📋 前往任务列表查看</button>
            <button class="btn btn-secondary" @click="clearBatchResults">清空结果</button>
          </div>
        </div>
      </section>

      <RecentTasks
        v-if="showEmptyState"
        @view-task="handleViewTask"
        @navigate="handleNavigate"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { ElMessage } from 'element-plus'
import taskApi from '@api/taskApi'
import { useAppStore } from '@stores/app'
import SysStatusBar from '@home/SysStatusBar.vue'
import RecentTasks from '@home/RecentTasks.vue'
import UploadArea from '@upload/UploadArea.vue'
import FileList from '@upload/FileList.vue'
import DetectionList from '@task/DetectionList.vue'

const store = useAppStore()
const isProcessing = ref(false)

const showEmptyState = computed(() => {
  return store.batchResults.length === 0 && store.isIdle && !store.error && store.batchTaskIds.length === 0
})

const handleFilesSelected = (files) => {
  store.addFiles(files)
}

const handleFileRemove = (idx) => {
  const file = store.selectedFiles[idx]
  if (file.preview) {
    URL.revokeObjectURL(file.preview)
  }
  store.removeFileAt(idx)
}

const handleClearFiles = () => {
  store.clearFiles()
}

const handleReset = () => {
  store.clearFiles()
  store.reset()
}

const handleNavigate = (page) => {
  store.setPage(page)
}

const handleViewTask = (task) => {
  store.pendingTaskId = task.id
  store.setPage('tasks')
}

const handleUpload = async () => {
  if (store.selectedFiles.length === 0) {
    ElMessage.warning('请先选择图片')
    return
  }

  isProcessing.value = true

  try {
    const uploadedTasks = []

    for (let i = 0; i < store.selectedFiles.length; i++) {
      const fileWrapper = store.selectedFiles[i]
      const file = fileWrapper.file || fileWrapper

      try {
        const res = await taskApi.uploadTask(file)
        if (res.success && res.data) {
          uploadedTasks.push({
            taskId: res.data.id,
            fileName: fileWrapper.name,
            status: res.data.status
          })
          ElMessage.success(`已上传 ${i + 1}/${store.selectedFiles.length}: ${fileWrapper.name}`)
        } else {
          ElMessage.error(`上传失败 ${fileWrapper.name}: ${res.error || '未知错误'}`)
        }
      } catch (err) {
        console.error('Upload error for', fileWrapper.name, err)
        ElMessage.error(`上传失败 ${fileWrapper.name}: ${err.message || err.detail || '未知错误'}`)
      }
    }

    if (uploadedTasks.length > 0) {
      store.clearFiles()
      ElMessage.success(`上传完成！共 ${uploadedTasks.length} 个任务已添加到任务列表`)
      store.setPage('tasks')
    } else {
      ElMessage.warning('没有任务上传成功')
    }
  } catch (err) {
    console.error('上传异常:', err)
    const errorMsg = err.detail || err.message || '处理失败'
    store.completeError(errorMsg)
    ElMessage.error(errorMsg)
  } finally {
    isProcessing.value = false
  }
}

const clearBatchResults = () => {
  store.batchResults = []
  store.clearFiles()
}

const goToTasks = () => {
  store.setPage('tasks')
}
</script>

<style scoped>
.home-page {
  padding-bottom: var(--spacing-xl);
}

.home-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 1200px;
  margin: 0 auto;
}

.upload-section {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 20px;
}

.result-section {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 16px;
  border-bottom: 1px solid var(--color-border);
}

.result-header h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.result-count {
  font-size: 12px;
  color: var(--color-text-tertiary);
}

.result-body {
  padding: 16px;
}

.error-state {
  background: rgba(248, 113, 113, 0.1);
  border: 1px solid var(--color-danger);
  border-radius: var(--radius-md);
  padding: 12px 16px;
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--color-danger);
}

.error-icon {
  font-size: 18px;
}

@media (max-width: 768px) {
  .upload-section {
    padding: 16px;
  }
}
</style>