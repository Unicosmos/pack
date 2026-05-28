<template>
  <div class="home-page">
    <PageContainer>
      <StatusBanner :status="store.systemStatus" />

      <section class="panel">
        <div class="panel-body">
          <UploadArea @files-selected="handleFilesSelected" />
          <FileList :files="store.selectedFiles" @remove="handleFileRemove" @clear="handleClearFiles" />

          <div class="btn-group" style="margin-top: var(--spacing-lg);">
            <button class="btn btn-success" :disabled="store.selectedFiles.length === 0 || isProcessing" @click="handleUpload">
              {{ isProcessing ? '上传中...' : (store.selectedFiles.length > 1 ? '📦 批量上传' : '📷 上传图片') }}
            </button>
            <button class="btn btn-secondary" @click="handleReset">🔄 重置</button>
          </div>
        </div>
      </section>

      <section v-if="store.error" class="error-state">
        <div class="error-icon">❌</div>
        <div class="error-text">{{ store.error }}</div>
      </section>

      <section v-if="store.batchResults.length > 0" class="panel">
        <div class="panel-header">
          <h2>📊 检测结果</h2>
          <span class="result-count">共 {{ store.batchResults.length }} 张图片</span>
        </div>
        <div class="panel-body">
          <DetectionList :results="store.batchResults" :mode="store.currentMode" />

          <div class="btn-group" style="margin-top: var(--spacing-lg);">
            <button class="btn btn-primary" @click="goToTasks">📋 前往任务列表查看</button>
            <button class="btn btn-secondary" @click="clearBatchResults">清空结果</button>
          </div>
        </div>
      </section>

      <section v-if="showEmptyState" class="panel">
        <div class="empty-state">
          <div class="empty-icon">📷</div>
          <p>请上传图片开始识别</p>
          <p class="empty-tip">识别完成后请前往<a href="#" @click.prevent="goToTasks">任务列表</a>查看</p>
        </div>
      </section>
    </PageContainer>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { tasks } from '@api/client'
import { useAppStore } from '@stores/app'
import PageContainer from '@layout/PageContainer.vue'
import StatusBanner from '@ui/StatusBanner.vue'
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
        const task = await tasks.upload(file)
        uploadedTasks.push({
          taskId: task.id,
          fileName: fileWrapper.name,
          status: task.status
        })
        ElMessage.success(`已上传 ${i + 1}/${store.selectedFiles.length}: ${fileWrapper.name}`)
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

.header-tip {
  font-size: var(--font-size-sm);
  margin-top: var(--spacing-sm);
  opacity: 0.9;
}

.header-tip a {
  color: white;
  text-decoration: underline;
}

.result-count {
  font-size: var(--font-size-sm);
  color: var(--color-text-tertiary);
  font-weight: normal;
}

.empty-tip {
  font-size: var(--font-size-sm);
  color: var(--color-text-tertiary);
  margin-top: var(--spacing-xs);
}

.empty-tip a {
  color: var(--color-primary);
}
</style>
