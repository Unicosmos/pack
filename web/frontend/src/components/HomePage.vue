<template>
  <div class="home-page">
    <StatusBanner :status="store.systemStatus" />

    <div class="header">
      <h1>📦 箱货检测与SKU匹配</h1>
      <p>上传图片，自动检测箱体并匹配SKU</p>
      <div class="header-tip">
        💡 检测结果可在<a href="#" @click.prevent="goToTasks">任务列表</a>中进行审核和匹配
      </div>
    </div>

    <div class="section">
      <UploadArea @files-selected="handleFilesSelected" />
      <FileList :files="store.selectedFiles" @remove="handleFileRemove" @clear="handleClearFiles" />

      <div class="btn-group">
        <button class="btn btn-success" :disabled="store.selectedFiles.length === 0 || isProcessing" @click="handleUpload">
          {{ isProcessing ? '上传中...' : (store.selectedFiles.length > 1 ? '📦 批量上传' : '🔍 上传图片') }}
        </button>
        <button class="btn btn-default" @click="handleReset">🔄 重置</button>
      </div>
    </div>

    <div v-if="store.error" class="section error">
      <div class="error-icon">❌</div>
      <div class="error-text">{{ store.error }}</div>
    </div>

    <div v-if="store.batchResults.length > 0" class="section">
      <div class="result-title">
        <h2>📊 检测结果</h2>
        <span class="result-count">共 {{ store.batchResults.length }} 张图片</span>
      </div>

      <DetectionList :results="store.batchResults" :mode="store.currentMode" @review="handleReview" />

      <div class="result-actions">
        <button class="btn btn-primary" @click="goToTasks">📋 前往任务列表进行审核和匹配</button>
        <button class="btn btn-default" @click="clearBatchResults">清空结果</button>
      </div>
    </div>

    <div v-if="showEmptyState" class="section empty">
      <div class="empty-icon">📷</div>
      <p>请上传图片开始检测</p>
      <p class="empty-tip">检测完成后请前往<a href="#" @click.prevent="goToTasks">任务列表</a>进行审核和匹配</p>
    </div>

    <ReviewDialog
      v-model="reviewDialogVisible"
      :task="currentTask"
      @submit="handleReviewSubmit"
    />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { detector, tasks } from '../api/client'
import { useAppStore } from '../stores/app'
import StatusBanner from './StatusBanner.vue'
import UploadArea from './upload/UploadArea.vue'
import FileList from './upload/FileList.vue'
import DetectionList from './result/DetectionList.vue'
import ReviewDialog from './result/ReviewDialog.vue'

const store = useAppStore()
const isProcessing = ref(false)

const reviewDialogVisible = ref(false)
const currentTask = ref(null)

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
      ElMessage.success(`上传完成！共 ${uploadedTasks.length} 个任务已添加到任务列表，请前往任务列表进行检测`)
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

const handleReview = (result, idx) => {
  if (!result.success) return

  const boxesWithMatch = result.boxes?.map((box, boxIdx) => ({
    ...box,
    match_result: result.matches?.[String(boxIdx)] || result.matches?.[boxIdx]
  })) || []

  currentTask.value = {
    id: idx,
    image_name: result.fileName,
    image_path: null,
    result: {
      ...result,
      detections: { boxes: boxesWithMatch },
      image_with_boxes: result.image_with_boxes
    }
  }
  reviewDialogVisible.value = true
}

const handleReviewSubmit = async ({ task, boxes, approvedCount, rejectedCount }) => {
  reviewDialogVisible.value = false
  
  if (task.result.taskId) {
    try {
      await tasks.reviewTask(task.result.taskId, boxes)
      ElMessage.success(`审核成功：通过 ${approvedCount} 个，拒绝 ${rejectedCount} 个，已保存到数据库`)
    } catch (err) {
      ElMessage.error('保存审核结果失败: ' + (err.message || '未知错误'))
    }
  } else {
    ElMessage.success(`审核成功：通过 ${approvedCount} 个，拒绝 ${rejectedCount} 个`)
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
  padding-bottom: 40px;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 25px 30px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  border-radius: 12px;
  margin-bottom: 20px;
}

.header h1 {
  font-size: 24px;
  font-weight: 600;
  margin: 0;
}

.header p {
  font-size: 14px;
  opacity: 0.9;
  margin-top: 5px;
}

.section {
  background: white;
  border-radius: 12px;
  padding: 25px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.btn-group {
  margin-top: 20px;
  display: flex;
  gap: 10px;
}

.btn {
  padding: 14px 32px;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-success {
  background: linear-gradient(135deg, #67c23a 0%, #52c41a 100%);
  color: white;
}

.btn-success:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(103, 194, 58, 0.4);
}

.btn-default {
  background: #909399;
  color: white;
}

.btn-default:hover {
  background: #7d8085;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.error {
  background: #fef0f0;
  border: 1px solid #fde2e2;
  padding: 20px;
  border-radius: 8px;
  color: #f56c6c;
  display: flex;
  align-items: center;
  gap: 12px;
}

.error-icon {
  font-size: 24px;
}

.error-text {
  font-size: 14px;
}

.empty {
  text-align: center;
  padding: 60px;
  color: #999;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 15px;
}

.result-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
}

.result-title h2 {
  margin: 0;
}

.result-count {
  font-size: 14px;
  color: #999;
  margin-left: 10px;
  font-weight: normal;
}

.header-tip {
  font-size: 13px;
  margin-top: 10px;
  opacity: 0.9;
}

.header-tip a {
  color: white;
  text-decoration: underline;
}

.result-actions {
  margin-top: 20px;
  display: flex;
  gap: 10px;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.empty-tip {
  font-size: 13px;
  color: #999;
  margin-top: 10px;
}

.empty-tip a {
  color: #667eea;
  text-decoration: none;
}

.empty-tip a:hover {
  text-decoration: underline;
}
</style>
