<template>
  <div class="task-list-page">
    <div class="header">
      <h1>📋 任务列表</h1>
      <p>查看历史检测任务记录，支持批量任务进度追踪</p>
    </div>

    <div class="main-container" :class="{ 'split-view': showReviewPanel }">
      <!-- 任务列表区域 -->
      <div class="task-list-section">
        <div class="section">
          <div class="stats-row">
            <div class="stat-card">
              <div class="stat-value">{{ stats.total || 0 }}</div>
              <div class="stat-label">总任务数</div>
            </div>
            <div class="stat-card success">
              <div class="stat-value">{{ stats.completed || 0 }}</div>
              <div class="stat-label">已完成</div>
            </div>
            <div class="stat-card warning">
              <div class="stat-value">{{ stats.pending || 0 }}</div>
              <div class="stat-label">进行中</div>
            </div>
            <div class="stat-card danger">
              <div class="stat-value">{{ stats.failed || 0 }}</div>
              <div class="stat-label">失败</div>
            </div>
          </div>
        </div>

        <div class="section">
          <div class="table-toolbar">
            <div class="filter-tabs">
              <button :class="{ active: statusFilter === null }" @click="filterByStatus(null)">全部</button>
              <button :class="{ active: statusFilter === 'completed' }" @click="filterByStatus('completed')">已完成</button>
              <button :class="{ active: statusFilter === 'pending' }" @click="filterByStatus('pending')">进行中</button>
              <button :class="{ active: statusFilter === 'failed' }" @click="filterByStatus('failed')">失败</button>
            </div>
            <div class="toolbar-right">
              <button 
                class="btn btn-primary" 
                @click="batchDetectTasks"
                :disabled="selectedTasks.length === 0"
              >
                🔍 批量识别 ({{ selectedTasks.length }}个)
              </button>
              <div class="export-dropdown">
                <button class="btn btn-success" @click="toggleExportMenu" :disabled="selectedTasks.length === 0">
                  📥 导出 ({{ selectedTasks.length }}个)
                </button>
                <div v-if="showExportMenu" class="export-menu">
                  <div class="export-option" @click="exportTasks('json', false)">
                    <span>📄 JSON格式</span>
                    <span class="option-desc">结构清晰，便于程序处理</span>
                  </div>
                  <div class="export-option" @click="exportTasks('csv', false)">
                    <span>📊 CSV格式</span>
                    <span class="option-desc">适合Excel打开，便于数据分析</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div v-if="loading" class="loading">加载中...</div>

          <div v-else-if="tasks.length === 0" class="empty-state">
            <div class="empty-icon">📭</div>
            <p>暂无任务记录</p>
          </div>

          <div v-else class="task-table-container">
            <!-- 表头行 -->
            <div class="task-row-header header-row">
              <div class="task-cell task-select">
                <input type="checkbox" @click.stop="selectAllTasks" :checked="selectedTaskIds.length === tasks.length && tasks.length > 0" />
              </div>
              <div class="task-cell task-id">ID</div>
              <div class="task-cell task-thumb">缩略图</div>
              <div class="task-cell task-name">图片名称</div>
              <div class="task-cell task-counts">统计</div>
              <div class="task-cell task-date">创建时间</div>
              <div class="task-cell task-actions">操作</div>
              <div class="task-cell task-toggle"></div>
            </div>
            
            <div 
              v-for="task in tasks" 
              :key="task.id" 
              class="task-row"
              :class="{ 'expanded': expandedTaskId === task.id, 'selected': selectedTaskIds.includes(task.id) }"
            >
              <!-- 任务行 -->
              <div class="task-row-header" @click="toggleTask(task)">
                <div class="task-cell task-select">
                  <input type="checkbox" :value="task.id" @click.stop="toggleTaskSelection(task.id)" :checked="selectedTaskIds.includes(task.id)" />
                </div>
                <div class="task-cell task-id">{{ task.id }}</div>
                <div class="task-cell task-thumb">
                  <img 
                    :src="getTaskImagePath(task)" 
                    :alt="task.image_name"
                    class="task-thumb-img"
                    @error="$event.target.style.display='none'"
                  />
                </div>
                <div class="task-cell task-name">{{ task.image_name }}</div>
                <div class="task-cell task-counts">
                  <span class="count-item">检测: {{ task.box_count || 0 }}</span>
                  <span class="count-item">匹配: {{ task.matched_count || 0 }}</span>
                </div>
                <div class="task-cell task-date">{{ formatDate(task.created_at) }}</div>
                <div class="task-cell task-actions">
                  <div class="action-buttons">
                    <button v-if="shouldShowDetect(task)" class="btn-small btn-primary" @click.stop="detectTask(task)">识别</button>
                    <button v-if="shouldShowReview(task)" class="btn-small btn-warning" @click.stop="openReview(task)">审核</button>
                    <button v-if="shouldShowReDetect(task)" class="btn-small btn-danger" @click.stop="reDetectTask(task)">重新识别</button>
                    <button class="btn-icon" @click.stop="deleteTask(task.id)" title="删除">🗑️</button>
                  </div>
                </div>
                <div class="task-cell task-toggle">
                  <span class="toggle-icon">{{ expandedTaskId === task.id ? '▼' : '▶' }}</span>
                </div>
              </div>

              <!-- 展开的详情 -->
              <transition name="slide">
                <div v-if="expandedTaskId === task.id" class="task-detail-expanded">
                  <div class="detail-main">
                    <!-- 左侧：任务详情和图片预览 -->
                    <div class="detail-content">
                      <div class="detail-actions">
                      <button v-if="shouldShowDetect(task)" class="btn btn-primary" @click="detectTask(task)">识别</button>
                    </div>

                      <div class="detail-grid">
                        <div class="detail-item">
                          <span class="detail-label">图片名称：</span>
                          <span class="detail-value">{{ task.image_name }}</span>
                        </div>
                        <div class="detail-item">
                          <span class="detail-label">状态：</span>
                          <span :class="['detail-value', 'status-badge', task.status]">{{ getStatusText(task.status) }}</span>
                        </div>
                        <div class="detail-item">
                          <span class="detail-label">检测数量：</span>
                          <span class="detail-value">{{ task.box_count || 0 }}</span>
                        </div>
                        <div class="detail-item">
                          <span class="detail-label">匹配数量：</span>
                          <span class="detail-value">{{ task.matched_count || 0 }}</span>
                        </div>
                        <div class="detail-item">
                          <span class="detail-label">未匹配数量：</span>
                          <span class="detail-value">{{ task.unmatched_count || 0 }}</span>
                        </div>
                        <div class="detail-item">
                          <span class="detail-label">创建时间：</span>
                          <span class="detail-value">{{ formatDate(task.created_at) }}</span>
                        </div>
                        <div class="detail-item" v-if="task.completed_at">
                          <span class="detail-label">完成时间：</span>
                          <span class="detail-value">{{ formatDate(task.completed_at) }}</span>
                        </div>
                      </div>

                      <!-- 原图和检测结果 -->
                      <div class="result-section">
                        <div class="preview-row">
                          <div class="preview-section" @click="openImageViewer(getTaskImagePath(task), task.image_name)">
                            <div class="preview-title">
                              <span>原图</span>
                              <span class="click-indicator">👆 点击查看大图</span>
                            </div>
                            <SkuImage
                              :image-path="getTaskImagePath(task)"
                              height="200px"
                              class="preview-img clickable"
                            />
                          </div>
                          <div v-if="getTaskPreviewPath(task)" class="preview-section" @click="openImageViewer(getTaskPreviewPath(task).url, task.image_name + ' (检测结果)')">
                            <div class="preview-title">
                              <span>检测结果（带框）</span>
                              <span class="click-indicator">👆 点击查看大图</span>
                            </div>
                            <SkuImage
                              :image-path="getTaskPreviewPath(task)"
                              height="200px"
                              class="preview-img clickable"
                            />
                          </div>
                        </div>

                        <div v-if="getDetectionBoxes(task).length > 0" class="detection-boxes-preview">
                          <h5>识别结果 ({{ getDetectionBoxes(task).length }}个)</h5>
                          <div class="boxes-grid">
                            <div v-for="(box, idx) in getDetectionBoxes(task)" :key="box.box_id" class="box-item" @click="openImageViewer(getBoxImageUrl(box), `箱体 ${idx + 1}`)">
                              <SkuImage
                                :image-path="getBoxImageUrl(box)"
                                :placeholder-icon="String(idx + 1)"
                                height="80px"
                                class="clickable"
                              />
                              <div class="box-info">
                                <span class="box-idx">箱体 {{ idx + 1 }}</span>
                                <span class="box-conf">置信度: {{ (box.confidence * 100).toFixed(1) }}%</span>
                                <span v-if="getMatchResultForTask(task, box.box_id)" class="box-match" :class="getMatchResultForTask(task, box.box_id).status">
                                  {{ getMatchResultForTask(task, box.box_id).sku_id || '未匹配' }}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <!-- 右侧：审核面板 -->
                    <div v-if="shouldShowReview(task) && task.detections" class="detail-review">
                      <div class="review-content">
                        <ReviewDialog
                          :task="{ ...task, result: { detections: { boxes: task.detections } } }"
                          :inline="true"
                          @update="handleInlineReviewUpdate(task, $event)"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </transition>
            </div>
          </div>

          <div class="pagination" v-if="total > pageSize">
            <button :disabled="page <= 1" @click="changePage(page - 1)">上一页</button>
            <span>第 {{ page }} / {{ totalPages }} 页</span>
            <button :disabled="page >= totalPages" @click="changePage(page + 1)">下一页</button>
          </div>
        </div>
      </div>

      <!-- 审核面板（右侧分栏） -->
      <transition name="slide-right">
        <div v-if="showReviewPanel" class="review-panel">
          <div class="review-panel-header">
            <h3>审核检测结果 #{{ selectedTaskForReview?.id }}</h3>
            <button class="btn-close" @click="closeReviewPanel">×</button>
          </div>
          <div class="review-panel-content">
            <ReviewDialog
              v-if="selectedTaskForReview"
              :task="selectedTaskForReview"
              @cancel="closeReviewPanel"
              @update="handleReviewUpdate"
            />
          </div>
        </div>
      </transition>
    </div>

    <!-- 大图查看器 -->
    <ImageViewer
      :visible="showImageViewer"
      :image-url="currentImageUrl"
      :image-name="currentImageName"
      @update:visible="showImageViewer = false"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox, ElLoading } from 'element-plus'
import { tasks as taskApi, getImageUrlFromPath } from '@api/client'
import SkuImage from '@sku/SkuImage.vue'
import ReviewDialog from '@task/ReviewDialog.vue'
import ImageViewer from '@ui/ImageViewer.vue'
import {
  getStatusText,
  formatDate,
  shouldShowDetect,
  shouldShowReview,
  shouldShowReDetect
} from '@utils/taskUtils'

const tasks = ref([])
const stats = ref({})
const loading = ref(false)
const submitting = ref(false)
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)
const statusFilter = ref(null)
const expandedTaskId = ref(null)
const showReviewPanel = ref(false)
const selectedTaskForReview = ref(null)

// 导出相关
const selectedTaskIds = ref([])
const showExportMenu = ref(false)
const exporting = ref(false)

const selectedTasks = computed(() => {
  return tasks.value.filter(task => selectedTaskIds.value.includes(task.id))
})

const toggleTaskSelection = (taskId) => {
  const index = selectedTaskIds.value.indexOf(taskId)
  if (index === -1) {
    selectedTaskIds.value.push(taskId)
  } else {
    selectedTaskIds.value.splice(index, 1)
  }
}

const selectAllTasks = () => {
  if (selectedTaskIds.value.length === tasks.value.length) {
    selectedTaskIds.value = []
  } else {
    selectedTaskIds.value = tasks.value.map(t => t.id)
  }
}

// 大图查看功能
const showImageViewer = ref(false)
const currentImageUrl = ref('')
const currentImageName = ref('')

const totalPages = computed(() => Math.ceil(total.value / pageSize.value) || 1)

const loadTasks = async () => {
  loading.value = true
  try {
    const res = await taskApi.list(page.value, pageSize.value, statusFilter.value)
    if (res.success) {
      tasks.value = res.tasks
      total.value = res.total
    }
  } catch (e) {
    ElMessage.error('加载任务列表失败: ' + (e.message || '未知错误'))
  } finally {
    loading.value = false
  }
}

const loadStats = async () => {
  try {
    const res = await taskApi.stats()
    if (res.success) {
      stats.value = res
    }
  } catch (e) {
    console.error('加载统计失败', e)
  }
}

const filterByStatus = (status) => {
  statusFilter.value = status
  page.value = 1
  loadTasks()
}

const changePage = (newPage) => {
  if (newPage >= 1 && newPage <= totalPages.value) {
    page.value = newPage
    loadTasks()
  }
}

const toggleTask = async (task) => {
  if (expandedTaskId.value === task.id) {
    expandedTaskId.value = null
  } else {
    // 如果是待审核状态，先获取检测结果数据
    if (shouldShowReview(task) && !task.detections) {
      try {
        const res = await taskApi.getDetections(task.id)
        if (res.success && res.boxes) {
          task.detections = res.boxes
        }
      } catch (e) {
        console.error('获取检测结果失败:', e)
      }
    }
    expandedTaskId.value = task.id
  }
}

const getTaskImagePath = (task) => {
  return `/api/tasks/${task.id}/image`
}

const getBoxImageUrl = (box) => {
  if (!box) return ''
  if (box.crop_base64) {
    return { url: 'data:image/jpeg;base64,' + box.crop_base64 }
  }
  if (box.crop_path) {
    return getImageUrlFromPath(box.crop_path)
  }
  return ''
}

const getTaskPreviewPath = (task) => {
  if (task.id && task.status === 'detected') {
    return { url: `/api/tasks/${task.id}/detection-image` }
  }
  return null
}

const getDetectionBoxes = (task) => {
  if (!task.detections) return []
  return task.detections
}

const getMatchResultForTask = (task, boxId) => {
  if (!task.detections) return null
  const box = task.detections.find(b => b.box_id === boxId || b.box_id === `box_${boxId}` || String(b.box_id) === String(boxId))
  return box ? box.match_result : null
}

const detectTask = async (task) => {
    submitting.value = true
    try {
      const res = await taskApi.detect(task.id)
      if (res && res.status === 'detected') {
        ElMessage.success('检测成功')
        await loadTasks()
        await loadStats()
        if (expandedTaskId.value === task.id) {
          expandedTaskId.value = null
          await new Promise(resolve => setTimeout(resolve, 100))
          expandedTaskId.value = task.id
        }
      } else {
        ElMessage.error('检测未成功执行')
      }
    } catch (e) {
      ElMessage.error('检测失败: ' + (e.detail || e.message || '未知错误'))
    } finally {
      submitting.value = false
    }
  }

const openReview = async (task) => {
  try {
    const res = await taskApi.getDetections(task.id)
    if (res.success && res.boxes) {
      const taskWithDetections = {
        ...task,
        result: {
          detections: {
            boxes: res.boxes
          }
        }
      }
      selectedTaskForReview.value = taskWithDetections
    } else {
      selectedTaskForReview.value = task
    }
    showReviewPanel.value = true
  } catch (e) {
    ElMessage.error('获取检测结果失败: ' + (e.message || '未知错误'))
    selectedTaskForReview.value = task
    showReviewPanel.value = true
  }
}

const closeReviewPanel = () => {
  showReviewPanel.value = false
  selectedTaskForReview.value = null
}

const handleReviewSave = async ({ task, boxes }) => {
  try {
    await taskApi.reviewTask(task.id, boxes)
    await loadTasks()
    await loadStats()
  } catch (e) {
    ElMessage.error('保存失败: ' + (e.detail || e.message || '未知错误'))
  }
}

const handleReviewUpdate = async ({ task, boxes, approvedCount, rejectedCount }) => {
  try {
    await taskApi.reviewTask(task.id, boxes)
    ElMessage.success(`更新成功：通过 ${approvedCount} 个，拒绝 ${rejectedCount} 个`)
    await loadTasks()
    await loadStats()
  } catch (e) {
    ElMessage.error('更新失败: ' + (e.detail || e.message || '未知错误'))
  }
}

const handleInlineReviewUpdate = async (task, { boxes, approvedCount, rejectedCount }) => {
  try {
    await taskApi.reviewTask(task.id, boxes)
    ElMessage.success(`更新成功：通过 ${approvedCount} 个，拒绝 ${rejectedCount} 个`)
    await loadTasks()
    await loadStats()
  } catch (e) {
    ElMessage.error('更新失败: ' + (e.detail || e.message || '未知错误'))
  }
}

const matchTask = async (task) => {
  submitting.value = true
  try {
    const res = await taskApi.matchTask(task.id)
    if (res) {
      ElMessage.success('匹配成功')
      await loadTasks()
      await loadStats()
      if (expandedTaskId.value === task.id) {
        expandedTaskId.value = null
        await new Promise(resolve => setTimeout(resolve, 100))
        expandedTaskId.value = task.id
      }
    }
  } catch (e) {
    ElMessage.error('匹配失败: ' + (e.detail || e.message || '未知错误'))
  } finally {
    submitting.value = false
  }
}

const reDetectTask = async (task) => {
  try {
    await ElMessageBox.confirm('确定要重新检测这个任务吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await detectTask(task)
  } catch {
    // 用户取消
  }
}

const batchDetectTasks = async () => {
  if (selectedTasks.value.length === 0) {
    ElMessage.warning('请先选择要识别的任务')
    return
  }
  
  submitting.value = true
  const loadingInstance = ElLoading.service({
    message: `正在识别 ${selectedTasks.value.length} 个任务...`
  })
  
  try {
    for (const task of selectedTasks.value) {
      if (shouldShowDetect(task) || shouldShowReDetect(task)) {
        await taskApi.detect(task.id)
      }
    }
    
    ElMessage.success('批量识别完成')
    selectedTaskIds.value = []
    await loadTasks()
    await loadStats()
  } catch (e) {
    ElMessage.error('批量识别失败: ' + (e.detail || e.message || '未知错误'))
  } finally {
    loadingInstance.close()
    submitting.value = false
  }
}

const deleteTask = async (id) => {
  try {
    await ElMessageBox.confirm('确定要删除这个任务吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    const res = await taskApi.delete(id)
    if (res.success) {
      ElMessage.success('删除成功')
      if (expandedTaskId.value === id) {
        expandedTaskId.value = null
      }
      await loadTasks()
      await loadStats()
    }
  } catch (e) {
    if (e !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

const openImageViewer = (imageUrl, imageName = '图片') => {
  currentImageUrl.value = imageUrl
  currentImageName.value = imageName
  showImageViewer.value = true
}

const toggleExportMenu = () => {
  showExportMenu.value = !showExportMenu.value
}

const exportTasks = async (format, includeImages) => {
  if (selectedTasks.value.length === 0) {
    ElMessage.warning('请先选择要导出的任务')
    return
  }

  exporting.value = true
  showExportMenu.value = false

  const taskIds = selectedTasks.value.map(task => task.id)

  try {
    const loadingInstance = ElLoading.service({
      lock: true,
      text: `正在导出 ${selectedTasks.value.length} 个任务...`,
      background: 'rgba(0, 0, 0, 0.7)'
    })

    const response = await fetch(`/api/tasks/batch/export?format=${format}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(taskIds)
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: '导出请求失败' }))
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }

    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-')
    const filename = `batch_export_${selectedTasks.value.length}_tasks_${timestamp}.${format === 'json' ? 'json' : 'csv'}`
    a.download = filename
    
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)

    loadingInstance.close()
    
    ElMessage({
      message: `成功导出 ${selectedTasks.value.length} 个任务！`,
      type: 'success',
      duration: 3000
    })
  } catch (e) {
    console.error('导出失败:', e)
    ElMessage({
      message: '导出失败: ' + (e.message || '未知错误'),
      type: 'error',
      duration: 5000
    })
  } finally {
    exporting.value = false
  }
}

const handleClickOutside = (event) => {
  const exportDropdown = document.querySelector('.export-dropdown')
  if (exportDropdown && !exportDropdown.contains(event.target)) {
    showExportMenu.value = false
  }
}

onMounted(() => {
  loadTasks()
  loadStats()
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<style scoped>
.task-list-page {
  padding-bottom: var(--spacing-xl);
}

.main-container {
  display: flex;
  gap: var(--spacing-lg);
}

.main-container.split-view {
  gap: 0;
}

.task-list-section {
  flex: 1;
  min-width: 0;
}

.main-container.split-view .task-list-section {
  flex: 0 0 45%;
  max-width: 45%;
}

.section {
  background: var(--color-bg-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
  box-shadow: var(--shadow-md);
}

.stats-row {
  display: flex;
  gap: var(--spacing-lg);
}

.stat-card {
  flex: 1;
  padding: var(--spacing-lg);
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-md);
  text-align: center;
}

.stat-card.success {
  background: rgba(103, 194, 58, 0.1);
}

.stat-card.warning {
  background: rgba(230, 162, 60, 0.1);
}

.stat-card.danger {
  background: rgba(245, 108, 108, 0.1);
}

.stat-value {
  font-size: 32px;
  font-weight: bold;
  color: var(--color-text-primary);
}

.stat-label {
  color: var(--color-text-secondary);
  margin-top: var(--spacing-xs);
}

.table-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
}

.toolbar-right {
  display: flex;
  gap: var(--spacing-md);
  align-items: center;
}

.export-dropdown {
  position: relative;
}

.export-menu {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: var(--spacing-xs);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
  min-width: 200px;
  z-index: 100;
}

.export-option {
  padding: var(--spacing-md);
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  transition: background-color var(--transition-fast);
}

.export-option:hover {
  background: var(--color-bg-tertiary);
}

.export-option span:first-child {
  font-weight: 500;
  color: var(--color-text-primary);
}

.option-desc {
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
}

.filter-tabs {
  display: flex;
  gap: var(--spacing-md);
}

.filter-tabs button {
  padding: var(--spacing-sm) var(--spacing-lg);
  background: var(--color-bg-tertiary);
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: var(--font-size-base);
  color: var(--color-text-secondary);
  transition: all var(--transition-fast);
}

.filter-tabs button:hover {
  background: var(--color-bg-secondary);
}

.filter-tabs button.active {
  background: var(--color-primary);
  color: white;
}

.task-table-container {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.task-row {
  border-bottom: 1px solid var(--color-border-light);
  transition: background-color var(--transition-fast);
}

.task-row:last-child {
  border-bottom: none;
}

.task-row.selected {
  background-color: rgba(102, 126, 234, 0.1);
}

.task-row-header {
  display: flex;
  align-items: center;
  padding: var(--spacing-md);
  background: var(--color-bg-tertiary);
  cursor: pointer;
  transition: background-color var(--transition-fast);
}

.task-row-header:hover {
  background: var(--color-bg-secondary);
}

.task-row.expanded .task-row-header {
  background: rgba(102, 126, 234, 0.1);
}

.task-row-header.header-row {
  font-weight: 600;
  color: var(--color-text-secondary);
  background: var(--color-bg-tertiary);
  cursor: default;
  font-size: var(--font-size-base);
}

.task-row-header.header-row .task-cell {
  font-weight: 600;
  font-size: var(--font-size-base);
}

.task-cell {
  padding: 0 var(--spacing-sm);
  font-size: var(--font-size-sm);
}

.task-select {
  width: 40px;
  text-align: center;
}

.task-id {
  width: 60px;
  font-weight: 600;
  color: var(--color-primary);
}

.task-name {
  flex: 2;
  min-width: 150px;
  font-weight: 500;
  font-size: var(--font-size-sm);
}

.task-status {
  width: 100px;
}

.task-counts {
  width: 140px;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
}

.count-item {
  white-space: nowrap;
}

.task-date {
  width: 160px;
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
}

.task-actions {
  flex: 1;
  min-width: 200px;
}

.task-toggle {
  width: 30px;
  text-align: center;
  color: var(--color-text-secondary);
  font-size: var(--font-size-xs);
}

.toggle-icon {
  display: inline-block;
  transition: transform var(--transition-fast);
}

.action-buttons {
  display: flex;
  gap: var(--spacing-md);
  flex-wrap: wrap;
}

.action-buttons .btn-small {
  padding: 6px 12px;
  font-size: var(--font-size-sm);
}

.btn-icon {
  background: transparent;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  padding: 6px 10px;
  font-size: 16px;
  transition: all var(--transition-fast);
}

.btn-icon:hover {
  background: var(--color-bg-tertiary);
}

.status-badge.pending,
.status-badge.warning {
  background: rgba(230, 162, 60, 0.1);
  color: var(--color-warning);
}

.status-badge.completed {
  background: rgba(103, 194, 58, 0.1);
  color: var(--color-success);
}

.status-badge.failed {
  background: rgba(245, 108, 108, 0.1);
  color: var(--color-danger);
}

/* 展开详情样式 */
.task-detail-expanded {
  background: var(--color-bg-primary);
  padding: 0;
  overflow: hidden;
}

.detail-content {
  padding: var(--spacing-lg);
}

.detail-actions {
  display: flex;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
  padding-bottom: var(--spacing-md);
  border-bottom: 1px solid var(--color-border-light);
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.detail-label {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
}

.detail-value {
  font-size: var(--font-size-base);
  color: var(--color-text-primary);
}

.result-section h5 {
  margin: var(--spacing-lg) 0 var(--spacing-md) 0;
  font-size: var(--font-size-base);
  color: var(--color-text-secondary);
}

.preview-row {
  display: flex;
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
}

.preview-section {
  flex: 1;
}

.preview-title {
  margin-bottom: var(--spacing-sm);
  font-size: var(--font-size-base);
  font-weight: 500;
  color: var(--color-text-secondary);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.click-indicator {
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
  font-weight: normal;
}

.preview-img {
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
}

.preview-img.clickable,
.box-item .clickable {
  cursor: pointer;
  transition: transform var(--transition-fast), box-shadow var(--transition-fast);
}

.preview-img.clickable:hover,
.box-item:hover .clickable {
  transform: scale(1.02);
  box-shadow: var(--shadow-lg);
}

.boxes-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--spacing-md);
  max-width: 100%;
}

.box-item {
  min-width: 200px;
  max-width: 300px;
  justify-self: start;
  cursor: pointer;
  transition: transform var(--transition-fast);
}

.box-item:hover {
  transform: translateY(-2px);
}

.task-thumb {
  width: 80px;
  height: 50px;
  overflow: hidden;
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  justify-content: center;
}

.task-thumb-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.box-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  margin-top: var(--spacing-xs);
  font-size: var(--font-size-xs);
}

.box-idx {
  font-weight: 600;
  color: var(--color-primary);
  font-size: var(--font-size-sm);
}

.box-conf {
  color: var(--color-text-secondary);
}

.box-status {
  color: var(--color-success);
}

.box-match {
  padding: 3px 6px;
  border-radius: var(--radius-sm);
  font-size: var(--font-size-xs);
  text-align: center;
}

.box-match.matched {
  background: rgba(103, 194, 58, 0.1);
  color: var(--color-success);
}

.box-match.low_conf {
  background: rgba(230, 162, 60, 0.1);
  color: var(--color-warning);
}

.box-match.unmatched {
  background: rgba(245, 108, 108, 0.1);
  color: var(--color-danger);
}

/* 并排布局样式 */
.detail-main {
  display: flex;
  gap: var(--spacing-lg);
  padding: var(--spacing-md);
}

.detail-content {
  flex: 0 0 45%;
  max-width: 45%;
}

.detail-review {
  flex: 1;
  min-width: 450px;
  border-left: 1px solid var(--color-border);
  padding-left: var(--spacing-lg);
}

.detail-review .review-header {
  margin-bottom: var(--spacing-md);
  padding-bottom: var(--spacing-md);
  border-bottom: 1px solid var(--color-border);
}

.detail-review .review-header h4 {
  margin: 0;
  font-size: var(--font-size-lg);
  color: var(--color-text-primary);
}

.detail-review .review-content {
  max-height: calc(100vh - 200px);
  overflow-y: auto;
}

/* 审核面板样式 */
.review-panel {
  position: fixed;
  top: 60px;
  right: 0;
  width: 55%;
  height: calc(100vh - 60px);
  background: var(--color-bg-primary);
  box-shadow: -4px 0 20px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.review-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-lg);
  border-bottom: 1px solid var(--color-border-light);
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%);
  color: white;
}

.review-panel-header h3 {
  margin: 0;
  font-size: var(--font-size-lg);
}

.review-panel-content {
  flex: 1;
  overflow: hidden;
}

.btn-close {
  background: transparent;
  border: none;
  font-size: var(--font-size-2xl);
  cursor: pointer;
  color: white;
  padding: 0;
  line-height: 1;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: background-color var(--transition-fast);
}

.btn-close:hover {
  background: rgba(255, 255, 255, 0.2);
}

/* 过渡动画 */
.slide-enter-active,
.slide-leave-active {
  transition: all var(--transition-normal);
}

.slide-enter-from {
  opacity: 0;
  max-height: 0;
  transform: translateY(-20px);
}

.slide-leave-to {
  opacity: 0;
  max-height: 0;
  transform: translateY(-20px);
}

.slide-right-enter-active,
.slide-right-leave-active {
  transition: all var(--transition-normal);
}

.slide-right-enter-from,
.slide-right-leave-to {
  transform: translateX(100%);
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .main-container.split-view .task-list-section {
    flex: 0 0 40%;
    max-width: 40%;
  }
  
  .review-panel {
    width: 60%;
  }
}

@media (max-width: 900px) {
  .detail-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .boxes-grid {
    grid-template-columns: repeat(2, minmax(150px, 1fr));
  }
  
  .box-item {
    width: auto;
  }
  
  .task-counts {
    display: none;
  }
}

@media (max-width: 768px) {
  .main-container.split-view {
    flex-direction: column;
  }
  
  .main-container.split-view .task-list-section {
    flex: 1;
    max-width: 100%;
  }
  
  .review-panel {
    position: fixed;
    width: 100%;
    height: 100vh;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
  }
  
  .detail-grid {
    grid-template-columns: 1fr;
  }
  
  .boxes-grid {
    grid-template-columns: repeat(2, minmax(120px, 1fr));
  }
  
  .box-item {
    width: auto;
  }
  
  .stats-row {
    flex-wrap: wrap;
  }
  
  .stat-card {
    flex: none;
    width: calc(50% - 10px);
  }
  
  .task-row-header {
    flex-wrap: wrap;
    gap: var(--spacing-md);
  }
  
  .task-name {
    width: 100%;
    order: -1;
  }
}
</style>
