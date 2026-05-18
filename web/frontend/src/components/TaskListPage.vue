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
            <button class="btn btn-primary" @click="loadTasks">🔄 刷新</button>
          </div>

          <div v-if="loading" class="loading">加载中...</div>

          <div v-else-if="tasks.length === 0" class="empty-state">
            <div class="empty-icon">📭</div>
            <p>暂无任务记录</p>
          </div>

          <div v-else class="task-table-container">
            <div 
              v-for="task in tasks" 
              :key="task.id" 
              class="task-row"
              :class="{ 'expanded': expandedTaskId === task.id }"
            >
              <!-- 任务行 -->
              <div class="task-row-header" @click="toggleTask(task)">
                <div class="task-cell task-id">{{ task.id }}</div>
                <div class="task-cell task-name">{{ task.image_name }}</div>
                <div class="task-cell task-status">
                  <span :class="['status-badge', getStatusBadgeClass(task)]">
                    {{ getStatusText(task.status, task.detection_status, task.review_status) }}
                  </span>
                </div>
                <div class="task-cell task-counts">
                  <span class="count-item">检测: {{ task.box_count || 0 }}</span>
                  <span class="count-item">匹配: {{ task.matched_count || 0 }}</span>
                </div>
                <div class="task-cell task-date">{{ formatDate(task.created_at) }}</div>
                <div class="task-cell task-actions">
                  <div class="action-buttons">
                    <button v-if="shouldShowDetect(task)" class="btn-small btn-primary" @click.stop="detectTask(task)">检测</button>
                    <button v-if="shouldShowReview(task)" class="btn-small btn-warning" @click.stop="openReview(task)">审核</button>
                    <button v-if="shouldShowMatch(task)" class="btn-small btn-success" @click.stop="matchTask(task)">匹配</button>
                    <button v-if="shouldShowReDetect(task)" class="btn-small btn-danger" @click.stop="reDetectTask(task)">重新检测</button>
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
                  <div class="detail-content">
                    <div class="detail-actions">
                      <button v-if="shouldShowDetect(task)" class="btn btn-primary" @click="detectTask(task)">检测</button>
                      <button v-if="shouldShowReview(task)" class="btn btn-warning" @click="openReview(task)">审核</button>
                      <button v-if="shouldShowMatch(task)" class="btn btn-success" @click="matchTask(task)">匹配</button>
                      <button v-if="shouldShowReDetect(task)" class="btn btn-danger" @click="reDetectTask(task)">重新检测</button>
                      <button class="btn btn-default" @click="saveTaskResult(task)">保存结果</button>
                    </div>

                    <div class="detail-grid">
                      <div class="detail-item">
                        <span class="detail-label">图片名称：</span>
                        <span class="detail-value">{{ task.image_name }}</span>
                      </div>
                      <div class="detail-item">
                        <span class="detail-label">主状态：</span>
                        <span :class="['detail-value', 'status-badge', task.status]">{{ task.status }}</span>
                      </div>
                      <div class="detail-item">
                        <span class="detail-label">检测状态：</span>
                        <span class="detail-value">{{ getDetectionStatusText(task.detection_status) }}</span>
                      </div>
                      <div class="detail-item">
                        <span class="detail-label">审核状态：</span>
                        <span class="detail-value">{{ getReviewStatusText(task.review_status) }}</span>
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
                        <h5>检测到的箱体 ({{ getDetectionBoxes(task).length }}个)</h5>
                        <div class="boxes-grid">
                          <div v-for="(box, idx) in getDetectionBoxes(task)" :key="box.box_id" class="box-item" @click="openImageViewer(box.crop_base64 ? 'data:image/jpeg;base64,' + box.crop_base64 : getTaskImagePath(task), `箱体 ${idx + 1}`)">
                            <SkuImage
                              :image-path="box.crop_base64 ? { url: 'data:image/jpeg;base64,' + box.crop_base64 } : ''"
                              :placeholder-icon="String(idx + 1)"
                              height="80px"
                              class="clickable"
                            />
                            <div class="box-info">
                              <span class="box-conf">置信度: {{ (box.confidence * 100).toFixed(1) }}%</span>
                              <div class="box-status" v-if="box.status">
                                状态: {{ box.status === 'approved' ? '✓ 已批准' : '✗ 已拒绝' }}
                              </div>
                              <span v-if="getMatchResultForTask(task, box.box_id)" class="box-match" :class="getMatchResultForTask(task, box.box_id).status">
                                {{ getMatchResultForTask(task, box.box_id).sku_id || '未匹配' }}
                              </span>
                            </div>
                          </div>
                        </div>
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
              @submit="handleReviewSubmit"
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
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { tasks as taskApi } from '../api/client'
import SkuImage from './result/SkuImage.vue'
import ReviewDialog from './result/ReviewDialog.vue'
import ImageViewer from './result/ImageViewer.vue'

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

const toggleTask = (task) => {
  if (expandedTaskId.value === task.id) {
    expandedTaskId.value = null
  } else {
    expandedTaskId.value = task.id
  }
}

const getTaskImagePath = (task) => {
  return `/api/tasks/${task.id}/image`
}

const getTaskPreviewPath = (task) => {
  if (task.result && task.result.image_with_boxes) {
    return { url: 'data:image/jpeg;base64,' + task.result.image_with_boxes }
  }
  return null
}

const getDetectionBoxes = (task) => {
  if (!task.result) return []
  
  const result = task.result
  if (result.detections && result.detections.boxes) {
    return result.detections.boxes
  }
  if (result.boxes) {
    return result.boxes.map((box, idx) => ({
      box_id: String(idx),
      ...box,
      crop_base64: result.crops ? result.crops[idx] : null
    }))
  }
  return []
}

const getMatchResultForTask = (task, boxId) => {
  if (task.result && task.result.matches) {
    return task.result.matches[boxId] || task.result.matches[parseInt(boxId)]
  }
  return null
}

const detectTask = async (task) => {
  submitting.value = true
  try {
    const res = await taskApi.detect(task.id)
    if (res) {
      ElMessage.success('检测成功')
      await loadTasks()
      await loadStats()
      // 如果当前任务正在展开，刷新展开的内容
      if (expandedTaskId.value === task.id) {
        expandedTaskId.value = null
        await new Promise(resolve => setTimeout(resolve, 100))
        expandedTaskId.value = task.id
      }
    }
  } catch (e) {
    ElMessage.error('检测失败: ' + (e.detail || e.message || '未知错误'))
  } finally {
    submitting.value = false
  }
}

const openReview = (task) => {
  selectedTaskForReview.value = task
  showReviewPanel.value = true
}

const closeReviewPanel = () => {
  showReviewPanel.value = false
  selectedTaskForReview.value = null
}

const handleReviewSubmit = async ({ task, boxes, approvedCount, rejectedCount }) => {
  try {
    await taskApi.reviewTask(task.id, boxes)
    ElMessage.success(`审核成功：通过 ${approvedCount} 个，拒绝 ${rejectedCount} 个`)
    await loadTasks()
    await loadStats()
    closeReviewPanel()
  } catch (e) {
    ElMessage.error('审核失败: ' + (e.detail || e.message || '未知错误'))
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

const saveTaskResult = async (task) => {
  try {
    const res = await taskApi.update(task.id, {
      status: task.status,
      detection_status: task.detection_status,
      review_status: task.review_status
    })
    
    if (res) {
      ElMessage.success('保存成功')
      await loadTasks()
    }
  } catch (e) {
    ElMessage.error('保存失败: ' + (e.message || '未知错误'))
  }
}

const openImageViewer = (imageUrl, imageName = '图片') => {
  currentImageUrl.value = imageUrl
  currentImageName.value = imageName
  showImageViewer.value = true
}

const shouldShowDetect = (task) => {
  return task.detection_status === 'pending' && task.status !== 'completed'
}

const shouldShowReview = (task) => {
  return task.detection_status === 'detected' && task.review_status === 'pending'
}

const shouldShowMatch = (task) => {
  return task.review_status === 'reviewed' && task.review_status !== 'matched'
}

const shouldShowReDetect = (task) => {
  return task.status === 'failed' || (task.detection_status !== 'pending')
}

const getStatusBadgeClass = (task) => {
  if (task.status === 'failed') return 'failed'
  if (task.status === 'completed' || task.review_status === 'matched') return 'completed'
  if (task.review_status === 'reviewed') return 'pending'
  if (task.detection_status === 'detected') return 'warning'
  return 'pending'
}

const getStatusText = (status, detectionStatus, reviewStatus) => {
  if (status === 'failed') return '失败'
  if (status === 'completed') return '已完成'
  if (reviewStatus === 'matched') return '已匹配'
  if (reviewStatus === 'reviewed') return '已审核'
  if (detectionStatus === 'detected') return '已检测'
  if (status === 'pending') return '进行中'
  return status
}

const getDetectionStatusText = (status) => {
  const map = {
    'pending': '待检测',
    'detected': '已完成',
    'error': '检测失败'
  }
  return map[status] || (status || '未知')
}

const getReviewStatusText = (status) => {
  const map = {
    'pending': '待审核',
    'reviewed': '已审核',
    'matched': '已匹配'
  }
  return map[status] || (status || '未知')
}

const formatDate = (dateStr) => {
  if (!dateStr) return '-'
  const d = new Date(dateStr)
  return d.toLocaleString('zh-CN')
}

onMounted(() => {
  loadTasks()
  loadStats()
})
</script>

<style scoped>
.task-list-page {
  padding-bottom: 40px;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 25px 30px;
  border-radius: 12px;
  margin-bottom: 20px;
}

.header h1 {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
}

.header p {
  font-size: 14px;
  opacity: 0.9;
  margin-top: 5px;
}

.main-container {
  display: flex;
  gap: 20px;
}

.main-container.split-view {
  gap: 0;
}

.task-list-section {
  flex: 1;
  min-width: 0;
}

.main-container.split-view .task-list-section {
  flex: 0 0 50%;
  max-width: 50%;
}

.section {
  background: white;
  border-radius: 12px;
  padding: 25px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.stats-row {
  display: flex;
  gap: 20px;
}

.stat-card {
  flex: 1;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 8px;
  text-align: center;
}

.stat-card.success {
  background: #e1f3d8;
}

.stat-card.warning {
  background: #faecd8;
}

.stat-card.danger {
  background: #fef0f0;
}

.stat-value {
  font-size: 32px;
  font-weight: bold;
  color: #333;
}

.stat-label {
  color: #666;
  margin-top: 5px;
}

.table-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.filter-tabs {
  display: flex;
  gap: 10px;
}

.filter-tabs button {
  padding: 8px 16px;
  background: #f5f5f5;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  color: #666;
  transition: all 0.2s;
}

.filter-tabs button:hover {
  background: #e0e0e0;
}

.filter-tabs button.active {
  background: #667eea;
  color: white;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.btn-primary {
  background: #667eea;
  color: white;
}

.btn-primary:hover {
  background: #5a70d6;
}

.btn-warning {
  background: #e6a23c;
  color: white;
}

.btn-warning:hover {
  background: #d39437;
}

.btn-success {
  background: #67c23a;
  color: white;
}

.btn-success:hover {
  background: #5aaf33;
}

.btn-danger {
  background: #f56c6c;
  color: white;
}

.btn-danger:hover {
  background: #de5c5c;
}

.btn-default {
  background: #909399;
  color: white;
}

.btn-default:hover {
  background: #80848c;
}

.btn-small {
  padding: 4px 10px;
  font-size: 12px;
}

.loading, .empty-state {
  text-align: center;
  padding: 40px 20px;
  color: #999;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.task-table-container {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
}

.task-row {
  border-bottom: 1px solid #e0e0e0;
}

.task-row:last-child {
  border-bottom: none;
}

.task-row-header {
  display: flex;
  align-items: center;
  padding: 15px;
  background: #fafafa;
  cursor: pointer;
  transition: background-color 0.2s;
}

.task-row-header:hover {
  background: #f0f0f0;
}

.task-row.expanded .task-row-header {
  background: #e8f0fe;
}

.task-cell {
  padding: 0 10px;
}

.task-id {
  width: 60px;
  font-weight: 600;
  color: #667eea;
}

.task-name {
  flex: 2;
  min-width: 150px;
  font-weight: 500;
}

.task-status {
  width: 100px;
}

.task-counts {
  width: 140px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: #666;
}

.count-item {
  white-space: nowrap;
}

.task-date {
  width: 160px;
  font-size: 13px;
  color: #666;
}

.task-actions {
  flex: 1;
  min-width: 200px;
}

.task-toggle {
  width: 30px;
  text-align: center;
  color: #666;
  font-size: 12px;
}

.toggle-icon {
  display: inline-block;
  transition: transform 0.2s;
}

.action-buttons {
  display: flex;
  gap: 5px;
  flex-wrap: wrap;
}

.btn-icon {
  padding: 6px 10px;
  background: transparent;
  border: none;
  cursor: pointer;
  font-size: 16px;
}

.btn-icon:hover {
  background: #fef0f0;
}

.status-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
}

.status-badge.pending {
  background: #faecd8;
  color: #e6a23c;
}

.status-badge.warning {
  background: #faecd8;
  color: #e6a23c;
}

.status-badge.completed {
  background: #e1f3d8;
  color: #67c23a;
}

.status-badge.failed {
  background: #fef0f0;
  color: #f56c6c;
}

/* 展开详情样式 */
.task-detail-expanded {
  background: white;
  padding: 0;
  overflow: hidden;
}

.detail-content {
  padding: 20px;
}

.detail-actions {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 1px solid #f0f0f0;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 15px;
  margin-bottom: 25px;
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.detail-label {
  font-size: 13px;
  color: #666;
}

.detail-value {
  font-size: 14px;
  color: #333;
}

.result-section {
  margin-top: 0;
}

.result-section h4 {
  margin: 0 0 15px 0;
  font-size: 16px;
  color: #333;
}

.result-section h5 {
  margin: 20px 0 12px 0;
  font-size: 14px;
  color: #666;
}

.preview-row {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
}

.preview-section {
  flex: 1;
}

.preview-title {
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 500;
  color: #666;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.click-indicator {
  font-size: 12px;
  color: #999;
  font-weight: normal;
}

.preview-img {
  border-radius: 6px;
  border: 1px solid #e0e0e0;
}

.preview-img.clickable,
.box-item .clickable {
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.preview-img.clickable:hover,
.box-item:hover .clickable {
  transform: scale(1.02);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.box-item {
  width: calc(20% - 8px);
  min-width: 120px;
  cursor: pointer;
  transition: transform 0.2s;
}

.box-item:hover {
  transform: translateY(-2px);
}

.detection-boxes-preview {
  margin-top: 10px;
}

.boxes-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.box-item {
  width: calc(20% - 8px);
  min-width: 120px;
}

.box-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 6px;
  font-size: 12px;
}

.box-conf {
  color: #666;
}

.box-status {
  color: #67c23a;
}

.box-match {
  padding: 3px 6px;
  border-radius: 3px;
  font-size: 11px;
  text-align: center;
}

.box-match.matched {
  background: #e1f3d8;
  color: #67c23a;
}

.box-match.low_conf {
  background: #faecd8;
  color: #e6a23c;
}

.box-match.unmatched {
  background: #fef0f0;
  color: #f56c6c;
}

/* 审核面板样式 */
.review-panel {
  position: fixed;
  top: 60px;
  right: 0;
  width: 50%;
  height: calc(100vh - 60px);
  background: white;
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
  padding: 20px;
  border-bottom: 1px solid #f0f0f0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.review-panel-header h3 {
  margin: 0;
  font-size: 18px;
}

.review-panel-content {
  flex: 1;
  overflow: hidden;
}

.btn-close {
  background: transparent;
  border: none;
  font-size: 28px;
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
  transition: background-color 0.2s;
}

.btn-close:hover {
  background: rgba(255, 255, 255, 0.2);
}

/* 分页样式 */
.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 20px;
  margin-top: 20px;
}

.pagination button {
  padding: 8px 16px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.pagination button:hover:not(:disabled) {
  background: #5a70d6;
}

.pagination button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

/* 过渡动画 */
.slide-enter-active,
.slide-leave-active {
  transition: all 0.3s ease;
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
  transition: all 0.3s ease;
}

.slide-right-enter-from {
  transform: translateX(100%);
}

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
  
  .box-item {
    width: calc(25% - 8px);
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
  
  .box-item {
    width: calc(50% - 5px);
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
    gap: 10px;
  }
  
  .task-name {
    width: 100%;
    order: -1;
  }
}
</style>
