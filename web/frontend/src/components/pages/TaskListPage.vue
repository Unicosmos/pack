<template>
  <div class="task-list-page">
    <PageContainer>
      <div class="main-container" :class="{ 'view-detail': viewMode === 'detail' }">

      <!-- 普通任务列表视图 -->
      <div v-if="viewMode === 'list'" class="task-list-section">
        <TaskStats :stats="stats" />

        <div class="section">
          <div class="table-toolbar">
            <FilterBar>
              <!-- 状态筛选 -->
              <FilterDropdown
                v-model="statusFilter"
                :options="statusOptions"
                placeholder="全部状态"
                @change="handleStatusChange"
              />

              <!-- 时间筛选 -->
              <TimeFilterDropdown
                v-model="timeFilter"
                :custom-start="customStart"
                :custom-end="customEnd"
                @update:customStart="customStart = $event"
                @update:customEnd="customEnd = $event"
                @change="handleTimeChange"
              />

              <!-- 批量操作 -->
              <ActionMenu
                :items="batchActions"
                :loading="detecting"
                :progress="detectProgress"
                @select="handleBatchAction"
              >
                <template #label>{{ detecting ? '识别中...' : `批量操作 (${selectedTasks.length}个)` }}</template>
              </ActionMenu>
            </FilterBar>
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
              <div class="task-cell task-status">状态</div>
              <div class="task-cell task-counts">统计</div>
              <div class="task-cell task-date">创建时间</div>
            </div>
            
            <div 
              v-for="task in tasks" 
              :key="task.id" 
              class="task-row"
              :class="{ 'selected': selectedTaskIds.includes(task.id) }"
            >
              <!-- 任务行 -->
              <div class="task-row-header" @click="openViewDetail(task)">
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
                <div class="task-cell task-status">
                  <button v-if="shouldShowDetect(task)" class="status-badge pending status-badge-btn" @click.stop="detectTask(task)" title="点击识别当前任务">
                    待识别，点击识别
                  </button>
                  <span v-else :class="['status-badge', getStatusBadgeClass(task)]">{{ getStatusText(task.status) }}</span>
                </div>
                <div class="task-cell task-counts">
                  <span class="count-item">检测: {{ task.box_count || 0 }}</span>
                  <span class="count-item">匹配: {{ task.matched_count || 0 }}</span>
                </div>
                <div class="task-cell task-date">{{ formatDate(task.created_at) }}</div>
              </div>
            </div>
          </div>

          <div class="pagination" v-if="total > pageSize">
            <button :disabled="page <= 1" @click="changePage(page - 1)">上一页</button>
            <span>第 {{ page }} / {{ totalPages }} 页</span>
            <button :disabled="page >= totalPages" @click="changePage(page + 1)">下一页</button>
          </div>
        </div>
      </div>

    </div>

    </PageContainer>
    
    <!-- 查看识别结果详情视图 -->
    <div v-if="viewMode === 'detail'" class="detail-view-container">
      <div class="detail-view-sidebar">
        <div class="detail-view-topbar">
          <div class="topbar-left">
            <h3>📊 任务列表</h3>
          </div>
          <button class="topbar-close" @click="closeViewDetail">×</button>
        </div>
        <div class="detail-view-nav">
          <div class="detail-nav-list">
            <div 
              v-for="(task, index) in tasks" 
              :key="task.id"
              class="detail-nav-item"
              :class="{ 'active': task.id === viewDetailTask?.id, 'reviewed': task.status === 'completed' }"
              @click="switchViewDetailTask(task)"
            >
              <div class="nav-item-thumb">
                <img :src="getTaskImagePath(task)" :alt="task.image_name" class="nav-item-thumb-img" @error="$event.target.style.display='none'" />
              </div>
              <div class="nav-item-info">
                <div class="nav-item-id">#{{ task.id }}</div>
                <div class="nav-item-name">{{ task.image_name }}</div>
                <span :class="['status-badge', getStatusBadgeClass(task)]">{{ getStatusText(task.status) }}</span>
              </div>
              <div v-if="task.status === 'completed'" class="nav-item-check">✓</div>
            </div>
          </div>
        </div>
      </div>
      <div class="detail-view-main">
        <div class="detail-view-main-topbar" v-if="viewDetailTask">
          <span class="mainbar-item"><strong>#{{ viewDetailTask.id }}</strong> {{ viewDetailTask.image_name }}</span>
          <span class="mainbar-divider">|</span>
          <span :class="['mainbar-status', getStatusBadgeClass(viewDetailTask)]">{{ getStatusText(viewDetailTask.status) }}</span>
          <span class="mainbar-divider">|</span>
          <span>检测 {{ viewDetailTask.box_count || 0 }}</span>
          <span class="mainbar-dot">·</span>
          <span>匹配 {{ viewDetailTask.matched_count || 0 }}</span>
          <span class="mainbar-spacer"></span>
          <span>创建 {{ formatDate(viewDetailTask.created_at) }}</span>
          <span class="mainbar-divider">|</span>
          <span>完成 {{ formatDate(viewDetailTask.completed_at || viewDetailTask.updated_at) }}</span>
        </div>
        <TaskDetailPanel
          v-if="viewDetailTask"
          :task="viewDetailTask"
          hide-info
          @view-image="openImageViewer"
          @match-box="handleBoxMatch"
          @delete-box="handleDeleteBox"
        />
      </div>
    </div>
    
    <!-- 匹配弹窗 -->
    <BoxMatchDialog
      :visible="showMatchDialog"
      :box="matchDialogBox"
      :box-index="matchDialogIndex"
      :task-id="viewDetailTask?.id"
      @close="showMatchDialog = false"
      @update="handleBoxMatchUpdate"
      @submit-review="handleSubmitReview"
    />
    
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
import { ElMessage, ElMessageBox, ElLoading } from 'element-plus'
import taskApi from '@api/taskApi'
import { useAppStore } from '@stores/app'
import PageContainer from '@layout/PageContainer.vue'
import TaskStats from '@task/TaskStats.vue'
import TaskDetailPanel from '@task/TaskDetailPanel.vue'
import BoxMatchDialog from '@task/BoxMatchDialog.vue'
import ImageViewer from '@ui/ImageViewer.vue'
import FilterBar from '@ui/FilterBar.vue'
import FilterDropdown from '@ui/FilterDropdown.vue'
import TimeFilterDropdown from '@ui/TimeFilterDropdown.vue'
import ActionMenu from '@ui/ActionMenu.vue'
import {
  getStatusText,
  getStatusBadgeClass,
  formatDate,
  shouldShowDetect,
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
const timeFilter = ref('all')
const customStart = ref('')
const customEnd = ref('')

const store = useAppStore()

// 筛选选项配置
const statusOptions = [
  { value: null, label: '全部状态' },
  { value: 'completed', label: '已完成' },
  { value: 'pending', label: '待识别' },
  { value: 'detected', label: '待审核' },
  { value: 'failed', label: '识别失败' }
]

// 批量操作选项
const batchActions = [
  { label: '批量识别', icon: '🔍', action: 'detect' },
  { divider: true },
  { label: '导出 JSON', icon: '📄', action: 'export-json' },
  { label: '导出 CSV', icon: '📊', action: 'export-csv' },
  { divider: true },
  { label: '批量删除', icon: '🗑️', action: 'delete' }
]

const timeOptions = [
  { value: 'all', label: '全部时间' },
  { value: 'today', label: '今日' },
  { value: 'week', label: '本周' },
  { value: 'month', label: '本月' },
  { value: 'custom', label: '自定义' }
]

// 视图模式
const viewMode = ref('list')
const viewDetailTask = ref(null)

// 导出相关
const selectedTaskIds = ref([])
const exporting = ref(false)

// 批量识别加载状态
const detecting = ref(false)
const detectProgress = ref(null)

// 批量删除加载状态
const deleting = ref(false)

// 匹配弹窗相关
const showMatchDialog = ref(false)
const matchDialogBox = ref(null)
const matchDialogIndex = ref(0)

const handleBoxMatch = ({ box, index }) => {
  matchDialogBox.value = box
  matchDialogIndex.value = index
  showMatchDialog.value = true
}

const handleBoxMatchUpdate = ({ boxId, skuId, boxIndex }) => {
  if (!viewDetailTask.value) return

  const boxes = viewDetailTask.value.detections || viewDetailTask.value.result?.detections?.boxes
  if (!boxes) return

  const box = boxes.find(b => b.box_id === boxId || b.box_id === `box_${boxId}` || String(b.box_id) === String(boxId))
  if (!box) return

  if (box.match_result) {
    box.match_result.sku_id = skuId
    box.match_result.status = 'matched'
  } else {
    box.match_result = { sku_id: skuId, status: 'matched' }
  }
  
  box.isModified = true

  ElMessage.success(`箱体 ${boxIndex + 1} 已更新为 ${skuId}`)
}

const handleDeleteBox = async ({ box, index }) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除箱体 ${index + 1} 吗？`,
      '确认删除',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
  } catch {
    return
  }

  if (!viewDetailTask.value) return

  const boxes = viewDetailTask.value.detections || viewDetailTask.value.result?.detections?.boxes
  if (!boxes) return

  const targetBox = boxes.find(b => b.box_id === box.box_id || b.box_id === `box_${box.box_id}` || String(b.box_id) === String(box.box_id))
  if (!targetBox) return

  targetBox.status = 'deleted'
  ElMessage.success(`箱体 ${index + 1} 已删除`)
}

const handleSubmitReview = async () => {
  if (!viewDetailTask.value) return

  const boxes = viewDetailTask.value.detections || viewDetailTask.value.result?.detections?.boxes
  if (!boxes || boxes.length === 0) {
    ElMessage.warning('没有检测结果可提交')
    return
  }

  const activeBoxes = boxes.filter(b => b.status !== 'deleted')
  if (activeBoxes.length === 0) {
    ElMessage.warning('所有箱体都已删除，请恢复至少一个箱体后再提交')
    return
  }

  try {
    const resultBoxes = activeBoxes.map((b, idx) => ({
      box_id: `box_${idx}`,
      status: b.status === 'deleted' ? 'deleted' : (b.status || 'approved'),
      is_manual_override: b.isModified || false,
      custom_sku: b.match_result?.sku_id
    }))

    const res = await taskApi.reviewTask(viewDetailTask.value.id, resultBoxes)
    if (res.success) {
      ElMessage.success(res.message || '审核提交成功')
      viewDetailTask.value.status = 'completed'
      await loadTasks()
      await loadStats()
    } else {
      ElMessage.error('提交审核失败')
    }
  } catch (e) {
    ElMessage.error('提交审核失败: ' + (e.detail || e.message || '未知错误'))
  }
}

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
    const customTime = timeFilter.value === 'custom'
    let startUtc = null
    let endUtc = null
    if (customTime && customStart.value) {
      startUtc = new Date(customStart.value).toISOString().slice(0, 16)
    }
    if (customTime && customEnd.value) {
      endUtc = new Date(customEnd.value).toISOString().slice(0, 16)
    }
    const res = await taskApi.listTasks({
      page: page.value,
      page_size: pageSize.value,
      status_filter: statusFilter.value,
      time_filter: customTime ? null : timeFilter.value,
      start_time: startUtc,
      end_time: endUtc
    })
    if (res.success) {
      tasks.value = res.data.tasks
      total.value = res.data.total
      
      // 如果当前正在查看详情，同步更新基本信息，保留detections数据
      if (viewDetailTask.value) {
        const updatedTask = tasks.value.find(t => t.id === viewDetailTask.value.id)
        if (updatedTask) {
          // 保留原有的detections和result数据，只更新基本信息（状态等）
          viewDetailTask.value = { 
            ...updatedTask, 
            detections: viewDetailTask.value.detections,
            result: viewDetailTask.value.result 
          }
        }
      }
    }
  } catch (e) {
    ElMessage.error('加载任务列表失败: ' + (e.message || '未知错误'))
  } finally {
    loading.value = false
  }
}

const loadStats = async () => {
  try {
    const customTime = timeFilter.value === 'custom'
    let startUtc = null
    let endUtc = null
    if (customTime && customStart.value) {
      startUtc = new Date(customStart.value).toISOString().slice(0, 16)
    }
    if (customTime && customEnd.value) {
      endUtc = new Date(customEnd.value).toISOString().slice(0, 16)
    }
    const res = await taskApi.getTaskStats(
      customTime ? null : timeFilter.value,
      startUtc,
      endUtc
    )
    if (res.success) {
      stats.value = res.data
    }
  } catch (e) {
    console.error('加载统计失败', e)
  }
}

const handleStatusChange = () => {
  page.value = 1
  loadTasks()
}

const applyCustomTime = () => {
  if (!customStart.value || !customEnd.value) return
  if (new Date(customEnd.value) <= new Date(customStart.value)) {
    ElMessage.warning('结束时间必须晚于开始时间')
    return
  }
  page.value = 1
  loadTasks()
  loadStats()
}

const handleTimeChange = (value) => {
  if (value && value.type === 'custom') {
    if (!customStart.value || !customEnd.value) return
    if (new Date(customEnd.value) <= new Date(customStart.value)) {
      ElMessage.warning('结束时间必须晚于开始时间')
      return
    }
  }
  page.value = 1
  loadTasks()
  loadStats()
}

const handleBatchAction = (action) => {
  if (action === 'detect') {
    batchDetectTasks()
  } else if (action === 'export-json') {
    exportTasks('json', false)
  } else if (action === 'export-csv') {
    exportTasks('csv', false)
  } else if (action === 'delete') {
    batchDeleteTasks()
  }
}

const changePage = (newPage) => {
  if (newPage >= 1 && newPage <= totalPages.value) {
    page.value = newPage
    loadTasks()
  }
}

const getTaskImagePath = (task) => {
  return `/api/tasks/${task.id}/image`
}

const detectTask = async (task) => {
  submitting.value = true
  try {
    const res = await taskApi.detectTask(task.id)
    if (res.success && res.data && res.data.status === 'detected') {
      ElMessage.success('检测成功')
      await loadTasks()
      await loadStats()
    } else {
      ElMessage.error('检测未成功执行')
    }
  } catch (e) {
    ElMessage.error('检测失败: ' + (e.detail || e.message || '未知错误'))
  } finally {
    submitting.value = false
  }
}

const openViewDetail = async (task) => {
  try {
    const res = await taskApi.getTaskDetections(task.id)
    if (res.success && res.data && res.data.boxes) {
      viewDetailTask.value = {
        ...task,
        result: {
          detections: {
            boxes: res.data.boxes
          }
        }
      }
    } else {
      viewDetailTask.value = task
    }
    viewMode.value = 'detail'
  } catch (e) {
    viewDetailTask.value = task
    viewMode.value = 'detail'
  }
}

const closeViewDetail = () => {
  viewMode.value = 'list'
  viewDetailTask.value = null
}

const switchViewDetailTask = (task) => {
  openViewDetail(task)
}

const batchDetectTasks = async () => {
  if (selectedTasks.value.length === 0) {
    ElMessage.warning('请先选择要识别的任务')
    return
  }
  
  detecting.value = true
  detectProgress.value = 0
  
  try {
    const total = selectedTasks.value.length
    let completed = 0
    
    for (const task of selectedTasks.value) {
      if (shouldShowDetect(task) || shouldShowReDetect(task)) {
        await taskApi.detectTask(task.id)
      }
      completed++
      detectProgress.value = Math.round((completed / total) * 100)
    }
    
    ElMessage.success('批量识别完成')
    selectedTaskIds.value = []
    await loadTasks()
    await loadStats()
  } catch (e) {
    ElMessage.error('批量识别失败: ' + (e.detail || e.message || '未知错误'))
  } finally {
    detecting.value = false
    detectProgress.value = null
  }
}

const batchDeleteTasks = async () => {
  if (selectedTasks.value.length === 0) {
    ElMessage.warning('请先选择要删除的任务')
    return
  }

  try {
    await ElMessageBox.confirm(
      `确定要删除选中的 ${selectedTasks.value.length} 个任务吗？此操作不可恢复。`,
      '确认批量删除',
      {
        confirmButtonText: '确定删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
  } catch {
    return
  }

  deleting.value = true
  try {
    const taskIds = selectedTasks.value.map(task => task.id)
    const res = await taskApi.batchDeleteTasks(taskIds)
    if (res.success) {
      ElMessage.success(res.data.message || `成功删除 ${selectedTasks.value.length} 个任务`)
      selectedTaskIds.value = []
      await loadTasks()
      await loadStats()
    } else {
      ElMessage.error('批量删除失败')
    }
  } catch (e) {
    ElMessage.error('批量删除失败: ' + (e.detail || e.message || '未知错误'))
  } finally {
    deleting.value = false
  }
}

const openImageViewer = (imageUrl, imageName = '图片') => {
  currentImageUrl.value = imageUrl
  currentImageName.value = imageName
  showImageViewer.value = true
}

const exportTasks = async (format, includeImages) => {
  if (selectedTasks.value.length === 0) {
    ElMessage.warning('请先选择要导出的任务')
    return
  }

  exporting.value = true

  const taskIds = selectedTasks.value.map(task => task.id)

  try {
    const loadingInstance = ElLoading.service({
      lock: true,
      text: `正在导出 ${selectedTasks.value.length} 个任务...`,
      background: 'rgba(0, 0, 0, 0.7)'
    })

    const response = await taskApi.batchExportTasks(taskIds, format)

    if (!response.success) {
      throw new Error(response.error || '导出请求失败')
    }

    const blob = response.data
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

const openPendingTask = () => {
  const pendingId = store.pendingTaskId
  if (pendingId === null) return

  store.pendingTaskId = null

  const task = tasks.value.find(t => t.id === pendingId)
  if (task) {
    openViewDetail(task)
  } else {
    taskApi.getTask(pendingId).then(res => {
      if (res.success && res.data) {
        openViewDetail(res.data)
      }
    })
  }
}

onMounted(async () => {
  await loadTasks()
  await loadStats()
  openPendingTask()
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

.task-list-section {
  flex: 1;
  min-width: 0;
}

.section {
  background: var(--color-bg-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
  box-shadow: var(--shadow-md);
}

.table-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
}

.time-custom-panel {
  padding: var(--spacing-sm) var(--spacing-md);
  border-top: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.time-row {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.time-label {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  min-width: 32px;
}

.time-input {
  padding: var(--spacing-xs) var(--spacing-sm);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
  flex: 1;
  color: var(--color-text-primary);
  background: var(--color-bg-primary);
  outline: none;
  transition: border-color var(--transition-fast);
}

.time-input:focus {
  border-color: var(--color-primary);
}

.time-custom-panel .btn-small {
  padding: var(--spacing-xs) var(--spacing-md);
  font-size: var(--font-size-sm);
  align-self: flex-end;
  margin-top: var(--spacing-xs);
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

.status-badge.pending {
  background: rgba(14, 165, 233, 0.1);
  color: var(--color-info);
}

.status-badge.detected {
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

.status-badge-btn {
  border: none;
  cursor: pointer;
  font: inherit;
  outline: none;
  transition: opacity var(--transition-fast);
}

.status-badge-btn:hover {
  opacity: 0.8;
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

.loading, .empty-state {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--color-text-secondary);
}

.empty-icon {
  font-size: 48px;
  margin-bottom: var(--spacing-md);
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: var(--spacing-md);
  margin-top: var(--spacing-lg);
}

.pagination button {
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.pagination button:hover:not(:disabled) {
  background: var(--color-primary);
  color: white;
  border-color: var(--color-primary);
}

.pagination button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 响应式设计 */
@media (max-width: 900px) {
  .task-counts {
    display: none;
  }
}

@media (max-width: 768px) {
  .task-row-header {
    flex-wrap: wrap;
    gap: var(--spacing-md);
  }
  
  .task-name {
    width: 100%;
    order: -1;
  }
}

/* 查看详情视图样式 - position fixed 实现真贴边 */
.detail-view-container {
  position: fixed;
  top: 60px;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  background: var(--color-bg-primary);
  z-index: 100;
}

.detail-view-sidebar {
  width: 320px;
  display: flex;
  flex-direction: column;
  background: var(--color-bg-tertiary);
  border-right: 1px solid var(--color-border);
  flex-shrink: 0;
}

.detail-view-topbar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--color-primary);
  color: white;
  flex-shrink: 0;
  min-height: 42px;
}

.topbar-left {
  flex-shrink: 0;
}

.topbar-left h3 {
  margin: 0;
  font-size: var(--font-size-sm);
  white-space: nowrap;
}

.topbar-close {
  background: transparent;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: white;
  padding: 0;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: background-color var(--transition-fast);
  flex-shrink: 0;
  line-height: 1;
}

.topbar-close:hover {
  background: rgba(255, 255, 255, 0.2);
}

.detail-view-main-topbar {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  background: var(--color-bg-primary);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
  min-height: 42px;
  font-size: 12px;
  color: var(--color-text-secondary);
}

.mainbar-item {
  font-weight: 500;
  color: var(--color-text-primary);
  white-space: nowrap;
}

.mainbar-item strong {
  color: var(--color-primary);
  margin-right: 4px;
}

.mainbar-divider {
  color: var(--color-border);
  font-size: 14px;
}

.mainbar-dot {
  color: var(--color-text-tertiary);
  font-size: 16px;
  line-height: 1;
}

.mainbar-status {
  display: inline-block;
  padding: 1px 8px;
  border-radius: var(--radius-xs);
  font-size: 11px;
  font-weight: 500;
}

.mainbar-status.pending { background: rgba(14, 165, 233, 0.1); color: var(--color-info); }
.mainbar-status.detected { background: rgba(230, 162, 60, 0.1); color: var(--color-warning); }
.mainbar-status.completed { background: rgba(103, 194, 58, 0.1); color: var(--color-success); }
.mainbar-status.failed { background: rgba(245, 108, 108, 0.1); color: var(--color-danger); }

.mainbar-spacer {
  flex: 1;
  min-width: 12px;
}

.detail-view-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: var(--color-bg-secondary);
  overflow: hidden;
}

.detail-view-main > :deep(.task-detail-panel) {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.detail-view-nav {
  display: flex;
  flex-direction: column;
  flex: 1;
  overflow: hidden;
}

.detail-nav-list {
  flex: 1;
  overflow-y: auto;
  padding: var(--spacing-sm);
}

.detail-nav-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  margin-bottom: var(--spacing-xs);
  background: var(--color-bg-secondary);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--transition-fast);
}

.detail-nav-item:hover {
  background: var(--color-bg-primary);
}

.detail-nav-item.active {
  background: var(--color-primary-light);
  border-left: 3px solid var(--color-primary);
}

.detail-nav-item.reviewed {
  opacity: 0.7;
}

.nav-item-thumb {
  width: 48px;
  height: 48px;
  flex-shrink: 0;
  border-radius: var(--radius-xs);
  overflow: hidden;
  background: var(--color-bg-primary);
}

.nav-item-thumb-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.nav-item-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-item-id {
  font-weight: 600;
  font-size: var(--font-size-sm);
  color: var(--color-text-primary);
}

.nav-item-name {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.nav-item-check {
  color: var(--color-success);
  font-size: 18px;
  flex-shrink: 0;
}

.nav-item-info .status-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: var(--radius-xs);
  font-size: var(--font-size-xs);
}
</style>
