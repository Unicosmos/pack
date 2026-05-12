<template>
  <div class="task-list-page">
    <div class="header">
      <h1>📋 任务列表</h1>
      <p>查看历史检测任务记录</p>
    </div>

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

      <table v-else class="task-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>图片名称</th>
            <th>状态</th>
            <th>检测数量</th>
            <th>匹配数量</th>
            <th>创建时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="task in tasks" :key="task.id">
            <td>{{ task.id }}</td>
            <td class="task-name">{{ task.image_name }}</td>
            <td>
              <span :class="['status-badge', task.status]">
                {{ getStatusText(task.status) }}
              </span>
            </td>
            <td>{{ task.box_count || 0 }}</td>
            <td>{{ task.matched_count || 0 }}</td>
            <td>{{ formatDate(task.created_at) }}</td>
            <td>
              <button class="btn-icon" @click="viewTask(task)" title="查看">👁️</button>
              <button class="btn-icon danger" @click="deleteTask(task.id)" title="删除">🗑️</button>
            </td>
          </tr>
        </tbody>
      </table>

      <div class="pagination" v-if="total > pageSize">
        <button :disabled="page <= 1" @click="changePage(page - 1)">上一页</button>
        <span>第 {{ page }} / {{ totalPages }} 页</span>
        <button :disabled="page >= totalPages" @click="changePage(page + 1)">下一页</button>
      </div>
    </div>

    <div v-if="selectedTask" class="modal-overlay" @click.self="selectedTask = null">
      <div class="modal">
        <div class="modal-header">
          <h3>任务详情 #{{ selectedTask.id }}</h3>
          <button class="btn-close" @click="selectedTask = null">×</button>
        </div>
        <div class="modal-body">
          <div class="detail-row">
            <span class="label">图片名称：</span>
            <span>{{ selectedTask.image_name }}</span>
          </div>
          <div class="detail-row">
            <span class="label">状态：</span>
            <span :class="['status-badge', selectedTask.status]">
              {{ getStatusText(selectedTask.status) }}
            </span>
          </div>
          <div class="detail-row">
            <span class="label">检测数量：</span>
            <span>{{ selectedTask.box_count || 0 }}</span>
          </div>
          <div class="detail-row">
            <span class="label">匹配数量：</span>
            <span>{{ selectedTask.matched_count || 0 }}</span>
          </div>
          <div class="detail-row">
            <span class="label">创建时间：</span>
            <span>{{ formatDate(selectedTask.created_at) }}</span>
          </div>
          <div v-if="selectedTask.result" class="detail-row">
            <span class="label">匹配结果：</span>
            <pre class="result-json">{{ JSON.stringify(selectedTask.result, null, 2) }}</pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { tasks as taskApi } from '../api/client'

const tasks = ref([])
const loading = ref(false)
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)
const statusFilter = ref(null)
const stats = ref({})
const selectedTask = ref(null)

const totalPages = computed(() => Math.ceil(total.value / pageSize.value))

const loadTasks = async () => {
  loading.value = true
  try {
    const res = await taskApi.list(page.value, pageSize.value, statusFilter.value)
    if (res.success) {
      tasks.value = res.tasks
      total.value = res.total
    }
  } catch (e) {
    ElMessage.error('加载任务列表失败')
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
  page.value = newPage
  loadTasks()
}

const viewTask = async (task) => {
  try {
    const res = await taskApi.get(task.id)
    if (res) {
      selectedTask.value = res
    }
  } catch (e) {
    ElMessage.error('获取任务详情失败')
  }
}

const deleteTask = async (id) => {
  if (!confirm('确定要删除这个任务吗？')) return

  try {
    const res = await taskApi.delete(id)
    if (res.success) {
      ElMessage.success('删除成功')
      loadTasks()
      loadStats()
    }
  } catch (e) {
    ElMessage.error('删除失败')
  }
}

const getStatusText = (status) => {
  const map = {
    'pending': '进行中',
    'completed': '已完成',
    'failed': '失败'
  }
  return map[status] || status
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
}

.filter-tabs button:hover {
  background: #e0e0e0;
}

.filter-tabs button.active {
  background: #667eea;
  color: white;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}

.btn-primary {
  background: #667eea;
  color: white;
}

.btn-primary:hover {
  background: #5a70d6;
}

.loading {
  text-align: center;
  padding: 40px;
  color: #999;
}

.empty-state {
  text-align: center;
  padding: 60px;
  color: #999;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 15px;
}

.task-table {
  width: 100%;
  border-collapse: collapse;
}

.task-table th,
.task-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #f0f0f0;
}

.task-table th {
  font-weight: 600;
  color: #333;
  background: #fafafa;
}

.task-table tbody tr:hover {
  background: #f5f5f5;
}

.task-name {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
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

.status-badge.completed {
  background: #e1f3d8;
  color: #67c23a;
}

.status-badge.failed {
  background: #fef0f0;
  color: #f56c6c;
}

.btn-icon {
  padding: 6px 10px;
  background: transparent;
  border: none;
  cursor: pointer;
  font-size: 16px;
}

.btn-icon.danger:hover {
  background: #fef0f0;
}

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
}

.pagination button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal {
  background: white;
  border-radius: 12px;
  width: 90%;
  max-width: 600px;
  max-height: 80vh;
  overflow: auto;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #f0f0f0;
}

.modal-header h3 {
  margin: 0;
}

.btn-close {
  background: transparent;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #999;
}

.btn-close:hover {
  color: #333;
}

.modal-body {
  padding: 20px;
}

.detail-row {
  margin-bottom: 15px;
}

.detail-row .label {
  font-weight: 600;
  color: #333;
}

.result-json {
  background: #f5f5f5;
  padding: 10px;
  border-radius: 6px;
  font-size: 12px;
  overflow: auto;
  max-height: 200px;
}
</style>
