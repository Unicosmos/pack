<template>
  <section class="recent-tasks">
    <div class="section-header">
      <h3>📋 最近任务</h3>
      <a href="#" class="view-all" @click.prevent="goToTasks">查看全部 →</a>
    </div>

    <div v-if="loading" class="loading-state">加载中...</div>

    <div v-else-if="tasks.length === 0" class="empty-state">
      <div class="empty-icon">📭</div>
      <p>暂无任务记录</p>
      <p class="empty-tip">上传图片后将自动创建识别任务</p>
    </div>

    <div v-else class="task-table">
      <div class="task-row header-row">
        <div class="task-cell cell-id">ID</div>
        <div class="task-cell cell-thumb">缩略图</div>
        <div class="task-cell cell-name">图片名称</div>
        <div class="task-cell cell-status">状态</div>
        <div class="task-cell cell-counts">统计</div>
        <div class="task-cell cell-date">创建时间</div>
      </div>

      <div
        v-for="task in tasks"
        :key="task.id"
        class="task-row data-row"
        @click="goToTaskDetail(task)"
      >
        <div class="task-cell cell-id">#{{ task.id }}</div>
        <div class="task-cell cell-thumb">
          <img
            :src="`/api/tasks/${task.id}/image`"
            :alt="task.image_name"
            class="task-thumb"
            @error="$event.target.style.display='none'"
          />
        </div>
        <div class="task-cell cell-name">{{ task.image_name }}</div>
        <div class="task-cell cell-status">
          <span :class="['tag', getStatusBadgeClass(task)]">{{ getStatusText(task.status) }}</span>
        </div>
        <div class="task-cell cell-counts">
          检测: {{ task.box_count || 0 }} · 匹配: {{ task.matched_count || 0 }}
        </div>
        <div class="task-cell cell-date">{{ formatDate(task.created_at) }}</div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import taskApi from '@api/taskApi'
import { getStatusBadgeClass, getStatusText, formatDate } from '@utils/taskUtils'

const emit = defineEmits(['view-task', 'navigate'])

const tasks = ref([])
const loading = ref(false)

const loadRecentTasks = async () => {
  loading.value = true
  try {
    const res = await taskApi.listTasks({ page: 1, page_size: 5 })
    if (res.success) {
      tasks.value = res.data.tasks || []
    }
  } catch (e) {
    console.error('加载最近任务失败', e)
  } finally {
    loading.value = false
  }
}

const goToTasks = () => {
  emit('navigate', 'tasks')
}

const goToTaskDetail = (task) => {
  emit('view-task', task)
}

onMounted(() => {
  loadRecentTasks()
})
</script>

<style scoped>
.recent-tasks {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 16px;
  border-bottom: 1px solid var(--color-border);
}

.section-header h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.view-all {
  color: var(--color-primary);
  text-decoration: none;
  font-size: 13px;
}

.view-all:hover {
  opacity: 0.8;
}

.loading-state,
.empty-state {
  text-align: center;
  padding: 32px 16px;
  color: var(--color-text-tertiary);
}

.empty-icon {
  font-size: 36px;
  margin-bottom: 8px;
}

.empty-state p {
  margin: 0 0 4px;
  font-size: 14px;
}

.empty-tip {
  font-size: 12px;
  color: var(--color-text-tertiary);
}

.task-table {
  font-size: 13px;
}

.task-row {
  display: flex;
  align-items: center;
  padding: 10px 16px;
  border-bottom: 1px solid var(--color-border-light);
  transition: background-color var(--transition-fast);
}

.task-row:last-child {
  border-bottom: none;
}

.task-row.data-row {
  cursor: pointer;
}

.task-row.data-row:hover {
  background: var(--color-bg-tertiary);
}

.header-row {
  font-weight: 600;
  color: var(--color-text-secondary);
  font-size: 12px;
  background: var(--color-bg-tertiary);
}

.task-cell {
  padding: 0 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cell-id {
  width: 50px;
  font-weight: 600;
  color: var(--color-primary);
}

.cell-thumb {
  width: 50px;
  height: 36px;
  padding: 0 6px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.task-thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 4px;
  background: var(--color-bg-tertiary);
}

.cell-name {
  flex: 1;
  min-width: 100px;
  font-weight: 500;
}

.cell-status {
  width: 80px;
  display: flex;
  align-items: center;
}

.cell-counts {
  width: 140px;
  color: var(--color-text-secondary);
  font-size: 12px;
}

.cell-date {
  width: 150px;
  color: var(--color-text-secondary);
  font-size: 12px;
  text-align: right;
}

.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
}

.tag.completed {
  background: rgba(74, 222, 128, 0.15);
  color: var(--color-success);
}

.tag.pending {
  background: rgba(14, 165, 233, 0.15);
  color: #38bdf8;
}

.tag.detected {
  background: rgba(251, 191, 36, 0.15);
  color: var(--color-warning);
}

.tag.failed {
  background: rgba(248, 113, 113, 0.15);
  color: var(--color-danger);
}

@media (max-width: 768px) {
  .cell-counts,
  .cell-date {
    display: none;
  }
}
</style>