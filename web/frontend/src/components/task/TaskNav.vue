<template>
  <div class="task-nav">
    <div class="nav-header">
      <h3>{{ title }}</h3>
      <button class="btn-close" @click="$emit('close')">×</button>
    </div>
    <div class="nav-list">
      <div 
        v-for="(task, index) in tasks" 
        :key="task.id"
        :class="{ 
          'nav-item': true,
          'active': task.id === activeId, 
          'reviewed': task.status === 'completed' 
        }"
        @click="$emit('select', task, index)"
      >
        <div class="nav-thumb">
          <img 
            :src="getTaskImagePath(task)" 
            :alt="task.image_name"
            class="nav-thumb-img"
            @error="$event.target.style.display='none'"
          />
        </div>
        <div class="nav-info">
          <div class="nav-id">#{{ task.id }}</div>
          <div class="nav-name">{{ task.image_name }}</div>
          <span :class="['status-badge', getStatusBadgeClass(task)]">
            {{ getStatusText(task.status) }}
          </span>
        </div>
        <div v-if="task.status === 'completed'" class="nav-check">✓</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { getStatusText, getStatusBadgeClass } from '@utils/taskUtils'

defineProps({
  title: {
    type: String,
    default: '任务列表'
  },
  tasks: {
    type: Array,
    required: true
  },
  activeId: {
    type: [Number, String],
    default: null
  }
})

defineEmits(['select', 'close'])

const getTaskImagePath = (task) => {
  return `/api/tasks/${task.id}/image`
}
</script>

<style scoped>
.task-nav {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--color-bg-tertiary);
}

.nav-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-md);
  border-bottom: 1px solid var(--color-border);
  background: var(--color-primary);
  color: white;
}

.nav-header h3 {
  margin: 0;
  font-size: var(--font-size-base);
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

.nav-list {
  flex: 1;
  overflow-y: auto;
  padding: var(--spacing-sm);
}

.nav-item {
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

.nav-item:hover {
  background: var(--color-bg-primary);
}

.nav-item.active {
  background: var(--color-primary-light);
  border-left: 3px solid var(--color-primary);
}

.nav-item.reviewed {
  opacity: 0.7;
}

.nav-thumb {
  width: 48px;
  height: 48px;
  flex-shrink: 0;
  border-radius: var(--radius-xs);
  overflow: hidden;
  background: var(--color-bg-primary);
}

.nav-thumb-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.nav-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-id {
  font-weight: 600;
  font-size: var(--font-size-sm);
  color: var(--color-text-primary);
}

.nav-name {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.nav-check {
  color: var(--color-success);
  font-size: 18px;
  flex-shrink: 0;
}

.status-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: var(--radius-xs);
  font-size: var(--font-size-xs);
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
</style>