<template>
  <div class="sys-status-bar" :class="statusClass">
    <div class="status-left">
      <span class="status-dot" :class="statusDotClass"></span>
      <span class="status-label">{{ statusLabel }}</span>
      <span class="status-divider">|</span>
      <span class="status-model">检测:</span>
      <FilterDropdown v-model="selectedDetector" :options="detectorOptions" class="model-dropdown" />
      <span v-if="matcherModel" class="status-divider">|</span>
      <span v-if="matcherModel" class="status-model">匹配:</span>
      <FilterDropdown v-if="matcherModel" v-model="selectedMatcher" :options="matcherOptions" class="model-dropdown" />
      <span class="status-divider">|</span>
      <span class="status-sku">SKU库: {{ skuCount }}个</span>
    </div>
    <div class="status-right">
      <a href="#" @click.prevent="goTo('tasks')">📋 查看任务列表 →</a>
      <a href="#" @click.prevent="goTo('skus')">📦 管理SKU库 →</a>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useAppStore } from '@stores/app'
import FilterDropdown from '@ui/FilterDropdown.vue'

const store = useAppStore()

const emit = defineEmits(['navigate'])

const statusDotClass = computed(() => {
  if (store.systemStatus === 'error' || store.systemStatus === 'init') return 'dot-error'
  if (store.systemStatus === 'no-sku') return 'dot-warning'
  return 'dot-ready'
})

const statusLabel = computed(() => {
  switch (store.systemStatus) {
    case 'init':
      return '系统初始化中...'
    case 'error':
      return '系统异常'
    case 'no-sku':
      return '检测就绪，SKU匹配未配置'
    default:
      return '系统就绪'
  }
})

const statusClass = computed(() => {
  if (store.systemStatus === 'error') return 'bar-error'
  if (store.systemStatus === 'init') return 'bar-warning'
  return ''
})

const detectorModel = computed(() => {
  return store.modelInfo || '未加载'
})

const matcherModel = computed(() => {
  return store.skuModelInfo || ''
})

const skuCount = computed(() => store.skuCount)

const detectorOptions = [
  { value: 'best.pt', label: 'best.pt' },
  { value: 'base.pt', label: 'base.pt' }
]

const matcherOptions = [
  { value: 'vits16_dino_finetuned.pth', label: '微调模型' },
  { value: 'vits16_dino.pth', label: '预训练模型' }
]

const selectedDetector = ref(store.modelInfo || 'best.pt')
const selectedMatcher = ref(store.skuModelInfo || 'vits16_dino_finetuned.pth')

const goTo = (page) => {
  emit('navigate', page)
}
</script>

<style scoped>
.sys-status-bar {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 10px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  transition: background-color var(--transition-normal), border-color var(--transition-normal);
}

.bar-error {
  background: rgba(248, 113, 113, 0.1);
  border-color: var(--color-danger);
}

.bar-warning {
  background: rgba(251, 191, 36, 0.1);
  border-color: var(--color-warning);
}

.status-left {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text-secondary);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.dot-ready {
  background: var(--color-success);
  box-shadow: 0 0 6px var(--color-success);
}

.dot-warning {
  background: var(--color-warning);
  box-shadow: 0 0 6px var(--color-warning);
}

.dot-error {
  background: var(--color-danger);
  box-shadow: 0 0 6px var(--color-danger);
}

.status-label {
  font-weight: 500;
  color: var(--color-text-primary);
  white-space: nowrap;
}

.status-divider {
  color: var(--color-border);
  font-size: 14px;
}

.status-model,
.status-sku {
  white-space: nowrap;
}

.status-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.status-right a {
  color: var(--color-primary);
  text-decoration: none;
  font-size: 13px;
  white-space: nowrap;
  transition: opacity var(--transition-fast);
}

.status-right a:hover {
  opacity: 0.8;
}

@media (max-width: 768px) {
  .sys-status-bar {
    flex-direction: column;
    gap: 8px;
    align-items: flex-start;
  }

  .status-right {
    width: 100%;
    justify-content: flex-start;
  }
}

.model-dropdown {
  display: inline-flex;
  vertical-align: middle;
}

.model-dropdown :deep(.dropdown-trigger) {
  padding: 1px 6px;
  font-size: 13px;
  border: none;
  background: transparent;
  color: var(--color-text-primary);
  font-weight: 500;
  gap: 2px;
}

.model-dropdown :deep(.dropdown-trigger:hover) {
  color: var(--color-primary);
  background: var(--color-bg-hover);
}

.model-dropdown :deep(.dropdown-arrow) {
  font-size: 10px;
  color: var(--color-text-tertiary);
}
</style>