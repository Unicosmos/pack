<template>
  <div v-if="status !== 'ready'" class="status-banner" :class="statusClass">
    <div class="status-content">
      <span class="status-icon">{{ statusIcon }}</span>
      <span class="status-text">{{ statusText }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  status: {
    type: String,
    default: 'ready'
  }
})

const statusClass = computed(() => {
  switch (props.status) {
    case 'init':
      return 'status-warning'
    case 'error':
      return 'status-error'
    case 'no-sku':
      return 'status-info'
    default:
      return ''
  }
})

const statusIcon = computed(() => {
  switch (props.status) {
    case 'init':
      return '⚠️'
    case 'error':
      return '❌'
    case 'no-sku':
      return 'ℹ️'
    default:
      return ''
  }
})

const statusText = computed(() => {
  switch (props.status) {
    case 'init':
      return '系统初始化中... 部分功能可能不可用'
    case 'error':
      return '系统异常，请检查后端服务'
    case 'no-sku':
      return 'SKU库未配置，匹配功能受限'
    default:
      return ''
  }
})
</script>

<style scoped>
.status-banner {
  padding: 12px 20px;
  text-align: center;
  font-size: 14px;
}

.status-warning {
  background: linear-gradient(135deg, #fef08a 0%, #fde047 100%);
  color: #854d0e;
}

.status-error {
  background: linear-gradient(135deg, #fecaca 0%, #f87171 100%);
  color: #991b1b;
}

.status-info {
  background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
  color: #1e40af;
}

.status-content {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.status-icon {
  font-size: 16px;
}

.status-text {
  font-weight: 500;
}
</style>