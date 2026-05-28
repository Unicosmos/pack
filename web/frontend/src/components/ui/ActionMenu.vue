<template>
  <div class="action-menu" ref="menuRef">
    <button class="menu-trigger" @click="toggle" :disabled="loading">
      <span class="menu-label"><slot name="label"></slot></span>
      <span v-if="loading" class="loading-indicator">
        <span class="loading-spinner"></span>
      </span>
      <span v-else class="dropdown-arrow" :class="{ open: isOpen }">▼</span>
      
      <!-- 进度条 -->
      <div v-if="loading && progress !== null" class="progress-wrapper">
        <div class="progress-bar" :style="{ width: progress + '%' }"></div>
      </div>
    </button>
    <transition name="dropdown">
      <div v-if="isOpen && !loading" class="menu-panel" @click.stop>
        <div
          v-for="(item, index) in items"
          :key="index"
          class="menu-item"
          :class="{ divider: item.divider }"
          @click="handleSelect(item)"
        >
          <template v-if="!item.divider">
            <span class="menu-icon">{{ item.icon }}</span>
            <span class="menu-text">{{ item.label }}</span>
          </template>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

defineProps({
  items: {
    type: Array,
    default: () => []
  },
  loading: {
    type: Boolean,
    default: false
  },
  progress: {
    type: Number,
    default: null
  }
})

const emit = defineEmits(['select'])

const isOpen = ref(false)
const menuRef = ref(null)

const toggle = () => {
  isOpen.value = !isOpen.value
}

const handleSelect = (item) => {
  if (item.divider) return
  emit('select', item.action)
  isOpen.value = false
}

const handleClickOutside = (event) => {
  if (menuRef.value && !menuRef.value.contains(event.target)) {
    isOpen.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<style scoped>
.action-menu {
  position: relative;
  display: inline-block;
}

.menu-trigger {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  cursor: pointer;
  font-size: var(--font-size-sm);
  color: var(--color-text-primary);
  transition: all var(--transition-fast);
  min-width: 100px;
}

.menu-trigger:hover {
  background: var(--color-bg-secondary);
  border-color: var(--color-primary);
}

.menu-label {
  flex: 1;
  text-align: left;
}

.dropdown-arrow {
  font-size: 10px;
  transition: transform var(--transition-fast);
}

.dropdown-arrow.open {
  transform: rotate(180deg);
}

.menu-trigger:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.loading-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
}

.loading-spinner {
  width: 14px;
  height: 14px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.progress-wrapper {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--color-border);
  border-radius: 0 0 var(--radius-md) var(--radius-md);
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: var(--color-primary);
  transition: width 0.3s ease;
}

.menu-panel {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  min-width: 160px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
  z-index: 100;
  overflow: hidden;
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  cursor: pointer;
  font-size: var(--font-size-sm);
  color: var(--color-text-primary);
  transition: background-color var(--transition-fast);
  min-height: 36px;
  box-sizing: border-box;
}

.menu-item:hover {
  background: var(--color-bg-tertiary);
}

.menu-item.divider {
  height: 1px;
  background: var(--color-border);
  margin: 4px 0;
  cursor: default;
}

.menu-icon {
  width: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.menu-text {
  flex: 1;
}

.dropdown-enter-active,
.dropdown-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.dropdown-enter-from,
.dropdown-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
