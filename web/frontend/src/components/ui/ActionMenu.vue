<template>
  <div class="action-menu" ref="menuRef">
    <button
      class="menu-trigger"
      :class="{ active: isOpen, disabled: loading || selectedCount === 0 }"
      @click="toggle"
      :disabled="loading || selectedCount === 0"
    >
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
          :class="{ divider: item.divider, danger: item.danger }"
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
import { ref, computed, onMounted, onUnmounted } from 'vue'

const props = defineProps({
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
  },
  selectedCount: {
    type: Number,
    default: 0
  }
})

const emit = defineEmits(['select'])

const isOpen = ref(false)
const menuRef = ref(null)

const toggle = () => {
  if (props.loading || props.selectedCount === 0) return
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
  margin-left: auto;
}

/* 当在 filter-bar 中时的样式 */
:global(.filter-bar) .action-menu {
  margin-left: auto;
}

.menu-trigger {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid var(--color-border);
  background: var(--color-bg-card);
  color: var(--color-text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
  position: relative;
}

.menu-trigger:hover:not(.disabled) {
  border-color: var(--color-text-tertiary);
  color: var(--color-text-primary);
}

.menu-trigger.active {
  border-color: var(--color-primary);
  color: var(--color-primary);
  background: rgba(59, 130, 246, 0.08);
}

.dark .menu-trigger {
  background: #0f172a;
}

.dark .menu-trigger:hover:not(.disabled) {
  border-color: #475569;
}

.menu-trigger.disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.menu-label {
  flex: 1;
  text-align: left;
}

.dropdown-arrow {
  font-size: 10px;
  transition: transform 0.2s;
}

.dropdown-arrow.open {
  transform: rotate(180deg);
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
  border-radius: 0 0 6px 6px;
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
  right: 0;
  left: auto;
  min-width: 130px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 4px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
  z-index: 50;
  overflow: hidden;
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  cursor: pointer;
  font-size: 12px;
  color: var(--color-text-secondary);
  transition: all 0.1s;
  border-radius: 4px;
  white-space: nowrap;
}

.menu-item:hover {
  background: var(--color-bg-hover);
  color: var(--color-text-primary);
}

.menu-item.danger {
  color: var(--color-danger);
}

.menu-item.danger:hover {
  background: rgba(239, 68, 68, 0.1);
}

.dark .menu-item:hover {
  background: rgba(255, 255, 255, 0.05);
}

.menu-item.divider {
  height: 1px;
  background: var(--color-border);
  margin: 4px 0;
  padding: 0;
  cursor: default;
  border-radius: 0;
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
