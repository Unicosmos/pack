<template>
  <div class="time-filter-dropdown" ref="dropdownRef">
    <button
      class="dropdown-trigger"
      :class="{ active: isOpen }"
      @click="toggle"
    >
      <span class="dropdown-label">{{ displayLabel }}</span>
      <span class="dropdown-arrow" :class="{ open: isOpen }">▼</span>
    </button>
    <transition name="dropdown">
      <div v-if="isOpen" class="dropdown-panel" @click.stop>
        <div
          v-for="option in quickOptions"
          :key="option.value"
          class="dropdown-item"
          :class="{ selected: modelValue === option.value }"
          @click="selectQuickOption(option.value)"
        >
          {{ option.label }}
        </div>
        <div class="custom-panel" v-if="showCustomPanel">
          <div class="time-row">
            <span class="time-label">开始</span>
            <input type="date" v-model="customStart" class="time-input" />
          </div>
          <div class="time-row">
            <span class="time-label">结束</span>
            <input type="date" v-model="customEnd" class="time-input" />
          </div>
          <button
            class="confirm-btn"
            :disabled="!customStart || !customEnd"
            @click="confirmCustom"
          >
            查询
          </button>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  modelValue: { type: String, default: 'all' },
  customStart: { type: String, default: '' },
  customEnd: { type: String, default: '' }
})

const emit = defineEmits(['update:modelValue', 'update:customStart', 'update:customEnd', 'change'])

const isOpen = ref(false)
const dropdownRef = ref(null)
const customStart = ref(props.customStart)
const customEnd = ref(props.customEnd)
const showCustomPanel = ref(false)

const quickOptions = [
  { value: 'all', label: '全部时间' },
  { value: 'today', label: '今日' },
  { value: 'week', label: '本周' },
  { value: 'month', label: '本月' },
  { value: 'custom', label: '自定义' }
]

const displayLabel = computed(() => {
  if (props.modelValue === 'custom') return '自定义'
  const option = quickOptions.find(o => o.value === props.modelValue)
  return option ? option.label : '全部时间'
})

const toggle = () => {
  isOpen.value = !isOpen.value
}

const selectQuickOption = (value) => {
  if (value === 'custom') {
    showCustomPanel.value = !showCustomPanel.value
    return
  }
  showCustomPanel.value = false
  emit('update:modelValue', value)
  emit('change', value)
  isOpen.value = false
}

const confirmCustom = () => {
  emit('update:modelValue', 'custom')
  emit('update:customStart', customStart.value)
  emit('update:customEnd', customEnd.value)
  emit('change', { type: 'custom', start: customStart.value, end: customEnd.value })
  isOpen.value = false
}

const handleClickOutside = (event) => {
  if (dropdownRef.value && !dropdownRef.value.contains(event.target)) {
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
.time-filter-dropdown {
  position: relative;
  display: inline-block;
}

.dropdown-trigger {
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
  position: relative;
  white-space: nowrap;
}

.dropdown-trigger:hover {
  border-color: var(--color-text-tertiary);
  color: var(--color-text-primary);
}

.dropdown-trigger.active {
  border-color: var(--color-primary);
  color: var(--color-primary);
  background: rgba(59, 130, 246, 0.08);
}

.dark .dropdown-trigger {
  background: #0f172a;
}

.dark .dropdown-trigger:hover {
  border-color: #475569;
}

.dropdown-label {
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

.dropdown-panel {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  min-width: 180px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 4px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
  z-index: 50;
  overflow: hidden;
}

.dropdown-item {
  padding: 6px 10px;
  border-radius: 4px;
  font-size: 12px;
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all 0.1s;
  white-space: nowrap;
}

.dropdown-item:hover {
  background: var(--color-bg-hover);
  color: var(--color-text-primary);
}

.dropdown-item.selected {
  color: var(--color-primary);
  background: rgba(59, 130, 246, 0.08);
}

.dark .dropdown-item:hover {
  background: rgba(255, 255, 255, 0.05);
}

.custom-panel {
  padding: 10px;
  border-top: 1px solid var(--color-border);
  margin-top: 4px;
}

.time-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.time-label {
  font-size: 11px;
  color: var(--color-text-tertiary);
  min-width: 36px;
}

.time-input {
  flex: 1;
  padding: 5px 8px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  color: var(--color-text-primary);
  font-size: 12px;
  outline: none;
}

.dark .time-input {
  background: #0f172a;
}

.time-input:focus {
  border-color: var(--color-primary);
}

.confirm-btn {
  width: 100%;
  padding: 6px;
  margin-top: 2px;
  background: var(--color-primary);
  color: #fff;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.2s;
}

.confirm-btn:hover:not(:disabled) {
  background: #2563eb;
}

.confirm-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
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
