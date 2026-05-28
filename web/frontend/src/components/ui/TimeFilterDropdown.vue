<template>
  <div class="time-filter-dropdown" ref="dropdownRef">
    <button class="dropdown-trigger" @click="toggle">
      <span class="dropdown-label">{{ displayLabel }}</span>
      <span class="dropdown-arrow" :class="{ open: isOpen }">▼</span>
    </button>
    <transition name="dropdown">
      <div v-if="isOpen" class="dropdown-panel" @click.stop>
        <div
          v-for="option in quickOptions"
          :key="option.value"
          class="dropdown-item"
          :class="{ active: modelValue === option.value }"
          @click="selectQuickOption(option.value)"
        >
          <span class="check">{{ modelValue === option.value ? '✓' : '' }}</span>
          <span>{{ option.label }}</span>
        </div>
        <div class="dropdown-divider"></div>
        <div class="custom-panel" v-if="modelValue === 'custom'">
          <div class="time-row">
            <span class="time-label">开始</span>
            <input type="datetime-local" v-model="customStart" class="time-input" />
          </div>
          <div class="time-row">
            <span class="time-label">结束</span>
            <input type="datetime-local" v-model="customEnd" class="time-input" />
          </div>
          <button
            class="btn btn-small btn-primary"
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

const quickOptions = [
  { value: 'all', label: '全部时间' },
  { value: 'today', label: '今日' },
  { value: 'week', label: '本周' },
  { value: 'month', label: '本月' },
  { value: 'custom', label: '自定义' }
]

const displayLabel = computed(() => {
  const option = quickOptions.find(o => o.value === props.modelValue)
  return option ? option.label : '全部时间'
})

const toggle = () => {
  isOpen.value = !isOpen.value
}

const selectQuickOption = (value) => {
  emit('update:modelValue', value)
  emit('change', value)
  if (value !== 'custom') {
    isOpen.value = false
  }
}

const confirmCustom = () => {
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

.dropdown-trigger:hover {
  background: var(--color-bg-secondary);
  border-color: var(--color-primary);
}

.dropdown-label {
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

.dropdown-panel {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  min-width: 180px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
  z-index: 100;
  overflow: hidden;
}

.dropdown-item {
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

.dropdown-item:hover {
  background: var(--color-bg-tertiary);
}

.dropdown-item.active {
  background: var(--color-primary-light);
  color: var(--color-primary);
}

.check {
  width: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--color-primary);
  font-weight: bold;
}

.dropdown-divider {
  height: 1px;
  background: var(--color-border);
  margin: 4px 0;
}

.custom-panel {
  padding: 8px 12px;
}

.time-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.time-label {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  min-width: 28px;
}

.time-input {
  flex: 1;
  padding: 6px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
  color: var(--color-text-primary);
  background: var(--color-bg-primary);
  outline: none;
  transition: border-color var(--transition-fast);
}

.time-input:focus {
  border-color: var(--color-primary);
}

.btn-small {
  width: 100%;
  padding: 8px 12px;
  margin-top: 4px;
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
