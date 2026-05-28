<template>
  <div class="filter-dropdown" ref="dropdownRef">
    <button class="dropdown-trigger" @click="toggle">
      <span class="dropdown-label">{{ displayLabel }}</span>
      <span class="dropdown-arrow" :class="{ open: isOpen }">▼</span>
    </button>
    <transition name="dropdown">
      <div v-if="isOpen" class="dropdown-panel" @click.stop>
        <div
          v-for="option in options"
          :key="option.value"
          class="dropdown-item"
          :class="{ active: modelValue === option.value }"
          @click="selectOption(option.value)"
        >
          <span class="check">{{ modelValue === option.value ? '✓' : '' }}</span>
          <span>{{ option.label }}</span>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  modelValue: { type: [String, Number, Boolean, null], default: null },
  options: { type: Array, default: () => [] },
  placeholder: { type: String, default: '请选择' }
})

const emit = defineEmits(['update:modelValue', 'change'])

const isOpen = ref(false)
const dropdownRef = ref(null)

const displayLabel = computed(() => {
  const option = props.options.find(o => o.value === props.modelValue)
  if (option) return option.label
  return props.placeholder
})

const toggle = () => {
  isOpen.value = !isOpen.value
}

const selectOption = (value) => {
  emit('update:modelValue', value)
  emit('change', value)
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
.filter-dropdown {
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
  min-width: 100%;
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
