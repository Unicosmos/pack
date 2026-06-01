<template>
  <div class="app-container">
    <nav class="nav-bar">
      <div class="nav-left">
        <h1>📦 Pack Web</h1>
        <span class="nav-title">{{ pageTitle }}</span>
      </div>
      <div class="nav-menu">
        <button :class="{ active: store.currentPage === 'home' }" @click="store.setPage('home')">
          🏠 首页
        </button>
        <button :class="{ active: store.currentPage === 'tasks' }" @click="store.setPage('tasks')">
          📋 任务列表
        </button>
        <button :class="{ active: store.currentPage === 'skus' }" @click="store.setPage('skus')">
          📦 SKU管理
        </button>
        <button :class="{ active: store.currentPage === 'skuReview' }" @click="store.setPage('skuReview')">
          🔍 SKU入库审核
        </button>
        <button class="theme-toggle" @click="toggleDarkMode" :title="isDark ? '切换到浅色模式' : '切换到深色模式'">
          {{ isDark ? '☀️' : '🌙' }}
        </button>
      </div>
    </nav>

    <main class="main-wrapper" :class="{ 'fullscreen': store.currentPage === 'tasks' || store.currentPage === 'skus' }">
      <HomePage v-if="store.currentPage === 'home'" />
      <TaskListPage v-else-if="store.currentPage === 'tasks'" />
      <SkuListPage v-else-if="store.currentPage === 'skus'" />
      <SkuReviewPage v-else-if="store.currentPage === 'skuReview'" />
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useAppStore } from '@stores/app'
import HomePage from '@pages/HomePage.vue'
import TaskListPage from '@pages/TaskListPage.vue'
import SkuListPage from '@pages/SkuListPage.vue'
import SkuReviewPage from '@pages/SkuReviewPage.vue'

const store = useAppStore()

const pageTitle = computed(() => {
  const titles = {
    home: '箱货检测与SKU匹配',
    tasks: '任务列表',
    skus: 'SKU管理',
    skuReview: 'SKU入库审核'
  }
  return titles[store.currentPage] || ''
})

// 使用 ref 存储深色模式状态
const isDark = ref(false)

// 从 localStorage 读取保存的主题设置
const loadTheme = () => {
  const saved = localStorage.getItem('darkMode')
  if (saved !== null) {
    isDark.value = saved === 'true'
  } else {
    // 默认检查系统偏好
    isDark.value = window.matchMedia('(prefers-color-scheme: dark)').matches
  }
  updateHtmlClass()
}

// 更新 html 元素的 class
const updateHtmlClass = () => {
  const html = document.documentElement
  if (isDark.value) {
    html.classList.add('dark')
  } else {
    html.classList.remove('dark')
  }
}

// 保存主题设置到 localStorage
const saveTheme = () => {
  localStorage.setItem('darkMode', String(isDark.value))
}

// 切换深色模式
const toggleDarkMode = () => {
  isDark.value = !isDark.value
  updateHtmlClass()
  saveTheme()
}

// 监听状态变化
watch(isDark, () => {
  updateHtmlClass()
  saveTheme()
})

onMounted(() => {
  store.fetchSystemHealth()
  loadTheme()
})
</script>

<style>
/* 全局主题切换相关样式 */
.app-container {
  min-height: 100vh;
  background: var(--color-bg-secondary);
  transition: background-color var(--transition-normal);
}

.theme-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  font-size: var(--font-size-lg);
  background: var(--color-bg-tertiary);
  transition: all var(--transition-fast);
}

.theme-toggle:hover {
  transform: scale(1.1);
  background: var(--color-primary);
}
</style>
