<template>
  <div class="app-container">
    <nav class="nav-bar">
      <div class="nav-left">
        <div class="nav-logo">📦</div>
        <div class="nav-title">Pack Web</div>
        <div class="nav-sub">{{ pageTitle }}</div>
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

    <main class="main-wrapper" :class="{ 'fullscreen': ['tasks', 'skus', 'skuReview'].includes(store.currentPage) }">
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
    skus: 'SKU匹配库管理',
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
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
  background: transparent;
  color: var(--color-text-secondary);
  transition: all .2s;
}

.theme-toggle:hover {
  background: rgba(255, 255, 255, 0.05);
  color: var(--color-text-primary);
}
</style>
