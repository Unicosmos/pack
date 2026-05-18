<template>
  <div class="app-container">
    <div class="nav-bar">
      <div class="nav-left">
        <h1>📦 Pack Web</h1>
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
          🔍 SKU审核
        </button>
      </div>
    </div>

    <div class="main-wrapper">
      <HomePage v-if="store.currentPage === 'home'" />
      <TaskListPage v-else-if="store.currentPage === 'tasks'" />
      <SkuListPage v-else-if="store.currentPage === 'skus'" />
      <SkuReviewPage v-else-if="store.currentPage === 'skuReview'" />
    </div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useAppStore } from './stores/app'
import HomePage from './components/HomePage.vue'
import TaskListPage from './components/TaskListPage.vue'
import SkuListPage from './components/SkuListPage.vue'
import SkuReviewPage from './components/SkuReviewPage.vue'

const store = useAppStore()

onMounted(() => {
  store.fetchSystemHealth()
})
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  background: #f5f5f5;
}

.nav-bar {
  background: white;
  padding: 0 30px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  height: 60px;
}

.nav-left h1 {
  margin: 0;
  font-size: 20px;
  color: #333;
}

.nav-menu {
  display: flex;
  gap: 10px;
}

.nav-menu button {
  padding: 8px 20px;
  background: transparent;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  color: #666;
  transition: all 0.3s;
}

.nav-menu button:hover {
  background: #f0f0f0;
}

.nav-menu button.active {
  background: #667eea;
  color: white;
}

.main-wrapper {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}
</style>
