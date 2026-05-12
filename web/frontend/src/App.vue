<template>
  <div class="app-container">
    <Login v-if="!store.isLoggedIn" @login-success="handleLoginSuccess" />

    <template v-else>
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
        </div>
        <div class="nav-right">
          <span class="user-info">{{ store.user?.username || '用户' }}</span>
          <button class="btn-logout" @click="handleLogout">退出</button>
        </div>
      </div>

      <div class="main-wrapper">
        <HomePage v-if="store.currentPage === 'home'" />
        <TaskListPage v-else-if="store.currentPage === 'tasks'" />
      </div>
    </template>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useAppStore } from './stores/app'
import Login from './components/Login.vue'
import HomePage from './components/HomePage.vue'
import TaskListPage from './components/TaskListPage.vue'

const store = useAppStore()

const handleLoginSuccess = (user) => {
  store.setUser(user)
}

const handleLogout = () => {
  store.logout()
}

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

.nav-right {
  display: flex;
  align-items: center;
  gap: 15px;
}

.user-info {
  color: #666;
  font-size: 14px;
}

.btn-logout {
  padding: 6px 16px;
  background: #f56c6c;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

.btn-logout:hover {
  background: #f78989;
}

.main-wrapper {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}
</style>
