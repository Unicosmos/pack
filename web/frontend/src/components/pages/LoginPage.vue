<template>
  <div class="login-page">
    <div class="login-card">
      <div class="login-header">
        <div class="login-logo">📦</div>
        <h1 class="login-title">仓储物品识别系统</h1>
        <p class="login-subtitle">欢迎回来，请登录以继续</p>
      </div>

      <form @submit.prevent="handleLogin" class="login-form">
        <div class="form-group">
          <label class="form-label">用户名</label>
          <input
            v-model="username"
            type="text"
            class="form-input"
            placeholder="请输入用户名"
            autocomplete="username"
            @keyup.enter="passwordInput.focus()"
          />
        </div>

        <div class="form-group">
          <label class="form-label">密码</label>
          <input
            ref="passwordInput"
            v-model="password"
            type="password"
            class="form-input"
            placeholder="请输入密码"
            autocomplete="current-password"
            @keyup.enter="handleLogin"
          />
        </div>

        <div v-if="errorMessage" class="error-message">{{ errorMessage }}</div>

        <button type="submit" class="login-button" :disabled="loading">
          {{ loading ? '登录中...' : '登录' }}
        </button>
      </form>

      <div class="login-footer">
        <p class="hint-text">提示：账号 admin / 密码 123456</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useAppStore } from '@stores/app'

const store = useAppStore()

const username = ref('admin')
const password = ref('123456')
const loading = ref(false)
const errorMessage = ref('')
const passwordInput = ref(null)

const handleLogin = async () => {
  if (!username.value.trim() || !password.value.trim()) {
    errorMessage.value = '请输入用户名和密码'
    return
  }

  loading.value = true
  errorMessage.value = ''

  await new Promise(resolve => setTimeout(resolve, 300))

  const result = store.login(username.value, password.value)
  if (!result.success) {
    errorMessage.value = result.message
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-bg-primary);
  padding: 20px;
}

.login-card {
  width: 100%;
  max-width: 400px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 12px;
  padding: 40px 32px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
}

.login-header {
  text-align: center;
  margin-bottom: 32px;
}

.login-logo {
  font-size: 48px;
  margin-bottom: 16px;
}

.login-title {
  font-size: 24px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 8px;
}

.login-subtitle {
  font-size: 14px;
  color: var(--color-text-secondary);
}

.login-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-label {
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-primary);
}

.form-input {
  width: 100%;
  padding: 12px 16px;
  font-size: 14px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  color: var(--color-text-primary);
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.form-input:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.1);
}

.form-input::placeholder {
  color: var(--color-text-tertiary);
}

.error-message {
  font-size: 13px;
  color: var(--color-danger);
  text-align: center;
  padding: 10px;
  background: rgba(239, 68, 68, 0.1);
  border-radius: 6px;
}

.login-button {
  width: 100%;
  padding: 12px;
  font-size: 15px;
  font-weight: 500;
  background: var(--color-primary);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s, opacity 0.2s;
}

.login-button:hover:not(:disabled) {
  background: #3b82f6;
}

.login-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.login-footer {
  margin-top: 24px;
  text-align: center;
}

.hint-text {
  font-size: 13px;
  color: var(--color-text-tertiary);
}
</style>
