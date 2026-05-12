<template>
  <div class="login-container">
    <div class="login-box">
      <h1>📦 Pack Web</h1>
      <p class="subtitle">箱货检测与SKU匹配系统</p>

      <div class="form-group">
        <label>用户名</label>
        <input
          type="text"
          v-model="username"
          placeholder="请输入用户名"
          @keyup.enter="handleLogin"
        />
      </div>

      <div class="form-group">
        <label>密码</label>
        <input
          type="password"
          v-model="password"
          placeholder="请输入密码"
          @keyup.enter="handleLogin"
        />
      </div>

      <div v-if="error" class="error-message">{{ error }}</div>
      <div v-if="success" class="success-message">{{ success }}</div>

      <button class="btn-login" @click="handleLogin" :disabled="loading">
        {{ loading ? '登录中...' : '登 录' }}
      </button>

      <div class="login-hint">
        默认账号: admin / admin123
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { auth } from '../api/client'

const emit = defineEmits(['login-success'])

const username = ref('')
const password = ref('')
const error = ref('')
const success = ref('')
const loading = ref(false)

// 页面加载时自动填充默认账号
onMounted(() => {
  username.value = 'admin'
  password.value = 'admin123'
  console.log('Login: 默认账号已填充')
})

const handleLogin = async () => {
  if (!username.value || !password.value) {
    error.value = '请输入用户名和密码'
    console.log('Login: 用户名或密码为空')
    return
  }

  loading.value = true
  error.value = ''
  success.value = ''

  console.log(`Login: 尝试登录，用户名: ${username.value}`)

  try {
    const result = await auth.login(username.value, password.value)
    console.log('Login: 登录API返回:', result)
    
    if (result && result.access_token) {
      console.log('Login: 获取access_token成功')
      
      const userInfo = await auth.getMe()
      console.log('Login: 获取用户信息:', userInfo)
      
      if (userInfo) {
        localStorage.setItem('user', JSON.stringify(userInfo))
        success.value = '登录成功！正在跳转...'
        console.log('Login: 登录成功，准备跳转')
        setTimeout(() => {
          window.location.reload()
        }, 500)
      } else {
        error.value = '获取用户信息失败'
        console.log('Login: 获取用户信息失败')
      }
    } else {
      error.value = '登录失败，请检查用户名和密码'
      console.log('Login: 登录失败，无access_token')
    }
  } catch (e) {
    error.value = '登录失败: ' + (e.message || '未知错误')
    console.error('Login: 登录异常:', e)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-container {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.login-box {
  background: white;
  padding: 40px;
  border-radius: 12px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
  width: 100%;
  max-width: 400px;
}

.login-box h1 {
  text-align: center;
  margin: 0 0 10px 0;
  color: #333;
}

.subtitle {
  text-align: center;
  color: #666;
  margin-bottom: 30px;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  color: #333;
  font-weight: 500;
}

.form-group input {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  box-sizing: border-box;
}

.form-group input:focus {
  outline: none;
  border-color: #667eea;
}

.error-message {
  color: #f56c6c;
  font-size: 14px;
  margin-bottom: 15px;
  text-align: center;
}

.success-message {
  color: #67c23a;
  font-size: 14px;
  margin-bottom: 15px;
  text-align: center;
}

.btn-login {
  width: 100%;
  padding: 14px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  cursor: pointer;
  transition: opacity 0.3s;
}

.btn-login:hover:not(:disabled) {
  opacity: 0.9;
}

.btn-login:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.login-hint {
  margin-top: 20px;
  text-align: center;
  color: #999;
  font-size: 12px;
}
</style>
