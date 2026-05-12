import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { detector } from '../api/client'

export const useAppStore = defineStore('app', () => {
  const states = ['IDLE', 'UPLOADED', 'PROCESSING', 'SUCCESS', 'ERROR', 'SYSTEM_INIT']

  const currentState = ref('IDLE')
  const selectedFile = ref(null)
  const previewUrl = ref('')
  const result = ref(null)
  const error = ref(null)
  const skuCount = ref(0)
  const systemStatus = ref('ready')
  const currentPage = ref('home')

  const user = ref(JSON.parse(localStorage.getItem('user') || 'null'))
  const isLoggedIn = computed(() => !!localStorage.getItem('token'))

  const isIdle = computed(() => currentState.value === 'IDLE')
  const isUploaded = computed(() => currentState.value === 'UPLOADED')
  const isProcessing = computed(() => currentState.value === 'PROCESSING')
  const isSuccess = computed(() => currentState.value === 'SUCCESS')
  const hasError = computed(() => currentState.value === 'ERROR')
  const isSystemInit = computed(() => currentState.value === 'SYSTEM_INIT')

  function setStatus(status) {
    if (states.includes(status)) {
      currentState.value = status
    }
  }

  function setUser(userData) {
    user.value = userData
  }

  function setPage(page) {
    currentPage.value = page
  }

  function logout() {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    user.value = null
    currentPage.value = 'home'
  }

  function uploadImage(file) {
    selectedFile.value = file
    previewUrl.value = URL.createObjectURL(file)
    error.value = null
    result.value = null
    setStatus('UPLOADED')
  }

  function startProcessing() {
    setStatus('PROCESSING')
    error.value = null
  }

  function completeSuccess(data) {
    result.value = data
    setStatus('SUCCESS')
  }

  function completeError(err) {
    error.value = err
    setStatus('ERROR')
  }

  function reset() {
    selectedFile.value = null
    previewUrl.value = ''
    result.value = null
    error.value = null
    setStatus('IDLE')
  }

  function removeFile() {
    selectedFile.value = null
    previewUrl.value = ''
    if (isSuccess.value || hasError.value) {
      setStatus('IDLE')
    }
  }

  async function fetchSystemHealth() {
    try {
      const res = await detector.health()
      console.log('Health check response:', res)
      systemStatus.value = res.status
      skuCount.value = res.sku_count || 0

      if (res.status === 'init') {
        setStatus('SYSTEM_INIT')
      } else if (!res.matcher_ready) {
        systemStatus.value = 'no-sku'
      } else {
        systemStatus.value = 'ready'
      }
    } catch (err) {
      console.error('Health check error:', err)
      systemStatus.value = 'error'
      setStatus('SYSTEM_INIT')
    }
  }

  return {
    currentState,
    selectedFile,
    previewUrl,
    result,
    error,
    skuCount,
    systemStatus,
    currentPage,
    user,
    isLoggedIn,
    isIdle,
    isUploaded,
    isProcessing,
    isSuccess,
    hasError,
    isSystemInit,
    setStatus,
    setUser,
    setPage,
    logout,
    uploadImage,
    startProcessing,
    completeSuccess,
    completeError,
    reset,
    removeFile,
    fetchSystemHealth
  }
})
