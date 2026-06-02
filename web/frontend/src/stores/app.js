import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { detector } from '@api/client'

export const useAppStore = defineStore('app', () => {
  const states = ['IDLE', 'UPLOADED', 'PROCESSING', 'SUCCESS', 'ERROR', 'SYSTEM_INIT']

  const currentState = ref('IDLE')
  const selectedFile = ref(null)
  const selectedFiles = ref([])
  const previewUrl = ref('')
  const result = ref(null)
  const error = ref(null)
  const skuCount = ref(0)
  const systemStatus = ref('ready')
  const modelInfo = ref('')
  const skuModelInfo = ref('')
  const detectorReady = ref(false)
  const matcherReady = ref(false)
  const savedPage = localStorage.getItem('currentPage')
  const currentPage = ref(savedPage || 'home')
  const refreshTrigger = ref(0)
  const pendingTaskId = ref(null)

  const batchTaskIds = ref([])
  const batchTasks = ref([])
  const batchResults = ref([])
  const currentBatchIndex = ref(0)
  const currentMode = ref('review')

  // 登录状态
  const isLoggedIn = ref(sessionStorage.getItem('isLoggedIn') === 'true')

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

  function setPage(page) {
    currentPage.value = page
    localStorage.setItem('currentPage', page)
    refreshTrigger.value++
  }

  function uploadImage(file) {
    selectedFile.value = file
    previewUrl.value = URL.createObjectURL(file)
    error.value = null
    result.value = null
    setStatus('UPLOADED')
  }

  function addFiles(files) {
    selectedFiles.value = [...selectedFiles.value, ...files]
  }

  function removeFileAt(index) {
    selectedFiles.value.splice(index, 1)
  }

  function clearFiles() {
    selectedFiles.value.forEach(file => {
      if (file.preview) {
        URL.revokeObjectURL(file.preview)
      }
    })
    selectedFiles.value = []
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
    batchTaskIds.value = []
    batchTasks.value = []
    setStatus('IDLE')
  }

  function removeFile() {
    selectedFile.value = null
    previewUrl.value = ''
    if (isSuccess.value || hasError.value) {
      setStatus('IDLE')
    }
  }

  function setBatchTaskIds(ids) {
    batchTaskIds.value = ids
  }

  function setBatchTasks(tasks) {
    batchTasks.value = tasks
  }

  const getModelName = (modelPath) => {
    if (!modelPath) return ''
    const parts = modelPath.replace(/\\/g, '/').split('/')
    return parts[parts.length - 1] || modelPath
  }

  async function fetchSystemHealth() {
    try {
      const res = await detector.health()
      skuCount.value = res.sku_count || 0
      modelInfo.value = getModelName(res.model_path) || ''
      skuModelInfo.value = getModelName(res.sku_model_path) || ''
      detectorReady.value = !!res.detector_ready
      matcherReady.value = !!res.matcher_ready

      if (res.status === 'init') {
        systemStatus.value = 'init'
        setStatus('SYSTEM_INIT')
      } else if (res.status === 'error') {
        systemStatus.value = 'error'
        setStatus('SYSTEM_INIT')
      } else if (!res.detector_ready) {
        systemStatus.value = 'error'
        setStatus('SYSTEM_INIT')
      } else if (!res.matcher_ready) {
        systemStatus.value = 'no-sku'
        setStatus('IDLE')
      } else {
        systemStatus.value = 'ready'
        setStatus('IDLE')
      }
    } catch (err) {
      console.error('Health check error:', err)
      systemStatus.value = 'error'
      setStatus('SYSTEM_INIT')
    }
  }

  // 登录/登出功能
  function login(username, password) {
    if (username === 'admin' && password === '123456') {
      isLoggedIn.value = true
      sessionStorage.setItem('isLoggedIn', 'true')
      return { success: true }
    }
    return { success: false, message: '账号或密码错误' }
  }

  function logout() {
    isLoggedIn.value = false
    sessionStorage.removeItem('isLoggedIn')
  }

  return {
    currentState,
    selectedFile,
    selectedFiles,
    previewUrl,
    result,
    error,
    skuCount,
    systemStatus,
    modelInfo,
    skuModelInfo,
    detectorReady,
    matcherReady,
    currentPage,
    pendingTaskId,
    batchTaskIds,
    batchTasks,
    batchResults,
    currentBatchIndex,
    currentMode,
    isLoggedIn,
    isIdle,
    isUploaded,
    isProcessing,
    isSuccess,
    hasError,
    isSystemInit,
    setStatus,
    setPage,
    uploadImage,
    addFiles,
    removeFileAt,
    clearFiles,
    startProcessing,
    completeSuccess,
    completeError,
    reset,
    removeFile,
    setBatchTaskIds,
    setBatchTasks,
    fetchSystemHealth,
    login,
    logout
  }
})
