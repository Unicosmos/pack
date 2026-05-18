import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { detector } from '../api/client'

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
  const currentPage = ref('home')
  const refreshTrigger = ref(0)

  const batchTaskIds = ref([])
  const batchTasks = ref([])
  const batchResults = ref([])
  const currentBatchIndex = ref(0)
  const currentMode = ref('review')

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

  async function fetchSystemHealth() {
    try {
      const res = await detector.health()
      console.log('Health check response:', res)
      skuCount.value = res.sku_count || 0

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

  return {
    currentState,
    selectedFile,
    selectedFiles,
    previewUrl,
    result,
    error,
    skuCount,
    systemStatus,
    currentPage,
    batchTaskIds,
    batchTasks,
    batchResults,
    currentBatchIndex,
    currentMode,
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
    fetchSystemHealth
  }
})
