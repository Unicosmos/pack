import client from './client'

export async function listTasks(params = {}) {
  try {
    const res = await client.get('/api/tasks', { params })
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function getTask(taskId) {
  try {
    const res = await client.get(`/api/tasks/${taskId}`)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function uploadTask(file) {
  try {
    const formData = new FormData()
    formData.append('file', file)
    const res = await client.post('/api/tasks/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function batchUploadTasks(formData) {
  try {
    const res = await client.post('/api/tasks/batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function getBatchStatus(taskIds) {
  try {
    const ids = taskIds.join(',')
    const res = await client.get(`/api/tasks/batch/${ids}`)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function batchDeleteTasks(taskIds) {
  try {
    const res = await client.post('/api/tasks/batch-delete', taskIds)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function getTaskStats(timeFilter = null, customStart = null, customEnd = null) {
  try {
    const params = {}
    if (timeFilter && timeFilter !== 'all') {
      params.time_filter = timeFilter
    }
    if (customStart) params.start_time = customStart
    if (customEnd) params.end_time = customEnd
    const res = await client.get('/api/tasks/stats/summary', { params })
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function detectTask(taskId) {
  try {
    const res = await client.post(`/api/tasks/${taskId}/detect`)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function getTaskDetections(taskId) {
  try {
    const res = await client.get(`/api/tasks/${taskId}/detections`)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function reviewTask(taskId, boxes) {
  try {
    const res = await client.put(`/api/tasks/${taskId}/review`, { boxes })
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function matchTask(taskId, matchThreshold = 0.85) {
  try {
    const res = await client.post(`/api/tasks/${taskId}/match?match_threshold=${matchThreshold}`)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function updateTask(taskId, data) {
  try {
    const res = await client.put(`/api/tasks/${taskId}`, data)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function exportTask(taskId, format = 'json', includeImages = false) {
  try {
    const res = await client.get(`/api/tasks/${taskId}/export`, {
      params: { format, include_images: includeImages },
      responseType: 'blob'
    })
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function batchExportTasks(taskIds, format = 'json') {
  try {
    const res = await client.post(`/api/tasks/batch/export?format=${format}`, taskIds, {
      headers: { 'Content-Type': 'application/json' },
      responseType: 'blob'
    })
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export default {
  listTasks,
  getTask,
  uploadTask,
  batchUploadTasks,
  getBatchStatus,
  batchDeleteTasks,
  getTaskStats,
  detectTask,
  getTaskDetections,
  reviewTask,
  matchTask,
  updateTask,
  exportTask,
  batchExportTasks
}
