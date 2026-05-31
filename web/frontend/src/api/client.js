import axios from 'axios'

const API_BASE = ''

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

client.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

client.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    let errorMessage = '请求失败'
    if (error.response) {
      const data = error.response.data
      if (data && data.detail) {
        errorMessage = data.detail
      } else {
        errorMessage = `服务器错误: ${error.response.status}`
      }
    } else if (error.request) {
      errorMessage = '网络连接失败，请检查网络'
    } else {
      errorMessage = error.message || '请求配置错误'
    }
    error.message = errorMessage
    return Promise.reject(error)
  }
)

function getImageUrl(type, id, filename) {
  const encodedFilename = encodeURIComponent(filename)
  switch (type) {
    case 'sku_library':
      return `/static/sku_images/${id}/${encodedFilename}`
    case 'sku_review_crop':
      return `/api/sku-review/crops-image/${id}/${encodedFilename}`
    case 'sku_review_output':
      return `/api/sku-review/sku-image/${id}/${encodedFilename}`
    default:
      return ''
  }
}

function getImageUrlFromPath(path) {
  if (!path) return ''
  if (typeof path === 'object') {
    if (path.url) {
      return path.url
    }
    if (path.path) {
      return getImageUrlFromPath(path.path)
    }
    return ''
  }
  const pathStr = String(path)
  if (pathStr.startsWith('/static/') || pathStr.startsWith('/api/')) {
    return pathStr
  }
  return `/api/sku-image?path=${encodeURIComponent(pathStr)}`
}

async function request(url, options = {}) {
  const headers = {
    ...options.headers,
  }

  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers,
  })

  return response
}

export const detector = {
  async health() {
    const response = await request('/api/health')
    return response.json()
  },

  async detectAndMatch(file, confThreshold = 0.5, matchThreshold = 0.85) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('conf_threshold', confThreshold)
    formData.append('match_threshold', matchThreshold)

    const response = await request('/api/detect-and-match', {
      method: 'POST',
      body: formData,
    })
    return response.json()
  },

  async detectOnly(file, confThreshold = 0.5) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('conf_threshold', confThreshold)

    const response = await request('/api/detect', {
      method: 'POST',
      body: formData,
    })
    return response.json()
  }
}

export const tasks = {
  async upload(file) {
    const formData = new FormData()
    formData.append('file', file)

    const response = await request('/api/tasks/upload', {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      let errorMsg = '上传失败'
      try {
        const errorData = await response.json()
        errorMsg = errorData.detail || errorMsg
      } catch (e) {
        errorMsg = response.statusText || errorMsg
      }
      throw new Error(errorMsg)
    }

    return response.json()
  },

  async batchUpload(formData) {
    const response = await request('/api/tasks/batch', {
      method: 'POST',
      body: formData,
    })
    return response.json()
  },

  async getBatchStatus(taskIds) {
    const ids = taskIds.join(',')
    const response = await request(`/api/tasks/batch/${ids}`)
    return response.json()
  },

  async list(page = 1, pageSize = 10, status = null, timeFilter = null, customStart = null, customEnd = null) {
    let url = `/api/tasks?page=${page}&page_size=${pageSize}`
    if (status) {
      url += `&status_filter=${status}`
    }
    if (timeFilter && timeFilter !== 'all') {
      url += `&time_filter=${timeFilter}`
    }
    if (customStart) {
      url += `&start_time=${encodeURIComponent(customStart)}`
    }
    if (customEnd) {
      url += `&end_time=${encodeURIComponent(customEnd)}`
    }
    const response = await request(url)
    return response.json()
  },

  async get(id) {
    const response = await request(`/api/tasks/${id}`)
    return response.json()
  },

  async delete(id) {
    const response = await request(`/api/tasks/${id}`, {
      method: 'DELETE',
    })
    return response.json()
  },

  async stats(timeFilter = null, customStart = null, customEnd = null) {
    let url = '/api/tasks/stats/summary'
    const params = []
    if (timeFilter && timeFilter !== 'all') {
      params.push(`time_filter=${timeFilter}`)
    }
    if (customStart) {
      params.push(`start_time=${encodeURIComponent(customStart)}`)
    }
    if (customEnd) {
      params.push(`end_time=${encodeURIComponent(customEnd)}`)
    }
    if (params.length > 0) {
      url += '?' + params.join('&')
    }
    const response = await request(url)
    return response.json()
  },

  async detect(taskId) {
    const response = await request(`/api/tasks/${taskId}/detect`, {
      method: 'POST',
    })
    return response.json()
  },

  async getDetections(taskId) {
    const response = await request(`/api/tasks/${taskId}/detections`)
    return response.json()
  },

  async reviewTask(taskId, boxes) {
    const response = await request(`/api/tasks/${taskId}/review`, {
      method: 'PUT',
      body: JSON.stringify({ boxes }),
    })
    return response.json()
  },

  async matchTask(taskId, matchThreshold = 0.85) {
    const response = await request(`/api/tasks/${taskId}/match?match_threshold=${matchThreshold}`, {
      method: 'POST',
    })
    return response.json()
  },

  async update(taskId, data) {
    const response = await request(`/api/tasks/${taskId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    })
    return response.json()
  },

  async exportTask(taskId, format = 'json', includeImages = false) {
    const url = `/api/tasks/${taskId}/export?format=${format}&include_images=${includeImages}`
    const response = await fetch(url)

    if (!response.ok) {
      let errorMsg = '导出失败'
      try {
        const errorData = await response.json()
        errorMsg = errorData.detail || errorMsg
      } catch (e) {
        errorMsg = response.statusText || errorMsg
      }
      throw new Error(errorMsg)
    }

    return response
  }
}

export const sku = {
  async list(page = 1, pageSize = 20, search = '', category = '', status = '') {
    let url = `/api/skus?page=${page}&page_size=${pageSize}`
    if (search) url += `&search=${encodeURIComponent(search)}`
    if (category) url += `&category=${encodeURIComponent(category)}`
    if (status) url += `&status=${encodeURIComponent(status)}`
    const response = await request(url)
    return response.json()
  },

  async getDetail(skuId) {
    const response = await request(`/api/skus/${skuId}`)
    return response.json()
  },

  async stats() {
    const response = await request('/api/skus/stats')
    return response.json()
  },

  async create(data) {
    const response = await request('/api/skus', {
      method: 'POST',
      body: JSON.stringify(data),
    })
    return response.json()
  },

  async update(skuId, data) {
    const response = await request(`/api/skus/${skuId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    })
    return response.json()
  },

  async delete(skuId) {
    const response = await request(`/api/skus/${skuId}`, {
      method: 'DELETE',
    })
    return response.json()
  },

  async batchDelete(skuIds) {
    const response = await request('/api/skus/batch-delete', {
      method: 'POST',
      body: JSON.stringify(skuIds),
    })
    return response.json()
  },

  async getCategories() {
    const response = await request('/api/skus/categories')
    return response.json()
  },

  async importCsv(file) {
    const formData = new FormData()
    formData.append('file', file)
    const response = await fetch('/api/skus/import', {
      method: 'POST',
      body: formData,
    })
    return response.json()
  },

  async exportCsv() {
    const response = await request('/api/skus/export/download')
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'sku_export.csv'
    a.click()
  },

  async syncFromCsv() {
    const response = await request('/api/skus/sync-from-csv', {
      method: 'POST',
    })
    return response.json()
  },

  async getImages(skuId) {
    const response = await request(`/api/skus/${skuId}/images`)
    return response.json()
  },

  async uploadImages(skuId, files) {
    const formData = new FormData()
    files.forEach(file => {
      formData.append('files', file)
    })
    const response = await fetch(`/api/skus/${skuId}/images/upload`, {
      method: 'POST',
      body: formData,
    })
    return response.json()
  },

  async deleteImage(skuId, filename) {
    const response = await request(`/api/skus/${skuId}/images/${filename}`, {
      method: 'DELETE',
    })
    return response.json()
  },

  async listImages(skuId) {
    const response = await request(`/api/skus/${skuId}/list-images`)
    return response.json()
  }
}

export const skuReview = {
  async getFolders() {
    const response = await request('/api/sku-review/folders')
    return response.json()
  },

  async getFolderImages(folderName) {
    const response = await request(`/api/sku-review/folder-images/${encodeURIComponent(folderName)}`)
    return response.json()
  },

  async getSkus(keyword = '') {
    const params = new URLSearchParams()
    if (keyword) params.set('keyword', keyword)
    const response = await request('/api/sku-review/skus?' + params.toString())
    return response.json()
  },

  async getSkuById(skuId) {
    const response = await request(`/api/sku/${encodeURIComponent(skuId)}`)
    return response.json()
  },

  async getSkuImages(skuId) {
    const response = await request(`/api/sku-review/sku-images/${encodeURIComponent(skuId)}`)
    return response.json()
  },

  async assignImages(skuId, imagePaths) {
    const response = await request('/api/sku-review/assign-images', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sku_id: skuId, image_paths: imagePaths })
    })
    return response.json()
  },

  async recallImages(skuId, imagePaths) {
    const response = await request('/api/sku-review/recall-images', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sku_id: skuId, image_paths: imagePaths })
    })
    return response.json()
  },

  async createSku(name) {
    const params = new URLSearchParams()
    if (name) params.set('name', name)
    const response = await request('/api/sku-review/create-sku?' + params.toString(), {
      method: 'POST'
    })
    return response.json()
  },

  async renameSku(oldId, newName) {
    const params = new URLSearchParams()
    params.set('old_id', oldId)
    params.set('new_name', newName)
    const response = await request('/api/sku-review/rename-sku?' + params.toString(), {
      method: 'PUT'
    })
    return response.json()
  },

  async deleteSku(skuId) {
    const response = await request(`/api/sku-review/delete-sku/${encodeURIComponent(skuId)}`, {
      method: 'DELETE'
    })
    return response.json()
  },

  async saveDatabase() {
    const response = await request('/api/sku-review/save-database', {
      method: 'POST'
    })
    return response.json()
  },

  async uploadFolder(folderName, files) {
    const formData = new FormData()
    files.forEach(file => {
      formData.append('files', file)
    })
    const response = await request(`/api/sku-review/upload-folder/${encodeURIComponent(folderName)}`, {
      method: 'POST',
      body: formData
    })
    return response.json()
  },

  async deleteFolder(folderName) {
    const response = await request(`/api/sku-review/delete-folder/${encodeURIComponent(folderName)}`, {
      method: 'DELETE'
    })
    return response.json()
  }
}

export const build = {
  async getStatus() {
    const response = await request('/api/build/status')
    return response.json()
  },

  async triggerBuild() {
    const response = await request('/api/build/library', {
      method: 'POST'
    })
    return response.json()
  },

  async getFeatureStatus() {
    const response = await request('/api/build/feature/status')
    return response.json()
  },

  async triggerFeatureExtract() {
    const response = await request('/api/build/feature/extract', {
      method: 'POST'
    })
    return response.json()
  },

  async getCombinedStatus() {
    const response = await request('/api/build/combined/status')
    return response.json()
  },

  async triggerCombinedBuild() {
    const response = await request('/api/build/combined/run', {
      method: 'POST'
    })
    return response.json()
  },

  async checkChange() {
    const response = await request('/api/build/check-change')
    return response.json()
  },

  async getInfo() {
    const response = await request('/api/build/info')
    return response.json()
  }
}

export const logs = {
  async list(params = {}) {
    const searchParams = new URLSearchParams()
    if (params.entity_type) searchParams.set('entity_type', params.entity_type)
    if (params.entity_id) searchParams.set('entity_id', params.entity_id)
    if (params.action) searchParams.set('action', params.action)
    if (params.page) searchParams.set('page', params.page)
    if (params.page_size) searchParams.set('page_size', params.page_size)

    const response = await request('/api/logs?' + searchParams.toString())
    return response.json()
  },

  async get(id) {
    const response = await request(`/api/logs/${id}`)
    return response.json()
  }
}

export { getImageUrl, getImageUrlFromPath }
export default client
