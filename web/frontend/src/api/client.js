const API_BASE = ''

async function request(url, options = {}) {
  const token = localStorage.getItem('token')
  const headers = {
    ...options.headers,
  }

  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers,
  })

  if (response.status === 401) {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    window.location.href = '/#/login'
    return null
  }

  return response
}

export const auth = {
  async login(username, password) {
    const response = await request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    })
    if (!response) return null
    const data = await response.json()
    if (data.access_token) {
      localStorage.setItem('token', data.access_token)
    }
    return data
  },

  async register(username, password, email) {
    const response = await request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, password, email }),
    })
    return response.json()
  },

  async getMe() {
    const response = await request('/api/auth/me')
    if (!response) return null
    return response.json()
  },

  logout() {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
  },

  isLoggedIn() {
    return !!localStorage.getItem('token')
  }
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

    const response = await fetch('/api/detect-and-match', {
      method: 'POST',
      body: formData,
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
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
    return response.json()
  },

  async list(page = 1, pageSize = 10, status = null) {
    let url = `/api/tasks?page=${page}&page_size=${pageSize}`
    if (status) {
      url += `&status_filter=${status}`
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

  async stats() {
    const response = await request('/api/tasks/stats/summary')
    return response.json()
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
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
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
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
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

// SKU审核 API
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
  }
}
