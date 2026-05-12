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
  async list(page = 1, pageSize = 20, search = '') {
    let url = `/api/skus?page=${page}&page_size=${pageSize}`
    if (search) {
      url += `&search=${search}`
    }
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

  async export() {
    const response = await request('/api/skus/export')
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'sku_library.csv'
    a.click()
  }
}
