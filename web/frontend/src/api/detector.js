import axios from 'axios'

const API_BASE = '/api'

export async function detectAndMatch(file, confThreshold = 0.5, matchThreshold = 0.85) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('conf_threshold', confThreshold)
  formData.append('match_threshold', matchThreshold)

  const response = await axios.post(`${API_BASE}/detect-and-match`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return response.data
}

export async function detectOnly(file, confThreshold = 0.5) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('conf_threshold', confThreshold)

  const response = await axios.post(`${API_BASE}/detect`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return response.data
}

export async function getSkuList() {
  const response = await axios.get(`${API_BASE}/skus`)
  return response.data
}

export async function getSystemHealth() {
  const response = await axios.get(`${API_BASE}/health`)
  return response.data
}