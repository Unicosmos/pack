import client from './client'

export async function listSkus(params = {}) {
  try {
    const res = await client.get('/api/skus', { params })
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function getSku(skuId) {
  try {
    const res = await client.get(`/api/skus/${skuId}`)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function createSku(data) {
  try {
    const res = await client.post('/api/skus', data)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function updateSku(skuId, data) {
  try {
    const res = await client.put(`/api/skus/${skuId}`, data)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function deleteSku(skuId) {
  try {
    const res = await client.delete(`/api/skus/${skuId}`)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function batchDeleteSkus(skuIds) {
  try {
    const res = await client.post('/api/skus/batch-delete', skuIds)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function getSkuCategories() {
  try {
    const res = await client.get('/api/skus/categories')
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function getSkuStats() {
  try {
    const res = await client.get('/api/skus/stats')
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function importSkuCsv(file) {
  try {
    const formData = new FormData()
    formData.append('file', file)
    const res = await client.post('/api/skus/import', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function exportSkuCsv() {
  try {
    const res = await client.get('/api/skus/export/download', {
      responseType: 'blob'
    })
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function syncSkuFromCsv() {
  try {
    const res = await client.post('/api/skus/sync-from-csv')
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function getSkuImages(skuId) {
  try {
    const res = await client.get(`/api/skus/${skuId}/images`)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function uploadSkuImages(skuId, files) {
  try {
    const formData = new FormData()
    files.forEach(file => {
      formData.append('files', file)
    })
    const res = await client.post(`/api/skus/${skuId}/images/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function deleteSkuImage(skuId, filename) {
  try {
    const res = await client.delete(`/api/skus/${skuId}/images/${filename}`)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function listSkuImages(skuId) {
  try {
    const res = await client.get(`/api/skus/${skuId}/list-images`)
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function triggerBuild() {
  try {
    const res = await client.post('/api/build/library')
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export async function getBuildStatus() {
  try {
    const res = await client.get('/api/build/status')
    return { success: true, data: res.data }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

export default {
  listSkus,
  getSku,
  createSku,
  updateSku,
  deleteSku,
  batchDeleteSkus,
  getSkuCategories,
  getSkuStats,
  importSkuCsv,
  exportSkuCsv,
  syncSkuFromCsv,
  getSkuImages,
  uploadSkuImages,
  deleteSkuImage,
  listSkuImages,
  triggerBuild,
  getBuildStatus
}
