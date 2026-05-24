export const getStatusBadgeClass = (task) => {
  if (task.status === 'failed') return 'failed'
  if (task.status === 'completed') return 'completed'
  if (task.status === 'detected') return 'warning'
  return 'pending'
}

export const getStatusText = (status) => {
  const map = {
    'pending': '待识别',
    'detected': '待审核',
    'completed': '已完成',
    'failed': '失败'
  }
  return map[status] || status
}

export const shouldShowDetect = (task) => {
  return task.status === 'pending'
}

export const shouldShowReview = (task) => {
  return task.status === 'detected' || task.status === 'completed'
}

export const shouldShowReDetect = (task) => {
  return task.status === 'failed'
}

export const getDetectionStatusText = (status) => {
  const map = {
    'pending': '待检测',
    'detected': '已完成',
    'error': '检测失败'
  }
  return map[status] || (status || '未知')
}

export const getReviewStatusText = (status) => {
  const map = {
    'pending': '待审核',
    'reviewed': '已审核',
    'matched': '已匹配'
  }
  return map[status] || (status || '未知')
}

export const formatDate = (dateStr) => {
  if (!dateStr) return '-'
  const d = new Date(dateStr)
  return d.toLocaleString('zh-CN')
}