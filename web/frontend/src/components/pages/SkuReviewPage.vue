<template>
  <div class="sku-review-page">
    <!-- 三栏主内容区 -->
    <div class="content">
      <!-- 左侧：文件夹面板 -->
      <div class="folder-panel" :class="{ collapsed: folderCollapsed }">
        <div class="folder-header">
          <div class="folder-title">
            <span>📁</span>
            <span v-if="!folderCollapsed">文件夹</span>
          </div>
          <button class="folder-toggle" @click="toggleFolderPanel">◀</button>
        </div>
        <div class="folder-actions">
          <input
            ref="folderFileInput"
            type="file"
            multiple
            accept="image/*"
            @change="handleFolderFileSelect"
            style="display: none"
          />
          <input
            ref="newFolderInput"
            type="file"
            multiple
            accept="image/*"
            @change="handleNewFolderFileSelect"
            style="display: none"
          />
          <button class="folder-btn upload" @click="handleUploadClick">
            <span>上传图片</span>
          </button>
          <button class="folder-btn new-folder" @click="handleNewFolderClick">
            <span>新建文件夹</span>
          </button>
          <button
            v-if="currentFolder"
            class="folder-btn delete"
            @click="confirmDeleteFolder"
          >
            <span>删除文件夹</span>
          </button>
        </div>
        <div class="folder-list">
          <div
            v-for="(folder, idx) in folders"
            :key="folder"
            :class="['folder-item', { active: currentFolderIndex === idx }]"
            @click="selectFolder(idx)"
          >
            <span class="folder-icon">📁</span>
            <span v-if="!folderCollapsed" class="folder-name">{{ folder }}</span>
            <span v-if="!folderCollapsed" class="folder-count">{{ folderImageCounts[folder] || 0 }}</span>
          </div>
        </div>
      </div>

      <!-- 中间：图片面板 -->
      <div class="image-panel">
        <div class="image-header">
          <div class="image-title">
            <span>📋</span>
            <span>待审核图片 - {{ currentFolder || '未选择' }}</span>
            <span class="image-count">({{ currentImages.length }}张)</span>
          </div>
        </div>
        <div class="image-grid">
          <div v-if="currentImages.length === 0" class="empty-state" style="grid-column: 1 / -1;">
            <div class="empty-icon">📂</div>
            <div>该文件夹暂无图片</div>
          </div>
          <div
            v-for="(img, idx) in currentImages"
            :key="idx"
            :class="['image-item', { selected: selectedCrops.includes(idx) }]"
            @click="toggleCropSelect(idx)"
          >
            <input type="checkbox" class="img-check" :checked="selectedCrops.includes(idx)" @click.stop>
            <img :src="encodeURI(img.url)" :alt="img.name" loading="lazy" @error="(e) => e.target.style.display = 'none'" />
            <div class="img-name">{{ img.name }}</div>
          </div>
        </div>
        <div class="image-footer">
          <div class="select-info">
            <span>✅</span>
            <span>已选择 <b>{{ selectedCrops.length }}</b> 张图片</span>
          </div>
          <button class="clear-btn" @click="clearSelection">🧹 清空</button>
        </div>
      </div>

      <!-- 右侧：SKU库面板 -->
      <div class="sku-panel">
        <div class="sku-header">
          <div class="sku-title">
            📦 SKU库 <span class="sku-title-hint">（双击SKU查看详情）</span>
          </div>
          <div class="sku-search-row">
            <input
              v-model="searchKeyword"
              type="text"
              class="sku-search"
              placeholder="搜索SKU编号或名称"
              @input="debounceSearch"
            />
            <button class="btn-add" @click="createSku">➕ 新增</button>
          </div>
        </div>
        <div class="sku-grid">
          <div v-if="filteredSkus.length === 0" class="empty-state">
            <div class="empty-icon">📦</div>
            <div>暂无SKU数据</div>
          </div>
          <div
            v-for="(sku, idx) in filteredSkus"
            :key="sku.id"
            :class="['sku-card', { targeted: selectedSkuId === sku.id }]"
            @click="selectSku(sku)"
            @dblclick="openSkuDetail(sku)"
          >
            <div class="sku-card-img">
              <img v-if="sku.cover_url" :src="encodeURI(sku.cover_url)" :alt="sku.name" @error="(e) => e.target.style.display = 'none'" />
              <span v-else>🖼️</span>
              <span class="sku-img-count">{{ sku.cnt }}张</span>
            </div>
            <div class="sku-card-info">
              <div class="sku-card-id">{{ sku.id }}</div>
              <div class="sku-card-name">{{ sku.name }}</div>
            </div>
          </div>
        </div>
        <div class="action-area">
          <div class="action-hint">{{ actionHint }}</div>
          <div class="action-btns">
            <button class="btn btn-confirm" @click="assignImages" :disabled="!canAssign">
              ✓ 确认归类
            </button>
            <button class="btn btn-delete" @click="handleDelete" :disabled="!selectedSkuId">
              🗑️ 删除
            </button>
          </div>
        </div>
        <div class="sync-area">
          <div class="sync-info">
            <span>审核库: <b>{{ libraryInfo?.sku_output?.sku_count || 0 }}</b> SKU / <b>{{ skuOutputImageCount }}</b> 图</span>
            <span>正式库: <b>{{ libraryInfo?.sku_library?.meta?.total_skus || 0 }}</b> SKU / <b>{{ libraryInfo?.sku_library?.meta?.total_images || 0 }}</b> 图</span>
          </div>
          <button
            class="btn-sync"
            @click="triggerCombinedBuild"
            :disabled="combinedStatus === 'running'"
          >
            {{ combinedStatus === 'running' ? '🔄 执行中...' : '🚀 一键同步并提取特征' }}
          </button>
          <div v-if="combinedStatus === 'running'" class="step-progress">
            <div class="step-item" :class="{ active: combinedStep >= 1, completed: combinedStep > 1 }">
              <div class="step-icon">{{ combinedStep > 1 ? '✓' : '1' }}</div>
              <div class="step-label">图片增强</div>
            </div>
            <div class="step-divider" :class="{ active: combinedStep > 1 }"></div>
            <div class="step-item" :class="{ active: combinedStep >= 2, completed: combinedStep > 2 }">
              <div class="step-icon">{{ combinedStep > 2 ? '✓' : '2' }}</div>
              <div class="step-label">特征提取</div>
            </div>
          </div>
          <div v-if="combinedStatus !== 'idle'" class="build-status">
            <span :class="['status-indicator', combinedStatus]"></span>
            <span>{{ combinedMessage }}</span>
          </div>
          <div v-if="combinedOutput" class="output-log">
            <pre>{{ combinedOutput }}</pre>
          </div>
          <div v-if="combinedStatus === 'completed' || combinedStatus === 'failed'" class="result-detail">
            <div v-if="combinedStatus === 'completed'" class="success-box">✅ 建库成功！</div>
            <div v-if="combinedStatus === 'failed'" class="error-box">❌ {{ combinedMessage }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 底部日志面板 -->
    <div class="log-panel">
      <div class="log-header">
        <span>📝</span>
        <span>操作日志</span>
      </div>
      <div class="log-content">
        <div v-for="(log, idx) in logs.slice(-50).reverse()" :key="idx" class="log-line">
          <span class="time">[{{ log.split(']')[0].replace('[', '') }}]</span>
          {{ log.split(']').slice(1).join(']') }}
        </div>
      </div>
    </div>

    <!-- 浮动提示 -->
    <div class="float-tip" :class="{ show: showTip }">
      💡 <b>提示：</b>先在左侧选择图片，再点击右侧目标SKU进行归类
    </div>

    <!-- SKU详情滑出面板 -->
    <div class="detail-overlay" :class="{ show: showSkuDetail }" @click="closeSkuDetail"></div>
    <div class="sku-detail-panel" :class="{ show: showSkuDetail }">
      <div class="sku-detail-header">
        <div class="sku-detail-title">{{ detailSku?.id }} - {{ detailSku?.name }}</div>
        <button class="sku-detail-close" @click="closeSkuDetail">✕</button>
      </div>
      <div class="sku-detail-body">
        <div class="sku-detail-info">
          <div class="info-item">
            <div class="info-label">SKU编号</div>
            <div class="info-value">{{ detailSku?.id }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">名称</div>
            <div class="info-value">{{ detailSku?.name }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">图片数量</div>
            <div class="info-value">{{ detailSku?.cnt || 0 }}</div>
          </div>
          <div class="info-item full">
            <div class="info-label">已有图片</div>
            <div class="detail-images-grid">
              <div v-for="(img, idx) in detailSkuImages" :key="idx" class="detail-img-item">
                <img :src="encodeURI(img.url)" :alt="img.name" @error="(e) => e.target.style.display = 'none'" />
              </div>
              <div v-if="detailSkuImages.length === 0" class="no-images">暂无图片</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 大图查看 -->
    <ImageViewer
      :visible="showImageViewer"
      :image-url="imageViewerUrl"
      :image-name="imageViewerName"
      @update:visible="showImageViewer = false"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue';
import { skuReview, build, getImageUrlFromPath } from '@api/client';
import ImageViewer from '@ui/ImageViewer.vue';
import { ElMessageBox, ElMessage } from 'element-plus';

// 数据状态
const folders = ref([]);
const currentFolderIndex = ref(0);
const currentFolder = ref('');
const currentImages = ref([]);
const selectedCrops = ref([]);
const skus = ref([]);
const selectedSkuId = ref('');
const selectedSku = ref(null);
const selectedSkuImages = ref([]);
const skuImagesData = ref([]);
const searchKeyword = ref('');
const newSkuInput = ref('');
const logs = ref([]);
const showToast = ref(false);
const toastMessage = ref('');
const toastType = ref('info');

// 建库状态
const buildStatus = ref('idle')
const buildMessage = ref('')
const libraryInfo = ref(null)
let pollInterval = null

// 特征提取状态
const featureStatus = ref('idle')
const featureMessage = ref('')
let featurePollInterval = null

// 合并任务状态
const combinedStatus = ref('idle')
const combinedStep = ref(0)
const combinedMessage = ref('')
const combinedOutput = ref('')
const skuOutputImageCount = ref(0)
let combinedPollInterval = null

// 文件夹管理状态
const folderFileInput = ref(null);
const newFolderInput = ref(null);

// 文件夹折叠状态
const folderCollapsed = ref(false);

// 浮动提示
const showTip = ref(false);
let tipTimer = null;

// 文件夹图片数量缓存
const folderImageCounts = ref({});

// SKU详情滑出面板
const showSkuDetail = ref(false);
const detailSku = ref(null);
const detailSkuImages = ref([]);

// 大图查看
const showImageViewer = ref(false);
const imageViewerUrl = ref('');
const imageViewerName = ref('');

// 计算属性
const filteredSkus = computed(() => {
  if (!searchKeyword.value) return skus.value;
  const kw = searchKeyword.value.toLowerCase();
  return skus.value.filter(s =>
    s.id.toLowerCase().includes(kw) ||
    s.name.toLowerCase().includes(kw)
  );
});

const actionHint = computed(() => {
  if (!selectedSkuId.value) return '操作提示：先在左侧选择图片，再点击右侧目标SKU';
  const sku = skus.value.find(s => s.id === selectedSkuId.value);
  if (selectedCrops.value.length > 0) {
    return `即将把 ${selectedCrops.value.length} 张图片归类至【${selectedSkuId.value}】(${sku?.name})`;
  }
  return `已选中 SKU: ${selectedSkuId.value}（${sku?.cnt}张）`;
});

const canAssign = computed(() => selectedCrops.value.length > 0 && selectedSkuId.value);

// 辅助函数：统一确认弹窗
const showConfirm = async (message) => {
  try {
    await ElMessageBox.confirm(message, '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    });
    return true;
  } catch (e) {
    return e === 'cancel' ? false : Promise.reject(e);
  }
};

// 辅助函数：显示操作结果消息
const showResult = (success, successMsg, errorMsg) => {
  if (success) {
    ElMessage.success(successMsg);
  } else {
    ElMessage.error(errorMsg || '操作失败');
  }
};

// 添加日志
const addLog = (msg) => {
  const time = new Date().toLocaleTimeString();
  logs.value.push(`[${time}] ${msg}`);
};

// 文件夹折叠
const toggleFolderPanel = () => {
  folderCollapsed.value = !folderCollapsed.value;
};

// 浮动提示
const showFloatTip = () => {
  showTip.value = true;
  clearTimeout(tipTimer);
  tipTimer = setTimeout(() => showTip.value = false, 3000);
};

// 加载文件夹
const loadFolders = async () => {
  try {
    const res = await skuReview.getFolders();
    if (res.success) {
      folders.value = res.folders;
      if (folders.value.length > 0 && !currentFolder.value) {
        currentFolder.value = folders.value[0];
        currentFolderIndex.value = 0;
        await loadFolderImages();
      }
      // 加载各文件夹图片数量
      await loadFolderImageCounts();
    }
  } catch (e) {
    addLog('⚠️ 加载文件夹失败');
  }
};

const loadFolderImageCounts = async () => {
  const counts = {};
  for (const folder of folders.value) {
    try {
      const res = await skuReview.getFolderImages(folder);
      if (res.success) {
        counts[folder] = res.images.length;
      }
    } catch (e) {
      counts[folder] = 0;
    }
  }
  folderImageCounts.value = counts;
};

const loadFolderImages = async () => {
  if (!currentFolder.value) return;
  try {
    const res = await skuReview.getFolderImages(currentFolder.value);
    if (res.success) {
      currentImages.value = res.images;
      selectedCrops.value = [];
      addLog(`切换文件夹 → ${currentFolder.value}（${currentImages.value.length}张图片）`);
    }
  } catch (e) {
    addLog('⚠️ 加载图片失败');
  }
};

const prevFolder = async () => {
  if (currentFolderIndex.value > 0) {
    currentFolderIndex.value--;
    currentFolder.value = folders.value[currentFolderIndex.value];
    await loadFolderImages();
  }
};

const nextFolder = async () => {
  if (currentFolderIndex.value < folders.value.length - 1) {
    currentFolderIndex.value++;
    currentFolder.value = folders.value[currentFolderIndex.value];
    await loadFolderImages();
  }
};

const selectFolder = async (idx) => {
  if (currentFolderIndex.value !== idx) {
    currentFolderIndex.value = idx;
    currentFolder.value = folders.value[idx];
    await loadFolderImages();
  }
};

const onFolderSelect = async () => {
  const idx = folders.value.indexOf(currentFolder.value);
  if (idx >= 0) {
    currentFolderIndex.value = idx;
    await loadFolderImages();
  }
};

// 加载SKU
const loadSkus = async () => {
  try {
    const res = await skuReview.getSkus(searchKeyword.value);
    if (res.success) {
      skus.value = res.skus;
    }
  } catch (e) {
    addLog('⚠️ 加载 SKU 列表失败');
  }
};

// 防抖搜索
let searchTimer = null;
const debounceSearch = () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(loadSkus, 300);
};

// 选择操作
const toggleCropSelect = (idx) => {
  const i = selectedCrops.value.indexOf(idx);
  if (i >= 0) {
    selectedCrops.value.splice(i, 1);
  } else {
    selectedCrops.value.push(idx);
    showFloatTip();
  }
  selectedCrops.value.sort((a, b) => a - b);
};

const deselectCrop = (pos) => {
  const idx = selectedCrops.value[pos];
  const i = selectedCrops.value.indexOf(idx);
  if (i >= 0) {
    selectedCrops.value.splice(i, 1);
  }
};

const clearSelection = () => {
  selectedCrops.value = [];
  addLog('🧹 已清空选择');
};

const selectSku = async (sku) => {
  selectedSkuId.value = sku.id;
  selectedSku.value = sku;
  selectedSkuImages.value = [];
  await loadSkuImages(sku.id);
  addLog(`选中 SKU: ${sku.id}（${sku.cnt}张图片）`);
};

const openSkuDetail = async (sku) => {
  detailSku.value = sku;
  showSkuDetail.value = true;
  try {
    const res = await skuReview.getSkuImages(sku.id);
    if (res.success) {
      detailSkuImages.value = res.images;
    }
  } catch (e) {
    detailSkuImages.value = [];
  }
};

const closeSkuDetail = () => {
  showSkuDetail.value = false;
  detailSku.value = null;
  detailSkuImages.value = [];
};

const loadSkuImages = async (skuId) => {
  try {
    const res = await skuReview.getSkuImages(skuId);
    if (res.success) {
      skuImagesData.value = res.images;
      selectedSkuImages.value = [];
    }
  } catch (e) {
    addLog('⚠️ 加载 SKU 图片失败');
  }
};

const getSkuImages = () => skuImagesData.value || [];

const toggleSkuImageSelect = (idx) => {
  const i = selectedSkuImages.value.indexOf(idx);
  if (i >= 0) {
    selectedSkuImages.value.splice(i, 1);
  } else {
    selectedSkuImages.value.push(idx);
  }
};

// 操作
const assignImages = async () => {
  try {
    const imagePaths = selectedCrops.value.map(idx => currentImages.value[idx].path);
    const res = await skuReview.assignImages(selectedSkuId.value, imagePaths);

    addLog(`✅ 归类 ${selectedCrops.value.length} 张图片至 ${selectedSkuId.value}`);
    showResult(res.success, res.message, res.detail);

    if (res.success) {
      selectedCrops.value = [];
      await loadSkus();
      await loadSkuImages(selectedSkuId.value);
      await loadFolderImageCounts();
    }
  } catch (e) {
    ElMessage.error('操作失败');
  }
};

const createSku = async () => {
  try {
    let res;
    if (newSkuInput.value.includes('|')) {
      const [oldId, newName] = newSkuInput.value.split('|');
      res = await skuReview.renameSku(oldId.trim(), newName.trim());
    } else {
      res = await skuReview.createSku(newSkuInput.value || undefined);
    }

    addLog(res.message);
    showResult(res.success, res.message, res.detail);

    if (res.success) {
      newSkuInput.value = '';
      await loadSkus();
    }
  } catch (e) {
    ElMessage.error('操作失败');
  }
};

const handleDelete = async () => {
  if (!selectedSkuId.value) return;

  const hasSelectedImages = selectedSkuImages.value.length > 0;

  if (hasSelectedImages) {
    if (!await showConfirm(`确定要删除选中的 ${selectedSkuImages.value.length} 张图片吗？`)) return

    try {
      const images = getSkuImages();
      const imagePaths = selectedSkuImages.value.map(idx => images[idx].path);
      const res = await skuReview.recallImages(selectedSkuId.value, imagePaths)

      addLog(`🗑️ 从 ${selectedSkuId.value} 删除 ${selectedSkuImages.value.length} 张图片`)
      showResult(res.success, '删除成功', res.detail)

      if (res.success) {
        selectedSkuImages.value = []
        await loadSkus()
        await loadSkuImages(selectedSkuId.value)
      }
    } catch (e) {
      ElMessage.error('操作失败: ' + (e.message || e))
    }
  } else {
    if (!await showConfirm(`确定要删除 SKU ${selectedSkuId.value} 及其所有图片吗？`)) return

    try {
      const res = await skuReview.deleteSku(selectedSkuId.value)

      addLog(`🗑️ 已删除 SKU: ${selectedSkuId.value}`)
      showResult(res.success, '删除成功', res.detail)

      if (res.success) {
        selectedSkuId.value = ''
        selectedSku.value = null
        skuImagesData.value = []
        selectedSkuImages.value = []
        await loadSkus()
      }
    } catch (e) {
      ElMessage.error('操作失败: ' + (e.message || e))
    }
  }
};

// 大图查看
const openLargeImage = (url, name, event) => {
  if (event) event.stopPropagation();
  imageViewerUrl.value = url;
  imageViewerName.value = name;
  showImageViewer.value = true;
};

// 建库操作
const triggerBuild = async () => {
  if (buildStatus.value === 'running') {
    ElMessage.warning('建库任务正在进行中');
    return;
  }

  buildStatus.value = 'running';
  buildMessage.value = '正在启动建库任务...';

  try {
    const res = await build.triggerBuild();
    if (res.success) {
      addLog('📦 建库任务已启动');
      buildMessage.value = res.message;
      startPolling();
    } else {
      buildStatus.value = 'idle';
      buildMessage.value = '';
      ElMessage.error(res.message || '启动失败');
    }
  } catch (e) {
    buildStatus.value = 'idle';
    buildMessage.value = '';
    ElMessage.error('启动失败: ' + e.message);
  }
};

const startPolling = () => {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const res = await build.getStatus();
      if (res.success) {
        buildStatus.value = res.status;
        buildMessage.value = res.message;

        if (res.status === 'completed') {
          addLog('✅ 建库完成');
          ElMessage.success('建库完成');
          stopPolling();
          await refreshLibraryInfo();
        } else if (res.status === 'failed') {
          addLog('❌ 建库失败: ' + res.message);
          ElMessage.error('建库失败: ' + res.message);
          stopPolling();
        }
      }
    } catch (e) {
      console.error('轮询失败:', e);
    }
  }, 2000);
};

const stopPolling = () => {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
};

const refreshLibraryInfo = async () => {
  try {
    const res = await build.getInfo()
    if (res.success) {
      libraryInfo.value = res.data
    }
  } catch (e) {
    console.error('获取库信息失败:', e)
  }
}

// 合并任务操作
const triggerCombinedBuild = async () => {
  if (combinedStatus.value === 'running') {
    ElMessage.warning('任务正在进行中')
    return
  }

  combinedStatus.value = 'running'
  combinedStep.value = 1
  combinedMessage.value = '正在启动...'
  combinedOutput.value = ''

  try {
    const res = await build.triggerCombinedBuild()
    if (res.success) {
      addLog('🚀 启动完整建库流程')
      combinedMessage.value = res.message
      startCombinedPolling()
    } else {
      combinedStatus.value = 'idle'
      combinedStep.value = 0
      combinedMessage.value = ''
      ElMessage.error(res.message || '启动失败')
    }
  } catch (e) {
    combinedStatus.value = 'idle'
    combinedStep.value = 0
    combinedMessage.value = ''
    ElMessage.error('启动失败: ' + e.message)
  }
}

const startCombinedPolling = () => {
  if (combinedPollInterval) clearInterval(combinedPollInterval)
  combinedPollInterval = setInterval(async () => {
    try {
      const res = await build.getCombinedStatus()
      if (res.success) {
        combinedStatus.value = res.status
        combinedStep.value = res.step || 0
        combinedMessage.value = res.message
        combinedOutput.value = res.output || ''

        if (res.status === 'completed') {
          addLog('✅ 完整建库完成！')
          ElMessage.success('建库成功！')
          stopCombinedPolling()
          await refreshLibraryInfo()
          await updateImageCount()
        } else if (res.status === 'failed') {
          addLog('❌ 建库失败: ' + res.message)
          ElMessage.error('建库失败')
          stopCombinedPolling()
        }
      }
    } catch (e) {
      console.error('轮询失败:', e)
    }
  }, 500)
}

const stopCombinedPolling = () => {
  if (combinedPollInterval) {
    clearInterval(combinedPollInterval)
    combinedPollInterval = null
  }
}

// 特征提取操作
const triggerFeatureExtract = async () => {
  if (featureStatus.value === 'running') {
    ElMessage.warning('特征提取任务正在进行中')
    return
  }

  featureStatus.value = 'running'
  featureMessage.value = '正在启动特征提取任务...'

  try {
    const res = await build.triggerFeatureExtract()
    if (res.success) {
      addLog('🔍 特征提取任务已启动')
      featureMessage.value = res.message
      startFeaturePolling()
    } else {
      featureStatus.value = 'idle'
      featureMessage.value = ''
      ElMessage.error(res.message || '启动失败')
    }
  } catch (e) {
    featureStatus.value = 'idle'
    featureMessage.value = ''
    ElMessage.error('启动失败: ' + e.message)
  }
}

const startFeaturePolling = () => {
  if (featurePollInterval) clearInterval(featurePollInterval)
  featurePollInterval = setInterval(async () => {
    try {
      const res = await build.getFeatureStatus()
      if (res.success) {
        featureStatus.value = res.status
        featureMessage.value = res.message

        if (res.status === 'completed') {
          addLog('✅ 特征提取完成')
          ElMessage.success('特征提取完成')
          stopFeaturePolling()
          await refreshLibraryInfo()
        } else if (res.status === 'failed') {
          addLog('❌ 特征提取失败: ' + res.message)
          ElMessage.error('特征提取失败: ' + res.message)
          stopFeaturePolling()
        }
      }
    } catch (e) {
      console.error('轮询失败:', e)
    }
  }, 2000)
}

const stopFeaturePolling = () => {
  if (featurePollInterval) {
    clearInterval(featurePollInterval)
    featurePollInterval = null
  }
}

// 获取审核库图片数量
const updateImageCount = async () => {
  try {
    const res = await build.checkChange()
    if (res.success) {
      skuOutputImageCount.value = res.image_count || 0
    }
  } catch (e) {
    console.error('获取图片数量失败:', e)
  }
}

// 文件夹管理方法
const handleUploadClick = () => {
  if (!currentFolder.value) {
    ElMessage.warning('请先选择一个文件夹')
    return
  }
  nextTick(() => folderFileInput.value?.click())
}

const handleFolderFileSelect = async (event) => {
  const files = Array.from(event.target.files)
  if (files.length === 0) return

  try {
    const res = await skuReview.uploadFolder(currentFolder.value, files)
    addLog(`📤 成功上传 ${res.saved_count} 张图片到文件夹: ${currentFolder.value}`)
    showResult(res.success, res.message || '上传成功', res.message)

    if (res.success) {
      await loadFolderImages()
      await loadFolderImageCounts()
      await updateImageCount()
    }
  } catch (e) {
    ElMessage.error('上传失败: ' + e.message)
  } finally {
    if (folderFileInput.value) {
      folderFileInput.value.value = ''
    }
  }
}

const handleNewFolderClick = async () => {
  const { value } = await ElMessageBox.prompt('请输入新文件夹名称', '新建文件夹', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    inputPlaceholder: '文件夹名称'
  }).catch(() => ({ value: null }))
  if (!value) return
  nextTick(() => {
    newFolderName.value = value
    newFolderInput.value?.click()
  })
}

const newFolderName = ref('')

const handleNewFolderFileSelect = async (event) => {
  const files = Array.from(event.target.files)
  if (files.length === 0) {
    newFolderName.value = ''
    return
  }

  try {
    const res = await skuReview.uploadFolder(newFolderName.value, files)
    addLog(`📤 成功上传 ${res.saved_count} 张图片到新文件夹: ${newFolderName.value}`)
    showResult(res.success, res.message || '上传成功', res.message)

    if (res.success) {
      await loadFolders()
      const idx = folders.value.indexOf(newFolderName.value)
      if (idx !== -1) {
        currentFolderIndex.value = idx
        await loadFolderImages()
      }
      await loadFolderImageCounts()
      await updateImageCount()
    }
  } catch (e) {
    ElMessage.error('上传失败: ' + e.message)
  } finally {
    newFolderName.value = ''
    if (newFolderInput.value) {
      newFolderInput.value.value = ''
    }
  }
}

const confirmDeleteFolder = async () => {
  if (!currentFolder.value) return

  if (!await showConfirm(`确定要删除文件夹 "${currentFolder.value}" 及其所有图片吗？`)) return

  try {
    const res = await skuReview.deleteFolder(currentFolder.value)
    addLog(`🗑️ 已删除文件夹: ${currentFolder.value}，包含 ${res.deleted_count} 张图片`)
    showResult(res.success, '删除成功', res.message)

    await loadFolders()
    if (folders.value.length > 0) {
      currentFolderIndex.value = Math.min(currentFolderIndex.value, folders.value.length - 1)
      await loadFolderImages()
    } else {
      currentFolder.value = ''
      currentFolderIndex.value = -1
      currentImages.value = []
    }
    await updateImageCount()
  } catch (e) {
    ElMessage.error('删除失败: ' + (e.message || e))
  }
}

onMounted(async () => {
  addLog('🚀 SKU审核系统初始化完成')
  loadFolders()
  loadSkus()
  refreshLibraryInfo()
  updateImageCount()

  try {
    const res = await build.getCombinedStatus()
    if (res.success) {
      if (res.status === 'running') {
        combinedStatus.value = 'running'
        combinedStep.value = res.step || 0
        combinedMessage.value = res.message
        combinedOutput.value = res.output || ''
        addLog('🔄 检测到正在运行的任务，继续跟踪...')
        startCombinedPolling()
      } else if (res.status === 'failed') {
        combinedStatus.value = 'failed'
        combinedStep.value = res.step || 0
        combinedMessage.value = res.message
        combinedOutput.value = res.output || ''
        addLog('❌ 检测到失败的任务: ' + res.message)
      } else {
        combinedStatus.value = 'idle'
        combinedStep.value = 0
        combinedMessage.value = ''
        combinedOutput.value = ''
      }
    }
  } catch (e) {
    console.error('检查任务状态失败:', e)
  }
});

onUnmounted(() => {
  stopCombinedPolling()
  stopFeaturePolling()
});
</script>

<style scoped>
.sku-review-page {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--color-bg-secondary);
}

.content {
  flex: 1;
  display: flex;
  overflow: hidden;
}

/* 左侧面板 - 文件夹 */
.folder-panel {
  width: 200px;
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  background: var(--color-bg-tertiary);
  transition: width 0.3s ease;
  flex-shrink: 0;
}

.dark .folder-panel {
  background: rgba(30, 41, 59, 0.4);
}

.folder-panel.collapsed {
  width: 80px;
}

.folder-header {
  padding: 12px;
  border-bottom: 1px solid var(--color-border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.folder-title {
  font-size: 13px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
}

.folder-panel.collapsed .folder-title span:not(:first-child) {
  display: none;
}

.folder-toggle {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  border: none;
  background: transparent;
  color: var(--color-text-secondary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.3s;
}

.folder-toggle:hover {
  background: rgba(255, 255, 255, 0.05);
}

.folder-panel.collapsed .folder-toggle {
  transform: rotate(180deg);
}

.folder-actions {
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  border-bottom: 1px solid var(--color-border);
}

.folder-panel.collapsed .folder-actions {
  padding: 10px 8px;
}

.folder-btn {
  padding: 8px;
  border-radius: 6px;
  border: none;
  font-size: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  transition: all 0.2s;
  white-space: nowrap;
}

.folder-panel.collapsed .folder-btn {
  padding: 10px 4px;
}

.folder-panel.collapsed .folder-btn span:not(:first-child) {
  display: none;
}

.folder-btn.upload {
  background: var(--color-primary);
  color: #fff;
}

.folder-btn.upload:hover {
  filter: brightness(1.1);
}

.folder-btn.delete {
  background: rgba(239, 68, 68, 0.15);
  color: var(--color-danger);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.folder-btn.delete:hover {
  background: rgba(239, 68, 68, 0.25);
}

.folder-btn.new-folder {
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
}

.folder-btn.new-folder:hover {
  background: var(--color-border);
}

.folder-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.folder-panel.collapsed .folder-list {
  padding: 8px 6px;
}

.folder-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  color: var(--color-text-secondary);
  transition: all 0.15s;
  margin-bottom: 2px;
  white-space: nowrap;
}

.folder-panel.collapsed .folder-item {
  flex-direction: column;
  gap: 4px;
  padding: 10px 4px;
  font-size: 10px;
  text-align: center;
}

.folder-item:hover {
  background: rgba(255, 255, 255, 0.03);
  color: var(--color-text-primary);
}

.folder-item.active {
  background: rgba(59, 130, 246, 0.1);
  color: var(--color-primary);
}

.folder-item .folder-icon {
  font-size: 14px;
}

.folder-panel.collapsed .folder-item .folder-icon {
  font-size: 18px;
}

.folder-item .folder-count {
  margin-left: auto;
  font-size: 11px;
  color: var(--color-text-tertiary);
}

.folder-panel.collapsed .folder-item .folder-count {
  margin-left: 0;
  font-size: 10px;
}

/* 中间面板 - 图片 */
.image-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--color-border);
  min-width: 0;
  min-height: 0;
}

.image-header {
  padding: 12px 16px;
  border-bottom: 1px solid var(--color-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.image-title {
  font-size: 14px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}

.image-count {
  font-size: 12px;
  color: var(--color-text-tertiary);
  font-weight: normal;
}

.image-grid {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 12px;
}

.image-item {
  aspect-ratio: 1;
  border-radius: 8px;
  background: var(--color-bg-primary);
  border: 2px solid var(--color-border);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-tertiary);
  font-size: 32px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: all 0.2s;
}

.image-item:hover {
  border-color: var(--color-primary);
}

.image-item.selected {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}

.image-item .img-check {
  position: absolute;
  top: 6px;
  left: 6px;
  width: 20px;
  height: 20px;
  accent-color: var(--color-primary);
  z-index: 2;
}

.image-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-item .img-name {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 6px 8px;
  background: rgba(0, 0, 0, 0.7);
  font-size: 11px;
  color: #fff;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.image-footer {
  padding: 10px 16px;
  border-top: 1px solid var(--color-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--color-bg-tertiary);
}

.dark .image-footer {
  background: rgba(30, 41, 59, 0.6);
}

.select-info {
  font-size: 13px;
  color: var(--color-text-secondary);
  display: flex;
  align-items: center;
  gap: 6px;
}

.select-info b {
  color: var(--color-primary);
}

.clear-btn {
  padding: 4px 10px;
  border-radius: 4px;
  border: 1px solid var(--color-border);
  background: transparent;
  color: var(--color-text-secondary);
  font-size: 12px;
  cursor: pointer;
}

.clear-btn:hover {
  color: var(--color-text-primary);
  border-color: var(--color-text-tertiary);
}

/* 右侧面板 - SKU库 */
.sku-panel {
  width: 380px;
  display: flex;
  flex-direction: column;
  background: var(--color-bg-secondary);
  min-height: 0;
}

.dark .sku-panel {
  background: rgba(30, 41, 59, 0.3);
}

.sku-header {
  padding: 12px 16px;
  border-bottom: 1px solid var(--color-border);
}

.sku-title {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.sku-title-hint {
  font-size: 11px;
  font-weight: normal;
  color: var(--color-text-tertiary);
  margin-left: 2px;
}

.sku-search-row {
  display: flex;
  gap: 8px;
}

.sku-search {
  flex: 1;
  padding: 8px 12px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  color: var(--color-text-primary);
  font-size: 13px;
  outline: none;
}

.sku-search:focus {
  border-color: var(--color-primary);
}

.btn-add {
  padding: 8px 14px;
  border-radius: 6px;
  border: none;
  background: var(--color-success);
  color: #fff;
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  white-space: nowrap;
}

.btn-add:hover {
  filter: brightness(1.1);
}

.sku-grid {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-content: flex-start;
}

.sku-card {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.2s;
  width: calc(50% - 4px);
}

.sku-card:hover {
  border-color: var(--color-primary);
}

.sku-card.targeted {
  border-color: var(--color-success);
  box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.3);
}

.sku-card-img {
  width: 100%;
  height: 100px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-tertiary);
  font-size: 28px;
  background: var(--color-bg-tertiary);
  position: relative;
}

.sku-card-img img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.sku-card-img .sku-img-count {
  position: absolute;
  bottom: 4px;
  right: 4px;
  background: rgba(0, 0, 0, 0.7);
  color: #fff;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
}

.sku-card-info {
  padding: 8px;
}

.sku-card-id {
  font-size: 12px;
  color: var(--color-primary);
  font-weight: 600;
  margin-bottom: 2px;
}

.sku-card-name {
  font-size: 12px;
  color: var(--color-text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 操作区 */
.action-area {
  padding: 12px 16px;
  border-top: 1px solid var(--color-border);
}

.action-hint {
  padding: 10px 12px;
  background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(251, 191, 36, 0.15) 100%);
  border-left: 4px solid var(--color-warning);
  border-radius: 0 6px 6px 0;
  font-size: 12px;
  color: var(--color-text-primary);
  margin-bottom: 10px;
}

.action-btns {
  display: flex;
  gap: 8px;
}

.btn {
  flex: 1;
  padding: 10px;
  border-radius: 6px;
  border: none;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.btn-confirm {
  background: var(--color-success);
  color: #fff;
}

.btn-confirm:hover {
  filter: brightness(1.1);
}

.btn-confirm:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.btn-delete {
  background: rgba(239, 68, 68, 0.15);
  color: var(--color-danger);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.btn-delete:hover {
  background: rgba(239, 68, 68, 0.25);
}

/* 同步区 */
.sync-area {
  padding: 12px 16px;
  border-top: 1px solid var(--color-border);
  background: var(--color-bg-tertiary);
}

.dark .sync-area {
  background: rgba(30, 41, 59, 0.5);
}

.sync-info {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--color-text-secondary);
  margin-bottom: 10px;
  flex-wrap: wrap;
  gap: 4px;
}

.sync-info span {
  display: flex;
  align-items: center;
  gap: 4px;
}

.btn-sync {
  width: 100%;
  padding: 12px;
  border-radius: 8px;
  border: none;
  background: linear-gradient(135deg, var(--color-primary), #8b5cf6);
  color: #fff;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.btn-sync:hover {
  filter: brightness(1.1);
  transform: translateY(-1px);
}

.btn-sync:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

/* 步骤进度条 */
.step-progress {
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 12px 0;
}

.step-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  opacity: 0.4;
  transition: all 0.3s ease;
}

.step-item.active {
  opacity: 1;
}

.step-item.completed {
  opacity: 1;
}

.step-icon {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 12px;
  background: var(--color-bg-tertiary);
  border: 2px solid var(--color-border);
}

.step-item.active .step-icon {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: white;
}

.step-item.completed .step-icon {
  background: var(--color-success);
  border-color: var(--color-success);
  color: white;
}

.step-label {
  font-size: 11px;
  color: var(--color-text-secondary);
}

.step-item.active .step-label,
.step-item.completed .step-label {
  color: var(--color-text-primary);
  font-weight: 500;
}

.step-divider {
  width: 40px;
  height: 2px;
  background: var(--color-border);
  margin: 0 6px;
  margin-bottom: 16px;
}

.step-divider.active {
  background: var(--color-success);
}

.build-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--color-bg-tertiary);
  border-radius: 6px;
  margin-top: 10px;
  font-size: 12px;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-indicator.running {
  background: var(--color-primary);
  animation: pulse 1.5s infinite;
}

.status-indicator.completed {
  background: var(--color-success);
}

.status-indicator.failed {
  background: var(--color-danger);
}

.output-log {
  margin-top: 8px;
  padding: 8px;
  background: var(--color-bg-tertiary);
  border-radius: 6px;
  max-height: 120px;
  overflow: auto;
  font-size: 11px;
}

.output-log pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: 'Consolas', 'Monaco', monospace;
  color: var(--color-text-secondary);
}

.result-detail {
  margin-top: 8px;
}

.success-box {
  background: rgba(34, 197, 94, 0.1);
  color: var(--color-success);
  font-weight: 500;
  text-align: center;
  font-size: 13px;
  padding: 8px;
  border-radius: 6px;
}

.error-box {
  background: rgba(239, 68, 68, 0.1);
  color: var(--color-danger);
  font-weight: 500;
  font-size: 12px;
  padding: 8px;
  border-radius: 6px;
}

/* 底部日志 */
.log-panel {
  height: 140px;
  border-top: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  background: var(--color-bg-tertiary);
  flex-shrink: 0;
}

.dark .log-panel {
  background: rgba(30, 41, 59, 0.3);
}

.log-header {
  padding: 8px 16px;
  border-bottom: 1px solid var(--color-border);
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-secondary);
  display: flex;
  align-items: center;
  gap: 6px;
}

.log-content {
  flex: 1;
  overflow-y: auto;
  padding: 8px 16px;
  font-size: 12px;
  font-family: monospace;
}

.log-line {
  padding: 3px 0;
  color: var(--color-text-tertiary);
}

.log-line .time {
  color: var(--color-text-secondary);
  margin-right: 8px;
}

/* 浮动提示 */
.float-tip {
  position: fixed;
  bottom: 180px;
  left: 50%;
  transform: translateX(-50%) translateY(100px);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 13px;
  color: var(--color-text-secondary);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  opacity: 0;
  transition: all 0.3s ease;
  z-index: 70;
  pointer-events: none;
}

.float-tip.show {
  transform: translateX(-50%) translateY(0);
  opacity: 1;
}

.float-tip b {
  color: var(--color-warning);
}

/* SKU详情滑出面板 */
.detail-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  opacity: 0;
  visibility: hidden;
  z-index: 80;
  transition: all 0.3s ease;
}

.detail-overlay.show {
  opacity: 1;
  visibility: visible;
}

.sku-detail-panel {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: var(--color-bg-primary);
  border-top: 1px solid var(--color-border);
  border-radius: 16px 16px 0 0;
  box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.5);
  transform: translateY(100%);
  transition: transform 0.3s ease;
  z-index: 90;
  max-height: 50vh;
  display: flex;
  flex-direction: column;
}

.sku-detail-panel.show {
  transform: translateY(0);
}

.sku-detail-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.sku-detail-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.sku-detail-close {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: var(--color-text-secondary);
  cursor: pointer;
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.sku-detail-close:hover {
  background: rgba(255, 255, 255, 0.05);
  color: var(--color-text-primary);
}

.sku-detail-body {
  padding: 20px;
  overflow-y: auto;
}

.sku-detail-info {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.info-item {
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 12px;
}

.info-item.full {
  grid-column: 1 / -1;
}

.info-label {
  font-size: 11px;
  color: var(--color-text-tertiary);
  margin-bottom: 4px;
  text-transform: uppercase;
}

.info-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.detail-images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
  gap: 8px;
  margin-top: 8px;
}

.detail-img-item {
  aspect-ratio: 1;
  border-radius: 6px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.detail-img-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.no-images {
  grid-column: 1 / -1;
  text-align: center;
  color: var(--color-text-tertiary);
  font-size: 13px;
  padding: 20px;
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-text-tertiary);
  gap: 8px;
  min-height: 200px;
}

.empty-icon {
  font-size: 40px;
}

/* 滚动条 */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: var(--color-text-tertiary);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--color-text-secondary);
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
</style>
