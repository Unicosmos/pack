<template>
  <div class="sku-review-page">
    <PageContainer>
      <div class="main-content">
      <!-- 左侧：文件夹导航栏 -->
      <div class="folder-nav">
        <div class="folder-nav-header">
          <h3>📂 文件夹</h3>
          <div class="folder-nav-info">
            {{ currentFolderIndex + 1 }} / {{ folders.length }}
          </div>
        </div>
        
        <!-- 文件夹操作 -->
        <div class="folder-actions">
          <input 
            v-if="uploadFolderName" 
            type="file" 
            ref="folderFileInput" 
            multiple 
            accept="image/*" 
            @change="handleFolderFileSelect"
            style="display: none"
          />
          <input 
            v-model="newFolderName" 
            type="text" 
            placeholder="输入文件夹名称"
            class="folder-input"
          />
          <div class="folder-action-buttons">
            <button class="btn btn-sm btn-primary" @click="prepareUploadFolder">
              📤 上传图片
            </button>
            <button 
              v-if="currentFolder" 
              class="btn btn-sm btn-danger" 
              @click="confirmDeleteFolder"
            >
              🗑️ 删除文件夹
            </button>
          </div>
        </div>

        <div class="folder-list">
          <div
            v-for="(folder, idx) in folders"
            :key="folder"
            :class="['folder-item', { active: currentFolderIndex === idx }]"
            @click="selectFolder(idx)"
          >
            <span class="folder-icon">📁</span>
            <span class="folder-name">{{ folder }}</span>
          </div>
        </div>
      </div>

      <!-- 右侧：待审核图片区域 -->
      <div class="left-panel">
        <div class="panel-header">
          <h2>📋 待审核图片 - {{ currentFolder }}</h2>
          <div class="crop-count">共 {{ currentImages.length }} 张图片</div>
        </div>
        <div class="crop-gallery">
          <div
            v-for="(img, idx) in currentImages"
            :key="idx"
            :class="['crop-item', { selected: selectedCrops.includes(idx) }]"
            @click="toggleCropSelect(idx)"
          >
            <img :src="encodeURI(img.url)" :alt="img.name" loading="lazy" @error="(e) => e.target.style.display = 'none'" />
            <div class="crop-name">{{ img.name }}</div>
          </div>
        </div>

        <div class="selected-crops">
          <div class="selected-header">
            <h3>✅ 已选择图片 ({{ selectedCrops.length }})</h3>
            <button class="btn btn-sm btn-secondary" @click="clearSelection">🧹 清空</button>
          </div>
          <div class="selected-gallery">
            <div
              v-for="(idx, i) in selectedCrops"
              :key="i"
              class="selected-item"
              @click="deselectCrop(i)"
            >
              <img 
                :src="encodeURI(currentImages[idx]?.url)" 
                :alt="currentImages[idx]?.name" 
                @error="(e) => e.target.style.display = 'none'"
                @click.stop="openLargeImage(currentImages[idx]?.url, currentImages[idx]?.name, $event)"
              />
              <div class="remove-badge">×</div>
            </div>
          </div>
        </div>
      </div>

      <!-- 右侧：SKU 库区域 -->
      <div class="right-panel">
        <div class="panel-header">
          <h2>🗂️ SKU 库</h2>
          <div class="sku-controls">
            <div class="search-box">
              <input v-model="searchKeyword" placeholder="搜索 SKU 编号或名称" @input="debounceSearch" />
            </div>
            <div class="new-sku-box">
              <input v-model="newSkuInput" placeholder="新增 SKU" />
              <button class="btn btn-success" @click="createSku">➕ 新增</button>
            </div>
          </div>
        </div>
        <div class="sku-gallery">
          <div
            v-for="(sku, idx) in filteredSkus"
            :key="sku.id"
            :class="['sku-item', { selected: selectedSkuId === sku.id }]"
            @click="selectSku(sku)"
          >
            <div class="sku-cover">
              <img v-if="sku.cover_url" :src="encodeURI(sku.cover_url)" :alt="sku.name" @error="(e) => e.target.style.display = 'none'" />
              <div v-else class="no-cover">📷</div>
            </div>
            <div class="sku-info">
              <div class="sku-id">{{ sku.id }}</div>
              <div class="sku-name">{{ sku.name }}</div>
              <div class="sku-count">{{ sku.cnt }}张</div>
            </div>
          </div>
        </div>

        <div class="action-hint">{{ actionHint }}</div>
        <div class="action-buttons">
          <button class="btn btn-success" @click="assignImages" :disabled="!canAssign">✅ 确认归类</button>
          <button class="btn btn-danger" @click="handleDelete" :disabled="!selectedSkuId">🗑️ 删除</button>
        </div>

        <!-- SKU 详情 -->
        <div v-if="selectedSkuId" class="sku-detail">
          <div class="detail-header">
            <h3>SKU 详情：{{ selectedSkuId }}</h3>
            <div v-if="selectedSku" class="sku-info-detail">
              <span class="sku-name-detail">{{ selectedSku.name }}</span>
              <span class="sku-count-detail">{{ getSkuImages().length }} 张图片</span>
            </div>
          </div>
          <div class="sku-detail-images">
            <div
              v-for="(img, idx) in getSkuImages()"
              :key="idx"
              :class="['detail-image', { selected: selectedSkuImages.includes(idx) }]"
              @click="toggleSkuImageSelect(idx)"
            >
              <img 
                :src="encodeURI(img.url)" 
                :alt="img.name" 
                @error="(e) => e.target.style.display = 'none'"
                @click.stop="openLargeImage(img.url, img.name, $event)"
              />
              <div class="image-name">{{ img.name }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="build-panel">
      <div class="panel-header">
        <h3>📦 SKU库建库</h3>
        <div class="stats-row">
          <span class="stat-item">审核库: <span class="stat-value">{{ libraryInfo?.sku_output?.sku_count || 0 }}</span> SKU / <span class="stat-value">{{ skuOutputImageCount }}</span> 图</span>
          <span class="stat-divider">|</span>
          <span class="stat-item">正式库: <span class="stat-value">{{ libraryInfo?.sku_library?.meta?.total_skus || 0 }}</span> SKU / <span class="stat-value">{{ libraryInfo?.sku_library?.meta?.total_images || 0 }}</span> 图</span>
          <span class="stat-divider">|</span>
          <span class="stat-item">SKU库特征文件: {{ libraryInfo?.sku_library?.has_features ? '✅ 已生成' : '❌ 未生成' }}</span>
        </div>
      </div>
      
      <button 
        class="btn btn-primary" 
        @click="triggerCombinedBuild" 
        :disabled="combinedStatus === 'running'"
        style="width: 100%; margin-bottom: 16px;"
      >
        {{ combinedStatus === 'running' ? '🔄 执行中...' : '一键同步并提取特征' }}
      </button>
      
      <!-- 步骤进度条 -->
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
      
      <!-- 状态信息 -->
      <div v-if="combinedStatus !== 'idle'" class="build-status">
        <span :class="['status-indicator', combinedStatus]"></span>
        <span>{{ combinedMessage }}</span>
      </div>
      
      <!-- 输出日志 -->
      <div v-if="combinedOutput" class="output-log">
        <pre>{{ combinedOutput }}</pre>
      </div>
      
      <!-- 成功/失败详情 -->
      <div v-if="combinedStatus === 'completed' || combinedStatus === 'failed'" class="result-detail">
        <div v-if="combinedStatus === 'completed'" class="success-box">
          ✅ 建库成功！
        </div>
        <div v-if="combinedStatus === 'failed'" class="error-box">
          ❌ {{ combinedMessage }}
        </div>
      </div>
      
    </div>

    <div class="log-panel">
      <div class="panel-header">
        <h3>📜 操作日志</h3>
      </div>
      <div class="logs">
        <div v-for="(log, idx) in logs.slice(-50).reverse()" :key="idx" class="log-item">
          {{ log }}
        </div>
      </div>
    </div>

    <div v-if="showToast" :class="['toast', toastType]">
      {{ toastMessage }}
    </div>

    </PageContainer>
    
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
import PageHeader from '@layout/PageHeader.vue';
import PageContainer from '@layout/PageContainer.vue';
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
const newFolderName = ref('')
const uploadFolderName = ref('')
const folderFileInput = ref(null);

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
    }
  } catch (e) {
    addLog('⚠️ 加载文件夹失败');
  }
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
const prepareUploadFolder = () => {
  if (!newFolderName.value.trim()) {
    ElMessage.error('请先输入文件夹名称')
    return
  }
  uploadFolderName.value = newFolderName.value.trim()
  nextTick(() => {
    folderFileInput.value?.click()
  })
}

const handleFolderFileSelect = async (event) => {
  const files = Array.from(event.target.files)
  if (files.length === 0) {
    uploadFolderName.value = ''
    return
  }

  try {
    const res = await skuReview.uploadFolder(uploadFolderName.value, files)
    addLog(`📤 成功上传 ${res.saved_count} 张图片到文件夹: ${uploadFolderName.value}`)
    showResult(res.success, res.message || '上传成功', res.message)
    
    if (res.success) {
      await loadFolders()
      const newIndex = folders.value.indexOf(uploadFolderName.value)
      if (newIndex !== -1) {
        currentFolderIndex.value = newIndex
        await loadFolderImages()
      }
      await updateImageCount()
    }
  } catch (e) {
    ElMessage.error('上传失败: ' + e.message)
  } finally {
    uploadFolderName.value = ''
    newFolderName.value = ''
    if (folderFileInput.value) {
      folderFileInput.value.value = ''
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
  max-width: 100%;
  margin: 0 auto;
  background: var(--color-bg-secondary);
  min-height: 100vh;
  box-sizing: border-box;
}

.main-content {
  display: flex;
  gap: 16px;
  margin-bottom: 20px;
}

.top-bar {
  display: flex;
  gap: 16px;
  background: var(--color-bg-primary);
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
  box-shadow: var(--shadow-sm);
}

.sku-controls {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.search-box input, .new-sku-box input {
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: 5px;
  width: 160px;
  font-size: 13px;
  transition: all var(--transition-fast);
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
}

.search-box input:focus, .new-sku-box input:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
}

.top-bar {
  display: flex;
  gap: 16px;
  margin-bottom: 20px;
  padding: 0 8px;
}

/* 文件夹导航栏 */
.folder-nav {
  width: 200px;
  background: var(--color-bg-primary);
  border-radius: 12px;
  box-shadow: var(--shadow-md);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.folder-nav-header {
  padding: 14px 16px;
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%);
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.folder-nav-header h3 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
}

.folder-nav-info {
  font-size: 12px;
  background: rgba(255, 255, 255, 0.2);
  padding: 2px 8px;
  border-radius: 10px;
}

.folder-list {
  flex: 1;
  overflow-y: auto;
  max-height: 500px;
}

.folder-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  cursor: pointer;
  border-bottom: 1px solid var(--color-border-light);
  transition: all var(--transition-fast);
}

.folder-item:hover {
  background: var(--color-bg-tertiary);
}

.folder-item.active {
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%);
  color: white;
}

.folder-item.active .folder-icon {
  transform: scale(1.1);
}

.folder-actions {
  padding: 12px 16px;
  border-bottom: 1px solid var(--color-border-light);
  background: var(--color-bg-tertiary);
}

.folder-input {
  width: 100%;
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  box-sizing: border-box;
}

.folder-input:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
}

.folder-action-buttons {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 10px;
}

.folder-action-buttons .btn {
  width: 100%;
  padding: 6px 12px;
  font-size: 12px;
}

.folder-icon {
  font-size: 16px;
  transition: transform var(--transition-fast);
}

.folder-name {
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.left-panel, .right-panel {
  flex: 1;
  background: var(--color-bg-primary);
  padding: 0;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: var(--shadow-md);
  display: flex;
  flex-direction: column;
}

.crop-count {
  font-size: 12px;
  color: var(--color-primary);
  background: rgba(102, 126, 234, 0.1);
  padding: 4px 10px;
  border-radius: 12px;
}

.panel-header {
  padding: 12px 16px;
  background: linear-gradient(135deg, var(--color-bg-tertiary) 0%, var(--color-bg-secondary) 100%);
  border-bottom: 1px solid var(--color-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}

.panel-header h2, .panel-header h3 {
  margin: 0;
  font-size: 16px;
  color: var(--color-text-primary);
  font-weight: 600;
}

.left-panel h2, .right-panel h2 {
  margin: 0;
  font-size: 16px;
}

.crop-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  grid-auto-rows: minmax(160px, auto);
  gap: 12px;
  padding: 20px;
  max-height: 380px;
  overflow-y: auto;
}

.crop-gallery::-webkit-scrollbar {
  width: 6px;
}

.crop-gallery::-webkit-scrollbar-track {
  background: var(--color-bg-tertiary);
  border-radius: 3px;
}

.crop-gallery::-webkit-scrollbar-thumb {
  background: var(--color-border);
  border-radius: 3px;
}

.crop-item {
  cursor: pointer;
  border: 2px solid transparent;
  border-radius: 8px;
  overflow: hidden;
  transition: all var(--transition-fast);
  background: var(--color-bg-tertiary);
}

.crop-item:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.crop-item.selected {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

.crop-item img {
  width: 100%;
  height: 130px;
  object-fit: cover;
}

.crop-name {
  font-size: 11px;
  padding: 6px;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--color-text-primary);
  background: var(--color-bg-primary);
}

.selected-crops {
  padding: 0 20px 20px;
  border-top: 1px solid var(--color-border);
}

.selected-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 16px;
}

.selected-crops h3 {
  margin: 0;
  font-size: 14px;
  color: var(--color-text-primary);
  font-weight: 600;
}

.selected-gallery {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin: 12px 0;
}

.selected-item {
  position: relative;
  width: 150px;
  height: 150px;
  cursor: pointer;
  transition: all var(--transition-fast);
}

.selected-item:hover {
  transform: scale(1.03);
}

.selected-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 10px;
  box-shadow: var(--shadow-md);
}

.remove-badge {
  position: absolute;
  top: -8px;
  right: -8px;
  background: var(--color-danger);
  color: white;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: bold;
  box-shadow: var(--shadow-md);
}

.sku-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
  grid-auto-rows: minmax(150px, auto);
  gap: 12px;
  padding: 20px;
  max-height: 360px;
  overflow-y: auto;
}

.sku-gallery::-webkit-scrollbar {
  width: 6px;
}

.sku-gallery::-webkit-scrollbar-track {
  background: var(--color-bg-tertiary);
  border-radius: 3px;
}

.sku-gallery::-webkit-scrollbar-thumb {
  background: var(--color-border);
  border-radius: 3px;
}

.sku-item {
  cursor: pointer;
  border: 2px solid var(--color-border);
  border-radius: 8px;
  overflow: hidden;
  background: var(--color-bg-tertiary);
  transition: all var(--transition-fast);
}

.sku-item:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.sku-item.selected {
  border-color: var(--color-primary);
  background: rgba(102, 126, 234, 0.1);
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

.sku-cover {
  width: 100% !important;
  min-height: 85px !important;
  height: 85px !important;
  background: var(--color-bg-tertiary) !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  flex-shrink: 0 !important;
}

.sku-cover img {
  width: 100% !important;
  min-height: 85px !important;
  height: 100% !important;
  object-fit: cover !important;
  flex-shrink: 0 !important;
}

.no-cover {
  font-size: 28px;
  color: var(--color-text-tertiary);
}

.sku-info {
  padding: 8px 10px;
  background: var(--color-bg-primary);
}

.sku-id {
  font-family: monospace;
  font-size: 12px;
  color: var(--color-primary);
  font-weight: 600;
}

.sku-name {
  font-size: 12px;
  color: var(--color-text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin: 2px 0;
}

.sku-count {
  font-size: 11px;
  color: var(--color-text-secondary);
}

.action-hint {
  padding: 14px 20px;
  background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(251, 191, 36, 0.15) 100%);
  border-left: 4px solid var(--color-warning);
  margin: 0 20px 16px;
  border-radius: 0 6px 6px 0;
  font-size: 13px;
  color: var(--color-text-primary);
}

.action-buttons {
  display: flex;
  gap: 10px;
  padding: 0 20px 20px;
  flex-wrap: wrap;
}

.sku-detail {
  border-top: 1px solid var(--color-border);
  padding: 0;
}

.detail-header {
  padding: 16px 20px;
  background: var(--color-bg-tertiary);
  border-bottom: 1px solid var(--color-border);
}

.detail-header h3 {
  margin: 0;
  font-size: 14px;
  color: var(--color-text-primary);
  font-weight: 600;
}

.sku-info-detail {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.sku-name-detail {
  font-size: 13px;
  color: var(--color-text-primary);
  font-weight: 500;
}

.sku-count-detail {
  font-size: 12px;
  color: var(--color-primary);
  background: rgba(102, 126, 234, 0.1);
  padding: 2px 8px;
  border-radius: 4px;
}

.sku-detail-images {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
  gap: 10px;
  padding: 20px;
  max-height: 240px;
  overflow-y: auto;
}

.sku-detail-images::-webkit-scrollbar {
  width: 6px;
}

.sku-detail-images::-webkit-scrollbar-track {
  background: var(--color-bg-tertiary);
  border-radius: 3px;
}

.sku-detail-images::-webkit-scrollbar-thumb {
  background: var(--color-border);
  border-radius: 3px;
}

.detail-image {
  cursor: pointer;
  border: 2px solid transparent;
  border-radius: 6px;
  overflow: hidden;
  transition: all var(--transition-fast);
  background: var(--color-bg-tertiary);
}

.detail-image:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.detail-image.selected {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

/* 步骤进度条 */
.step-progress {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 16px;
  padding: 12px 0;
}

.step-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
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
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 14px;
  background: var(--color-bg-tertiary);
  border: 2px solid var(--color-border);
  transition: all 0.3s ease;
}

.step-item.active .step-icon {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: white;
  animation: pulse 1.5s infinite;
}

.step-item.completed .step-icon {
  background: #22c55e;
  border-color: #22c55e;
  color: white;
}

.step-label {
  font-size: 12px;
  color: var(--color-text-secondary);
}

.step-item.active .step-label,
.step-item.completed .step-label {
  color: var(--color-text-primary);
  font-weight: 500;
}

.step-divider {
  width: 60px;
  height: 3px;
  background: var(--color-border);
  margin: 0 8px;
  margin-bottom: 24px;
  transition: background 0.3s ease;
}

.step-divider.active {
  background: #22c55e;
}

@keyframes pulse {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4);
  }
  50% {
    box-shadow: 0 0 0 8px rgba(102, 126, 234, 0);
  }
}

/* 结果详情 */
.result-detail {
  margin-top: 12px;
  padding: 12px;
  border-radius: 8px;
}

.success-box {
  background: rgba(34, 197, 94, 0.1);
  color: #22c55e;
  font-weight: 500;
  text-align: center;
  font-size: 14px;
}

.error-box {
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  font-weight: 500;
  font-size: 13px;
}

/* 统计行 */
.stats-row {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 12px;
  color: var(--color-text-secondary);
}

.stats-row .stat-item {
  display: flex;
  align-items: center;
}

.stats-row .stat-value {
  color: var(--color-primary);
  font-weight: 600;
  margin: 0 2px;
}

.stats-row .stat-divider {
  color: var(--color-border);
}

/* 输出日志 */
.output-log {
  margin-top: 12px;
  padding: 10px;
  background: var(--color-bg-tertiary);
  border-radius: 6px;
  max-height: 200px;
  overflow: auto;
  font-size: 11px;
  line-height: 1.4;
}

.output-log pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: 'Consolas', 'Monaco', monospace;
  color: var(--color-text-secondary);
}

.detail-image img {
  width: 100%;
  height: 70px;
  object-fit: cover;
}

.image-name {
  font-size: 10px;
  padding: 4px;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--color-text-primary);
  background: var(--color-bg-primary);
}

.build-panel {
  background: var(--color-bg-primary);
  padding: 16px 20px;
  border-radius: 12px;
  box-shadow: var(--shadow-md);
  margin-bottom: 20px;
}

.build-panel .panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.build-panel h3 {
  font-size: 14px;
  margin: 0;
}

.build-actions {
  display: flex;
  gap: 10px;
  margin-bottom: 12px;
}

.build-actions .btn {
  flex: 1;
}

.build-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: var(--color-bg-tertiary);
  border-radius: 8px;
  margin-bottom: 12px;
  font-size: 13px;
}

.status-indicator {
  width: 10px;
  height: 10px;
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

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.library-info {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
}

.info-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.info-label {
  font-size: 13px;
  color: var(--color-text-secondary);
}

.info-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-primary);
  font-family: monospace;
}

.log-panel {
  background: var(--color-bg-primary);
  padding: 0;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: var(--shadow-md);
}

.log-panel h3 {
  font-size: 14px;
  margin: 0;
}

.logs {
  max-height: 160px;
  overflow-y: auto;
  font-size: 12px;
  font-family: monospace;
  padding: 16px 20px;
}

.logs::-webkit-scrollbar {
  width: 6px;
}

.logs::-webkit-scrollbar-track {
  background: var(--color-bg-tertiary);
  border-radius: 3px;
}

.logs::-webkit-scrollbar-thumb {
  background: var(--color-border);
  border-radius: 3px;
}

.log-item {
  padding: 6px 0;
  border-bottom: 1px solid var(--color-border-light);
  color: var(--color-text-primary);
}

.log-item:last-child {
  border-bottom: none;
}

.toast {
  position: fixed;
  bottom: 30px;
  right: 30px;
  padding: 14px 28px;
  border-radius: 8px;
  color: white;
  font-size: 14px;
  animation: slideIn 0.3s;
  box-shadow: var(--shadow-lg);
  z-index: 1000;
}

.toast.info {
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%);
}

.toast.success {
  background: linear-gradient(135deg, var(--color-success) 0%, #059669 100%);
}

.toast.error {
  background: linear-gradient(135deg, var(--color-danger) 0%, #dc2626 100%);
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}


</style>
