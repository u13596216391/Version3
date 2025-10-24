<template>
  <div class="data-view-page">
    <!-- 页面标题 -->
    <div class="page-header">
      <h2 class="title">数据查看与管理</h2>
      <p class="subtitle">上传CSV/ZIP文件，查看和筛选微震与支架阻力数据</p>
    </div>

    <!-- 标签页 -->
    <div class="tabs-container">
      <button 
        v-for="tab in tabs" 
        :key="tab.key"
        :class="['tab-btn', { active: activeTab === tab.key }]"
        @click="activeTab = tab.key"
      >
        {{ tab.label }}
      </button>
    </div>

    <!-- 文件上传标签页 -->
    <div v-if="activeTab === 'upload'" class="tab-content">
      <div class="upload-section">
        <h3 class="section-title">上传数据文件</h3>
        
        <div class="upload-form">
          <div class="form-group">
            <label>数据类型:</label>
            <select v-model="uploadForm.dataType" class="select-input">
              <option value="microseismic">微震数据</option>
              <option value="support_resistance">支架阻力</option>
            </select>
          </div>

          <div class="form-group">
            <label>选择文件 (CSV或ZIP):</label>
            <input 
              type="file" 
              @change="handleFileSelect" 
              accept=".csv,.zip"
              class="file-input"
              ref="fileInput"
            />
            <p class="file-hint">支持CSV和ZIP格式文件</p>
          </div>

          <div v-if="uploadForm.file" class="file-info">
            <p><strong>文件名:</strong> {{ uploadForm.file.name }}</p>
            <p><strong>文件大小:</strong> {{ formatFileSize(uploadForm.file.size) }}</p>
            <p><strong>文件类型:</strong> {{ uploadForm.file.type || '未知' }}</p>
          </div>

          <button 
            @click="uploadFile" 
            :disabled="!uploadForm.file || uploading"
            class="btn-upload"
          >
            {{ uploading ? '上传中...' : '开始上传' }}
          </button>
        </div>

        <!-- 上传结果 -->
        <div v-if="uploadResult" class="upload-result" :class="{ success: uploadResult.success, error: !uploadResult.success }">
          <div class="result-icon">{{ uploadResult.success ? '✓' : '✗' }}</div>
          <div class="result-content">
            <p class="result-message">{{ uploadResult.message }}</p>
            <div v-if="uploadResult.success && uploadResult.data">
              <p>解析记录数: <strong>{{ uploadResult.data.parsed_count }}</strong></p>
              <p>上传时间: {{ formatDateTime(uploadResult.data.upload_time) }}</p>
            </div>
            <p v-if="uploadResult.error" class="error-detail">{{ uploadResult.error }}</p>
          </div>
        </div>

        <!-- 上传历史 -->
        <div v-if="uploadedFiles.length > 0" class="upload-history">
          <h4 class="section-subtitle">上传历史</h4>
          <div class="files-list">
            <div v-for="file in uploadedFiles" :key="file.id" class="file-item">
              <div class="file-icon">📄</div>
              <div class="file-details">
                <p class="file-name">{{ file.filename }}</p>
                <p class="file-meta">
                  {{ file.data_type_display }} | {{ formatFileSize(file.file_size) }} | 
                  {{ formatDateTime(file.upload_time) }}
                </p>
              </div>
              <div class="file-status" :class="file.parse_status">
                {{ file.parse_status === 'success' ? `✓ ${file.parsed_count}条` : file.parse_status }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 微震数据标签页 -->
    <div v-if="activeTab === 'microseismic'" class="tab-content">
      <div class="data-section">
        <div class="section-header">
          <h3 class="section-title">微震数据列表</h3>
          <button @click="toggleFilters('microseismic')" class="btn-filter">
            {{ showMicroseismicFilters ? '隐藏筛选' : '显示筛选' }}
          </button>
        </div>

        <!-- 筛选器 -->
        <div v-if="showMicroseismicFilters" class="filters-panel">
          <div class="filter-row">
            <div class="filter-group">
              <label>开始日期:</label>
              <input v-model="microseismicFilters.startDate" type="date" class="date-input" />
            </div>
            <div class="filter-group">
              <label>结束日期:</label>
              <input v-model="microseismicFilters.endDate" type="date" class="date-input" />
            </div>
            <div class="filter-group">
              <label>源文件:</label>
              <input v-model="microseismicFilters.sourceFile" type="text" class="text-input" placeholder="文件名" />
            </div>
          </div>
          <div class="filter-row">
            <div class="filter-group">
              <label>X坐标范围:</label>
              <input v-model="microseismicFilters.minX" type="number" class="number-input" placeholder="最小值" />
              <span class="range-separator">~</span>
              <input v-model="microseismicFilters.maxX" type="number" class="number-input" placeholder="最大值" />
            </div>
            <div class="filter-group">
              <label>Y坐标范围:</label>
              <input v-model="microseismicFilters.minY" type="number" class="number-input" placeholder="最小值" />
              <span class="range-separator">~</span>
              <input v-model="microseismicFilters.maxY" type="number" class="number-input" placeholder="最大值" />
            </div>
          </div>
          <div class="filter-actions">
            <button @click="fetchMicroseismicData" class="btn-apply">应用筛选</button>
            <button @click="resetMicroseismicFilters" class="btn-reset">重置</button>
          </div>
        </div>

        <!-- 数据统计 -->
        <div v-if="microseismicData.total_count > 0" class="data-stats">
          <span>共 <strong>{{ microseismicData.total_count }}</strong> 条记录</span>
          <span>第 <strong>{{ microseismicData.page }}</strong> / {{ microseismicData.total_pages }} 页</span>
        </div>

        <!-- 数据表格 -->
        <div v-if="loadingMicroseismic" class="loading">
          <div class="spinner"></div>
          <p>加载中...</p>
        </div>

        <div v-else-if="microseismicData.data.length > 0" class="data-table-container">
          <table class="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>时间</th>
                <th>X坐标</th>
                <th>Y坐标</th>
                <th>Z坐标</th>
                <th>能量</th>
                <th>震级</th>
                <th>源文件</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in microseismicData.data" :key="item.id">
                <td>{{ item.id }}</td>
                <td>{{ formatDateTime(item.timestamp) }}</td>
                <td>{{ item.event_x.toFixed(2) }}</td>
                <td>{{ item.event_y.toFixed(2) }}</td>
                <td>{{ item.event_z?.toFixed(2) || '--' }}</td>
                <td>{{ item.energy ? item.energy.toExponential(2) : '--' }}</td>
                <td>{{ item.magnitude?.toFixed(2) || '--' }}</td>
                <td class="truncate">{{ item.source_file || '--' }}</td>
              </tr>
            </tbody>
          </table>

          <!-- 分页 -->
          <div class="pagination">
            <button 
              @click="changeMicroseismicPage(microseismicData.page - 1)" 
              :disabled="!microseismicData.has_previous"
              class="page-btn"
            >
              上一页
            </button>
            <span class="page-info">{{ microseismicData.page }} / {{ microseismicData.total_pages }}</span>
            <button 
              @click="changeMicroseismicPage(microseismicData.page + 1)" 
              :disabled="!microseismicData.has_next"
              class="page-btn"
            >
              下一页
            </button>
          </div>
        </div>

        <div v-else class="empty-state">
          <p>暂无数据，请先上传文件</p>
        </div>
      </div>
    </div>

    <!-- 支架阻力数据标签页 -->
    <div v-if="activeTab === 'support'" class="tab-content">
      <div class="data-section">
        <div class="section-header">
          <h3 class="section-title">支架阻力数据列表</h3>
          <button @click="toggleFilters('support')" class="btn-filter">
            {{ showSupportFilters ? '隐藏筛选' : '显示筛选' }}
          </button>
        </div>

        <!-- 筛选器 -->
        <div v-if="showSupportFilters" class="filters-panel">
          <div class="filter-row">
            <div class="filter-group">
              <label>开始日期:</label>
              <input v-model="supportFilters.startDate" type="date" class="date-input" />
            </div>
            <div class="filter-group">
              <label>结束日期:</label>
              <input v-model="supportFilters.endDate" type="date" class="date-input" />
            </div>
            <div class="filter-group">
              <label>测站ID:</label>
              <input v-model="supportFilters.stationId" type="text" class="text-input" placeholder="STATION_001" />
            </div>
            <div class="filter-group">
              <label>源文件:</label>
              <input v-model="supportFilters.sourceFile" type="text" class="text-input" placeholder="文件名" />
            </div>
          </div>
          <div class="filter-row">
            <div class="filter-group">
              <label>阻力范围 (MPa):</label>
              <input v-model="supportFilters.minResistance" type="number" class="number-input" placeholder="最小值" />
              <span class="range-separator">~</span>
              <input v-model="supportFilters.maxResistance" type="number" class="number-input" placeholder="最大值" />
            </div>
          </div>
          <div class="filter-actions">
            <button @click="fetchSupportData" class="btn-apply">应用筛选</button>
            <button @click="resetSupportFilters" class="btn-reset">重置</button>
          </div>
        </div>

        <!-- 数据统计 -->
        <div v-if="supportData.total_count > 0" class="data-stats">
          <span>共 <strong>{{ supportData.total_count }}</strong> 条记录</span>
          <span>第 <strong>{{ supportData.page }}</strong> / {{ supportData.total_pages }} 页</span>
        </div>

        <!-- 数据表格 -->
        <div v-if="loadingSupport" class="loading">
          <div class="spinner"></div>
          <p>加载中...</p>
        </div>

        <div v-else-if="supportData.data.length > 0" class="data-table-container">
          <table class="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>时间</th>
                <th>测站ID</th>
                <th>阻力值 (MPa)</th>
                <th>压力等级</th>
                <th>源文件</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in supportData.data" :key="item.id">
                <td>{{ item.id }}</td>
                <td>{{ formatDateTime(item.timestamp) }}</td>
                <td>{{ item.station_id }}</td>
                <td>{{ item.resistance.toFixed(2) }}</td>
                <td>{{ item.pressure_level || '--' }}</td>
                <td class="truncate">{{ item.source_file || '--' }}</td>
              </tr>
            </tbody>
          </table>

          <!-- 分页 -->
          <div class="pagination">
            <button 
              @click="changeSupportPage(supportData.page - 1)" 
              :disabled="!supportData.has_previous"
              class="page-btn"
            >
              上一页
            </button>
            <span class="page-info">{{ supportData.page }} / {{ supportData.total_pages }}</span>
            <button 
              @click="changeSupportPage(supportData.page + 1)" 
              :disabled="!supportData.has_next"
              class="page-btn"
            >
              下一页
            </button>
          </div>
        </div>

        <div v-else class="empty-state">
          <p>暂无数据，请先上传文件</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const activeTab = ref('upload')
const tabs = [
  { key: 'upload', label: '📤 文件上传' },
  { key: 'microseismic', label: '🌍 微震数据' },
  { key: 'support', label: '🔧 支架阻力' },
]

// 上传相关
const uploadForm = ref({
  dataType: 'microseismic',
  file: null
})
const fileInput = ref(null)
const uploading = ref(false)
const uploadResult = ref(null)
const uploadedFiles = ref([])

// 微震数据相关
const showMicroseismicFilters = ref(false)
const loadingMicroseismic = ref(false)
const microseismicFilters = ref({
  startDate: '',
  endDate: '',
  sourceFile: '',
  minX: '',
  maxX: '',
  minY: '',
  maxY: ''
})
const microseismicData = ref({
  data: [],
  total_count: 0,
  page: 1,
  page_size: 50,
  total_pages: 0,
  has_next: false,
  has_previous: false
})

// 支架阻力数据相关
const showSupportFilters = ref(false)
const loadingSupport = ref(false)
const supportFilters = ref({
  startDate: '',
  endDate: '',
  stationId: '',
  sourceFile: '',
  minResistance: '',
  maxResistance: ''
})
const supportData = ref({
  data: [],
  total_count: 0,
  page: 1,
  page_size: 50,
  total_pages: 0,
  has_next: false,
  has_previous: false
})

// 文件上传
const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file) {
    uploadForm.value.file = file
    uploadResult.value = null
  }
}

const uploadFile = async () => {
  if (!uploadForm.value.file) return

  uploading.value = true
  uploadResult.value = null

  const formData = new FormData()
  formData.append('file', uploadForm.value.file)
  formData.append('data_type', uploadForm.value.dataType)

  try {
    const response = await axios.post('/api/data/upload/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })

    uploadResult.value = {
      success: response.data.success,
      message: response.data.message || '上传成功',
      data: response.data.uploaded_file
    }

    // 重新加载上传历史
    fetchUploadedFiles()

    // 清空文件选择
    uploadForm.value.file = null
    if (fileInput.value) {
      fileInput.value.value = ''
    }
  } catch (error) {
    uploadResult.value = {
      success: false,
      message: '上传失败',
      error: error.response?.data?.error || error.message
    }
  } finally {
    uploading.value = false
  }
}

const fetchUploadedFiles = async () => {
  try {
    const response = await axios.get('/api/data/uploaded-files/')
    if (response.data.success) {
      uploadedFiles.value = response.data.files
    }
  } catch (error) {
    console.error('获取上传历史失败:', error)
  }
}

// 微震数据
const toggleFilters = (type) => {
  if (type === 'microseismic') {
    showMicroseismicFilters.value = !showMicroseismicFilters.value
  } else {
    showSupportFilters.value = !showSupportFilters.value
  }
}

const fetchMicroseismicData = async (page = 1) => {
  loadingMicroseismic.value = true

  try {
    const params = {
      page,
      page_size: 50,
      ...Object.fromEntries(
        Object.entries(microseismicFilters.value).filter(([_, v]) => v !== '')
      )
    }

    const response = await axios.get('/api/data/microseismic/', { params })

    if (response.data.success) {
      microseismicData.value = response.data
    }
  } catch (error) {
    console.error('获取微震数据失败:', error)
  } finally {
    loadingMicroseismic.value = false
  }
}

const resetMicroseismicFilters = () => {
  microseismicFilters.value = {
    startDate: '',
    endDate: '',
    sourceFile: '',
    minX: '',
    maxX: '',
    minY: '',
    maxY: ''
  }
  fetchMicroseismicData()
}

const changeMicroseismicPage = (page) => {
  fetchMicroseismicData(page)
}

// 支架阻力数据
const fetchSupportData = async (page = 1) => {
  loadingSupport.value = true

  try {
    const params = {
      page,
      page_size: 50,
      ...Object.fromEntries(
        Object.entries(supportFilters.value).filter(([_, v]) => v !== '')
      )
    }

    const response = await axios.get('/api/data/support-resistance/', { params })

    if (response.data.success) {
      supportData.value = response.data
    }
  } catch (error) {
    console.error('获取支架阻力数据失败:', error)
  } finally {
    loadingSupport.value = false
  }
}

const resetSupportFilters = () => {
  supportFilters.value = {
    startDate: '',
    endDate: '',
    stationId: '',
    sourceFile: '',
    minResistance: '',
    maxResistance: ''
  }
  fetchSupportData()
}

const changeSupportPage = (page) => {
  fetchSupportData(page)
}

// 工具函数
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
}

const formatDateTime = (timestamp) => {
  if (!timestamp) return '--'
  const date = new Date(timestamp)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

// 初始化
onMounted(() => {
  fetchUploadedFiles()
  fetchMicroseismicData()
  fetchSupportData()
})
</script>

<style scoped>
.data-view-page {
  min-height: 100vh;
  padding: 2rem;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

.page-header {
  text-align: center;
  margin-bottom: 2rem;
}

.title {
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.5rem;
}

.subtitle {
  color: #94a3b8;
  font-size: 1rem;
}

/* 标签页 */
.tabs-container {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
  background: rgba(30, 41, 59, 0.8);
  padding: 1rem;
  border-radius: 12px;
  backdrop-filter: blur(10px);
}

.tab-btn {
  flex: 1;
  padding: 0.75rem 1.5rem;
  background: rgba(51, 65, 85, 0.6);
  border: none;
  border-radius: 8px;
  color: #cbd5e1;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.tab-btn:hover {
  background: rgba(71, 85, 105, 0.8);
  transform: translateY(-2px);
}

.tab-btn.active {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

.tab-content {
  background: rgba(30, 41, 59, 0.8);
  border-radius: 12px;
  padding: 2rem;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* 上传部分 */
.section-title {
  color: #e2e8f0;
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
}

.section-subtitle {
  color: #cbd5e1;
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 1rem;
}

.upload-form {
  background: rgba(51, 65, 85, 0.4);
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  color: #cbd5e1;
  font-size: 0.875rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.select-input {
  width: 100%;
  max-width: 300px;
  padding: 0.5rem 1rem;
  background: rgba(51, 65, 85, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 6px;
  color: #e2e8f0;
  font-size: 0.875rem;
}

.file-input {
  display: block;
  width: 100%;
  max-width: 500px;
  padding: 0.5rem;
  background: rgba(51, 65, 85, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 6px;
  color: #e2e8f0;
  font-size: 0.875rem;
}

.file-hint {
  color: #94a3b8;
  font-size: 0.75rem;
  margin-top: 0.5rem;
}

.file-info {
  background: rgba(51, 65, 85, 0.4);
  padding: 1rem;
  border-radius: 6px;
  margin-bottom: 1rem;
}

.file-info p {
  color: #cbd5e1;
  margin-bottom: 0.5rem;
}

.file-info strong {
  color: #60a5fa;
}

.btn-upload {
  padding: 0.75rem 2rem;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  border: none;
  border-radius: 6px;
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-upload:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

.btn-upload:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.upload-result {
  display: flex;
  gap: 1rem;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.upload-result.success {
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.3);
}

.upload-result.error {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.result-icon {
  font-size: 2rem;
}

.upload-result.success .result-icon {
  color: #10b981;
}

.upload-result.error .result-icon {
  color: #ef4444;
}

.result-content {
  flex: 1;
}

.result-message {
  color: #e2e8f0;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.result-content p {
  color: #cbd5e1;
  font-size: 0.875rem;
  margin-bottom: 0.25rem;
}

.result-content strong {
  color: #60a5fa;
}

.error-detail {
  color: #fca5a5 !important;
  margin-top: 0.5rem;
}

/* 上传历史 */
.upload-history {
  margin-top: 2rem;
}

.files-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: rgba(51, 65, 85, 0.4);
  border-radius: 8px;
  transition: all 0.3s ease;
}

.file-item:hover {
  background: rgba(51, 65, 85, 0.6);
  transform: translateX(4px);
}

.file-icon {
  font-size: 2rem;
}

.file-details {
  flex: 1;
}

.file-name {
  color: #e2e8f0;
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.file-meta {
  color: #94a3b8;
  font-size: 0.75rem;
}

.file-status {
  padding: 0.25rem 0.75rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
}

.file-status.success {
  background: rgba(16, 185, 129, 0.2);
  color: #10b981;
}

.file-status.failed {
  background: rgba(239, 68, 68, 0.2);
  color: #ef4444;
}

.file-status.pending {
  background: rgba(251, 191, 36, 0.2);
  color: #fbbf24;
}

/* 数据部分 */
.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.btn-filter {
  padding: 0.5rem 1.5rem;
  background: rgba(59, 130, 246, 0.2);
  border: 1px solid rgba(59, 130, 246, 0.4);
  border-radius: 6px;
  color: #60a5fa;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-filter:hover {
  background: rgba(59, 130, 246, 0.3);
}

/* 筛选器 */
.filters-panel {
  background: rgba(51, 65, 85, 0.4);
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 1.5rem;
}

.filter-row {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}

.filter-row:last-child {
  margin-bottom: 0;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  flex: 1;
  min-width: 200px;
}

.filter-group label {
  color: #cbd5e1;
  font-size: 0.875rem;
  font-weight: 500;
}

.date-input,
.text-input,
.number-input {
  padding: 0.5rem;
  background: rgba(51, 65, 85, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 6px;
  color: #e2e8f0;
  font-size: 0.875rem;
}

.number-input {
  max-width: 120px;
}

.range-separator {
  color: #94a3b8;
  margin: 0 0.5rem;
  align-self: flex-end;
  padding-bottom: 0.5rem;
}

.filter-actions {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
}

.btn-apply,
.btn-reset {
  padding: 0.5rem 1.5rem;
  border: none;
  border-radius: 6px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-apply {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
}

.btn-apply:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

.btn-reset {
  background: rgba(100, 116, 139, 0.4);
  color: #cbd5e1;
}

.btn-reset:hover {
  background: rgba(100, 116, 139, 0.6);
}

/* 数据统计 */
.data-stats {
  display: flex;
  gap: 2rem;
  padding: 1rem;
  background: rgba(51, 65, 85, 0.4);
  border-radius: 6px;
  margin-bottom: 1rem;
  color: #cbd5e1;
  font-size: 0.875rem;
}

.data-stats strong {
  color: #60a5fa;
  font-size: 1.125rem;
}

/* 数据表格 */
.data-table-container {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
}

.data-table thead {
  background: rgba(30, 41, 59, 0.8);
}

.data-table th {
  padding: 0.75rem;
  text-align: left;
  color: #cbd5e1;
  font-weight: 600;
  font-size: 0.875rem;
  border-bottom: 2px solid rgba(148, 163, 184, 0.2);
}

.data-table td {
  padding: 0.75rem;
  color: #e2e8f0;
  font-size: 0.875rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}

.data-table tbody tr:hover {
  background: rgba(51, 65, 85, 0.4);
}

.data-table td.truncate {
  max-width: 200px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 分页 */
.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
  margin-top: 1.5rem;
}

.page-btn {
  padding: 0.5rem 1rem;
  background: rgba(59, 130, 246, 0.2);
  border: 1px solid rgba(59, 130, 246, 0.4);
  border-radius: 6px;
  color: #60a5fa;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.page-btn:hover:not(:disabled) {
  background: rgba(59, 130, 246, 0.3);
}

.page-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.page-info {
  color: #cbd5e1;
  font-weight: 600;
}

/* 加载和空状态 */
.loading,
.empty-state {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 3rem;
  color: #94a3b8;
  gap: 1rem;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(59, 130, 246, 0.2);
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
