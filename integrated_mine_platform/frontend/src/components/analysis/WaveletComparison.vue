<template>
  <div class="wavelet-comparison">
    <div class="control-panel">
      <div class="form-group">
        <label>选择数据集:</label>
        <select v-model="dataset" @change="onDatasetChange" class="select-input">
          <option value="">请选择数据集</option>
          <option v-for="ds in datasets" :key="ds.id" :value="ds.id">
            {{ ds.name }} ({{ ds.count }}条)
          </option>
        </select>
      </div>
      <div class="form-group">
        <label>测站ID:</label>
        <select v-model="stationId" class="select-input" :disabled="!dataset">
          <option value="">请选择测站</option>
          <option v-for="station in stations" :key="station.station_id" :value="station.station_id">
            {{ station.station_id }} ({{ station.count }}条数据)
          </option>
        </select>
      </div>
      <div class="form-group">
        <label>开始日期:</label>
        <input v-model="startDate" type="date" class="date-input" />
      </div>
      <div class="form-group">
        <label>结束日期:</label>
        <input v-model="endDate" type="date" class="date-input" />
      </div>
      <button @click="fetchComparison" class="btn-analyze" :disabled="loading">
        {{ loading ? '对比中...' : '开始对比' }}
      </button>
    </div>

    <!-- 小波选择 -->
    <div class="wavelet-selector">
      <label class="selector-label">选择要对比的小波基函数:</label>
      <div class="checkbox-group">
        <label v-for="w in availableWavelets" :key="w.value" class="checkbox-label">
          <input 
            type="checkbox" 
            :value="w.value" 
            v-model="selectedWavelets"
            class="checkbox-input"
          />
          <span class="checkbox-text">{{ w.label }}</span>
        </label>
      </div>
    </div>

    <div v-if="loading" class="loading">
      <div class="spinner"></div>
      <p>正在对比不同小波基函数的分析效果...</p>
    </div>

    <div v-else-if="result" class="result-panel">
      <!-- 对比统计 -->
      <div class="comparison-header">
        <h3 class="section-title">小波基函数对比分析</h3>
        <p class="section-subtitle">
          测站: <strong>{{ result.station_id }}</strong> | 
          数据点数: <strong>{{ result.count }}</strong>
        </p>
      </div>

      <div class="comparison-grid">
        <div 
          v-for="item in result.comparison" 
          :key="item.wavelet"
          class="comparison-card"
        >
          <div class="card-header">
            <h4 class="wavelet-name">{{ getWaveletLabel(item.wavelet) }}</h4>
          </div>
          <div class="card-body">
            <div class="metric-row">
              <span class="metric-label">检测事件数:</span>
              <span class="metric-value">{{ item.event_count }}</span>
            </div>
            <div class="metric-row">
              <span class="metric-label">异常阈值:</span>
              <span class="metric-value">{{ item.threshold.toFixed(2) }} MPa</span>
            </div>
            <div class="metric-row">
              <span class="metric-label">信噪比估计:</span>
              <span class="metric-value">{{ item.snr.toFixed(2) }} dB</span>
            </div>
          </div>
          <div class="card-footer">
            <button @click="selectWavelet(item.wavelet)" class="btn-select">
              查看详细分析
            </button>
          </div>
        </div>
      </div>

      <!-- 对比图表 -->
      <div class="charts-section">
        <div class="chart-container">
          <h4 class="chart-title">事件检测数量对比</h4>
          <div class="bar-chart">
            <div 
              v-for="item in result.comparison" 
              :key="'bar-' + item.wavelet"
              class="bar-item"
            >
              <div class="bar-label">{{ item.wavelet }}</div>
              <div class="bar-wrapper">
                <div 
                  class="bar-fill" 
                  :style="{ width: getBarWidth(item.event_count, maxEventCount) }"
                ></div>
                <span class="bar-value">{{ item.event_count }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="chart-container">
          <h4 class="chart-title">信噪比对比</h4>
          <div class="bar-chart">
            <div 
              v-for="item in result.comparison" 
              :key="'snr-' + item.wavelet"
              class="bar-item"
            >
              <div class="bar-label">{{ item.wavelet }}</div>
              <div class="bar-wrapper">
                <div 
                  class="bar-fill snr" 
                  :style="{ width: getBarWidth(item.snr, maxSNR) }"
                ></div>
                <span class="bar-value">{{ item.snr.toFixed(2) }} dB</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 推荐说明 -->
      <div class="recommendation-box">
        <h4 class="recommendation-title">💡 选择建议</h4>
        <ul class="recommendation-list">
          <li><strong>Daubechies (db4, db8):</strong> 最常用的小波基，适合大多数工程信号分析</li>
          <li><strong>Symlets (sym4, sym8):</strong> 对称性更好，相位特性优秀</li>
          <li><strong>Coiflets (coif4):</strong> 在时域和频域都具有良好的局部化特性</li>
          <li><strong>高信噪比:</strong> 表示去噪效果好，信号保真度高</li>
          <li><strong>适中事件数:</strong> 过多可能误报，过少可能漏检</li>
        </ul>
      </div>
    </div>

    <div v-else-if="error" class="error-message">
      <i class="error-icon">⚠</i>
      {{ error }}
    </div>

    <div v-else class="empty-state">
      <p>请选择数据集、输入测站ID、选择日期范围和小波基函数，然后点击"开始对比"按钮</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import axios from 'axios'
import { useDatasetStore } from '@/stores/datasetStore'

const datasetStore = useDatasetStore()

const datasets = ref([])
const dataset = ref('')
const stations = ref([])
const stationId = ref('')
const startDate = ref('')
const endDate = ref('')
const loading = ref(false)
const result = ref(null)
const error = ref(null)

// 监听数据集选择变化，保存到store
watch(dataset, (newValue) => {
  if (newValue) {
    const selected = datasets.value.find(d => d.id === newValue)
    if (selected) {
      datasetStore.setSupportDataset(selected)
    }
  }
})

const availableWavelets = [
  { value: 'db4', label: 'Daubechies 4 (db4)' },
  { value: 'db8', label: 'Daubechies 8 (db8)' },
  { value: 'sym4', label: 'Symlets 4 (sym4)' },
  { value: 'sym8', label: 'Symlets 8 (sym8)' },
  { value: 'coif4', label: 'Coiflets 4 (coif4)' },
]

const selectedWavelets = ref(['db4', 'sym4', 'coif4'])

// 加载数据集列表
const fetchDatasets = async () => {
  try {
    const response = await axios.get('/api/data/datasets/', {
      params: { data_type: 'support_resistance' }
    })
    if (response.data.datasets) {
      datasets.value = response.data.datasets
    }
  } catch (err) {
    console.error('加载数据集失败:', err)
  }
}

// 当数据集改变时，加载该数据集的测站列表
const onDatasetChange = async () => {
  stationId.value = ''
  stations.value = []
  
  if (!dataset.value) {
    return
  }
  
  try {
    const response = await axios.get('/api/data/datasets/stations/', {
      params: { dataset_id: dataset.value }
    })
    if (response.data.success && response.data.stations) {
      stations.value = response.data.stations
    }
  } catch (err) {
    console.error('加载测站列表失败:', err)
  }
}

const maxEventCount = computed(() => {
  if (!result.value?.comparison) return 1
  return Math.max(...result.value.comparison.map(item => item.event_count))
})

const maxSNR = computed(() => {
  if (!result.value?.comparison) return 1
  return Math.max(...result.value.comparison.map(item => item.snr))
})

const getWaveletLabel = (wavelet) => {
  return availableWavelets.find(w => w.value === wavelet)?.label || wavelet
}

const getBarWidth = (value, max) => {
  return `${(value / max * 100).toFixed(1)}%`
}

const fetchComparison = async () => {
  if (!dataset.value || !stationId.value || !startDate.value || !endDate.value) {
    error.value = '请选择数据集、输入测站ID并选择开始和结束日期'
    return
  }

  if (selectedWavelets.value.length === 0) {
    error.value = '请至少选择一个小波基函数'
    return
  }

  loading.value = true
  error.value = null
  result.value = null

  try {
    const params = new URLSearchParams()
    params.append('dataset_id', dataset.value)
    params.append('station_id', stationId.value)
    params.append('start_date', startDate.value)
    params.append('end_date', endDate.value)
    selectedWavelets.value.forEach(w => {
      params.append('wavelets[]', w)
    })

    const response = await axios.get('/api/analysis/support/wavelet-comparison/', {
      params: params
    })
    
    if (response.data.success) {
      result.value = response.data
    } else {
      error.value = response.data.message || '对比分析失败'
    }
  } catch (err) {
    error.value = err.response?.data?.error || '对比分析失败,请重试'
  } finally {
    loading.value = false
  }
}

const selectWavelet = (wavelet) => {
  // 可以触发跳转到DWT分析页面或打开详细分析
  alert(`查看 ${wavelet} 的详细分析功能即将开放`)
}

onMounted(() => {
  fetchDatasets()
})
</script>

<style scoped>
.wavelet-comparison {
  padding: 1rem;
}

.control-panel {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  align-items: flex-end;
  flex-wrap: wrap;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-group label {
  color: #cbd5e1;
  font-size: 0.875rem;
  font-weight: 500;
}

.text-input, .date-input {
  padding: 0.5rem 1rem;
  background: rgba(51, 65, 85, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 6px;
  color: #e2e8f0;
  font-size: 0.875rem;
  min-width: 150px;
}

.text-input::placeholder {
  color: #64748b;
}

.text-input:focus, .date-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
}

.wavelet-selector {
  background: rgba(51, 65, 85, 0.4);
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.selector-label {
  color: #cbd5e1;
  font-size: 0.875rem;
  font-weight: 600;
  display: block;
  margin-bottom: 0.75rem;
}

.checkbox-group {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  color: #e2e8f0;
  font-size: 0.875rem;
}

.checkbox-input {
  width: 18px;
  height: 18px;
  cursor: pointer;
  accent-color: #3b82f6;
}

.checkbox-text {
  user-select: none;
}

.btn-analyze {
  padding: 0.5rem 2rem;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  border: none;
  border-radius: 6px;
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  height: fit-content;
}

.btn-analyze:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

.btn-analyze:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 3rem;
  color: #94a3b8;
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

.result-panel {
  animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.comparison-header {
  background: rgba(51, 65, 85, 0.4);
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.section-title {
  color: #e2e8f0;
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.section-subtitle {
  color: #94a3b8;
  font-size: 0.875rem;
}

.section-subtitle strong {
  color: #60a5fa;
}

.comparison-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.comparison-card {
  background: rgba(51, 65, 85, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  overflow: hidden;
  transition: all 0.3s ease;
}

.comparison-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
  border-color: rgba(59, 130, 246, 0.5);
}

.card-header {
  background: rgba(30, 41, 59, 0.8);
  padding: 1rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}

.wavelet-name {
  color: #60a5fa;
  font-size: 1.125rem;
  font-weight: 600;
  margin: 0;
}

.card-body {
  padding: 1rem;
}

.metric-row {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}

.metric-row:last-child {
  border-bottom: none;
}

.metric-label {
  color: #94a3b8;
  font-size: 0.875rem;
}

.metric-value {
  color: #e2e8f0;
  font-weight: 600;
  font-size: 0.875rem;
}

.card-footer {
  padding: 1rem;
  background: rgba(30, 41, 59, 0.4);
}

.btn-select {
  width: 100%;
  padding: 0.5rem;
  background: rgba(59, 130, 246, 0.2);
  border: 1px solid rgba(59, 130, 246, 0.4);
  border-radius: 6px;
  color: #60a5fa;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-select:hover {
  background: rgba(59, 130, 246, 0.3);
  border-color: rgba(59, 130, 246, 0.6);
}

.charts-section {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 2rem;
  margin-bottom: 2rem;
}

.chart-container {
  background: rgba(51, 65, 85, 0.4);
  padding: 1.5rem;
  border-radius: 8px;
}

.chart-title {
  color: #e2e8f0;
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 1rem;
}

.bar-chart {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.bar-item {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.bar-label {
  color: #cbd5e1;
  font-size: 0.875rem;
  font-weight: 600;
  min-width: 60px;
}

.bar-wrapper {
  flex: 1;
  position: relative;
  height: 30px;
  background: rgba(30, 41, 59, 0.6);
  border-radius: 4px;
  overflow: hidden;
}

.bar-fill {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
  transition: width 0.5s ease;
}

.bar-fill.snr {
  background: linear-gradient(90deg, #10b981 0%, #059669 100%);
}

.bar-value {
  position: absolute;
  right: 0.5rem;
  top: 50%;
  transform: translateY(-50%);
  color: white;
  font-size: 0.75rem;
  font-weight: 600;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
}

.recommendation-box {
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.3);
  border-radius: 8px;
  padding: 1.5rem;
}

.recommendation-title {
  color: #34d399;
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 1rem;
}

.recommendation-list {
  color: #94a3b8;
  font-size: 0.875rem;
  line-height: 1.8;
  margin: 0;
  padding-left: 1.5rem;
}

.recommendation-list li {
  margin-bottom: 0.5rem;
}

.recommendation-list strong {
  color: #cbd5e1;
}

.error-message {
  padding: 1rem;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 8px;
  color: #fca5a5;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.error-icon {
  font-size: 1.5rem;
}

.empty-state {
  text-align: center;
  padding: 3rem;
  color: #94a3b8;
  font-size: 1rem;
}
</style>
