<template>
  <div class="support-dwt-analysis">
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
      <div class="form-group">
        <label>小波基函数:</label>
        <select v-model="wavelet" class="select-input">
          <option value="db4">Daubechies 4 (db4)</option>
          <option value="db8">Daubechies 8 (db8)</option>
          <option value="sym4">Symlets 4 (sym4)</option>
          <option value="sym8">Symlets 8 (sym8)</option>
          <option value="coif4">Coiflets 4 (coif4)</option>
        </select>
      </div>
      <button @click="fetchAnalysis" class="btn-analyze" :disabled="loading">
        {{ loading ? '分析中...' : '开始分析' }}
      </button>
    </div>

    <div v-if="loading" class="loading">
      <div class="spinner"></div>
      <p>正在进行DWT小波分析...</p>
    </div>

    <div v-else-if="result" class="result-panel">
      <!-- 统计信息 -->
      <div class="stats-grid">
        <div class="stat-card">
          <span class="stat-label">数据点数</span>
          <span class="stat-value">{{ result.count }}</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">平均阻力</span>
          <span class="stat-value">{{ result.statistics.mean.toFixed(2) }} MPa</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">标准差</span>
          <span class="stat-value">{{ result.statistics.std.toFixed(2) }} MPa</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">最小/最大</span>
          <span class="stat-value">
            {{ result.statistics.min.toFixed(2) }} / {{ result.statistics.max.toFixed(2) }} MPa
          </span>
        </div>
        <div class="stat-card highlight">
          <span class="stat-label">检测到异常事件</span>
          <span class="stat-value">{{ result.statistics.event_count }}</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">异常阈值</span>
          <span class="stat-value">{{ result.statistics.threshold.toFixed(2) }} MPa</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">分解层数</span>
          <span class="stat-value">{{ result.statistics.decomposition_levels }}</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">小波基函数</span>
          <span class="stat-value">{{ result.wavelet }}</span>
        </div>
      </div>

      <!-- DWT分析图 -->
      <div v-if="result.dwt_plot" class="plot-container">
        <h3 class="plot-title">支架阻力DWT小波分析</h3>
        <img :src="result.dwt_plot" alt="DWT分析图" class="plot-image" />
        <p class="plot-description">
          小波变换(DWT)分析可以同时提供时间和频率信息，用于检测支架阻力的异常变化和趋势。
          上图显示原始数据，中图显示小波去噪结果，下图标注检测到的压力异常事件。
        </p>
      </div>

      <!-- 压力分布图 -->
      <div v-if="result.distribution_plot" class="plot-container">
        <h3 class="plot-title">支架阻力分布直方图</h3>
        <img :src="result.distribution_plot" alt="压力分布图" class="plot-image" />
        <p class="plot-description">
          直方图显示支架阻力值的统计分布，红色虚线为平均值，绿色虚线为中位数。
        </p>
      </div>

      <!-- 异常事件列表 -->
      <div v-if="result.events && result.events.length > 0" class="events-section">
        <h3 class="section-title">检测到的异常事件 (前20个)</h3>
        <div class="events-table-container">
          <table class="events-table">
            <thead>
              <tr>
                <th>序号</th>
                <th>时间</th>
                <th>阻力值 (MPa)</th>
                <th>超过阈值</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(event, idx) in result.events" :key="idx">
                <td>{{ idx + 1 }}</td>
                <td>{{ formatDateTime(event.timestamp) }}</td>
                <td class="value-cell">{{ event.value.toFixed(2) }}</td>
                <td class="threshold-cell">
                  {{ (event.value - result.statistics.threshold).toFixed(2) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div v-else-if="error" class="error-message">
      <i class="error-icon">⚠</i>
      {{ error }}
    </div>

    <div v-else class="empty-state">
      <p>请选择数据集、输入测站ID、选择日期范围并点击"开始分析"按钮</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import axios from 'axios'
import { useDatasetStore } from '@/stores/datasetStore'

const datasetStore = useDatasetStore()

const datasets = ref([])
const dataset = ref('')
const stations = ref([])
const stationId = ref('')
const startDate = ref('')
const endDate = ref('')
const wavelet = ref('db4')
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

const fetchAnalysis = async () => {
  if (!dataset.value || !stationId.value || !startDate.value || !endDate.value) {
    error.value = '请选择数据集、输入测站ID并选择开始和结束日期'
    return
  }

  loading.value = true
  error.value = null
  result.value = null

  try {
    const response = await axios.get('/api/analysis/support/dwt/', {
      params: {
        dataset_id: dataset.value,
        station_id: stationId.value,
        start_date: startDate.value,
        end_date: endDate.value,
        wavelet: wavelet.value
      }
    })
    
    if (response.data.success) {
      result.value = response.data
    } else {
      error.value = response.data.message || '分析失败'
    }
  } catch (err) {
    error.value = err.response?.data?.error || '分析失败,请重试'
  } finally {
    loading.value = false
  }
}

const formatDateTime = (timestamp) => {
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

onMounted(() => {
  fetchDatasets()
})
</script>

<style scoped>
.support-dwt-analysis {
  padding: 1rem;
}

.control-panel {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
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

.text-input, .date-input, .select-input {
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

.text-input:focus, .date-input:focus, .select-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
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

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.stat-card {
  background: rgba(51, 65, 85, 0.6);
  padding: 1rem;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  border: 1px solid rgba(148, 163, 184, 0.1);
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.stat-card.highlight {
  background: rgba(239, 68, 68, 0.1);
  border-color: rgba(239, 68, 68, 0.3);
}

.stat-card.highlight .stat-value {
  color: #f87171;
}

.stat-label {
  color: #94a3b8;
  font-size: 0.875rem;
}

.stat-value {
  color: #60a5fa;
  font-size: 1.5rem;
  font-weight: 700;
}

.plot-container {
  margin-bottom: 2rem;
  background: rgba(51, 65, 85, 0.4);
  padding: 1.5rem;
  border-radius: 8px;
}

.plot-title {
  color: #e2e8f0;
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1rem;
}

.plot-image {
  width: 100%;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.plot-description {
  color: #94a3b8;
  font-size: 0.875rem;
  margin-top: 1rem;
  font-style: italic;
  line-height: 1.6;
}

.events-section {
  background: rgba(51, 65, 85, 0.4);
  padding: 1.5rem;
  border-radius: 8px;
  margin-top: 2rem;
}

.section-title {
  color: #e2e8f0;
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1rem;
}

.events-table-container {
  overflow-x: auto;
}

.events-table {
  width: 100%;
  border-collapse: collapse;
}

.events-table thead {
  background: rgba(30, 41, 59, 0.6);
}

.events-table th {
  padding: 0.75rem;
  text-align: left;
  color: #cbd5e1;
  font-weight: 600;
  font-size: 0.875rem;
  border-bottom: 2px solid rgba(148, 163, 184, 0.2);
}

.events-table td {
  padding: 0.75rem;
  color: #e2e8f0;
  font-size: 0.875rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}

.events-table tbody tr:hover {
  background: rgba(51, 65, 85, 0.4);
}

.value-cell {
  color: #60a5fa;
  font-weight: 600;
}

.threshold-cell {
  color: #f87171;
  font-weight: 600;
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
