<template>
  <div class="microseismic-density">
    <div class="control-panel">
      <div class="form-group">
        <label>选择数据集:</label>
        <select v-model="selectedDataset" class="select-input">
          <option value="">请选择数据集</option>
          <option v-for="dataset in datasets" :key="dataset.id" :value="dataset.id">
            {{ dataset.name }} ({{ dataset.count }}条记录)
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
        <label>分析类型:</label>
        <select v-model="analysisType" class="select-input">
          <option value="frequency">频次核密度</option>
          <option value="energy">能量核密度</option>
        </select>
      </div>
      <button @click="fetchAnalysis" class="btn-analyze" :disabled="loading">
        {{ loading ? '分析中...' : '开始分析' }}
      </button>
    </div>

    <div v-if="loading" class="loading">
      <div class="spinner"></div>
      <p>正在生成核密度图...</p>
    </div>

    <div v-else-if="result" class="result-panel">
      <div class="stats-grid">
        <div class="stat-card">
          <span class="stat-label">事件总数</span>
          <span class="stat-value">{{ result.statistics.total_events }}</span>
        </div>
        <div class="stat-card" v-if="result.statistics.avg_energy">
          <span class="stat-label">平均能量</span>
          <span class="stat-value">{{ result.statistics.avg_energy.toExponential(2) }} J</span>
        </div>
        <div class="stat-card" v-if="result.statistics.max_energy">
          <span class="stat-label">最大能量</span>
          <span class="stat-value">{{ result.statistics.max_energy.toExponential(2) }} J</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">X坐标范围</span>
          <span class="stat-value">
            {{ result.statistics.x_range[0].toFixed(1) }} ~ {{ result.statistics.x_range[1].toFixed(1) }} m
          </span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Y坐标范围</span>
          <span class="stat-value">
            {{ result.statistics.y_range[0].toFixed(1) }} ~ {{ result.statistics.y_range[1].toFixed(1) }} m
          </span>
        </div>
        <div class="stat-card" v-if="result.dataset_name">
          <span class="stat-label">数据集</span>
          <span class="stat-value">{{ result.dataset_name }}</span>
        </div>
      </div>

      <div v-if="result.density_plot" class="plot-container">
        <h3 class="plot-title">{{ analysisType === 'frequency' ? '微震频次核密度图' : '微震能量核密度图' }}</h3>
        <img :src="result.density_plot" alt="核密度图" class="plot-image" />
        <p class="plot-description">
          核密度估计显示了微震事件在空间上的分布密度。颜色越亮表示该区域微震活动越密集。
        </p>
      </div>

      <div v-if="result.scatter_plot" class="plot-container">
        <h3 class="plot-title">微震散点图</h3>
        <img :src="result.scatter_plot" alt="散点图" class="plot-image" />
      </div>
    </div>

    <div v-else-if="error" class="error-message">
      <i class="error-icon">⚠</i>
      {{ error }}
    </div>

    <div v-else class="empty-state">
      <p>请选择日期范围并点击"开始分析"按钮</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import axios from 'axios'
import { useDatasetStore } from '@/stores/datasetStore'

const datasetStore = useDatasetStore()

const selectedDataset = ref('')
const datasets = ref([])
const startDate = ref('')
const endDate = ref('')
const analysisType = ref('frequency')
const loading = ref(false)
const result = ref(null)
const error = ref(null)

// 监听数据集选择变化，保存到store
watch(selectedDataset, (newValue) => {
  if (newValue) {
    const selected = datasets.value.find(d => d.id === newValue)
    if (selected) {
      datasetStore.setMicroseismicDataset(selected)
    }
  }
})

const fetchDatasets = async () => {
  try {
    const response = await axios.get('/api/data/datasets/', {
      params: { data_type: 'microseismic' }
    })
    datasets.value = response.data.datasets || []
  } catch (err) {
    console.error('获取数据集列表失败:', err)
  }
}

const fetchAnalysis = async () => {
  if (!startDate.value || !endDate.value) {
    error.value = '请选择开始和结束日期'
    return
  }

  if (!selectedDataset.value) {
    error.value = '请选择要分析的数据集'
    return
  }

  loading.value = true
  error.value = null
  result.value = null

  try {
    const params = {
      start_date: startDate.value,
      end_date: endDate.value,
      analysis_type: analysisType.value,
      dataset_id: selectedDataset.value
    }

    const response = await axios.get('/api/analysis/microseismic/density/', { params })
    
    if (response.data.success) {
      result.value = response.data
    } else {
      error.value = response.data.message || '分析失败'
    }
  } catch (err) {
    error.value = err.response?.data?.error || '分析失败,请重试'
    console.error('分析错误:', err.response?.data)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  // 默认加载数据集列表
  fetchDatasets()
})
</script>

<style scoped>
.microseismic-density {
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

.date-input, .select-input {
  padding: 0.5rem 1rem;
  background: rgba(51, 65, 85, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 6px;
  color: #e2e8f0;
  font-size: 0.875rem;
  min-width: 150px;
}

.date-input:focus, .select-input:focus {
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
