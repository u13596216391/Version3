<template>
  <div class="microseismic-scatter">
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
          <option value="frequency">频次分析</option>
          <option value="energy">能量分析</option>
        </select>
      </div>
      <button @click="fetchAnalysis" class="btn-analyze" :disabled="loading">
        {{ loading ? '分析中...' : '开始分析' }}
      </button>
    </div>

    <div v-if="loading" class="loading">
      <div class="spinner"></div>
      <p>正在生成散点图...</p>
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
        <div class="stat-card" v-if="result.dataset_name">
          <span class="stat-label">数据集</span>
          <span class="stat-value">{{ result.dataset_name }}</span>
        </div>
      </div>

      <div v-if="result.scatter_plot" class="plot-container">
        <img :src="result.scatter_plot" alt="微震散点图" class="plot-image" />
      </div>

      <div v-if="result.density_plot" class="plot-container">
        <img :src="result.density_plot" alt="核密度图" class="plot-image" />
      </div>
    </div>

    <div v-else-if="error" class="error-message">
      {{ error }}
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

    const response = await axios.get('/api/analysis/microseismic/scatter/', { params })
    
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
.microseismic-scatter {
  padding: 1rem;
}

.control-panel {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
  align-items: flex-end;
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
}

.plot-image {
  width: 100%;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.error-message {
  padding: 1rem;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 8px;
  color: #fca5a5;
}
</style>
