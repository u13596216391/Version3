<template>
  <div class="deprecated-component p-4">
    <div class="bg-yellow-50 p-4 rounded-md border border-yellow-200">
      <h3 class="font-bold text-yellow-800">已弃用的组件：MicroseismicPredictor</h3>
      <p class="mt-2 text-yellow-700">微震预测现已整合到新的 <router-link to="/predictor" class="text-violet-600 underline">深度学习预测模块</router-link>。请在该页面中切换到“微震预测”标签以继续使用。</p>
    </div>
  </div>
</template>

<script setup>
// 保留组件作为向后兼容的占位提示
</script>

<style scoped>
.deprecated-component { }
</style>
          <div class="config-item">
            <label>时间列:</label>
            <input v-model="config.time_column" type="text" placeholder="如: timestamp" />
          </div>
          <div class="config-item">
            <label>训练轮数:</label>
            <input v-model.number="config.epochs" type="number" min="10" max="1000" />
          </div>
          <div class="config-item">
            <label>批次大小:</label>
            <input v-model.number="config.batch_size" type="number" min="8" max="256" />
          </div>
          <div class="config-item">
            <label>学习率:</label>
            <input v-model.number="config.learning_rate" type="number" step="0.0001" />
          </div>
          <div class="config-item">
            <label>训练集比例:</label>
            <input v-model.number="config.train_ratio" type="number" step="0.05" min="0.5" max="0.95" />
          </div>
          <div class="config-item full-width">
            <label>
              <input v-model="config.enable_lstm" type="checkbox" />
              启用LSTM模型
            </label>
          </div>
          <div class="config-item full-width">
            <label>
              <input v-model="config.enable_mamba" type="checkbox" />
              启用Mamba模型
            </label>
          </div>
        </div>
      </div>

      <button 
        @click="startTraining" 
        class="btn-train" 
        :disabled="(!selectedFile && !selectedDataset) || isTraining"
      >
        {{ isTraining ? '训练中...' : '开始训练' }}
      </button>
    </div>

    <div v-if="taskId" class="status-section">
      <h3>训练状态</h3>
      <div class="status-card" :class="statusClass">
        <div class="status-icon">{{ statusIcon }}</div>
        <div class="status-info">
          <div class="status-text">{{ statusMessage }}</div>
          <div v-if="trainingStatus === 'running'" class="progress-bar">
            <div class="progress-fill"></div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="results.length > 0" class="results-section">
      <h3>训练结果</h3>
      <div class="results-grid">
        <div v-for="result in results" :key="result.model_name" class="result-card">
          <h4>{{ result.model_name }}</h4>
          <div class="metrics-grid">
            <div class="metric-item" v-for="(value, key) in result.metrics" :key="key">
              <span class="metric-label">{{ formatMetricName(key) }}:</span>
              <span class="metric-value">{{ formatMetricValue(value) }}</span>
            </div>
          </div>
          
          <div v-if="result.plot_data" class="chart-container">
            <canvas :ref="`chart-${result.model_name}`"></canvas>
          </div>
        </div>
      </div>
    </div>

    <div v-if="error" class="error-message">
      <i class="error-icon">⚠</i>
      {{ error }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import Chart from 'chart.js/auto'
import { useDatasetStore } from '@/stores/datasetStore'

const datasetStore = useDatasetStore()

// 数据来源选择
const dataSource = ref('dataset') // 'dataset' 或 'upload'
const datasets = ref([])
const selectedDataset = ref('')

const selectedFile = ref(null)
const config = ref({
  target_column: 'energy',
  time_column: 'timestamp',
  epochs: 100,
  batch_size: 32,
  learning_rate: 0.001,
  train_ratio: 0.8,
  enable_lstm: true,
  enable_mamba: true
})

const isTraining = ref(false)
const taskId = ref(null)
const trainingStatus = ref('')
const statusMessage = ref('')
const results = ref([])
const error = ref(null)
const charts = ref([])

let statusCheckInterval = null

// 加载数据集列表
const fetchDatasets = async () => {
  try {
    const response = await axios.get('/api/data/datasets/', {
      params: { data_type: 'microseismic' }
    })
    datasets.value = response.data.datasets || []
    
    // 如果store中有选中的数据集，自动选择
    if (datasetStore.selectedMicroseismicDataset) {
      const found = datasets.value.find(d => d.id === datasetStore.selectedMicroseismicDataset.id)
      if (found) {
        selectedDataset.value = found.id
        dataSource.value = 'dataset'
      }
    }
  } catch (err) {
    console.error('获取数据集列表失败:', err)
  }
}

const statusClass = computed(() => {
  return {
    'status-running': trainingStatus.value === 'running',
    'status-success': trainingStatus.value === 'success',
    'status-failed': trainingStatus.value === 'failed'
  }
})

const statusIcon = computed(() => {
  switch (trainingStatus.value) {
    case 'running': return '⏳'
    case 'success': return '✅'
    case 'failed': return '❌'
    default: return 'ℹ️'
  }
})

const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file && file.name.endsWith('.csv')) {
    selectedFile.value = file
    error.value = null
  } else {
    error.value = '请选择CSV格式的文件'
  }
}

const startTraining = async () => {
  // 验证数据源
  if (dataSource.value === 'upload' && !selectedFile.value) {
    error.value = '请选择CSV文件'
    return
  }
  
  if (dataSource.value === 'dataset' && !selectedDataset.value) {
    error.value = '请选择数据集'
    return
  }

  isTraining.value = true
  error.value = null
  results.value = []
  charts.value.forEach(chart => chart.destroy())
  charts.value = []

  try {
    const formData = new FormData()
    
    // 根据数据源添加不同的参数
    if (dataSource.value === 'upload') {
      formData.append('file', selectedFile.value)
    } else {
      formData.append('dataset_id', selectedDataset.value)
    }
    
    formData.append('target_column', config.value.target_column)
    formData.append('time_column', config.value.time_column)
    formData.append('epochs', config.value.epochs)
    formData.append('batch_size', config.value.batch_size)
    formData.append('learning_rate', config.value.learning_rate)
    formData.append('train_ratio', config.value.train_ratio)
    formData.append('enable_lstm', config.value.enable_lstm)
    formData.append('enable_mamba', config.value.enable_mamba)

    const response = await axios.post('/api/predictor/start-training/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })

    taskId.value = response.data.task_id
    trainingStatus.value = 'running'
    statusMessage.value = '训练任务已启动，正在处理数据...'

    // 开始轮询状态
    startStatusPolling()
  } catch (err) {
    error.value = err.response?.data?.error || '启动训练失败'
    isTraining.value = false
  }
}
    error.value = '请先选择数据文件'
    return
  }

  isTraining.value = true
  error.value = null

  const formData = new FormData()
  formData.append('file', selectedFile.value)
  formData.append('config', JSON.stringify(config.value))

  try {
    const response = await axios.post('/api/predictor/start-training/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })

    taskId.value = response.data.task_id
    trainingStatus.value = 'running'
    statusMessage.value = '训练任务已启动，正在处理...'

    // 开始轮询状态
    startStatusPolling()
  } catch (err) {
    error.value = err.response?.data?.error || '启动训练失败'
    isTraining.value = false
  }
}

const startStatusPolling = () => {
  statusCheckInterval = setInterval(async () => {
    try {
      const response = await axios.get(`/api/predictor/status/${taskId.value}/`)
      trainingStatus.value = response.data.status
      statusMessage.value = response.data.message || '训练进行中...'

      if (response.data.status === 'success') {
        clearInterval(statusCheckInterval)
        isTraining.value = false
        await fetchResults()
      } else if (response.data.status === 'failed') {
        clearInterval(statusCheckInterval)
        isTraining.value = false
        error.value = response.data.error || '训练失败'
      }
    } catch (err) {
      console.error('检查状态失败:', err)
    }
  }, 3000) // 每3秒检查一次
}

const fetchResults = async () => {
  try {
    const response = await axios.get(`/api/predictor/results/${taskId.value}/`)
    results.value = response.data

    // 延迟渲染图表以确保DOM已更新
    setTimeout(() => {
      results.value.forEach(result => {
        if (result.plot_data) {
          renderChart(result)
        }
      })
    }, 100)
  } catch (err) {
    error.value = '获取结果失败'
  }
}

const renderChart = (result) => {
  const canvasRef = `chart-${result.model_name}`
  const canvas = document.querySelector(`[data-ref="${canvasRef}"]`)
  
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: result.plot_data.actuals.map((_, idx) => idx),
      datasets: [
        {
          label: 'Actual',
          data: result.plot_data.actuals,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.1
        },
        {
          label: 'Predicted',
          data: result.plot_data.predictions,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: `${result.model_name} - 预测结果对比`
        }
      },
      scales: {
        y: {
          beginAtZero: false
        }
      }
    }
  })

  charts.value.push(chart)
}

const formatMetricName = (name) => {
  const nameMap = {
    mse: 'MSE',
    rmse: 'RMSE',
    mae: 'MAE',
    r2: 'R²',
    mape: 'MAPE'
  }
  return nameMap[name] || name.toUpperCase()
}

const formatMetricValue = (value) => {
  if (typeof value === 'number') {
    return value.toFixed(4)
  }
  return value
}

onMounted(() => {
  fetchDatasets()
})

onUnmounted(() => {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval)
  }
  charts.value.forEach(chart => chart.destroy())
})
</script>

<style scoped>
.microseismic-predictor {
  padding: 1.5rem;
}

h3 {
  color: #e2e8f0;
  margin-bottom: 1rem;
  font-size: 1.25rem;
}

.upload-section {
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.dataset-section {
  margin-bottom: 1.5rem;
}

.option-tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.tab-btn {
  flex: 1;
  padding: 0.5rem 1rem;
  background: rgba(51, 65, 85, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 6px;
  color: #cbd5e1;
  cursor: pointer;
  transition: all 0.3s;
}

.tab-btn:hover {
  background: rgba(71, 85, 105, 0.8);
}

.tab-btn.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: transparent;
}

.dataset-select {
  margin-top: 1rem;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  color: #cbd5e1;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}

.select-input {
  width: 100%;
  padding: 0.5rem;
  background: rgba(51, 65, 85, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 6px;
  color: #e2e8f0;
  font-size: 0.875rem;
}

.dataset-info {
  padding: 0.75rem;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.3);
  border-radius: 6px;
  color: #10b981;
  font-size: 0.875rem;
  margin-top: 0.5rem;
}

.upload-area {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.btn-upload {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s;
}

.btn-upload:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.file-name {
  color: #94a3b8;
  font-size: 0.875rem;
}

.config-section {
  margin-bottom: 1.5rem;
}

.config-section h4 {
  color: #cbd5e1;
  font-size: 1rem;
  margin-bottom: 1rem;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.config-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.config-item.full-width {
  grid-column: 1 / -1;
  flex-direction: row;
  align-items: center;
}

.config-item label {
  color: #cbd5e1;
  font-size: 0.875rem;
}

.config-item input[type="text"],
.config-item input[type="number"] {
  padding: 0.5rem;
  background: rgba(51, 65, 85, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 4px;
  color: #e2e8f0;
}

.config-item input[type="checkbox"] {
  margin-right: 0.5rem;
}

.btn-train {
  width: 100%;
  padding: 1rem;
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-train:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
}

.btn-train:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.status-section {
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.status-card {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  border-radius: 6px;
  border-left: 4px solid;
}

.status-card.status-running {
  background: rgba(59, 130, 246, 0.1);
  border-left-color: #3b82f6;
}

.status-card.status-success {
  background: rgba(34, 197, 94, 0.1);
  border-left-color: #22c55e;
}

.status-card.status-failed {
  background: rgba(239, 68, 68, 0.1);
  border-left-color: #ef4444;
}

.status-icon {
  font-size: 2rem;
}

.status-info {
  flex: 1;
}

.status-text {
  color: #e2e8f0;
  margin-bottom: 0.5rem;
}

.progress-bar {
  height: 4px;
  background: rgba(148, 163, 184, 0.2);
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  animation: progress 2s ease-in-out infinite;
}

@keyframes progress {
  0% { width: 0%; }
  50% { width: 70%; }
  100% { width: 100%; }
}

.results-section {
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 1.5rem;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 1.5rem;
}

.result-card {
  background: rgba(51, 65, 85, 0.4);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 6px;
  padding: 1.5rem;
}

.result-card h4 {
  color: #a78bfa;
  margin-bottom: 1rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.metric-item {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem;
  background: rgba(30, 41, 59, 0.4);
  border-radius: 4px;
}

.metric-label {
  color: #94a3b8;
  font-size: 0.875rem;
}

.metric-value {
  color: #e2e8f0;
  font-weight: 600;
}

.chart-container {
  height: 300px;
  margin-top: 1rem;
}

.chart-container canvas {
  width: 100% !important;
  height: 100% !important;
}

.error-message {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 6px;
  padding: 1rem;
  color: #fca5a5;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.error-icon {
  font-size: 1.5rem;
}
</style>
