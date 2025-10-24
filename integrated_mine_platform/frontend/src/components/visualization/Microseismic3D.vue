<template>
  <div class="microseismic-3d">
    <div class="control-panel">
      <div class="form-group">
        <label>选择数据集:</label>
        <select v-model="dataset" class="select-input">
          <option value="">请选择数据集</option>
          <option v-for="ds in datasets" :key="ds.id" :value="ds.id">
            {{ ds.name }} ({{ ds.count }}条)
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
      <button @click="load3DData" class="btn-load" :disabled="loading">
        {{ loading ? '加载中...' : '加载三维数据' }}
      </button>
    </div>

    <div v-if="loading" class="loading">
      <div class="spinner"></div>
      <p>正在加载三维数据...</p>
    </div>

    <div v-else-if="chartData" class="chart-section">
      <div id="3d-plot" class="plot-container"></div>
      <div class="controls-panel">
        <div class="control-group">
          <label>能量阈值:</label>
          <input 
            v-model.number="energyThreshold" 
            type="range" 
            min="0" 
            :max="maxEnergy" 
            step="100"
            @input="updatePlot"
          />
          <span>{{ energyThreshold.toFixed(0) }} J</span>
        </div>
        <div class="control-group">
          <label>点大小:</label>
          <input 
            v-model.number="pointSize" 
            type="range" 
            min="1" 
            max="20" 
            step="1"
            @input="updatePlot"
          />
          <span>{{ pointSize }}</span>
        </div>
        <div class="control-group">
          <label>
            <input v-model="showReferenceLines" type="checkbox" @change="updatePlot" />
            显示参考线
          </label>
        </div>
      </div>
    </div>

    <div v-else-if="error" class="error-message">
      <i class="error-icon">⚠</i>
      {{ error }}
    </div>

    <div v-else class="empty-state">
      <p>请选择数据集和日期范围，然后点击"加载三维数据"按钮</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import Plotly from 'plotly.js-dist'

const datasets = ref([])
const dataset = ref('')
const startDate = ref('')
const endDate = ref('')
const loading = ref(false)
const chartData = ref(null)
const error = ref(null)

const energyThreshold = ref(0)
const maxEnergy = ref(10000)
const pointSize = ref(5)
const showReferenceLines = ref(true)

let plotDiv = null

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

const load3DData = async () => {
  if (!dataset.value || !startDate.value || !endDate.value) {
    error.value = '请选择数据集和日期范围'
    return
  }

  loading.value = true
  error.value = null
  chartData.value = null

  try {
    // 查询微震数据
    const response = await axios.get('/api/data/microseismic/', {
      params: {
        dataset_id: dataset.value,
        start_date: startDate.value,
        end_date: endDate.value,
        limit: 10000 // 限制数据量
      }
    })

    if (response.data.results && response.data.results.length > 0) {
      const data = response.data.results
      chartData.value = {
        x: data.map(d => d.event_x),
        y: data.map(d => d.event_y),
        z: data.map(d => d.event_z || -820), // 默认深度
        energy: data.map(d => d.energy || 0),
        timestamp: data.map(d => d.timestamp)
      }

      maxEnergy.value = Math.max(...chartData.value.energy)
      energyThreshold.value = 0

      // 延迟渲染以确保DOM已更新
      setTimeout(() => {
        renderPlot()
      }, 100)
    } else {
      error.value = '该时间范围内没有数据'
    }
  } catch (err) {
    error.value = err.response?.data?.error || '加载数据失败'
  } finally {
    loading.value = false
  }
}

const renderPlot = () => {
  plotDiv = document.getElementById('3d-plot')
  if (!plotDiv || !chartData.value) return

  const filtered = filterData()

  const trace = {
    x: filtered.x,
    y: filtered.y,
    z: filtered.z,
    mode: 'markers',
    type: 'scatter3d',
    name: '微震事件',
    marker: {
      size: pointSize.value,
      color: filtered.energy,
      colorscale: 'Hot',
      showscale: true,
      colorbar: {
        title: 'Energy (J)',
        thickness: 20,
        len: 0.7
      },
      opacity: 0.6
    },
    text: filtered.energy.map((e, i) => 
      `时间: ${new Date(filtered.timestamp[i]).toLocaleString()}<br>` +
      `坐标: (${filtered.x[i].toFixed(1)}, ${filtered.y[i].toFixed(1)}, ${filtered.z[i].toFixed(1)})<br>` +
      `能量: ${e.toFixed(2)} J`
    ),
    hoverinfo: 'text'
  }

  const traces = [trace]

  // 添加参考线
  if (showReferenceLines.value) {
    // 胶运巷道 (绿色)
    traces.push({
      x: [0, 1750],
      y: [0, 0],
      z: [-820, -820],
      mode: 'lines',
      type: 'scatter3d',
      name: '胶运巷道',
      line: { color: 'green', width: 4 },
      hoverinfo: 'name'
    })

    // 辅运巷道 (橙色)
    traces.push({
      x: [0, 1750],
      y: [300, 300],
      z: [-820, -820],
      mode: 'lines',
      type: 'scatter3d',
      name: '辅运巷道',
      line: { color: 'orange', width: 4 },
      hoverinfo: 'name'
    })

    // 工作面边界 (红色) - 垂直线
    traces.push({
      x: [1750, 1750],
      y: [0, 300],
      z: [-820, -820],
      mode: 'lines',
      type: 'scatter3d',
      name: '工作面边界',
      line: { color: 'red', width: 4 },
      hoverinfo: 'name'
    })

    // 开切眼位置 (灰色)
    traces.push({
      x: [0, 0],
      y: [0, 300],
      z: [-820, -820],
      mode: 'lines',
      type: 'scatter3d',
      name: '开切眼位置',
      line: { color: 'gray', width: 4 },
      hoverinfo: 'name'
    })

    // 坚硬顶板分界线 (紫色)
    traces.push({
      x: [965, 965],
      y: [0, 300],
      z: [-820, -820],
      mode: 'lines',
      type: 'scatter3d',
      name: '坚硬顶板分界线',
      line: { color: 'purple', width: 4 },
      hoverinfo: 'name'
    })
  }

  const layout = {
    title: {
      text: '微震事件三维分布图',
      font: { size: 18, color: '#e2e8f0' }
    },
    scene: {
      xaxis: { 
        title: 'X 坐标 (m)',
        backgroundcolor: 'rgba(30, 41, 59, 0.8)',
        gridcolor: 'rgba(148, 163, 184, 0.3)',
        showbackground: true,
        color: '#e2e8f0'
      },
      yaxis: { 
        title: 'Y 坐标 (m)',
        backgroundcolor: 'rgba(30, 41, 59, 0.8)',
        gridcolor: 'rgba(148, 163, 184, 0.3)',
        showbackground: true,
        color: '#e2e8f0'
      },
      zaxis: { 
        title: 'Z 坐标 (m)',
        backgroundcolor: 'rgba(30, 41, 59, 0.8)',
        gridcolor: 'rgba(148, 163, 184, 0.3)',
        showbackground: true,
        color: '#e2e8f0'
      },
      aspectmode: 'manual',
      aspectratio: { x: 3, y: 1, z: 1 },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.2 }
      }
    },
    paper_bgcolor: 'rgba(30, 41, 59, 0.6)',
    plot_bgcolor: 'rgba(30, 41, 59, 0.6)',
    font: { color: '#e2e8f0' },
    showlegend: true,
    legend: {
      x: 1,
      y: 1,
      bgcolor: 'rgba(30, 41, 59, 0.8)',
      bordercolor: 'rgba(148, 163, 184, 0.3)',
      borderwidth: 1
    },
    margin: { l: 0, r: 0, t: 50, b: 0 }
  }

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['toImage']
  }

  Plotly.newPlot(plotDiv, traces, layout, config)
}

const filterData = () => {
  if (!chartData.value) return { x: [], y: [], z: [], energy: [], timestamp: [] }

  const indices = chartData.value.energy
    .map((e, i) => e >= energyThreshold.value ? i : -1)
    .filter(i => i !== -1)

  return {
    x: indices.map(i => chartData.value.x[i]),
    y: indices.map(i => chartData.value.y[i]),
    z: indices.map(i => chartData.value.z[i]),
    energy: indices.map(i => chartData.value.energy[i]),
    timestamp: indices.map(i => chartData.value.timestamp[i])
  }
}

const updatePlot = () => {
  if (plotDiv && chartData.value) {
    renderPlot()
  }
}

onMounted(() => {
  fetchDatasets()
})

onUnmounted(() => {
  if (plotDiv) {
    Plotly.purge(plotDiv)
  }
})
</script>

<style scoped>
.microseismic-3d {
  padding: 1.5rem;
}

.control-panel {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  align-items: flex-end;
  flex-wrap: wrap;
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 1.5rem;
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

.select-input, .date-input {
  padding: 0.5rem 1rem;
  background: rgba(51, 65, 85, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 6px;
  color: #e2e8f0;
  min-width: 200px;
}

.btn-load {
  padding: 0.5rem 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-load:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.btn-load:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem;
  gap: 1rem;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 4px solid rgba(148, 163, 184, 0.2);
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading p {
  color: #cbd5e1;
  font-size: 1rem;
}

.chart-section {
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 1.5rem;
}

.plot-container {
  width: 100%;
  height: 700px;
  margin-bottom: 1rem;
}

.controls-panel {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  padding: 1rem;
  background: rgba(51, 65, 85, 0.4);
  border-radius: 6px;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.control-group label {
  color: #cbd5e1;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.control-group input[type="range"] {
  width: 100%;
}

.control-group span {
  color: #94a3b8;
  font-size: 0.875rem;
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

.empty-state {
  padding: 3rem;
  text-align: center;
  color: #94a3b8;
}
</style>
