<template>
  <div class="realtime-chart-container">
    <div class="chart-header">
      <div class="chart-status">
        <span class="status-dot" :class="{ 'active': isActive }"></span>
        <span class="status-text">{{ isActive ? '实时更新中' : '连接断开' }}</span>
        <span class="update-time">{{ lastUpdateTime }}</span>
      </div>
      <div class="chart-stats">
        <span class="stat-item">当前: <strong>{{ currentValue }}</strong></span>
        <span class="stat-item">平均: <strong>{{ avgValue }}</strong></span>
        <span class="stat-item">最大: <strong>{{ maxValue }}</strong></span>
      </div>
    </div>
    
    <div v-if="loading && chartData.times.length === 0" class="loading-state">
      <div class="spinner"></div>
      <p>加载数据中...</p>
    </div>
    
    <div v-else-if="error" class="error-state">
      <p>{{ error }}</p>
      <button @click="fetchData" class="retry-btn">重试</button>
    </div>
    
    <div v-else class="chart-wrapper">
      <v-chart :option="chartOption" autoresize />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent
} from 'echarts/components'
import axios from 'axios'

use([
  CanvasRenderer,
  LineChart,
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent
])

const props = defineProps({
  type: {
    type: String,
    required: true
  },
  maxPoints: {
    type: Number,
    default: 50  // 默认50个点，可通过props自定义
  }
})

const chartData = ref({
  times: [],
  values: []
})

const loading = ref(false)
const error = ref(null)
const isActive = ref(true)
const lastUpdateTime = ref('')
// 移除硬编码的maxDataPoints，改为使用props
// const maxDataPoints = 50  // 删除这行

let refreshInterval = null

const typeConfig = {
  microseismic: {
    name: '微震能量',
    unit: 'J',
    color: '#00d4ff',
    yAxisName: '能量(J)'
  },
  support_resistance: {
    name: '支架阻力',
    unit: 'MPa',
    color: '#10b981',
    yAxisName: '阻力(MPa)'
  },
  gas: {
    name: '瓦斯浓度',
    unit: '%',
    color: '#f59e0b',
    yAxisName: '浓度(%)'
  },
  temperature: {
    name: '环境温度',
    unit: '°C',
    color: '#ef4444',
    yAxisName: '温度(°C)'
  }
}

const config = computed(() => typeConfig[props.type] || typeConfig.microseismic)

const currentValue = computed(() => {
  const values = chartData.value.values
  if (values.length === 0) return '--'
  return `${values[values.length - 1].toFixed(2)} ${config.value.unit}`
})

const avgValue = computed(() => {
  const values = chartData.value.values
  if (values.length === 0) return '--'
  const avg = values.reduce((sum, val) => sum + val, 0) / values.length
  return `${avg.toFixed(2)} ${config.value.unit}`
})

const maxValue = computed(() => {
  const values = chartData.value.values
  if (values.length === 0) return '--'
  return `${Math.max(...values).toFixed(2)} ${config.value.unit}`
})

const fetchData = async () => {
  try {
    loading.value = true
    error.value = null
    
    const response = await axios.get('/api/monitoring/realtime/', {
      params: {
        type: props.type,
        hours: 1
      }
    })
    
    const data = response.data.data || []
    
    if (data.length > 0) {
      // 使用props.maxPoints限制数据点数量
      const recentData = data.slice(-props.maxPoints)
      
      chartData.value = {
        times: recentData.map(item => {
          const date = new Date(item.timestamp)
          return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
        }),
        values: recentData.map(item => item.value)
      }
      
      lastUpdateTime.value = new Date().toLocaleTimeString('zh-CN')
      isActive.value = true
    } else {
      // 如果没有数据，保持现有数据或显示空状态
      if (chartData.value.times.length === 0) {
        error.value = '暂无数据，请启动模拟器生成数据'
      }
    }
  } catch (err) {
    console.error('获取监控数据失败:', err)
    error.value = '获取数据失败，请检查后端服务'
    isActive.value = false
  } finally {
    loading.value = false
  }
}

const chartOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(10, 14, 39, 0.95)',
    borderColor: config.value.color,
    borderWidth: 1,
    textStyle: {
      color: '#fff'
    },
    formatter: (params) => {
      const param = params[0]
      return `
        <div style="padding: 8px;">
          <div style="font-weight: bold; margin-bottom: 4px;">${config.value.name}</div>
          <div>时间: ${param.name}</div>
          <div>数值: <span style="color: ${config.value.color};">${param.value} ${config.value.unit}</span></div>
        </div>
      `
    }
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    top: '10%',
    containLabel: true
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    data: chartData.value.times,
    axisLine: {
      lineStyle: {
        color: 'rgba(148, 163, 184, 0.3)'
      }
    },
    axisLabel: {
      color: '#94a3b8',
      interval: Math.floor(chartData.value.times.length / 10) || 0
    },
    axisTick: {
      show: false
    }
  },
  yAxis: {
    type: 'value',
    name: config.value.yAxisName,
    nameTextStyle: {
      color: '#94a3b8',
      padding: [0, 0, 0, -10]
    },
    axisLine: {
      show: false
    },
    splitLine: {
      lineStyle: {
        color: 'rgba(148, 163, 184, 0.1)',
        type: 'dashed'
      }
    },
    axisLabel: {
      color: '#94a3b8',
      formatter: (value) => value.toFixed(1)
    }
  },
  series: [
    {
      name: config.value.name,
      type: 'line',
      smooth: true,
      symbol: 'circle',
      symbolSize: props.maxPoints <= 20 ? 8 : 6,  // 数据点少时使用更大的符号
      showSymbol: props.maxPoints <= 20,  // 数据点少时始终显示符号
      data: chartData.value.values,
      itemStyle: {
        color: config.value.color,
        borderWidth: 2,
        borderColor: '#fff'
      },
      lineStyle: {
        color: config.value.color,
        width: props.maxPoints <= 20 ? 3 : 2  // 数据点少时使用更粗的线条
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: `${config.value.color}60` },
            { offset: 1, color: `${config.value.color}00` }
          ]
        }
      }
    }
  ]
}))

watch(() => props.type, () => {
  // 切换类型时重新获取数据
  chartData.value = { times: [], values: [] }
  fetchData()
})

onMounted(() => {
  fetchData()
  // 每5秒自动刷新
  refreshInterval = setInterval(fetchData, 5000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<style scoped>
.realtime-chart-container {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding: 0.75rem;
  background: rgba(51, 65, 85, 0.4);
  border-radius: 6px;
}

.chart-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #6b7280;
  animation: pulse 2s ease-in-out infinite;
}

.status-dot.active {
  background: #10b981;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-text {
  color: #cbd5e1;
  font-weight: 500;
}

.update-time {
  color: #94a3b8;
  font-size: 0.75rem;
}

.chart-stats {
  display: flex;
  gap: 1.5rem;
  font-size: 0.875rem;
}

.stat-item {
  color: #94a3b8;
}

.stat-item strong {
  color: #60a5fa;
  margin-left: 0.25rem;
}

.chart-wrapper {
  flex: 1;
  min-height: 250px;
}

.loading-state,
.error-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
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

.retry-btn {
  padding: 0.5rem 1.5rem;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  border: none;
  border-radius: 6px;
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.retry-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}
</style>
