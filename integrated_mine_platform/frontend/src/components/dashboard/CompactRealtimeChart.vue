<template>
  <div class="compact-realtime-chart">
    <!-- 数据状态 -->
    <div class="stats-row">
      <div class="stat-item">
        <span class="stat-label">当前</span>
        <span class="stat-value" :style="{ color: config.color }">{{ currentValue }}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">平均</span>
        <span class="stat-value">{{ avgValue }}</span>
      </div>
      <div class="stat-item">
        <span class="status-indicator" :class="{ 'active': isActive }"></span>
        <span class="stat-label text-xs">{{ isActive ? '在线' : '离线' }}</span>
      </div>
    </div>

    <!-- 迷你图表 -->
    <div v-if="loading && chartData.values.length === 0" class="loading-mini">
      <div class="spinner-mini"></div>
    </div>
    <div v-else-if="error" class="error-mini">
      <p class="text-xs text-gray-500">{{ error }}</p>
    </div>
    <div v-else class="chart-mini">
      <v-chart :option="chartOption" autoresize />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import {
  TooltipComponent,
  GridComponent
} from 'echarts/components'
import axios from 'axios'

use([
  CanvasRenderer,
  LineChart,
  TooltipComponent,
  GridComponent
])

const props = defineProps({
  type: {
    type: String,
    required: true
  }
})

const chartData = ref({
  times: [],
  values: []
})

const loading = ref(false)
const error = ref(null)
const isActive = ref(true)
const maxPoints = 10  // 固定显示10个数据点

let refreshInterval = null

const typeConfig = {
  microseismic: {
    name: '微震能量',
    unit: 'J',
    color: '#00d4ff'
  },
  support_resistance: {
    name: '支架阻力',
    unit: 'MPa',
    color: '#10b981'
  },
  gas: {
    name: '瓦斯浓度',
    unit: '%',
    color: '#f59e0b'
  },
  temperature: {
    name: '环境温度',
    unit: '°C',
    color: '#ef4444'
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
      const recentData = data.slice(-maxPoints)
      
      chartData.value = {
        times: recentData.map(item => {
          const date = new Date(item.timestamp)
          return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
        }),
        values: recentData.map(item => item.value)
      }
      
      isActive.value = true
    } else {
      if (chartData.value.times.length === 0) {
        error.value = '暂无数据'
      }
    }
  } catch (err) {
    console.error('获取监控数据失败:', err)
    error.value = '加载失败'
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
      color: '#fff',
      fontSize: 12
    },
    formatter: (params) => {
      const param = params[0]
      return `${param.name}<br/><span style="color: ${config.value.color};">${param.value} ${config.value.unit}</span>`
    }
  },
  grid: {
    left: '5%',
    right: '5%',
    bottom: '5%',
    top: '5%',
    containLabel: false
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    data: chartData.value.times,
    show: false
  },
  yAxis: {
    type: 'value',
    show: false
  },
  series: [
    {
      name: config.value.name,
      type: 'line',
      smooth: true,
      symbol: 'circle',
      symbolSize: 8,
      showSymbol: true,
      data: chartData.value.values,
      itemStyle: {
        color: config.value.color,
        borderWidth: 2,
        borderColor: '#0a0e27'
      },
      lineStyle: {
        color: config.value.color,
        width: 3
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: `${config.value.color}80` },
            { offset: 1, color: `${config.value.color}10` }
          ]
        }
      }
    }
  ]
}))

onMounted(() => {
  fetchData()
  refreshInterval = setInterval(fetchData, 5000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<style scoped>
.compact-realtime-chart {
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.stats-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  background: rgba(51, 65, 85, 0.3);
  border-radius: 6px;
  border: 1px solid rgba(100, 116, 139, 0.2);
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.stat-item:last-child {
  flex-direction: row;
  gap: 0.5rem;
}

.stat-label {
  font-size: 0.75rem;
  color: #94a3b8;
  font-weight: 500;
}

.stat-value {
  font-size: 1rem;
  font-weight: 700;
  color: #60a5fa;
}

.status-indicator {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #6b7280;
}

.status-indicator.active {
  background: #10b981;
  box-shadow: 0 0 8px #10b981;
  animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
  0%, 100% { 
    opacity: 1;
    box-shadow: 0 0 8px #10b981;
  }
  50% { 
    opacity: 0.6;
    box-shadow: 0 0 12px #10b981;
  }
}

.chart-mini {
  flex: 1;
  min-height: 120px;
}

.loading-mini,
.error-mini {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #94a3b8;
}

.spinner-mini {
  width: 24px;
  height: 24px;
  border: 3px solid rgba(59, 130, 246, 0.2);
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
