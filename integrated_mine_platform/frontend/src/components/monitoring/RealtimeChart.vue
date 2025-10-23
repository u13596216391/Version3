<template>
  <div class="w-full h-full min-h-[300px]">
    <v-chart :option="chartOption" autoresize />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent
} from 'echarts/components'
import axios from 'axios'

use([
  CanvasRenderer,
  LineChart,
  TitleComponent,
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

let refreshInterval = null

const fetchData = async () => {
  try {
    const response = await axios.get('/api/monitoring/realtime/', {
      params: {
        type: props.type,
        hours: 1
      }
    })
    
    const data = response.data.data || []
    chartData.value = {
      times: data.map(item => new Date(item.timestamp).toLocaleTimeString()),
      values: data.map(item => item.value)
    }
  } catch (error) {
    console.error('获取监控数据失败:', error)
  }
}

const chartOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(10, 14, 39, 0.9)',
    borderColor: '#0066ff',
    textStyle: {
      color: '#fff'
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
        color: '#0066ff'
      }
    },
    axisLabel: {
      color: '#8B9AAA'
    }
  },
  yAxis: {
    type: 'value',
    axisLine: {
      lineStyle: {
        color: '#0066ff'
      }
    },
    splitLine: {
      lineStyle: {
        color: 'rgba(0, 102, 255, 0.1)'
      }
    },
    axisLabel: {
      color: '#8B9AAA'
    }
  },
  series: [
    {
      type: 'line',
      smooth: true,
      symbol: 'circle',
      symbolSize: 6,
      data: chartData.value.values,
      itemStyle: {
        color: '#00d4ff'
      },
      lineStyle: {
        color: '#00d4ff',
        width: 2
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(0, 212, 255, 0.3)' },
            { offset: 1, color: 'rgba(0, 212, 255, 0)' }
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
