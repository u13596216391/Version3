<template>
  <div class="w-full h-full min-h-[300px]">
    <v-chart :option="chartOption" autoresize />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components'

use([
  CanvasRenderer,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
])

const props = defineProps({
  data: {
    type: Array,
    default: () => []
  }
})

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
  legend: {
    data: ['已完成', '失败'],
    textStyle: {
      color: '#fff'
    },
    top: 10
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    top: '15%',
    containLabel: true
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    data: props.data.map(item => {
      const date = new Date(item.timestamp)
      return `${date.getMonth() + 1}/${date.getDate()}`
    }),
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
      name: '已完成',
      type: 'line',
      smooth: true,
      data: props.data.map(item => item.completed_tasks),
      itemStyle: {
        color: '#00d4ff'
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
    },
    {
      name: '失败',
      type: 'line',
      smooth: true,
      data: props.data.map(item => item.failed_tasks),
      itemStyle: {
        color: '#ff4d4f'
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(255, 77, 79, 0.3)' },
            { offset: 1, color: 'rgba(255, 77, 79, 0)' }
          ]
        }
      }
    }
  ]
}))
</script>
