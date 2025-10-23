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
import { PieChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent
} from 'echarts/components'

use([
  CanvasRenderer,
  PieChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent
])

const props = defineProps({
  data: {
    type: Array,
    default: () => []
  }
})

const levelColors = {
  info: '#0066ff',
  warning: '#faad14',
  danger: '#ff7a45',
  critical: '#ff4d4f'
}

const levelNames = {
  info: '信息',
  warning: '警告',
  danger: '危险',
  critical: '严重'
}

const chartOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: {
    trigger: 'item',
    backgroundColor: 'rgba(10, 14, 39, 0.9)',
    borderColor: '#0066ff',
    textStyle: {
      color: '#fff'
    }
  },
  legend: {
    orient: 'vertical',
    right: '10%',
    top: 'center',
    textStyle: {
      color: '#fff'
    }
  },
  series: [
    {
      name: '预警级别',
      type: 'pie',
      radius: ['40%', '70%'],
      center: ['40%', '50%'],
      avoidLabelOverlap: false,
      itemStyle: {
        borderRadius: 10,
        borderColor: '#0a0e27',
        borderWidth: 2
      },
      label: {
        show: true,
        color: '#fff',
        formatter: '{b}: {c}'
      },
      emphasis: {
        label: {
          show: true,
          fontSize: 16,
          fontWeight: 'bold'
        }
      },
      data: props.data.map(item => ({
        name: levelNames[item.level] || item.level,
        value: item.count,
        itemStyle: {
          color: levelColors[item.level] || '#0066ff'
        }
      }))
    }
  ]
}))
</script>
