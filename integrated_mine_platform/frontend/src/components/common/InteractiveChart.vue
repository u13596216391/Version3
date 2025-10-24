<template>
  <div class="interactive-chart bg-white p-6 rounded-xl shadow-lg border border-gray-100">
    <h3 class="text-xl font-bold text-gray-800 mb-4">{{ title }}</h3>
    <canvas ref="chartCanvas"></canvas>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, onUnmounted } from 'vue'
import Chart from 'chart.js/auto'

const props = defineProps({
  title: {
    type: String,
    default: '预测结果对比'
  },
  plotData: {
    type: Object,
    required: true
  }
})

const chartCanvas = ref(null)
let chartInstance = null

const renderChart = () => {
  if (!chartCanvas.value || !props.plotData) return

  // 销毁旧图表
  if (chartInstance) {
    chartInstance.destroy()
  }

  const ctx = chartCanvas.value.getContext('2d')

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: props.plotData.labels || props.plotData.actuals?.map((_, idx) => idx) || [],
      datasets: [
        {
          label: '实际值',
          data: props.plotData.actuals || [],
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          tension: 0.4,
          pointRadius: 2,
          pointHoverRadius: 5
        },
        {
          label: '预测值',
          data: props.plotData.predictions || [],
          borderColor: 'rgb(147, 51, 234)',
          backgroundColor: 'rgba(147, 51, 234, 0.1)',
          tension: 0.4,
          pointRadius: 2,
          pointHoverRadius: 5
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 2,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            usePointStyle: true,
            padding: 15,
            font: {
              size: 12
            }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          padding: 12,
          titleFont: {
            size: 14
          },
          bodyFont: {
            size: 13
          },
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || ''
              if (label) {
                label += ': '
              }
              if (context.parsed.y !== null) {
                label += context.parsed.y.toFixed(4)
              }
              return label
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: '时间步',
            font: {
              size: 13
            }
          },
          grid: {
            display: false
          }
        },
        y: {
          title: {
            display: true,
            text: props.plotData.y_label || '数值',
            font: {
              size: 13
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.05)'
          }
        }
      }
    }
  })
}

watch(() => props.plotData, () => {
  renderChart()
}, { deep: true })

onMounted(() => {
  renderChart()
})

onUnmounted(() => {
  if (chartInstance) {
    chartInstance.destroy()
  }
})
</script>
