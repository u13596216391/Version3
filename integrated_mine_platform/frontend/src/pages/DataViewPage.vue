<template>
  <div class="data-view-page">
    <div class="tech-card mb-6">
      <h2 class="text-2xl font-bold glow-text mb-4">数据库数据查看</h2>
      <p class="text-gray-400">查看系统中所有数据表的记录</p>
    </div>

    <!-- 数据库表概览 -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
      <div v-for="(tables, appName) in databaseData" :key="appName" class="tech-card">
        <h3 class="text-lg font-bold text-tech-cyan mb-4">{{ formatAppName(appName) }}</h3>
        <div class="space-y-3">
          <div 
            v-for="(info, tableName) in tables" 
            :key="tableName"
            class="flex justify-between items-center p-3 rounded bg-tech-dark/30 hover:bg-tech-dark/50 cursor-pointer transition-all"
            @click="loadTableData(appName, tableName)"
          >
            <div>
              <div class="font-semibold">{{ info.verbose_name }}</div>
              <div class="text-xs text-gray-500">{{ tableName }}</div>
            </div>
            <div class="text-tech-cyan font-bold">{{ info.count }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 数据详情表格 -->
    <div v-if="selectedTable" class="tech-card">
      <div class="flex justify-between items-center mb-4">
        <h3 class="text-xl font-bold text-tech-cyan">
          {{ selectedTable.appName }} - {{ selectedTable.tableName }}
        </h3>
        <button @click="selectedTable = null" class="tech-button">
          关闭
        </button>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-primary-500/30">
              <th class="text-left p-3">ID</th>
              <th class="text-left p-3">数据内容</th>
              <th class="text-left p-3">时间</th>
            </tr>
          </thead>
          <tbody>
            <tr 
              v-for="row in tableData" 
              :key="row.id"
              class="border-b border-primary-500/10 hover:bg-primary-500/10 transition-colors"
            >
              <td class="p-3">{{ row.id }}</td>
              <td class="p-3">{{ formatRowData(row) }}</td>
              <td class="p-3 text-gray-400">{{ formatTimestamp(row) }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-if="tableData.length === 0" class="text-center text-gray-400 py-8">
        暂无数据
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const databaseData = ref({})
const selectedTable = ref(null)
const tableData = ref([])

const formatAppName = (appName) => {
  const names = {
    'predictor_app': '支架阻力预测',
    'microseismic_app': '微震预测',
    'monitoring_app': '实时监控',
    'dashboard_app': '数据大屏'
  }
  return names[appName] || appName
}

const loadTableData = async (appName, tableName) => {
  selectedTable.value = { appName, tableName }
  // 这里可以添加具体的表数据加载逻辑
  tableData.value = []
  console.log(`Loading data for ${appName}.${tableName}`)
}

const formatRowData = (row) => {
  const keys = Object.keys(row).filter(k => !['id', 'timestamp', 'created_at', 'updated_at'].includes(k))
  return keys.slice(0, 3).map(k => `${k}: ${row[k]}`).join(', ')
}

const formatTimestamp = (row) => {
  const timestamp = row.timestamp || row.created_at || row.updated_at
  if (!timestamp) return '-'
  return new Date(timestamp).toLocaleString('zh-CN')
}

const fetchDatabaseOverview = async () => {
  try {
    const response = await axios.get('/api/dashboard/data-view/')
    databaseData.value = response.data
  } catch (error) {
    console.error('获取数据库概览失败:', error)
  }
}

onMounted(() => {
  fetchDatabaseOverview()
})
</script>

<style scoped>
table {
  @apply min-w-full;
}

thead th {
  @apply text-tech-cyan font-semibold;
}
</style>
