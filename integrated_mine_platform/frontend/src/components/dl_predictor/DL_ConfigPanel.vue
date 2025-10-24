<template>
  <div class="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700">
    <h2 class="text-2xl font-bold text-white mb-2">深度学习预测配置</h2>
    <p class="text-gray-400 mb-8">选择数据源、模型类型并调整参数，开始您的预测任务。</p>
    
    <!-- 标签页切换 -->
    <div class="tabs-container mb-8">
      <button 
        v-for="tab in tabs" 
        :key="tab.key"
        :class="['tab-btn', { active: activeTab === tab.key }]"
        @click="activeTab = tab.key"
      >
        {{ tab.label }}
      </button>
    </div>

    <div class="space-y-10">
      <!-- 1. 数据源选择 -->
      <div>
        <h3 class="text-lg font-semibold text-gray-200 mb-4 border-b border-gray-700 pb-2">1. 数据源</h3>
        <div class="grid grid-cols-1 gap-6">
          <!-- 数据源类型选择 -->
          <div>
            <div class="flex gap-4 mb-4">
              <button 
                :class="['source-btn', { active: dataSource === 'dataset' }]"
                @click="dataSource = 'dataset'"
              >
                📊 选择数据集
              </button>
              <button 
                :class="['source-btn', { active: dataSource === 'upload' }]"
                @click="dataSource = 'upload'"
              >
                📤 上传文件
              </button>
            </div>

            <!-- 数据集选择 -->
            <div v-if="dataSource === 'dataset'">
              <label class="block text-sm font-semibold text-gray-300 mb-2">选择数据集</label>
              <select v-model="selectedDataset" class="w-full px-4 py-2 bg-gray-700 border border-gray-600 text-gray-200 rounded-md focus:ring-violet-500 focus:border-violet-500">
                <option value="">请选择数据集</option>
                <option v-for="ds in datasets" :key="ds.id" :value="ds.id">
                  {{ ds.name }} ({{ ds.count }}条记录)
                </option>
              </select>
            </div>

            <!-- 文件上传 -->
            <div v-else>
              <label class="block text-sm font-semibold text-gray-300 mb-2">上传数据文件</label>
              <div class="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-600 border-dashed rounded-md hover:border-violet-400 transition-colors bg-gray-700/30">
                <div class="space-y-1 text-center">
                  <svg class="mx-auto h-12 w-12 text-gray-500" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                  </svg>
                  <div class="flex text-sm text-gray-400">
                    <label :for="'file-upload-' + activeTab" class="relative cursor-pointer bg-gray-800 rounded-md font-medium text-violet-400 hover:text-violet-300">
                      <span>{{ activeTab === 'microseismic' ? '选择ZIP文件' : '选择CSV文件' }}</span>
                      <input 
                        :id="'file-upload-' + activeTab" 
                        type="file" 
                        class="sr-only" 
                        @change="handleFileChange"
                        :accept="activeTab === 'microseismic' ? '.zip' : '.csv'"
                      >
                    </label>
                    <p class="pl-1">或拖拽到此处</p>
                  </div>
                  <p v-if="selectedFile" class="text-sm text-green-400 font-semibold mt-2">✓ {{ selectedFile.name }}</p>
                  <p v-else class="text-xs text-gray-500">
                    {{ activeTab === 'microseismic' ? 'ZIP包应包含微震CSV文件' : '支架阻力CSV文件' }}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 2. 模型选择 -->
      <div>
        <h3 class="text-lg font-semibold text-gray-200 mb-4 border-b border-gray-700 pb-2">2. 模型选择 (可多选)</h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div 
            v-for="model in availableModels" 
            :key="model.id"
            @click="toggleModel(model.id)"
            :class="['model-card', { 'selected': config.models.includes(model.id) }]"
          >
            <span class="font-bold text-lg">{{ model.name }}</span>
            <p class="text-xs">{{ model.desc }}</p>
          </div>
        </div>
      </div>

      <!-- 3. 训练参数 -->
      <div>
        <h3 class="text-lg font-semibold text-gray-200 mb-4 border-b border-gray-700 pb-2">3. 训练参数</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
          <ParameterControl 
            label="时间窗口大小" 
            description="输入模型的时间序列长度" 
            v-model="config.hyperparameters.window_size" 
            :min="12" 
            :max="72" 
          />
          <ParameterControl 
            label="训练周期 (Epochs)" 
            description="模型训练的总轮数" 
            v-model="config.hyperparameters.epochs" 
            :min="10" 
            :max="200" 
          />
          <ParameterControl 
            label="批处理大小" 
            description="每次训练迭代处理的样本数" 
            v-model="config.hyperparameters.batch_size" 
            :min="8" 
            :max="128" 
          />
          <ParameterControl 
            label="学习率" 
            description="控制模型参数更新的幅度" 
            v-model="config.hyperparameters.learning_rate" 
            :min="0.0001" 
            :max="0.01" 
            :step="0.0001" 
          />
          <ParameterControl 
            label="测试集比例" 
            description="用于最终评估的数据比例" 
            v-model="config.hyperparameters.test_size" 
            :min="0.1" 
            :max="0.5" 
            :step="0.05" 
          />
          <ParameterControl 
            label="隐藏层维度" 
            description="模型内部的特征维度" 
            v-model="config.hyperparameters.d_model" 
            :min="32" 
            :max="256" 
            :step="32" 
          />
        </div>
      </div>
    </div>

    <!-- 提交按钮 -->
    <div class="mt-10">
      <button 
        @click="submitTraining" 
        :disabled="isLoading || (!selectedFile && !selectedDataset) || config.models.length === 0" 
        class="w-full submit-button"
      >
        <span v-if="isLoading">
          <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          正在启动训练...
        </span>
        <span v-else>🚀 开始训练</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import axios from 'axios'
import ParameterControl from '@/components/common/ParameterControl.vue'
import { useDatasetStore } from '@/stores/datasetStore'

const emit = defineEmits(['start-training'])
const datasetStore = useDatasetStore()

// 标签页
const tabs = [
  { key: 'microseismic', label: '🌍 微震预测' },
  { key: 'support', label: '🔧 支架阻力预测' }
]
const activeTab = ref('microseismic')

// 数据源
const dataSource = ref('dataset') // 默认使用数据集选择
const datasets = ref([])
const selectedDataset = ref('')
const selectedFile = ref(null)
const isLoading = ref(false)

// 可选模型
const availableModels = ref([
  { id: 'LSTM', name: 'LSTM', desc: '经典时序模型' },
  { id: 'GRU', name: 'GRU', desc: 'LSTM变体' },
  { id: 'Mamba', name: 'Mamba', desc: '状态空间模型' },
  { id: 'CNN-LSTM', name: 'CNN-LSTM', desc: '混合卷积模型' },
  { id: 'Transformer', name: 'Transformer', desc: '注意力机制' }
])

// 配置
const config = reactive({
  models: ['LSTM', 'Mamba'],
  hyperparameters: {
    window_size: 24,
    epochs: 50,
    batch_size: 32,
    learning_rate: 0.001,
    test_size: 0.2,
    d_model: 64
  }
})

// 加载数据集列表
const fetchDatasets = async () => {
  try {
    const dataType = activeTab.value === 'microseismic' ? 'microseismic' : 'support_resistance'
    const response = await axios.get('/api/data/datasets/', {
      params: { data_type: dataType }
    })
    datasets.value = response.data.datasets || []
    
    // 检查store中是否有选中的数据集
    const storeDataset = activeTab.value === 'microseismic' 
      ? datasetStore.selectedMicroseismicDataset 
      : datasetStore.selectedSupportDataset
    
    if (storeDataset) {
      const found = datasets.value.find(d => d.id === storeDataset.id)
      if (found) {
        selectedDataset.value = found.id
        dataSource.value = 'dataset'
      }
    }
  } catch (error) {
    console.error('获取数据集失败:', error)
  }
}

const toggleModel = (modelId) => {
  const index = config.models.indexOf(modelId)
  if (index > -1) {
    config.models.splice(index, 1)
  } else {
    config.models.push(modelId)
  }
}

const handleFileChange = (event) => {
  selectedFile.value = event.target.files[0]
}

const submitTraining = async () => {
  // 验证数据源
  if (dataSource.value === 'upload') {
    if (!selectedFile.value) {
      alert('请先上传数据文件！')
      return
    }
  } else {
    if (!selectedDataset.value || selectedDataset.value === '') {
      alert('请先选择数据集！')
      return
    }
  }
  
  if (config.models.length === 0) {
    alert('请至少选择一个模型！')
    return
  }
  
  isLoading.value = true
  const formData = new FormData()
  
  // 根据数据源添加参数
  if (dataSource.value === 'upload') {
    formData.append('file', selectedFile.value)
    console.log('DEBUG - 上传文件模式:', selectedFile.value.name)
  } else {
    // 只在有效的 dataset_id 时才添加
    if (selectedDataset.value && selectedDataset.value !== '') {
      formData.append('dataset_id', selectedDataset.value)
      console.log('DEBUG - 数据集模式, dataset_id:', selectedDataset.value)
    } else {
      alert('请选择一个有效的数据集！')
      isLoading.value = false
      return
    }
  }
  
  formData.append('data_type', activeTab.value)
  formData.append('config', JSON.stringify(config))
  
  console.log('DEBUG - 发送的FormData:')
  for (let [key, value] of formData.entries()) {
    console.log(`  ${key}:`, value)
  }

  try {
    const response = await axios.post('/api/predictor/start-training/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    emit('start-training', response.data.task_id)
  } catch (error) {
    console.error('启动训练失败:', error)
    console.error('错误详情:', error.response?.data)
    alert('启动训练失败: ' + (error.response?.data?.error || error.message))
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  fetchDatasets()
})
</script>

<style scoped>
.tabs-container {
  @apply flex gap-2 bg-gray-700 p-1 rounded-lg;
}

.tab-btn {
  @apply flex-1 py-2 px-4 rounded-md text-sm font-medium text-gray-400 transition-all duration-200;
}

.tab-btn.active {
  @apply bg-gray-800 text-violet-400 shadow-sm;
}

.source-btn {
  @apply flex-1 py-3 px-6 border-2 border-gray-600 rounded-lg text-center cursor-pointer transition-all duration-200 hover:border-violet-400 font-medium text-gray-300 bg-gray-700/50;
}

.source-btn.active {
  @apply border-violet-500 bg-violet-900/30 text-violet-300 shadow-md;
}

.model-card {
  @apply p-4 border-2 border-gray-600 rounded-lg text-center cursor-pointer transition-all duration-200 hover:border-violet-400 hover:shadow-md bg-gray-700/50 text-gray-300;
}

.model-card.selected {
  @apply border-violet-500 bg-violet-900/30 text-violet-300 shadow-lg scale-105;
}

.submit-button {
  @apply flex justify-center items-center w-full py-3 px-4 border border-transparent rounded-lg shadow-sm text-base font-medium text-white bg-violet-600 hover:bg-violet-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-violet-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors;
}
</style>
