import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDatasetStore = defineStore('dataset', () => {
  // 存储选中的数据集信息
  const selectedMicroseismicDataset = ref(null)
  const selectedSupportDataset = ref(null)

  // 设置微震数据集
  function setMicroseismicDataset(dataset) {
    selectedMicroseismicDataset.value = dataset
    console.log('设置微震数据集:', dataset)
  }

  // 设置支架阻力数据集
  function setSupportDataset(dataset) {
    selectedSupportDataset.value = dataset
    console.log('设置支架阻力数据集:', dataset)
  }

  // 清空选择
  function clearSelection() {
    selectedMicroseismicDataset.value = null
    selectedSupportDataset.value = null
  }

  return {
    selectedMicroseismicDataset,
    selectedSupportDataset,
    setMicroseismicDataset,
    setSupportDataset,
    clearSelection
  }
})
