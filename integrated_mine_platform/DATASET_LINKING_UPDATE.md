# 数据集联动与导航更新说明

## 更新内容

### 1. ✅ 顶部导航栏添加"高级可视化"链接

**修改文件**: `frontend/src/App.vue`

在导航菜单中添加了"高级可视化"选项，用户可以直接从顶部导航栏跳转到高级可视化页面。

```javascript
const routes = [
  { path: '/', name: '数据大屏' },
  { path: '/predictor', name: '支架阻力预测' },
  { path: '/microseismic', name: '微震预测' },
  { path: '/monitoring', name: '实时监控' },
  { path: '/data-view', name: '数据查看' },
  { path: '/analysis', name: '多源数据分析' },
  { path: '/visualization', name: '高级可视化' },  // ← 新增
]
```

---

### 2. ✅ 创建数据集状态管理Store

**新建文件**: `frontend/src/stores/datasetStore.js`

使用Pinia创建全局状态管理，用于在不同页面间共享选中的数据集信息。

**功能**:
- `selectedMicroseismicDataset` - 存储选中的微震数据集
- `selectedSupportDataset` - 存储选中的支架阻力数据集
- `setMicroseismicDataset(dataset)` - 设置微震数据集
- `setSupportDataset(dataset)` - 设置支架阻力数据集
- `clearSelection()` - 清空所有选择

---

### 3. ✅ 分析模块自动保存数据集选择

**修改文件**:
- `frontend/src/components/analysis/MicroseismicScatter.vue`
- `frontend/src/components/analysis/MicroseismicDensity.vue`
- `frontend/src/components/analysis/SupportDWTAnalysis.vue`
- `frontend/src/components/analysis/WaveletComparison.vue`

**实现逻辑**:
当用户在分析页面选择数据集时，通过`watch`监听数据集选择变化，自动将选中的数据集保存到全局store中。

```javascript
import { useDatasetStore } from '@/stores/datasetStore'

const datasetStore = useDatasetStore()

// 监听数据集选择变化，保存到store
watch(selectedDataset, (newValue) => {
  if (newValue) {
    const selected = datasets.value.find(d => d.id === newValue)
    if (selected) {
      datasetStore.setMicroseismicDataset(selected) // 或 setSupportDataset
    }
  }
})
```

---

### 4. ✅ 预测模块支持数据集选择

**修改文件**:
- `frontend/src/components/predictor/MicroseismicPredictor.vue`
- `frontend/src/components/predictor/SupportPredictor.vue`

**新增功能**:

#### 4.1 双模式数据源选择
用户可以选择两种数据来源方式：
1. **从数据集选择** - 使用已上传的数据集
2. **上传CSV文件** - 临时上传新文件

```vue
<div class="option-tabs">
  <button :class="['tab-btn', { active: dataSource === 'dataset' }]">
    从数据集选择
  </button>
  <button :class="['tab-btn', { active: dataSource === 'upload' }]">
    上传CSV文件
  </button>
</div>
```

#### 4.2 自动加载数据集列表
组件挂载时自动从后端获取数据集列表：
- 微震预测：获取 `data_type=microseismic` 的数据集
- 支架阻力预测：获取 `data_type=support_resistance` 的数据集

#### 4.3 智能联动
如果用户从分析页面选择了数据集，跳转到预测页面时：
- 自动切换到"从数据集选择"模式
- 自动选中之前选择的数据集
- 无需重新选择，直接配置参数即可开始训练

```javascript
// 如果store中有选中的数据集，自动选择
if (datasetStore.selectedMicroseismicDataset) {
  const found = datasets.value.find(d => d.id === datasetStore.selectedMicroseismicDataset.id)
  if (found) {
    selectedDataset.value = found.id
    dataSource.value = 'dataset'
  }
}
```

#### 4.4 训练API更新
训练请求支持两种参数格式：
- 使用文件：`formData.append('file', selectedFile.value)`
- 使用数据集：`formData.append('dataset_id', selectedDataset.value)`

---

## 使用流程示例

### 场景1：从分析到预测的完整流程

1. **在"多源数据分析"页面**
   - 用户选择微震数据集 "2024-01-15_微震数据"
   - 进行散点图或核密度分析
   - 系统自动保存选中的数据集到store

2. **跳转到"支架阻力预测"或"微震预测"页面**
   - 页面自动加载数据集列表
   - 自动识别之前选择的数据集
   - 自动切换到"从数据集选择"模式
   - 数据集下拉框已选中"2024-01-15_微震数据"

3. **配置训练参数并开始训练**
   - 配置训练轮数、批次大小等参数
   - 点击"开始训练"
   - 使用选中的数据集进行模型训练

### 场景2：直接上传文件训练

1. **在"预测"页面**
   - 切换到"上传CSV文件"标签
   - 点击"选择CSV文件"按钮
   - 选择本地CSV文件
   - 配置参数并开始训练

---

## 技术实现细节

### Store状态管理
```javascript
// 定义
const selectedMicroseismicDataset = ref(null)
const selectedSupportDataset = ref(null)

// 使用
import { useDatasetStore } from '@/stores/datasetStore'
const datasetStore = useDatasetStore()

// 设置
datasetStore.setMicroseismicDataset(dataset)

// 读取
if (datasetStore.selectedMicroseismicDataset) {
  console.log('选中的数据集:', datasetStore.selectedMicroseismicDataset)
}
```

### 数据集API
```javascript
// 获取微震数据集列表
GET /api/data/datasets/?data_type=microseismic

// 获取支架阻力数据集列表
GET /api/data/datasets/?data_type=support_resistance
```

### 预测训练API
```javascript
// 使用文件训练
POST /api/predictor/start-training/
FormData: {
  file: <CSV文件>,
  target_column: 'energy',
  epochs: 100,
  ...
}

// 使用数据集训练
POST /api/predictor/start-training/
FormData: {
  dataset_id: 123,
  target_column: 'energy',
  epochs: 100,
  ...
}
```

---

## UI改进

### 新增样式类
- `.option-tabs` - 数据源选择标签容器
- `.tab-btn` - 标签按钮
- `.tab-btn.active` - 激活状态的标签
- `.dataset-section` - 数据集选择区域
- `.dataset-select` - 数据集下拉框容器
- `.dataset-info` - 数据集选择提示信息

### 视觉效果
- 标签按钮渐变背景：`linear-gradient(135deg, #667eea 0%, #764ba2 100%)`
- 选中提示：绿色背景 `rgba(16, 185, 129, 0.1)` + 绿色边框
- Hover动效：向上移动2px + 阴影增强

---

## 测试建议

### 测试场景1：数据集联动
1. 进入"多源数据分析"页面
2. 选择一个微震数据集
3. 进行任意分析操作
4. 跳转到"微震预测"页面
5. ✅ 验证：数据集是否自动选中

### 测试场景2：切换数据源
1. 在预测页面选择"从数据集选择"
2. 选择数据集A
3. 切换到"上传CSV文件"
4. 上传文件B
5. 切换回"从数据集选择"
6. ✅ 验证：数据集A是否仍然选中

### 测试场景3：跨模块切换
1. 在支架阻力分析页面选择数据集A
2. 跳转到微震预测页面
3. ✅ 验证：不应该自动选中任何数据集（类型不匹配）
4. 跳转到支架阻力预测页面
5. ✅ 验证：数据集A应该被自动选中

### 测试场景4：导航链接
1. 访问任意页面
2. 点击顶部导航栏的"高级可视化"
3. ✅ 验证：是否正确跳转到 `/visualization` 路由
4. ✅ 验证：页面是否正常显示三个标签页

---

## 部署步骤

### 1. 安装依赖（如需要）
```bash
cd frontend
npm install
```

### 2. 重新构建前端
```bash
cd C:\Users\m1359\Desktop\tzb\Version3\integrated_mine_platform
docker-compose stop frontend
docker-compose rm -f frontend
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

### 3. 验证部署
```bash
# 检查前端容器状态
docker-compose ps

# 查看前端日志
docker-compose logs frontend
```

### 4. 访问测试
- 主页面：http://localhost/
- 多源数据分析：http://localhost/analysis
- 预测模块：http://localhost/predictor
- 高级可视化：http://localhost/visualization

---

## 注意事项

1. **Store持久化**
   - 当前store数据仅在内存中，刷新页面后会丢失
   - 如需持久化，可以使用 `pinia-plugin-persistedstate` 插件

2. **数据集类型匹配**
   - 微震数据集只会在微震相关模块中被记忆和使用
   - 支架阻力数据集只会在支架阻力相关模块中被记忆和使用

3. **后端支持**
   - 确保后端预测API支持 `dataset_id` 参数
   - 如果后端尚未支持，需要先更新后端代码

4. **错误处理**
   - 数据集不存在时会显示空列表
   - 训练失败会显示具体错误信息
   - 网络错误会在控制台输出日志

---

## 文件清单

### 新建文件
- `frontend/src/stores/datasetStore.js` - Pinia状态管理store

### 修改文件
- `frontend/src/App.vue` - 添加导航链接
- `frontend/src/components/analysis/MicroseismicScatter.vue` - 添加watch和store导入
- `frontend/src/components/analysis/MicroseismicDensity.vue` - 添加watch和store导入
- `frontend/src/components/analysis/SupportDWTAnalysis.vue` - 添加watch和store导入
- `frontend/src/components/analysis/WaveletComparison.vue` - 添加watch和store导入
- `frontend/src/components/predictor/MicroseismicPredictor.vue` - 添加数据集选择功能
- `frontend/src/components/predictor/SupportPredictor.vue` - 添加数据集选择功能

---

## 完成状态

- ✅ 导航栏添加"高级可视化"链接
- ✅ 创建数据集状态管理Store
- ✅ 分析模块保存数据集选择
- ✅ 预测模块支持数据集联动
- ✅ 预测模块支持双模式数据源
- ✅ UI样式优化和视觉反馈
- ✅ 文档编写完成

**状态**: 代码已完成，等待构建部署和测试验证
