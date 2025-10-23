# 代码迁移指南

## 需要复制的文件和目录

### 1. 支架阻力预测模块 (predictor_app)

**源路径**: `G:\tzb\Version3\Old_version\Old_version\mine_project\predictor_app\`

**目标路径**: `G:\tzb\Version3\integrated_mine_platform\predictor_app\`

**需要复制的文件**:
```
predictor_app/
├── __init__.py          ✅ 直接复制
├── admin.py             ✅ 直接复制
├── apps.py              ✅ 直接复制
├── models.py            ✅ 直接复制（包含TrainingRun、ModelResult模型）
├── views.py             ✅ 直接复制（包含API端点）
├── urls.py              ✅ 直接复制
├── services.py          ✅ 直接复制（核心业务逻辑）
├── predictor_app_services.py  ✅ 如果存在则复制
├── migrations/          ⚠️ 不要复制！会重新生成
└── models_lib/          ✅ 完整复制（所有ML模型）
    ├── __init__.py
    ├── lstm_model.py
    ├── gru_model.py
    ├── transformer_model.py
    └── ... (其他模型文件)
```

### 2. 微震预测模块 (microseismic_app)

**源路径**: `G:\tzb\Version3\Old_version\Old_version\mine_project\microseismic_app\`

**目标路径**: `G:\tzb\Version3\integrated_mine_platform\microseismic_app\`

**需要复制的文件**:
```
microseismic_app/
├── __init__.py          ✅ 直接复制
├── admin.py             ✅ 直接复制
├── apps.py              ✅ 直接复制
├── models.py            ✅ 直接复制（包含MSTrainingRun、MSModelResult模型）
├── views.py             ✅ 直接复制
├── urls.py              ✅ 直接复制
├── services.py          ✅ 直接复制
├── microseismic_app_services.py  ✅ 如果存在则复制
├── migrations/          ⚠️ 不要复制！
├── models_lib/          ✅ 完整复制（LSTM、Transformer、Mamba模型）
└── preprocessor/        ✅ 完整复制（预处理模块）
    ├── __init__.py
    ├── preprocess.py    （ZIP文件处理）
    └── dataloader.py    （数据加载器）
```

### 3. 媒体文件（可选）

**源路径**: `G:\tzb\Version3\Old_version\Old_version\mine_project\media\`

**目标路径**: `G:\tzb\Version3\integrated_mine_platform\media\`

⚠️ **注意**: 这些是训练结果文件，文件很大，可以选择性复制或不复制（会重新生成）

---

## 复制步骤（PowerShell命令）

### 步骤1: 复制 predictor_app

```powershell
# 进入项目目录
cd G:\tzb\Version3\integrated_mine_platform

# 复制整个 predictor_app 目录（排除migrations）
$source = "G:\tzb\Version3\Old_version\Old_version\mine_project\predictor_app"
$dest = "G:\tzb\Version3\integrated_mine_platform\predictor_app"

# 创建目标目录
New-Item -ItemType Directory -Force -Path $dest

# 复制所有Python文件
Copy-Item "$source\*.py" -Destination $dest -Force

# 复制 models_lib 目录
Copy-Item "$source\models_lib" -Destination $dest -Recurse -Force

# 创建空的 migrations 目录
New-Item -ItemType Directory -Force -Path "$dest\migrations"
New-Item -ItemType File -Force -Path "$dest\migrations\__init__.py"
```

### 步骤2: 复制 microseismic_app

```powershell
# 复制整个 microseismic_app 目录
$source = "G:\tzb\Version3\Old_version\Old_version\mine_project\microseismic_app"
$dest = "G:\tzb\Version3\integrated_mine_platform\microseismic_app"

# 创建目标目录
New-Item -ItemType Directory -Force -Path $dest

# 复制所有Python文件
Copy-Item "$source\*.py" -Destination $dest -Force

# 复制 models_lib 目录
Copy-Item "$source\models_lib" -Destination $dest -Recurse -Force

# 复制 preprocessor 目录
Copy-Item "$source\preprocessor" -Destination $dest -Recurse -Force

# 创建空的 migrations 目录
New-Item -ItemType Directory -Force -Path "$dest\migrations"
New-Item -ItemType File -Force -Path "$dest\migrations\__init__.py"
```

### 步骤3: 数据库迁移

```powershell
# 激活虚拟环境（如果有）
# .\venv\Scripts\Activate.ps1

# 生成迁移文件
python manage.py makemigrations predictor_app
python manage.py makemigrations microseismic_app

# 应用迁移
python manage.py migrate
```

### 步骤4: 验证复制结果

```powershell
# 检查目录结构
Get-ChildItem -Path "G:\tzb\Version3\integrated_mine_platform\predictor_app" -Recurse -Name
Get-ChildItem -Path "G:\tzb\Version3\integrated_mine_platform\microseismic_app" -Recurse -Name
```

---

## 一键复制脚本

我已经创建了自动化脚本，请查看：`copy_apps.bat`

使用方法：
```powershell
# 在 integrated_mine_platform 目录下运行
.\copy_apps.bat
```

---

## 注意事项

### ⚠️ 需要手动检查的内容

1. **导入路径**: 如果原代码中有相对导入，可能需要调整
2. **配置文件**: 检查是否有硬编码的路径需要修改
3. **依赖包**: 确保 requirements.txt 包含所有需要的包
4. **静态文件**: Chart.js 等前端库的引用

### ✅ 已经配置好的内容

1. **settings.py**: 已经注册 predictor_app 和 microseismic_app
2. **urls.py**: 已经配置好路由映射
3. **CORS**: 已经配置跨域支持
4. **Media文件**: 已经配置 MEDIA_ROOT 和 MEDIA_URL

### 🔧 可能需要的依赖包

确保 requirements.txt 包含以下包（如果原项目使用）：
- torch (深度学习)
- numpy, pandas (数据处理)
- scikit-learn (机器学习)
- matplotlib, seaborn (可视化)
- Chart.js (前端图表 - 通过npm安装)

---

## 测试步骤

复制完成后，按以下步骤测试：

1. **启动后端**:
   ```powershell
   python manage.py runserver
   ```

2. **测试API**:
   - http://127.0.0.1:8000/api/predictor/
   - http://127.0.0.1:8000/api/microseismic/

3. **检查Admin**:
   - http://127.0.0.1:8000/admin
   - 查看是否显示新的模型

4. **前端集成**: 
   - 更新 PredictorPage.vue
   - 更新 MicroseismicPage.vue

---

## 遇到问题？

### ImportError: No module named 'xxx'
解决: `pip install xxx` 或更新 requirements.txt

### 数据库迁移错误
解决: 删除 db.sqlite3 重新迁移，或检查模型定义

### 路径错误
解决: 检查 settings.py 中的 MEDIA_ROOT, STATIC_ROOT 等配置

### CORS错误
解决: 已在 settings.py 配置，检查前端请求地址是否正确
