# 集成矿山智能预测平台 - 项目说明

## 项目概述

本项目是一个集成了两个现有矿山预测系统的全新智能平台，采用深蓝色科技美术风格，提供数据大屏展示和完整的预测分析功能。

## 已完成功能

### 1. 项目基础架构 ✅
- Django 5.0 后端框架
- Vue 3 + Vite 前端框架
- TailwindCSS 样式系统
- ECharts 数据可视化

### 2. 数据大屏模块 ✅
- 实时统计数据展示
- 任务执行趋势图表
- 预警级别分布可视化
- 系统健康状态监控
- 深蓝色科技风格UI设计

### 3. 实时监控模块 ✅
- 微震能量监测
- 支架阻力监测
- 瓦斯浓度监测
- 环境温度监测
- 实时数据图表展示

### 4. 数据查看模块 ✅
- 数据库表概览
- 数据记录浏览
- 多应用数据统一管理

### 5. 后端API ✅
- Dashboard API (数据大屏)
- Monitoring API (实时监控)
- 数据查看API
- RESTful API设计

## 待集成功能

### 1. 支架阻力预测 ⏳
需要从 `Old_version/mine_project/predictor_app/` 集成：
- 多模型训练功能
- 数据预处理
- 结果可视化
- GPU加速支持

### 2. 微震预测 ⏳
需要从 `Old_version/mine_project/microseismic_app/` 集成：
- ZIP文件预处理
- LSTM/Transformer/Mamba模型
- 时间序列预测
- 结果分析

### 3. 综采监控 ⏳
需要从 `Old_version/version2/` 集成：
- 实时数据接收
- MEA/HSA算法
- 多页面应用

## 如何继续开发

### 集成原有算法的步骤

#### 集成支架阻力预测：
1. 复制 `Old_version/mine_project/predictor_app/` 目录
2. 将其放入新项目根目录
3. 更新 `settings.py` 中的 INSTALLED_APPS（已配置）
4. 迁移数据库模型：
   ```bash
   python manage.py makemigrations predictor_app
   python manage.py migrate
   ```
5. 在前端创建对应页面组件
6. 集成 `models_lib/` 中的ML模型

#### 集成微震预测：
1. 复制 `Old_version/mine_project/microseismic_app/` 目录
2. 将其放入新项目根目录
3. 更新配置（已在settings.py中）
4. 迁移数据库：
   ```bash
   python manage.py makemigrations microseismic_app
   python manage.py migrate
   ```
5. 创建前端页面
6. 集成预处理和模型代码

### 前端页面开发
当前已有占位页面：
- `PredictorPage.vue` - 支架阻力预测页面框架
- `MicroseismicPage.vue` - 微震预测页面框架
- `MonitoringPage.vue` - 监控页面（已实现基础功能）

您需要在这些页面中添加：
- 文件上传组件
- 参数配置面板
- 训练任务监控
- 结果展示图表

## 设计风格指南

### 颜色方案
```css
主色调：#0056e0 (深蓝)
科技蓝：#0066ff
青色：#00d4ff
紫色：#7c3aed
背景暗色：#0a0e27
背景更暗：#050812
```

### 组件样式类
- `.tech-card` - 科技风格卡片
- `.tech-button` - 科技风格按钮
- `.tech-input` - 科技风格输入框
- `.glow-text` - 发光文字效果
- `.gradient-border` - 渐变边框

### 动画效果
- 悬停阴影发光
- 平滑过渡动画
- 数据流动效果
- 脉冲动画

## 数据库迁移

初始化数据库：
```bash
python manage.py makemigrations
python manage.py migrate
```

创建测试数据：
```bash
python manage.py shell
```
然后执行：
```python
from dashboard_app.models import SystemStatistics, AlertRecord
from monitoring_app.models import MonitoringData

# 创建初始统计数据
SystemStatistics.objects.create(
    total_tasks=150,
    running_tasks=5,
    completed_tasks=130,
    failed_tasks=15,
    predictor_tasks=80,
    microseismic_tasks=70,
    total_data_records=5000,
    system_health='normal'
)

# 创建测试预警
AlertRecord.objects.create(
    level='warning',
    title='支架阻力异常',
    message='3号支架阻力超出正常范围',
    source='支架阻力监测系统'
)
```

## 启动说明

### 首次启动
1. 安装后端依赖：`pip install -r requirements.txt`
2. 安装前端依赖：`cd frontend && npm install`
3. 迁移数据库：`python manage.py migrate`
4. 启动开发服务器：`.\start_development.bat`

### 日常开发
- 直接运行 `start_development.bat`
- 或手动启动前后端服务

## 文件结构说明

```
integrated_mine_platform/
├── integrated_mine/         # Django配置
├── dashboard_app/           # 数据大屏（已完成）
├── monitoring_app/          # 实时监控（已完成）
├── predictor_app/          # 支架阻力预测（待集成）
├── microseismic_app/       # 微震预测（待集成）
├── frontend/               # Vue前端（已完成框架）
│   ├── src/
│   │   ├── components/    # 可复用组件
│   │   ├── pages/         # 页面组件
│   │   ├── router/        # 路由配置
│   │   └── style.css      # 全局样式
├── media/                  # 媒体文件
├── requirements.txt        # Python依赖
├── README.md              # 用户文档
└── PROJECT_GUIDE.md       # 本文件（开发指南）
```

## 下一步工作

1. **复制原有app代码**：将 `predictor_app` 和 `microseismic_app` 复制到新项目
2. **数据库迁移**：运行 makemigrations 和 migrate
3. **前端集成**：完善预测页面的UI和逻辑
4. **测试**：确保所有功能正常工作
5. **优化**：性能优化和用户体验改进

## 技术支持

如有问题，请查看：
- Django文档：https://docs.djangoproject.com/
- Vue 3文档：https://vuejs.org/
- TailwindCSS文档：https://tailwindcss.com/
- ECharts文档：https://echarts.apache.org/

祝开发顺利！🚀
