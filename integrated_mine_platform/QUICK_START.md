# 快速部署指南

## 📋 两个问题的答案

### 1️⃣ 代码复制位置

#### 自动复制（推荐）
```powershell
cd G:\tzb\Version3\integrated_mine_platform
.\copy_apps.bat
```

#### 手动复制
从这里复制：
- `G:\tzb\Version3\Old_version\Old_version\mine_project\predictor_app\`
- `G:\tzb\Version3\Old_version\Old_version\mine_project\microseismic_app\`

到这里：
- `G:\tzb\Version3\integrated_mine_platform\predictor_app\`
- `G:\tzb\Version3\integrated_mine_platform\microseismic_app\`

详细说明请查看：`MIGRATION_GUIDE.md`

---

### 2️⃣ Docker一键部署

#### ✅ 是的，已经完全支持Docker一键部署！

**最简单的方式：**
```powershell
cd G:\tzb\Version3\integrated_mine_platform
.\docker_deploy.bat
```

**访问应用：**
- 前端：http://localhost
- 后端：http://localhost:8000
- 管理：http://localhost:8000/admin

详细说明请查看：`DOCKER_DEPLOY.md`

---

## 🚀 三种部署方式对比

### 方式一：本地开发（开发调试）
```powershell
# 1. 复制代码
.\copy_apps.bat

# 2. 安装依赖
pip install -r requirements.txt
cd frontend && npm install

# 3. 启动服务
.\start_development.bat
```

**优点**：
- ✅ 热更新，开发方便
- ✅ 可以直接调试代码
- ✅ 启动速度快

**缺点**：
- ❌ 需要配置Python和Node环境
- ❌ 环境依赖复杂

---

### 方式二：Docker Compose（推荐生产）
```powershell
# 一键部署
.\docker_deploy.bat
```

**优点**：
- ✅ 环境完全隔离
- ✅ 一键部署，无需配置
- ✅ 适合生产环境
- ✅ 易于迁移和扩展

**缺点**：
- ❌ 首次构建时间较长（10-15分钟）
- ❌ 需要安装Docker Desktop

---

### 方式三：Docker手动构建（高级用户）
```powershell
# 后端
docker build -t mine-backend .
docker run -d -p 8000:8000 mine-backend

# 前端
cd frontend
docker build -t mine-frontend .
docker run -d -p 80:80 mine-frontend
```

**优点**：
- ✅ 完全控制
- ✅ 可以自定义配置

**缺点**：
- ❌ 命令复杂
- ❌ 需要手动管理容器

---

## 📦 文件清单

### 代码迁移相关
- ✅ `MIGRATION_GUIDE.md` - 详细的代码迁移指南
- ✅ `copy_apps.bat` - 自动复制脚本

### Docker部署相关
- ✅ `Dockerfile` - 后端Docker镜像
- ✅ `frontend/Dockerfile` - 前端Docker镜像
- ✅ `docker-compose.yml` - Docker编排配置
- ✅ `frontend/nginx.conf` - Nginx配置
- ✅ `.dockerignore` - Docker忽略文件
- ✅ `DOCKER_DEPLOY.md` - Docker部署详细指南
- ✅ `docker_deploy.bat` - Docker一键部署脚本

### 项目文档
- ✅ `README.md` - 用户使用手册
- ✅ `PROJECT_GUIDE.md` - 开发指南
- ✅ `QUICK_START.md` - 本文件（快速开始）

---

## 🎯 推荐流程

### 第一次部署（推荐Docker）

1. **安装Docker Desktop**
   - 下载：https://www.docker.com/products/docker-desktop
   - 安装并启动

2. **复制原有代码**（可选，如果需要预测功能）
   ```powershell
   cd G:\tzb\Version3\integrated_mine_platform
   .\copy_apps.bat
   ```

3. **一键Docker部署**
   ```powershell
   .\docker_deploy.bat
   ```

4. **访问应用**
   - 打开浏览器：http://localhost
   - 查看数据大屏、监控等功能

5. **创建管理员账号**（可选）
   ```powershell
   docker-compose exec backend python manage.py createsuperuser
   ```

---

### 日常开发（推荐本地）

1. **启动开发服务器**
   ```powershell
   .\start_development.bat
   ```

2. **访问应用**
   - 前端：http://localhost:5173
   - 后端：http://127.0.0.1:8000

3. **修改代码**
   - 前端自动热更新
   - 后端需要刷新浏览器

---

## 🔧 常见操作

### Docker常用命令

```powershell
# 查看运行状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 进入容器
docker-compose exec backend bash
docker-compose exec frontend sh

# 查看资源使用
docker stats
```

### 数据库操作

```powershell
# 本地开发
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser

# Docker环境
docker-compose exec backend python manage.py makemigrations
docker-compose exec backend python manage.py migrate
docker-compose exec backend python manage.py createsuperuser
```

---

## ❓ 遇到问题？

### Docker相关
- **Docker未运行**：启动Docker Desktop
- **端口被占用**：修改docker-compose.yml中的端口
- **构建失败**：检查网络连接，或使用国内镜像

### 代码迁移相关
- **找不到模块**：运行 `pip install -r requirements.txt`
- **数据库错误**：删除db.sqlite3，重新migrate
- **路径错误**：检查MIGRATION_GUIDE.md中的路径

### 前端相关
- **npm install失败**：使用淘宝镜像 `npm config set registry https://registry.npmmirror.com`
- **页面空白**：检查浏览器控制台错误
- **API调用失败**：确认后端正在运行

---

## 📞 获取帮助

1. 查看对应的详细文档：
   - `MIGRATION_GUIDE.md` - 代码迁移
   - `DOCKER_DEPLOY.md` - Docker部署
   - `PROJECT_GUIDE.md` - 项目开发

2. 检查日志：
   ```powershell
   # 本地开发：查看终端输出
   # Docker：docker-compose logs -f
   ```

3. 重置环境：
   ```powershell
   # Docker
   docker-compose down -v
   docker-compose up -d --build
   
   # 本地
   # 删除 db.sqlite3 和 migrations/
   # 重新运行 migrate
   ```

---

## 🎉 完成！

现在您可以：
1. ✅ 使用Docker一键部署整个平台
2. ✅ 轻松复制原有代码到新项目
3. ✅ 选择最适合的开发/部署方式

祝使用愉快！🚀
