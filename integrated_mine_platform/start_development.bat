@echo off
chcp 65001 >nul
echo ========================================
echo 集成矿山智能预测平台 - 开发环境启动
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查Node.js是否安装
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到Node.js，请先安装Node.js 16+
    pause
    exit /b 1
)

echo [信息] 正在启动后端服务...
start "Django后端" cmd /k "python manage.py runserver"

echo [信息] 等待2秒后启动前端服务...
timeout /t 2 /nobreak >nul

echo [信息] 正在启动前端服务...
start "Vue前端" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo 启动完成！
echo ========================================
echo.
echo 后端地址: http://127.0.0.1:8000
echo 前端地址: http://localhost:5173
echo 管理后台: http://127.0.0.1:8000/admin
echo.
echo 按任意键关闭此窗口...
pause >nul
