-- PostgreSQL 初始化脚本
-- 创建必要的扩展和初始配置

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 设置默认字符集
ALTER DATABASE mine_platform_db SET timezone TO 'Asia/Shanghai';

-- 创建初始用户权限（如果需要）
-- CREATE ROLE mine_app WITH LOGIN PASSWORD 'mine_app_password';
-- GRANT ALL PRIVILEGES ON DATABASE mine_platform_db TO mine_app;
