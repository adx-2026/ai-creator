#!/bin/bash

# ==========================================
# AI Creator Space 一键更新部署脚本
# ==========================================

# 定义变量
PROJECT_DIR="/opt/ai-creater"
SERVICE_NAME="ai-creater.service"
BRANCH="main" 

echo "🚀 开始更新项目: AI Creator Space..."
echo "------------------------------------------"

# 1. 进入项目目录
cd $PROJECT_DIR || { echo "❌ 错误: 找不到目录 $PROJECT_DIR"; exit 1; }

# 2. 从 GitHub 拉取最新代码
echo "📦 正在从 GitHub ($BRANCH 分支) 拉取最新代码..."
git pull origin $BRANCH

# 3. 更新 Python 虚拟环境依赖
echo "🐍 正在检查并更新 Python 依赖..."
source venv/bin/activate
pip install -r requirements.txt
deactivate

# 4. 重启 Systemd 后台服务
echo "🔄 正在重启后台服务 ($SERVICE_NAME)..."
sudo systemctl restart $SERVICE_NAME

# 5. 检查服务运行状态
echo "------------------------------------------"
if sudo systemctl is-active --quiet $SERVICE_NAME; then
    echo "✅ 更新成功！服务已正常运行。"
else
    echo "❌ 警告：服务启动失败，请使用 'sudo journalctl -u $SERVICE_NAME -n 50' 查看报错日志。"
fi
echo "=========================================="