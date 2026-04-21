#!/bin/bash

# ==========================================
# AI Creator Space 一键更新部署脚本
# ==========================================

# 定义变量
PROJECT_DIR="/opt/ai-creator"
SERVICE_NAME="ai-creator.service"
BRANCH="main" 

echo "🚀 开始更新项目: AI Creator Space..."
echo "------------------------------------------"

# 1. 进入项目目录
cd $PROJECT_DIR || { echo "❌ 错误: 找不到目录 $PROJECT_DIR"; exit 1; }

# 2. 同步服务器上的本地变更（public_templates.json）
echo "📤 正在检查并提交服务器本地变更..."
if ! git diff --quiet public_templates.json; then
    git add public_templates.json
    git commit -m "sync: update public_templates.json from server"
    git push origin $BRANCH
    echo "✅ public_templates.json 已提交并推送。"
else
    echo "ℹ️  public_templates.json 无变更，跳过提交。"
fi

# 3. 从 GitHub 拉取最新代码
echo "📦 正在从 GitHub ($BRANCH 分支) 拉取最新代码..."
git pull --no-edit origin $BRANCH

# 4. 更新 Python 虚拟环境依赖
echo "🐍 正在检查并更新 Python 依赖..."
source venv/bin/activate
pip install -r requirements.txt | grep -v "Requirement already satisfied"
deactivate

# 5. 重启 Systemd 后台服务
echo "🔄 正在重启后台服务 ($SERVICE_NAME)..."
sudo systemctl restart $SERVICE_NAME

# 6. 检查服务运行状态
echo "------------------------------------------"
if sudo systemctl is-active --quiet $SERVICE_NAME; then
    echo "✅ 更新成功！服务已正常运行。"
else
    echo "❌ 警告：服务启动失败，请使用 'sudo journalctl -u $SERVICE_NAME -n 50' 查看报错日志。"
fi
echo "=========================================="