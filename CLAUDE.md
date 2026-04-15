# 项目约定

## 新增 Provider
1. 在 app/providers/ 创建新文件，继承 ImageProvider 基类
2. 在 app/providers/__init__.py 的工厂函数中注册
3. 在 app/credits.py 中添加积分定价
4. 在 config.py 的模型列表中注册

## 新增 Pipeline
1. 在 app/pipelines/ 创建新文件
2. 在 app/routes/pipeline_routes.py 注册路由
3. 在前端添加对应 composable

## 分支规范
所有修改提交到 claude/feature-xxx 分支，开 Draft PR
