# AI Creator Space

AI 多模态创作平台 — 支持文生图、图生图、图片裂变、AI 尺寸重绘、图案提取、电商场景生图、3D 刺绣转换、视频生成等多种创作模式，内置异步任务队列与积分计费系统。

## 功能特性

- **多创作模式**: 文生图 (T2I)、图生图 (I2I)、图片裂变、AI 尺寸重绘、图案提取生图、电商场景生图、3D 刺绣转换、多图/多视频批量生成
- **多 AI 引擎**: 支持 Gemini、通义万相 (Qwen)、MiniMax、豆包 (Doubao)、万象视频 (Wan) 等多家模型
- **异步任务队列**: 后台队列处理 + 并发控制，支持子任务重试
- **积分计费系统**: 按模型计费，实时预估与扣费，余额不足自动拦截
- **阿里云限流器**: 自动限速、重试与退避，按模型分组
- **用户认证**: Session Token 认证，多用户隔离工作空间
- **提示词模板**: 公共/私有模板库，快捷加载与管理
- **拖拽/粘贴上传**: 支持批量文件拖拽、Ctrl+V 粘贴图片
- **打包下载**: 单图下载（支持自定义文件名）与 ZIP 打包下载

## 项目结构

```
ai-creater/
├── main.py                     # FastAPI 入口（注册路由 + 启动队列）
├── run.py                      # Uvicorn 启动脚本
├── config.json                 # 用户与模型配置
├── config_example.json         # 配置示例
├── requirements.txt            # Python 依赖
│
├── app/                        # 后端核心模块
│   ├── __init__.py
│   ├── config.py               # 环境变量加载、模型列表、用户配置
│   ├── auth.py                 # Session 管理 & OAuth2 依赖
│   ├── credits.py              # 积分定价表、计费函数、扣费锁
│   ├── rate_limiter.py         # 阿里云 API 限流器（per-model 令牌桶 + 重试退避）
│   ├── models.py               # 子任务构建、进度刷新、结果规范化
│   ├── settings.py             # 用户设置持久化
│   ├── job_queue.py            # 任务队列 (JobQueue) + 后台消费者 (process_queue)
│   │
│   ├── providers/              # AI 模型适配层
│   │   ├── __init__.py         # ImageProvider 基类 + get_provider_for_model 工厂
│   │   ├── gemini.py           # Google Gemini
│   │   ├── qwen.py             # 通义万相 (DashScope)
│   │   ├── minimax.py          # MiniMax REST API
│   │   ├── doubao.py           # 豆包 (Volcengine Ark)
│   │   └── wan_video.py        # 万象视频 (DashScope + OSS 上传)
│   │
│   ├── pipelines/              # 多步骤 AI 流水线
│   │   ├── __init__.py
│   │   ├── extract.py          # 图案提取 (VL→图像生成)
│   │   ├── ecommerce.py        # 电商场景 (VL→文本→图像)
│   │   └── threed.py           # 3D 刺绣转换 (VL→图像)
│   │
│   └── routes/                 # API 路由
│       ├── __init__.py
│       ├── auth_routes.py      # POST /api/login, GET /api/me, POST /api/logout
│       ├── credit_routes.py    # GET/POST /api/credit
│       ├── model_routes.py     # GET /api/models
│       ├── template_routes.py  # GET/POST /api/templates, DELETE /api/templates/{scope}/{name}
│       ├── settings_routes.py  # GET/POST /api/settings
│       ├── job_routes.py       # POST /api/jobs, GET /api/jobs, retry, delete, download
│       ├── file_routes.py      # GET /api/images/*, GET /api/videos/*
│       └── pipeline_routes.py  # POST /api/ecommerce/*, POST /api/threed/*
│
├── static/                     # 前端静态文件
│   ├── index.html              # HTML 模板（Vue 3 模板 + ES Module 入口）
│   │
│   ├── css/
│   │   └── app.css             # 自定义样式
│   │
│   └── js/
│       ├── main.js             # Vue 根组件：组装所有 composable，返回模板绑定
│       ├── api.js              # Token 管理 & authFetch 请求封装
│       ├── constants.js        # 模式标签、比例选项、积分定价常量
│       │
│       └── composables/        # Vue 3 Composition API 模块
│           ├── useAuth.js      # 登录/登出/会话检查
│           ├── useCredits.js   # 积分余额查询 & 消耗预估
│           ├── useForm.js      # 表单状态、文件处理、任务计算、任务提交
│           ├── useJobs.js      # 任务列表管理、删除、重试
│           ├── useTemplates.js # 模板 CRUD
│           ├── useDownload.js  # 下载命名弹窗 & 文件下载
│           ├── useEcommerce.js # 电商场景 3 步流程状态机
│           └── useThreed.js    # 3D 转换 2 步流程状态机
│
└── users/                      # 用户数据目录（自动创建）
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 前端框架 | Vue 3 Composition API (CDN ESM, 无构建工具) |
| 样式 | Tailwind CSS (CDN) + Font Awesome |
| AI 模型 | Gemini / 通义万相 / MiniMax / 豆包 / 万象视频 |
| 认证 | Session Token + OAuth2PasswordBearer |
| 任务队列 | asyncio Queue + 后台消费者协程 |
| 限流 | 自研令牌桶 (per-model) + 指数退避重试 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

复制 `config_example.json` 为 `config.json`，配置用户账号与密码。

在 `.env` 文件中配置 AI 模型 API Key（参考各 provider 源码中的环境变量名）：

```env
GEMINI_API_KEY=your_key
DASHSCOPE_API_KEY=your_key
MINIMAX_GROUP_ID=your_group
MINIMAX_API_KEY=your_key
ARK_API_KEY=your_key
```

### 3. 启动

```bash
python run.py
```

浏览器打开 `http://127.0.0.1:8000`，使用 `config.json` 中配置的账号登录即可。

## 架构概览

```
浏览器 (Vue 3 SPA)
    │
    ├── ES Modules: main.js → composables/*
    │
    ▼
FastAPI (main.py)
    ├── routes/*          ← API 路由层
    ├── providers/*       ← AI 模型适配层
    ├── pipelines/*       ← 多步骤 AI 流水线
    ├── job_queue.py      ← 异步任务队列
    ├── credits.py        ← 积分计费
    └── rate_limiter.py   ← 限流控制
```

- **前端**: Vue 3 Composition API + ES Modules（无构建工具），composable 模式拆分状态与逻辑
- **后端**: FastAPI APIRouter 模式，按职责拆分为 routes / providers / pipelines
- **Provider 模式**: `ImageProvider` 抽象基类 + 工厂函数 `get_provider_for_model()`，新增模型只需实现一个 provider
- **Pipeline 模式**: 多步骤 AI 流水线（VL 理解 → 文本生成 → 图像生成），每步独立可测试

## Claude Code Routine API 开发流程

本项目支持通过 Claude Code Routine API 进行自动化开发。当 Routine 被触发时，AI Agent 会按照以下步骤执行：

### 执行步骤

1. **读取项目规范** — 阅读根目录 CLAUDE.md，严格遵守架构约定和分支规范
2. **获取待处理 Issue** — 查找带有 `claude` label 且状态为 open 的 Issue，选择编号最小的一个处理。如果没有符合条件的 Issue，则不执行任何修改
3. **实现需求** — 按 CLAUDE.md 的架构约定实现，只读取与本次任务直接相关的文件
4. **提交 PR** — 按分支规范提交代码并开 PR，PR 描述包含修改文件列表、实现说明、测试步骤和 `Closes #issue号`

### 使用方式

通过 GitHub API 创建带有 `claude` label 的 Issue 即可触发开发流程。
