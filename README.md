# Script Killer AI - 剧本杀 AI 助手

基于 LangGraph 和 RAG 的智能剧本杀系统

## 📦 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

**注意**：安装完成后 IDE 的"未解析的引用"错误会自动消失。

### 2. 配置环境变量

编辑 `.env` 文件，设置你的智谱 API 密钥：

```bash
ZHIPU_API_KEY=your_actual_api_key_here
```

### 3. 启动应用

```powershell
uvicorn app.main:app --reload
```

访问 http://127.0.0.1:8000 查看 API 文档。

## 📁 项目结构

```
script_killer_ai/
├── app/
│   ├── __init__.py          # 模块初始化
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # Pydantic 配置管理
│   ├── graph/               # LangGraph 状态机
│   │   ├── state.py        # State Schema
│   │   ├── nodes.py        # 4 个 Agent 节点
│   │   └── workflow.py     # 工作流编译
│   ├── rag/                 # RAG 检索模块
│   │   ├── embedding.py    # BGE 模型封装
│   │   ├── retriever.py    # 检索器逻辑
│   │   └── ingest.py       # 数据入库脚本
│   ├── tools/               # 自定义工具
│   │   └── game_tools.py   # 游戏工具函数
│   └── utils/               # 工具模块
│       └── llm_client.py   # 智谱 API 封装
├── data/                    # 数据目录
├── .env                     # 环境变量配置
├── .gitignore              # Git 忽略配置
├── requirements.txt        # Python 依赖
└── README.md               # 本文件
```

## 🎯 核心功能

- **FastAPI** 应用框架
- **LangGraph** 状态机（4 个 Agent：主持人、侦探、嫌疑人、证人）
- **RAG** 检索增强生成（支持 BGE Embedding）
- **智谱 AI** 大模型集成
- **剧本杀专用工具**（角色生成、线索校验等）

## ⚠️ 注意事项

1. **依赖未安装时的错误**：IDE 显示的"未解析的引用"错误是因为还没安装依赖包，执行 `pip install -r requirements.txt` 后会自动解决。

2. **API 密钥安全**：`.env` 文件包含敏感信息，不要提交到版本控制系统。

3. **Git 版本控制**：已配置 `.gitignore`，自动忽略敏感文件和缓存。

## 🔧 开发指南

### 安装依赖后验证

```powershell
python -c "from app.main import app; print('✓ 应用加载成功')"
```

### 运行数据入库脚本

```powershell
python -m app.rag.ingest
```

### 测试 API

```powershell
curl http://127.0.0.1:8000/health
```

## 📝 下一步

1. ✅ 安装依赖包
2. ✅ 配置 API 密钥
3. ✅ 启动应用测试
4. 🔲 实现具体的业务逻辑（参考各文件的 TODO 注释）
5. 🔲 准备剧本杀数据放入 `data/` 目录
6. 🔲 运行数据入库脚本

祝你使用愉快！🎉
