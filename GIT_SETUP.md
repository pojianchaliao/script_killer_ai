# Git 版本控制设置指南

## 🚀 快速开始

### 方法一：使用 PowerShell 脚本（推荐）

在项目根目录打开 PowerShell，执行：

```powershell
.\init_git.ps1
```

脚本会自动完成：
- ✅ 初始化 Git 仓库
- ✅ 配置用户信息（会提示输入）
- ✅ 添加所有文件
- ✅ 创建首次提交

### 方法二：手动命令

在 PowerShell 中依次执行以下命令：

```powershell
# 1. 进入项目目录
cd C:\Users\zbz\PycharmProjects\script_killer_ai

# 2. 初始化 Git 仓库
git init

# 3. 配置用户信息（替换为你的信息）
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 4. 添加所有文件
git add .

# 5. 创建首次提交
git commit -m "Initial commit: 项目初始化

- FastAPI 应用框架
- LangGraph 状态机定义
- RAG 检索模块
- 智谱 AI 集成
- 剧本杀游戏工具"
```

## 📋 .gitignore 说明

已创建 `.gitignore` 文件，忽略以下内容：

- **Python 缓存**: `__pycache__/`, `*.pyc`
- **虚拟环境**: `.venv/`, `venv/`
- **IDE 配置**: `.idea/`, `.vscode/`
- **环境变量**: `.env.local`, `.env.*.local`
- **日志文件**: `*.log`
- **数据库**: `*.db`, `*.sqlite`
- **测试文件**: `.pytest_cache/`, `.coverage`
- **临时文件**: `tempfile_*.py`
- **向量数据**: `data/vector_store/`, `*.faiss`

## 🔗 关联远程仓库

如果你有 GitHub/Gitee 仓库，可以关联：

```powershell
# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/yourusername/script_killer_ai.git

# 重命名分支为 main
git branch -M main

# 推送到远程仓库
git push -u origin main
```

## 📊 常用 Git 命令

```powershell
# 查看状态
git status

# 查看提交历史
git log --oneline

# 查看文件变化
git diff

# 添加文件
git add <filename>
git add .  # 添加所有文件

# 提交更改
git commit -m "提交信息"

# 拉取远程更新
git pull

# 推送本地更改
git push

# 创建新分支
git checkout -b feature-name

# 切换分支
git checkout branch-name
```

## ⚠️ 注意事项

1. **敏感信息保护**：
   - `.env` 文件包含 API 密钥，已添加到 `.gitignore`
   - 不要将真实密钥提交到仓库
   - 使用 `.env.example` 作为示例模板

2. **大文件处理**：
   - 如果有大型模型文件或数据集，建议使用 Git LFS
   - 安装：`git lfs install`
   - 跟踪文件：`git lfs track "*.bin"`

3. **依赖管理**：
   - `requirements.txt` 已加入版本控制
   - 定期更新：`pip freeze > requirements.txt`

## 🎯 下一步

1. ✅ 初始化 Git 仓库
2. ✅ 配置用户信息
3. ✅ 创建首次提交
4. 🔲 创建远程仓库（可选）
5. 🔲 推送到远程仓库（可选）
