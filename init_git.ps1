# Git 仓库初始化脚本
# 在 PowerShell 中执行：.\init_git.ps1

Write-Host "=== 初始化 Git 仓库 ===" -ForegroundColor Green

# 检查是否已存在 .git 目录
if (Test-Path .git) {
    Write-Host "Git 仓库已存在！" -ForegroundColor Yellow
} else {
    # 初始化 Git 仓库
    git init
    Write-Host "✓ Git 仓库初始化成功" -ForegroundColor Green
}

# 配置 Git 用户信息（请修改为你的信息）
Write-Host "`n=== 配置 Git 用户信息 ===" -ForegroundColor Green
Write-Host "请设置你的 Git 用户名和邮箱：" -ForegroundColor Cyan

$username = Read-Host "输入你的 Git 用户名"
$email = Read-Host "输入你的 Git 邮箱"

if ($username -and $email) {
    git config user.name $username
    git config user.email $email
    Write-Host "✓ Git 用户信息配置成功" -ForegroundColor Green
} else {
    Write-Host "⚠ 跳过用户信息配置，请稍后手动配置" -ForegroundColor Yellow
}

# 添加所有文件到暂存区
Write-Host "`n=== 添加文件到暂存区 ===" -ForegroundColor Green
git add .
Write-Host "✓ 文件已添加到暂存区" -ForegroundColor Green

# 首次提交
Write-Host "`n=== 创建首次提交 ===" -ForegroundColor Green
git commit -m "Initial commit: 项目初始化

- FastAPI 应用框架
- LangGraph 状态机定义
- RAG 检索模块
- 智谱 AI 集成
- 剧本杀游戏工具"

Write-Host "`n=== Git 仓库初始化完成！ ===" -ForegroundColor Green
Write-Host "`n后续操作建议：" -ForegroundColor Cyan
Write-Host "1. 创建远程仓库并关联：" -ForegroundColor White
Write-Host "   git remote add origin <your-repo-url>"
Write-Host "   git branch -M main"
Write-Host "   git push -u origin main"
Write-Host ""
Write-Host "2. 查看 Git 状态：" -ForegroundColor White
Write-Host "   git status"
Write-Host ""
Write-Host "3. 查看提交历史：" -ForegroundColor White
Write-Host "   git log --oneline"
