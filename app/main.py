"""
FastAPI 应用主入口
"""
from fastapi import FastAPI
from .config import settings

app = FastAPI(
    title="Script Killer AI",
    description="剧本杀 AI 助手 - 基于 LangGraph 和 RAG 的智能系统",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    print(f"应用启动，环境：{settings.APP_ENV}")
    # TODO: 初始化 LLM 客户端、RAG 检索器等


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    print("应用关闭，清理资源中...")
    # TODO: 清理连接和资源


@app.get("/")
async def root():
    """根路由"""
    return {
        "message": "欢迎使用 Script Killer AI",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


# TODO: 添加更多 API 路由
# from .routes import chat, game, character
# app.include_router(chat.router, prefix="/api/chat")
# app.include_router(game.router, prefix="/api/game")
