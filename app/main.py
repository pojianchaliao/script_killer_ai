"""
FastAPI 应用主入口

@Java 程序员提示:
- FastAPI 类似 Java 的 Spring Boot，是 Web 框架
- @app 装饰器类似 Java 的注解 (@GetMapping 等)
- async/await 是异步编程，类似 Java 的 CompletableFuture
- 模块即文件，import 就是导入其他文件的类/函数
"""
from fastapi import FastAPI  # 导入 FastAPI 类
from .config import settings  # 从 config.py 导入 settings 对象


# ==================== 创建 FastAPI 应用实例 ====================
# 类似 Java: @SpringBootApplication
# 这一行创建了 Web 应用实例
app = FastAPI(
    title="Script Killer AI",  # 应用标题
    description="剧本杀 AI 助手 - 基于 LangGraph 和 RAG 的智能系统",  # 描述
    version="1.0.0"  # 版本号
)


# ==================== 应用生命周期回调 ====================
# @app.on_event("startup") 是装饰器
# 类似 Java 的 @EventListener(ApplicationStartedEvent.class)
@app.on_event("startup")
async def startup_event():
    """
    应用启动时初始化
    
    @Java 程序员提示:
    - async def 定义异步函数 (协程)
    - 类似 Java 的 CompletableFuture 或 Reactor
    - 启动时自动执行，类似 Spring 的 @PostConstruct
    """
    # f-string 格式化字符串，类似 Java: "文本 " + variable
    print(f"应用启动，环境：{settings.APP_ENV}")
    
    # TODO: 初始化 LLM 客户端、RAG 检索器等
    # 这里可以初始化全局资源


# 应用关闭时的回调
@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭时清理资源
    
    @Java 程序员提示:
    - 类似 Java 的 @PreDestroy 或 DisposableBean
    - 用于关闭数据库连接、释放资源等
    """
    print("应用关闭，清理资源中...")
    
    # TODO: 清理连接和资源


# ==================== HTTP 路由定义 ====================
# @app.get("/") 是路由装饰器
# 类似 Java 的 @GetMapping("/")
@app.get("/")
async def root():
    """
    根路由 - 访问 http://localhost:8000/ 时触发
    
    Returns:
        dict: 返回 JSON 响应
    
    @Java 程序员提示:
    - 函数返回值会自动转换为 JSON
    - 类似 Java 的 @RestController 返回对象
    - Python 的字典 {key: value} 会自动转为 JSON 对象
    """
    # 返回字典，自动转为 JSON
    return {
        "message": "欢迎使用 Script Killer AI",
        "version": "1.0.0",
        "status": "running"
    }


# 健康检查接口
@app.get("/health")
async def health_check():
    """
    健康检查 - 用于监控和负载均衡
    
    Returns:
        dict: 健康状态
    
    @Java 程序员提示:
    - 类似 Spring Boot Actuator 的 /health 端点
    - Kubernetes 和负载均衡器会定期调用此接口
    """
    return {"status": "healthy"}  # 返回健康状态


# ==================== 路由模块化 (注释示例) ====================
# TODO: 添加更多 API 路由
# 类似 Java 的 @Configuration + @ComponentScan

# 方式 1: 直接导入并注册路由
# from .routes import chat, game, character
# app.include_router(chat.router, prefix="/api/chat")
# app.include_router(game.router, prefix="/api/game")

# 方式 2: 使用 APIRouter (推荐)
# from fastapi import APIRouter
# chat_router = APIRouter()
# @chat_router.get("/messages")
# async def get_messages(): ...
# app.include_router(chat_router, prefix="/api/chat")

# @Java 程序员提示:
# - include_router 类似 Spring 的 importResource
# - prefix 设置路由前缀，类似 @RequestMapping("/api/chat")
# - 每个模块可以有自己 router，实现代码组织化
