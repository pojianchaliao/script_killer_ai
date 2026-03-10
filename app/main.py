"""
FastAPI 应用主入口 / 单一函数启动

@Java 程序员提示:
- FastAPI 类似 Java 的 Spring Boot，是 Web 框架
- @app 装饰器类似 Java 的注解 (@GetMapping 等)
- async/await 是异步编程，类似 Java 的 CompletableFuture
- 模块即文件，import 就是导入其他文件的类/函数
"""
from fastapi import FastAPI  # 导入 FastAPI 类

# 支持两种导入方式：
# 1. python -m app.main (相对导入)
# 2. 直接运行 main.py (绝对导入)
try:
    from .config import settings  # 从 config.py 导入 settings 对象
except ImportError:
    from config import settings  # 直接运行时使用绝对导入

from typing import Optional, Dict, Any  # 类型注解
import os


# ==================== 创建 FastAPI 应用实例 ====================
# 类似 Java: @SpringBootApplication
# 这一行创建了 Web 应用实例
app = FastAPI(
    title="Script Killer AI",  # 应用标题
    description="剧本杀 AI 助手 - 基于 LangGraph 和 RAG 的智能系统",  # 描述
    version="1.0.0"  # 版本号
)


# ==================== 向量库管理 ====================
def check_and_create_vector_store():
    """
    检查向量库是否存在，不存在则自动创建
    
    @Java 程序员提示:
    - 这是初始化方法，类似 Java 的 @PostConstruct
    - 使用懒加载模式：首次使用时才创建
    - 确保向量库始终可用
    """
    from app.rag.ingest_new import DocumentIngestor
    import chromadb
    from pathlib import Path
    
    vector_store_path = settings.VECTOR_STORE_PATH
    
    # 检查向量库目录是否存在
    if not os.path.exists(vector_store_path):
        print(f"⚠️  向量库不存在：{vector_store_path}")
        print("🔄 开始自动创建向量库...")
        
        try:
            # 创建入库工具实例
            ingestor = DocumentIngestor(
                data_dir=str(Path(__file__).parent / "data"),
                vector_store_path=vector_store_path
            )
            
            # 使用父子文档策略处理 JSON 数据
            json_file_path = str(Path(__file__).parent / "data" / "romance_three_kingdoms.json")
            
            if os.path.exists(json_file_path):
                ingestor.ingest_json_with_parent_child(
                    json_file_path=json_file_path,
                    collection_name="three_kingdoms_parent_child"
                )
                print(f"✅ 向量库创建完成！路径：{vector_store_path}")
            else:
                print(f"❌ 数据文件不存在：{json_file_path}")
                return False
                
        except Exception as e:
            print(f"❌ 创建向量库失败：{e}")
            return False
    else:
        print(f"✓ 向量库已存在：{vector_store_path}")
    
    return True


# ==================== 统一启动函数 ====================
def start_game(
    game_id: str = "game_001",
    initial_phase: str = "intro",
    auto_create_vector_store: bool = True,
    enable_rag_character_selection: bool = True  # 新增参数：启用 RAG 角色选择
) -> Dict[str, Any]:
    """
    单一函数启动剧本杀游戏
    
    Args:
        game_id: 游戏 ID，默认 "game_001"
        initial_phase: 初始阶段 ("intro", "investigation", "interrogation", "conclusion")
        auto_create_vector_store: 是否自动创建向量库（默认 True）
        enable_rag_character_selection: 是否启用 RAG 角色选择（默认 True）
    
    Returns:
        Dict[str, Any]: 游戏状态和结果
    
    @Java 程序员提示:
    - 这是统一的入口函数，类似 Java 的 main 方法
    - 整合了所有初始化逻辑
    - 支持同步调用，返回完整游戏状态
    """
    import asyncio
    from app.graph.workflow import run_game_turn
    from app.graph.state import GameState
    
    print("\n" + "="*60)
    print("🎮 剧本杀 AI 系统启动中...")
    print("="*60)
    
    # 步骤 1: 检查并创建向量库
    if auto_create_vector_store:
        vector_store_ready = check_and_create_vector_store()
        if not vector_store_ready:
            return {
                "success": False,
                "error": "向量库创建失败",
                "game_state": None
            }
    
    # 步骤 2: RAG 驱动的角色选择（如果启用）
    player_input = None
    rag_context = []
    historical_event = None
    rag_characters = []
    characters = {}
    active_character = None
    
    if enable_rag_character_selection and initial_phase == 'intro':
        print("\n" + "="*60)
        print("🎭 RAG 驱动的角色选择")
        print("="*60)
        
        try:
            from app.graph.rag_generator import RAGCharacterGenerator
            from colorama import Fore, Style
            
            # 创建 RAG 生成器
            print(f"{Fore.YELLOW}⚙️  正在初始化 RAG 系统...{Style.RESET_ALL}")
            generator = RAGCharacterGenerator()
            print(f"{Fore.GREEN}✅ RAG 系统初始化完成{Style.RESET_ALL}\n")
            
            # 选择历史事件
            print(f"{Fore.YELLOW}🔍 正在检索三国历史事件...{Style.RESET_ALL}")
            query = "三国 谋杀 阴谋 刺杀"
            historical_event = generator.select_historical_event(query)
            
            if historical_event:
                print(f"\n{Fore.GREEN}✅ 选中历史事件：{historical_event['event']}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}   戏剧价值：{historical_event['dramatic_value']}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}   历史事实：{historical_event['historical_fact']}{Style.RESET_ALL}\n")
                
                # 生成角色
                print(f"{Fore.YELLOW}🎭 正在从历史事件中提取角色...{Style.RESET_ALL}")
                rag_characters = generator.extract_characters_from_event(historical_event, num_characters=4)
                
                if rag_characters:
                    # 显示角色选择
                    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}🎭 请选择您的角色 (输入数字 1-4){Style.RESET_ALL}")
                    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
                    
                    for i, char in enumerate(rag_characters, 1):
                        print(f"{Fore.YELLOW}{i}. {char['name']} - {char['role_type']}{Style.RESET_ALL}")
                        print(f"   背景：{char['background']}")
                        print(f"   目标：{char['target']}")
                        if char.get('secrets'):
                            print(f"   秘密：{', '.join(char['secrets'])}")
                        print()
                    
                    # 获取玩家输入
                    print(f"{Fore.GREEN}💬 请在下方输入您选择的角色编号:{Style.RESET_ALL}")
                    try:
                        player_choice = input("\n>>> ").strip()
                        
                        if player_choice:
                            selection_index = int(player_choice)
                            if 1 <= selection_index <= len(rag_characters):
                                selected_char = rag_characters[selection_index - 1]
                                
                                print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
                                print(f"{Fore.GREEN}✅ 您选择了：{selected_char['name']}{Style.RESET_ALL}")
                                print(f"{Fore.GREEN}   身份：{selected_char['role_type']}{Style.RESET_ALL}")
                                print(f"{Fore.GREEN}   背景：{selected_char['background']}{Style.RESET_ALL}")
                                print(f"{Fore.GREEN}   目标：{selected_char['target']}{Style.RESET_ALL}")
                                print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}\n")
                                
                                # 创建角色对象
                                character_id = f"char_{selection_index}"
                                character_data = {
                                    'character_id': character_id,
                                    'name': selected_char['name'],
                                    'background': selected_char.get('background', ''),
                                    'role_type': selected_char.get('role_type', ''),
                                    'relationships': selected_char.get('relationships', {}),
                                    'secrets': selected_char.get('secrets', []),
                                    'target': selected_char.get('target', ''),
                                    'alibi': None,
                                    'historical_basis': selected_char.get('historical_basis', '')
                                }
                                
                                characters = {character_id: character_data}
                                active_character = character_id
                                player_input = player_choice
                                
                                # 存储 RAG 上下文
                                rag_context = [historical_event]
                            else:
                                print(f"{Fore.RED}❌ 无效的选择，请输入 1-{len(rag_characters)}{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}❌ 未收到有效输入{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}❌ 无效输入，请输入数字{Style.RESET_ALL}")
                    except KeyboardInterrupt:
                        print(f"\n{Fore.RED}❌ 游戏中断{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ 无法生成角色{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ 无法检索到历史事件，使用默认事件{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ RAG 角色选择失败：{e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
    
    # 步骤 3: 初始化游戏状态
    initial_state = GameState(
        game_id=game_id,
        current_phase=initial_phase,
        turn_count=0,
        messages=[],
        characters=characters,
        active_character=active_character,
        clues=[],
        collected_clues=[],
        retrieved_contexts=rag_context if rag_context else [],
        hypotheses=[],
        conclusion=None,
        errors=[],
        metadata={},
        player_input=player_input,
        player_choices=[{
            'type': 'character_selection',
            'selection': rag_characters[int(player_input) - 1] if player_input and rag_characters else {},
            'turn': 0
        }] if player_input and rag_characters else []
    )
    
    # 如果有历史事件，也存入状态
    if historical_event:
        initial_state['historical_event'] = historical_event
        initial_state['rag_characters'] = rag_characters
    
    print(f"\n✓ 游戏状态初始化完成")
    print(f"  - 游戏 ID: {game_id}")
    print(f"  - 初始阶段：{initial_phase}")
    if active_character:
        print(f"  - 已选角色：{characters[active_character]['name']}")
    if historical_event:
        print(f"  - 历史事件：{historical_event['event']}")
    
    # 步骤 3: 运行游戏工作流
    print("\n🚀 开始运行游戏工作流...")
    print("="*60)
    
    try:
        # 异步运行工作流
        print("⏳ 正在执行工作流...")
        final_state = asyncio.run(run_game_turn(initial_state))
        
        print("\n" + "="*60)
        print("✅ 游戏运行完成！")
        print("="*60)
        print(f"  - 最终阶段：{final_state.get('current_phase')}")
        print(f"  - 总回合数：{final_state.get('turn_count')}")
        print(f"  - 消息数量：{len(final_state.get('messages', []))}")
        print(f"  - 收集线索：{len(final_state.get('collected_clues', []))}")
        
        # 显示所有生成的消息
        messages = final_state.get('messages', [])
        if messages:
            print("\n" + "="*60)
            print("📋 对话历史 (完整内容):")
            print("="*60)
            for i, msg in enumerate(messages, 1):
                agent_type = msg.get('agent_type', 'unknown')
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                print(f"\n{'='*60}")
                print(f"[{i}] {agent_type} ({role}) - {len(content)} 字符:")
                print(f"{'='*60}")
                # 显示完整内容，不再截断
                print(content)
                print()
        
        # 检查是否有错误
        errors = final_state.get('errors', [])
        if errors:
            print("\n" + "="*60)
            print("❌ 错误列表:")
            print("="*60)
            for error in errors:
                print(f"  - {error}")
        
        return {
            "success": True,
            "game_state": final_state,
            "message": "游戏运行成功"
        }
        
    except Exception as e:
        error_msg = f"游戏运行失败：{str(e)}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg,
            "game_state": None
        }


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
    
    # 自动检查并创建向量库
    check_and_create_vector_store()


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


# ==================== 玩家输入接口 ====================
@app.post("/game/input")
async def player_input(game_id: str, input_text: str, choice_type: str = "character"):
    """
    接收玩家输入的接口
    
    Args:
        game_id: 游戏 ID
        input_text: 玩家输入的文本（如角色编号 1-4）
        choice_type: 选择类型（"character" 角色选择，"action" 行动选择等）
    
    Returns:
        dict: 处理结果
    
    @Java 程序员提示:
    - POST 请求，类似 Java 的 @PostMapping
    - 参数会自动从查询字符串或请求体中提取
    """
    print(f"\n💬 收到玩家输入:")
    print(f"  - 游戏 ID: {game_id}")
    print(f"  - 输入内容：{input_text}")
    print(f"  - 选择类型：{choice_type}")
    
    # TODO: 将玩家输入存储到游戏状态中
    # 这需要结合数据库或内存存储
    
    return {
        "success": True,
        "message": f"已收到您的选择：{input_text}",
        "game_id": game_id
    }


# ==================== 脚本直接运行入口 ====================
if __name__ == "__main__":
    """
    直接运行此脚本即可启动剧本杀游戏
    
    @Java 程序员提示:
    - 类似 Java 的 public static void main(String[] args)
    - 可以直接运行：python -m app.main
    - 也可以导入后调用 start_game() 函数
    """
    import uvicorn
    
    # 方式 1: 直接调用 start_game 函数（推荐）
    print("\n" + "="*60)
    print("🎮 使用单一函数启动剧本杀 AI")
    print("="*60)
    
    result = start_game(
        game_id="game_001",
        initial_phase="intro",
        auto_create_vector_store=True
    )
    
    if result["success"]:
        print("\n✅ 游戏启动成功！")
        # 可以在这里继续处理游戏状态
    else:
        print(f"\n❌ 游戏启动失败：{result.get('error')}")
    
    # 方式 2: 启动 FastAPI 服务器（如果需要 Web 访问）
    # 取消下面的注释即可启用 Web 服务器
    # print("\n启动 FastAPI 服务器...")
    # uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


