"""
LangGraph Nodes - 4 个 Agent 的具体逻辑框架
每个 Node 代表一个 Agent 的处理逻辑

@Java 程序员提示:
- Node 是 LangGraph 的基本执行单元，类似 Java 的 Service 方法
- async def 定义异步函数，类似 Java 的 CompletableFuture
- 每个 Node 接收 GameState，返回需要更新的字段
- 类似状态模式：每个 Node 是状态转换的一个动作
"""
from typing import Dict, Any, List  # 导入类型注解
from .state import GameState, Message, AgentType  # 导入状态定义
from ..utils.llm_client import call_llm  # 导入 LLM 调用函数
from ..rag.retriever import retrieve_context  # 导入 RAG 检索函数


# ==================== 主持人 Node ====================
async def narrator_node(state: GameState) -> Dict[str, Any]:
    """
    主持人 Agent - 负责剧情推进和游戏流程控制
    
    Args:
        state: 当前游戏状态 (GameState 对象)
        
    Returns:
        Dict[str, Any]: 需要更新的状态字段
        - 类似 Java: Map<String, Object>
        - 只返回需要更新的字段，不是完整状态
        
    @Java 程序员提示:
    - async def 定义协程函数，可以 await 异步操作
    - 类似 Java 的 CompletableFuture 或 Spring WebFlux
    - Node 的输入是完整状态，输出是增量更新
    """
    try:
        # .get() 方法安全获取字典值，类似 Java 的 map.getOrDefault(key, default)
        # state.get("current_phase", "intro") 如果键不存在，返回默认值 "intro"
        current_phase = state.get("current_phase", "intro")
        
        # f-string 格式化字符串 (Python 3.6+)
        # 类似 Java 的 String.format() 或 "文本 " + variable
        prompt = f"""
        你是一位专业的剧本杀主持人。当前游戏阶段：{current_phase}
        
        请根据以下信息生成主持人的叙述：
        - 游戏 ID: {state.get('game_id')}
        - 回合数：{state.get('turn_count', 0)}
        - 已收集线索：{len(state.get('collected_clues', []))}
        
        请推进剧情并引导玩家进行下一步。
        """
        
        # await 等待异步函数完成
        # 类似 Java 的 CompletableFuture.get() 或 reactor 的 block()
        response = await call_llm(prompt=prompt)
        
        # 创建 Message 对象 (实际是字典)
        # TypedDict 在运行时还是 dict，可以直接创建
        new_message = Message(
            role="assistant",
            content=response,
            agent_type=AgentType.NARRATOR
        )
        
        # 返回增量更新
        # 类似 Java 的 Map.of("key", value)
        return {
            # 追加新消息到消息列表
            # state["messages"] + [new_message] 类似 Java 的 list.add()
            "messages": state["messages"] + [new_message],
            
            # 回合数 +1
            "turn_count": state.get("turn_count", 0) + 1
        }
        
    except Exception as e:
        # 异常处理
        # str(e) 将异常转为字符串，类似 e.getMessage()
        return {"errors": state.get("errors", []) + [f"Narrator error: {str(e)}"]}


# ==================== 侦探 Node ====================
async def detective_node(state: GameState) -> Dict[str, Any]:
    """
    侦探 Agent - 负责推理分析和线索整合
    
    @Java 程序员提示:
    - 结构与 narrator_node 类似
    - 每个 Node 专注于特定职责，类似单一职责原则
    """
    try:
        # 获取线索信息
        clues = state.get("clues", [])  # 所有线索
        collected = state.get("collected_clues", [])  # 已收集线索
        
        prompt = f"""
        你是一位经验丰富的侦探。请分析以下线索：
        
        可用线索数量：{len(clues)}
        已收集线索：{collected}
        
        请进行推理分析，提出可能的假设。
        """
        
        response = await call_llm(prompt=prompt)
        
        new_message = Message(
            role="assistant",
            content=response,
            agent_type=AgentType.DETECTIVE
        )
        
        # 只更新消息列表
        return {
            "messages": state["messages"] + [new_message]
        }
        
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Detective error: {str(e)}"]}


# ==================== 嫌疑人 Node ====================
async def suspect_node(state: GameState) -> Dict[str, Any]:
    """
    嫌疑人 Agent - 扮演被询问的嫌疑人角色
    
    @Java 程序员提示:
    - 这个角色会"扮演"特定角色，类似角色扮演
    - 根据角色设定回应，保持角色一致性
    """
    try:
        # 获取当前活跃角色
        active_char = state.get("active_character")
        characters = state.get("characters", {})
        
        prompt = f"""
        你现在扮演嫌疑人角色：{active_char}
        
        角色信息：
        {characters.get(active_char, {})}
        
        请根据角色设定回应询问。
        """
        
        response = await call_llm(prompt=prompt)
        
        new_message = Message(
            role="assistant",
            content=response,
            agent_type=AgentType.SUSPECT
        )
        
        return {
            "messages": state["messages"] + [new_message]
        }
        
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Suspect error: {str(e)}"]}


# ==================== 证人 Node ====================
async def witness_node(state: GameState) -> Dict[str, Any]:
    """
    证人 Agent - 提供证词和目击信息
    
    @Java 程序员提示:
    - 这个 Node 会使用 RAG 检索结果
    - retrieved_contexts 是从向量数据库检索的相关上下文
    """
    try:
        # 获取 RAG 检索结果
        retrieved = state.get("retrieved_contexts", [])
        
        # 条件判断和列表切片
        # retrieved[:3] 取前 3 个元素，类似 Java 的 list.subList(0, 3)
        if retrieved:
            # 列表推导式 (List Comprehension)
            # 类似 Java Stream: retrieved.stream().limit(3).map(ctx -> ctx.get("content", "")).collect(...)
            context_text = "\n".join([ctx.get("content", "") for ctx in retrieved[:3]])
        else:
            context_text = "暂无相关证据"
        
        prompt = f"""
        你是一位目击证人。请参考以下背景信息：
        
        {context_text}
        
        请提供你的证词。
        """
        
        response = await call_llm(prompt=prompt)
        
        new_message = Message(
            role="assistant",
            content=response,
            agent_type=AgentType.WITNESS
        )
        
        # 返回更新，包括检索结果
        return {
            "messages": state["messages"] + [new_message],
            "retrieved_contexts": retrieved
        }
        
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Witness error: {str(e)}"]}


# ==================== Node 设计模式总结 ====================
# @Java 程序员提示:
# 
# 1. 每个 Node 都是纯函数 (尽量):
#    - 输入：GameState
#    - 输出：Dict[str, Any] (增量更新)
#    - 无副作用 (side-effect free)
#
# 2. 错误处理模式:
#    - try-except 捕获所有异常
#    - 返回 errors 字段而不是抛出异常
#    - 保证工作流不会中断
#
# 3. 状态更新模式:
#    - 只返回需要更新的字段
#    - LangGraph 会自动合并到完整状态
#    - 类似 Git 的增量提交
#
# 4. 异步编程:
#    - async/await 处理 I/O 操作
#    - 提高并发性能
#    - 类似 Java 的 Reactive Programming
