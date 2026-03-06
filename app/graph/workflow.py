"""
LangGraph Workflow - 编译状态机图
定义 Agent 之间的流转逻辑

@Java 程序员提示:
- LangGraph 类似 Java 的状态机框架 (如 Spring State Machine)
- StateGraph 定义了状态的转换规则
- Node 是状态动作，Edge 是状态转换
- 类似工作流引擎：BPMN、Activiti 的简化版
"""
from langgraph.graph import StateGraph, END  # 导入 LangGraph 核心类
from typing import Dict, Callable  # 类型注解
from .state import GameState  # 导入状态定义
from .nodes import (
    narrator_node,      # 主持人 Node
    detective_node,     # 侦探 Node
    suspect_node,       # 嫌疑人 Node
    witness_node        # 证人 Node
)


# ==================== 创建工作流 ====================
def create_workflow() -> StateGraph:
    """
    创建并编译 LangGraph 工作流
    
    Returns:
        StateGraph: 编译后的工作流图
    
    @Java 程序员提示:
    - 类似 Java 的工厂方法模式
    - 返回 StateGraph 对象，可以多次使用
    - 工作流定义了 Agent 的执行顺序
    """
    # 创建工作流图，传入状态类型
    # 类似 Java: new StateGraph<GameState>()
    workflow = StateGraph(GameState)
    
    # ========== 添加 Node (节点) ==========
    # add_node(name, function) 注册执行节点
    # 类似 Java 的 serviceRegistry.register("name", service)
    workflow.add_node("narrator", narrator_node)
    workflow.add_node("detective", detective_node)
    workflow.add_node("suspect", suspect_node)
    workflow.add_node("witness", witness_node)
    
    # ========== 设置入口点 ==========
    # set_entry_point 设置工作流的起始节点
    # 类似 Java 的 workflow.startWith("narrator")
    workflow.set_entry_point("narrator")
    
    # ========== 添加 Edge (边) ==========
    # add_edge(from_node, to_node) 定义节点之间的连接
    # 类似 Java 的 workflow.from("narrator").to("detective")
    # 这是无条件边，执行完 from_node 后必定执行 to_node
    
    # 主持人 -> 侦探
    workflow.add_edge("narrator", "detective")
    
    # 侦探 -> 嫌疑人
    workflow.add_edge("detective", "suspect")
    
    # 嫌疑人 -> 证人
    workflow.add_edge("suspect", "witness")
    
    # 证人 -> 主持人 (形成循环)
    workflow.add_edge("witness", "narrator")
    
    # ========== 编译工作流 ==========
    # compile() 将图结构编译为可执行对象
    # 类似 Java 的 workflowBuilder.build()
    app = workflow.compile()
    
    return app


# ==================== 条件边路由函数 ====================
def get_conditional_edges(state: GameState) -> str:
    """
    条件边路由函数 - 根据状态决定下一个节点
    
    Args:
        state: 当前游戏状态
        
    Returns:
        str: 下一个节点的名称
    
    @Java 程序员提示:
    - 类似 Java 的状态模式：根据状态决定转换
    - 或者类似路由策略：Router Pattern
    - 返回节点名称字符串
    """
    # 获取当前阶段
    phase = state.get("current_phase", "intro")
    
    # 条件判断 (类似 Java 的 switch-case)
    if phase in ["intro", "conclusion"]:
        # 开场或结尾阶段，执行主持人
        return "narrator"
    
    elif phase == "investigation":
        # 调查阶段，执行侦探
        return "detective"
    
    elif phase == "interrogation":
        # 审问阶段，执行嫌疑人
        return "suspect"
    
    else:
        # 其他阶段，执行证人
        return "witness"


# ==================== 全局工作流实例 ====================
# 在模块级别创建单例工作流
# 类似 Java: public static final StateGraph workflow = createWorkflow();
workflow_app = create_workflow()


# ==================== 运行游戏回合 ====================
async def run_game_turn(
    initial_state: GameState,
    custom_workflow: StateGraph = None
) -> GameState:
    """
    运行游戏回合的便捷函数
    
    Args:
        initial_state: 初始游戏状态
        custom_workflow: 自定义工作流 (可选，用于测试)
        
    Returns:
        GameState: 最终的游戏状态
    
    @Java 程序员提示:
    - async def 定义异步函数
    - 类似 Java 的 CompletableFuture<GameState>
    - initial_state 是输入，final_state 是输出
    - 工作流会自动执行所有 Node
    """
    # 选择使用的工作流
    # custom_workflow or workflow_app 类似 Java 的 Optional.orElse()
    app = custom_workflow or workflow_app
    
    # ainvoke 是异步调用
    # 类似 Java 的 CompletableFuture.get()
    # 工作流会从入口点开始，按照边定义执行所有 Node
    final_state = await app.ainvoke(initial_state)
    
    return final_state


# ==================== 工作流执行流程示例 ====================
# @Java 程序员提示:
# 
# 假设初始状态:
# initial_state = {
#     "game_id": "game_001",
#     "current_phase": "intro",
#     "turn_count": 0,
#     "messages": [],
#     ...
# }
#
# 执行流程:
# 1. 从 "narrator" 开始 (入口点)
# 2. narrator_node 执行，返回更新
# 3. 沿边到 "detective"
# 4. detective_node 执行，返回更新
# 5. 沿边到 "suspect"
# 6. suspect_node 执行，返回更新
# 7. 沿边到 "witness"
# 8. witness_node 执行，返回更新
# 9. 沿边回到 "narrator" (循环)
#
# 每次执行都会累积更新到 state
# 最终返回完整的 GameState
