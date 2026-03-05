"""
LangGraph Workflow - 编译状态机图
定义 Agent 之间的流转逻辑
"""
from langgraph.graph import StateGraph, END
from typing import Dict, Callable
from .state import GameState
from .nodes import (
    narrator_node,
    detective_node,
    suspect_node,
    witness_node
)


def create_workflow() -> StateGraph:
    """
    创建并编译 LangGraph 工作流
    
    Returns:
        编译后的 StateGraph
    """
    workflow = StateGraph(GameState)
    
    workflow.add_node("narrator", narrator_node)
    workflow.add_node("detective", detective_node)
    workflow.add_node("suspect", suspect_node)
    workflow.add_node("witness", witness_node)
    
    workflow.set_entry_point("narrator")
    
    workflow.add_edge("narrator", "detective")
    workflow.add_edge("detective", "suspect")
    workflow.add_edge("suspect", "witness")
    workflow.add_edge("witness", "narrator")
    
    app = workflow.compile()
    
    return app


def get_conditional_edges(state: GameState) -> str:
    """条件边路由函数"""
    phase = state.get("current_phase", "intro")
    
    if phase in ["intro", "conclusion"]:
        return "narrator"
    elif phase == "investigation":
        return "detective"
    elif phase == "interrogation":
        return "suspect"
    else:
        return "witness"


workflow_app = create_workflow()


async def run_game_turn(
    initial_state: GameState,
    custom_workflow: StateGraph = None
) -> GameState:
    """运行游戏回合"""
    app = custom_workflow or workflow_app
    final_state = await app.ainvoke(initial_state)
    return final_state
