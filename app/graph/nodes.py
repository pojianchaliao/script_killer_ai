"""
LangGraph Nodes - 4 个 Agent 的具体逻辑框架
每个 Node 代表一个 Agent 的处理逻辑
"""
from typing import Dict, Any, List
from .state import GameState, Message, AgentType
from ..utils.llm_client import call_llm
from ..rag.retriever import retrieve_context


async def narrator_node(state: GameState) -> Dict[str, Any]:
    """
    主持人 Agent - 负责剧情推进和游戏流程控制
    
    Args:
        state: 当前游戏状态
        
    Returns:
        更新后的状态
    """
    try:
        # TODO: 实现主持人逻辑
        current_phase = state.get("current_phase", "intro")
        
        prompt = f"""
        你是一位专业的剧本杀主持人。当前游戏阶段：{current_phase}
        
        请根据以下信息生成主持人的叙述：
        - 游戏 ID: {state.get('game_id')}
        - 回合数：{state.get('turn_count', 0)}
        - 已收集线索：{len(state.get('collected_clues', []))}
        
        请推进剧情并引导玩家进行下一步。
        """
        
        response = await call_llm(prompt=prompt)
        
        new_message = Message(
            role="assistant",
            content=response,
            agent_type=AgentType.NARRATOR
        )
        
        return {
            "messages": state["messages"] + [new_message],
            "turn_count": state.get("turn_count", 0) + 1
        }
        
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Narrator error: {str(e)}"]}


async def detective_node(state: GameState) -> Dict[str, Any]:
    """
    侦探 Agent - 负责推理分析和线索整合
    """
    try:
        clues = state.get("clues", [])
        collected = state.get("collected_clues", [])
        
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
        
        return {
            "messages": state["messages"] + [new_message]
        }
        
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Detective error: {str(e)}"]}


async def suspect_node(state: GameState) -> Dict[str, Any]:
    """
    嫌疑人 Agent - 扮演被询问的嫌疑人角色
    """
    try:
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


async def witness_node(state: GameState) -> Dict[str, Any]:
    """
    证人 Agent - 提供证词和目击信息
    """
    try:
        retrieved = state.get("retrieved_contexts", [])
        
        if retrieved:
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
        
        return {
            "messages": state["messages"] + [new_message],
            "retrieved_contexts": retrieved
        }
        
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Witness error: {str(e)}"]}
