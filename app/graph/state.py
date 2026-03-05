"""
LangGraph 状态机定义 - State Schema
使用 TypedDict 定义状态结构
"""
from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum


class AgentType(str, Enum):
    """Agent 类型枚举"""
    NARRATOR = "narrator"  # 主持人
    DETECTIVE = "detective"  # 侦探
    SUSPECT = "suspect"  # 嫌疑人
    WITNESS = "witness"  # 证人


class Message(TypedDict):
    """消息结构"""
    role: str
    content: str
    agent_type: Optional[AgentType]


class CharacterInfo(TypedDict):
    """角色信息结构"""
    character_id: str
    name: str
    background: str
    relationships: Dict[str, str]
    secrets: List[str]
    alibi: Optional[str]


class Clue(TypedDict):
    """线索信息结构"""
    clue_id: str
    description: str
    type: str
    location: str
    related_characters: List[str]
    verified: bool


class GameState(TypedDict):
    """游戏状态 Schema - LangGraph State"""
    # 基础信息
    game_id: str
    current_phase: str
    turn_count: int
    
    # 对话历史
    messages: List[Message]
    
    # 角色信息
    characters: Dict[str, CharacterInfo]
    active_character: Optional[str]
    
    # 线索管理
    clues: List[Clue]
    collected_clues: List[str]
    
    # RAG 检索结果
    retrieved_contexts: List[Dict[str, Any]]
    rag_query: Optional[str]
    
    # 推理结果
    hypotheses: List[Dict[str, Any]]
    conclusion: Optional[str]
    
    # 错误处理
    errors: List[str]
    metadata: Dict[str, Any]
