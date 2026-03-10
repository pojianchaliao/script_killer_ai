"""
LangGraph 状态机定义 - State Schema
使用 TypedDict 定义状态结构

@Java 程序员提示:
- TypedDict 类似 Java 的接口或 POJO，用于定义数据结构
- Enum 是枚举类型，类似 Java 的 enum
- 这是 LangGraph 的核心，定义了工作流的状态格式
- 类似 Java 的状态模式 (State Pattern)
"""
from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum


# ==================== 枚举类型定义 ====================
class AgentType(str, Enum):
    """
    Agent 类型枚举
    
    @Java 程序员提示:
    - str, Enum 表示枚举值是字符串类型
    - 类似 Java: public enum AgentType { NARRATOR("narrator"), ... }
    - 使用时：AgentType.NARRATOR 或 AgentType.NARRATOR.value
    """
    NARRATOR = "narrator"  # 主持人 - 负责剧情推进
    DETECTIVE = "detective"  # 侦探 - 负责推理分析
    SUSPECT = "suspect"  # 嫌疑人 - 被询问角色
    WITNESS = "witness"  # 证人 - 提供证词


# ==================== TypedDict 数据结构定义 ====================
# TypedDict 用于定义字典的 Schema
# 类似 Java 的 POJO 或 Interface

class Message(TypedDict):
    """
    消息结构 - 定义对话消息的格式
    
    @Java 程序员提示:
    - 类似 Java 类：
      public class Message {
          String role;
          String content;
          AgentType agentType;
      }
    - TypedDict 在运行时还是 dict，但静态类型检查会验证字段
    """
    role: str  # 角色："user", "assistant", "system"
    content: str  # 消息内容
    agent_type: Optional[AgentType]  # Agent 类型，可以为 None


class CharacterInfo(TypedDict):
    """
    角色信息结构 - 定义 RPG 角色的完整信息
    
    @Java 程序员提示:
    - Dict[str, str] 类似 Java 的 Map<String, String>
    - List[str] 类似 Java 的 List<String> 或 ArrayList<String>
    - Optional[str] 类似 Java 的 @Nullable String
    """
    character_id: str  # 角色唯一 ID
    name: str  # 角色姓名
    background: str  # 背景故事
    role_type: str  # 角色身份（如武将、谋士、商人等）
    relationships: Dict[str, str]  # 与其他角色的关系 {角色 ID: 关系描述}
    secrets: List[str]  # 角色秘密列表
    alibi: Optional[str]  # 不在场证明，可以为 None
    target: str  # 角色目标
    historical_basis: str  # 历史依据


class Clue(TypedDict):
    """
    线索信息结构 - 定义剧本杀线索的格式
    
    @Java 程序员提示:
    - 类似 Java 的 POJO，包含线索的所有属性
    - bool 是布尔类型，值是 True/False
    """
    clue_id: str  # 线索唯一 ID
    description: str  # 线索描述
    type: str  # 线索类型 (物证、证言、文件等)
    location: str  # 线索发现地点
    related_characters: List[str]  # 相关角色 ID 列表
    verified: bool  # 是否已验证


# ==================== 游戏状态主 Schema ====================
class GameState(TypedDict):
    """
    游戏状态 Schema - LangGraph State 的完整定义
    
    这是 LangGraph 工作流的核心状态对象
    每次 Agent 执行都会接收和返回这个状态
    
    @Java 程序员提示:
    - 类似 Java 的状态对象或上下文对象
    - LangGraph 的 State 类似 Spring State Machine 的 StateContext
    - 所有 Agent Node 都接收 GameState，返回部分字段的更新
    
    字段说明:
    - 基础信息：游戏 ID、阶段、回合数
    - 对话历史：所有消息记录
    - 角色信息：角色定义和当前活跃角色
    - 线索管理：所有线索和已收集线索
    - RAG 检索：检索到的上下文
    - 推理结果：假设和结论
    - 错误处理：错误日志和元数据
    """
    
    # ---------- 基础信息 ----------
    game_id: str  # 游戏会话唯一标识
    current_phase: str  # 当前阶段："intro", "gameplay", "gameover"
    turn_count: int  # 回合数计数器
    
    # ---------- 对话历史 ----------
    # List[Message] 表示 Message 对象的列表
    messages: List[Message]  # 所有对话消息的历史记录
    
    # ---------- 角色管理 ----------
    # Dict[str, CharacterInfo] 类似 Java: Map<String, CharacterInfo>
    characters: Dict[str, CharacterInfo]  # 所有角色信息 {角色 ID: 角色对象}
    active_character: Optional[str]  # 当前活跃角色的 ID，可以为 None
    
    # ---------- 线索管理 ----------
    clues: List[Clue]  # 所有可用线索列表
    collected_clues: List[str]  # 玩家已收集的线索 ID 列表
    
    # ---------- RAG 检索结果 ----------
    # List[Dict[str, Any]] 表示字典列表，Any 表示任意类型
    retrieved_contexts: List[Dict[str, Any]]  # RAG 检索到的上下文
    rag_query: Optional[str]  # 最后一次检索查询
    
    # ---------- 推理结果 ----------
    hypotheses: List[Dict[str, Any]]  # 推理假设列表
    conclusion: Optional[str]  # 最终结论
    
    # ---------- 错误处理 ----------
    errors: List[str]  # 错误消息列表
    metadata: Dict[str, Any]  # 元数据，存储其他信息
    
    # ---------- 玩家输入 ----------
    player_input: Optional[str]  # 玩家最新输入
    player_choices: List[Dict[str, Any]]  # 玩家历史选择记录
    
    # ---------- RAG 检索 ----------
    rag_context: List[Dict[str, Any]]  # RAG 检索到的上下文（带分数）
    historical_event: Optional[Dict[str, Any]]  # 当前核心历史事件
    rag_characters: List[Dict[str, Any]]  # 从 RAG 中提取的候选角色列表
    
    # ---------- RPG 新增字段 ----------
    is_alive: bool  # 玩家是否存活
    death_reason: Optional[str]  # 死亡原因（如果已死亡）
    game_over: bool  # 游戏是否结束


# ==================== 使用示例 (注释) ====================
# 创建游戏状态实例:
# game_state = GameState(
#     game_id="game_001",
#     current_phase="intro",
#     turn_count=0,
#     messages=[],
#     characters={},
#     active_character=None,
#     clues=[],
#     collected_clues=[],
#     retrieved_contexts=[],
#     hypotheses=[],
#     conclusion=None,
#     errors=[],
#     metadata={}
# )

# @Java 程序员提示:
# - Python 的字典创建使用 {key: value}
# - 类似 Java 的 Builder 模式或构造函数
# - TypedDict 在运行时会检查必需字段
