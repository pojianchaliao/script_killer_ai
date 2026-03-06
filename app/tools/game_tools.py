"""
Game Tools - 剧本杀专用工具函数
提供角色生成、线索生成、剧情分支等工具

@Java 程序员提示:
- 这是工具类模块，类似 Java 的 Utils 类
- 包含多个异步函数 (async def)
- 使用 TypedDict 定义的数据结构
- 每个函数都是无状态的纯函数
"""
from typing import Dict, List, Any, Optional  # 类型注解
from ..graph.state import CharacterInfo, Clue, GameState  # 导入状态类型


# ==================== 角色生成工具 ====================
async def generate_character(
    character_type: str,
    story_background: str,
    relationships: Optional[Dict[str, str]] = None
) -> CharacterInfo:
    """
    生成角色信息
    
    Args:
        character_type: 角色类型 ("侦探", "嫌疑人", "证人" 等)
        story_background: 故事背景描述
        relationships: 与其他角色的关系 {角色名：关系描述}
        
    Returns:
        CharacterInfo: 生成的角色信息
        
    @Java 程序员提示:
    - async def 定义异步函数
    - 类似 Java 的 CompletableFuture<CharacterInfo>
    - Optional[Dict] 表示参数可以为 None
    - 使用 LLM 生成角色设定
    """
    # 导入 LLM 调用函数 (在函数内部导入，避免循环依赖)
    from ..utils.llm_client import call_llm
    
    # 构建提示词 (Prompt)
    # f-string 格式化多行字符串
    prompt = f"""
    请生成一个剧本杀角色设定。
    
    角色类型：{character_type}
    故事背景：{story_background}
    关系网络：{relationships}
    
    请生成包含以下内容的详细角色设定：
    1. 角色姓名
    2. 详细背景故事
    3. 性格特点
    4. 秘密信息（2-3 个）
    5. 不在场证明（如果有）
    """
    
    # TODO: 调用 LLM 生成角色信息
    # 实际实现会调用 call_llm(prompt) 获取 LLM 生成的内容
    # 然后解析为 CharacterInfo 对象
    
    # 这里返回示例数据
    return CharacterInfo(
        character_id=f"char_{character_type}_001",  # 生成唯一 ID
        name="示例角色",  # 角色姓名
        background=story_background,  # 背景故事
        relationships=relationships or {},  # 关系网络，如果为 None 则使用空字典
        secrets=["秘密 1", "秘密 2"],  # 秘密列表
        alibi=None  # 不在场证明，初始为 None
    )


# ==================== 线索生成工具 ====================
async def generate_clue(
    clue_type: str,
    location: str,
    related_characters: List[str],
    difficulty: str = "medium"
) -> Clue:
    """
    生成线索
    
    Args:
        clue_type: 线索类型 (物证、证言、文件等)
        location: 线索发现地点
        related_characters: 相关角色列表
        difficulty: 难度 ("easy", "medium", "hard")
        
    Returns:
        Clue: 生成的线索信息
        
    @Java 程序员提示:
    - List[str] 类似 Java 的 List<String>
    - 默认参数类似 Java 的方法重载
    - difficulty="medium" 表示默认值是 "medium"
    """
    # 导入 LLM 调用函数
    from ..utils.llm_client import call_llm
    
    # 构建提示词
    prompt = f"""
    请生成一个剧本杀线索。
    
    类型：{clue_type}
    地点：{location}
    相关角色：{related_characters}
    难度：{difficulty}
    
    请生成线索描述，确保线索既有价值又不会太明显。
    """
    
    # TODO: 调用 LLM 生成线索
    # 实际会根据难度调整线索的隐晦程度
    
    # 返回示例数据
    return Clue(
        clue_id=f"clue_{clue_type}_{location}_001",  # 唯一 ID
        description="线索描述内容",  # 线索描述
        type=clue_type,  # 类型
        location=location,  # 地点
        related_characters=related_characters,  # 相关角色
        verified=False  # 是否已验证，初始为 False
    )


# ==================== 线索一致性校验 ====================
def validate_clue_consistency(
    clue: Clue,
    other_clues: List[Clue],
    characters: Dict[str, CharacterInfo]
) -> bool:
    """
    校验线索的一致性 - 确保线索不矛盾
    
    Args:
        clue: 待校验的线索
        other_clues: 其他线索列表
        characters: 角色信息字典
        
    Returns:
        bool: 是否一致 (True/False)
        
    @Java 程序员提示:
    - 这是同步函数 (没有 async)
    - 类似 Java 的 boolean 方法
    - 用于验证线索是否与已有信息矛盾
    - 类似数据完整性检查
    """
    # 遍历其他线索
    for other in other_clues:
        # 检查地点是否一致
        # ["location"] 访问字典值，类似 Java 的 map.get("location")
        if clue["location"] != other["location"]:
            # 地点不同，可能是不同线索
            # TODO: 添加更复杂的逻辑
            pass
    
    # TODO: 实现更完整的一致性检查
    # 例如：检查线索与角色背景是否矛盾
    # 检查线索之间是否有逻辑冲突
    
    return True  # 默认返回一致


# ==================== 剧情分支生成 ====================
async def generate_plot_branch(
    current_state: GameState,
    player_choice: str
) -> Dict[str, Any]:
    """
    根据玩家选择生成分支剧情
    
    Args:
        current_state: 当前游戏状态
        player_choice: 玩家的选择
        
    Returns:
        Dict[str, Any]: 剧情发展信息
        
    @Java 程序员提示:
    - 这是动态剧情生成
    - 类似互动小说 (Interactive Fiction)
    - 根据玩家选择改变故事走向
    - 类似 Java 的规则引擎
    """
    # 导入 LLM 调用函数
    from ..utils.llm_client import call_llm
    
    # 构建提示词
    prompt = f"""
    当前游戏状态：
    - 阶段：{current_state.get('current_phase', 'unknown')}
    - 回合：{current_state.get('turn_count', 0)}
    - 已收集线索：{len(current_state.get('collected_clues', []))}
    
    玩家选择：{player_choice}
    
    请生成合理的剧情分支。
    """
    
    # TODO: 调用 LLM 生成剧情分支
    
    # 返回示例数据
    return {
        "plot_development": "剧情发展描述",  # 剧情发展
        "new_clues": [],  # 新线索列表
        "character_impacts": {}  # 对角色的影响
    }


# ==================== 嫌疑指数计算 ====================
def calculate_suspicion_level(
    character_id: str,
    clues: List[Clue],
    hypotheses: List[Dict[str, Any]]
) -> float:
    """
    计算角色的嫌疑指数
    
    Args:
        character_id: 角色 ID
        clues: 线索列表
        hypotheses: 推理假设列表
        
    Returns:
        float: 嫌疑指数 (0.0-1.0)
        
    @Java 程序员提示:
    - float 是浮点数类型
    - 返回值范围 0.0 (完全无辜) 到 1.0 (最大嫌疑)
    - 类似风险评分系统
    - 可以基于规则或机器学习
    """
    # 基础嫌疑值
    suspicion = 0.5  # 初始为中间值
    
    # TODO: 实现更复杂的计算逻辑
    # 例如:
    # 1. 检查线索是否指向该角色
    # 2. 检查假设中该角色的提及频率
    # 3. 检查角色的秘密和动机
    
    # 示例：遍历线索，增加或减少嫌疑值
    for clue in clues:
        if character_id in clue.get("related_characters", []):
            # 如果线索与该角色相关，增加嫌疑
            suspicion += 0.1
    
    # 确保在 0-1 范围内
    suspicion = max(0.0, min(1.0, suspicion))
    
    return suspicion


# ==================== 游戏结束检查 ====================
def check_game_completion(state: GameState) -> Dict[str, Any]:
    """
    检查游戏是否应该结束
    
    Args:
        state: 当前游戏状态
        
    Returns:
        Dict[str, Any]: 游戏结束信息
        {
            "is_finished": bool,      # 是否结束
            "reason": str,            # 结束原因
            "winner": str,            # 获胜者
            "final_score": int        # 最终得分
        }
        
    @Java 程序员提示:
    - 类似游戏状态机检查
    - 检查胜利条件
    - 返回游戏结果
    - 类似 Java 的游戏引擎逻辑
    """
    # TODO: 实现完整的游戏结束检查
    
    # 示例：检查是否已收集足够线索
    collected_clues = state.get("collected_clues", [])
    if len(collected_clues) >= 10:
        # 线索足够，游戏结束
        return {
            "is_finished": True,
            "reason": "已收集足够线索",
            "winner": "侦探",
            "final_score": 100
        }
    
    # 默认：游戏继续
    return {
        "is_finished": False,
        "reason": "",
        "winner": None,
        "final_score": 0
    }


# ==================== 工具函数使用示例 ====================
# @Java 程序员提示:
# 
# 使用方式 1: 生成角色
# character = await generate_character(
#     character_type="嫌疑人",
#     story_background="一个富有的商人...",
#     relationships={"侦探": "老朋友", "受害者": "商业对手"}
# )
#
# 使用方式 2: 生成线索
# clue = await generate_clue(
#     clue_type="物证",
#     location="书房",
#     related_characters=["嫌疑人 A", "受害者"],
#     difficulty="hard"
# )
#
# 使用方式 3: 计算嫌疑指数
# suspicion = calculate_suspicion_level(
#     character_id="char_001",
#     clues=clue_list,
#     hypotheses=hypothesis_list
# )
# print(f"嫌疑指数：{suspicion}")
