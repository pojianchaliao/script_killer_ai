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
from .rag_generator import RAGCharacterGenerator  # 导入 RAG 角色生成器
from .dynamic_events import DynamicEventGenerator, generate_dynamic_event  # 导入动态事件生成器
import json  # 导入 JSON 解析
import re  # 导入正则表达式


# ==================== 辅助函数 ====================
def generate_reference_speeches(char_name: str, char_role: str, char_target: str, event_name: str, historical_event: dict = None, turn_count: int = 0, message_history: list = None) -> list:
    """
    使用 LLM 根据角色性格和历史背景生成参考发言选项
    
    Args:
        char_name: 角色名称
        char_role: 角色身份
        char_target: 角色目标
        event_name: 当前历史事件
        historical_event: 完整的历史事件信息（可选）
        turn_count: 当前回合数
        message_history: 消息历史记录（可选）
        
    Returns:
        list: 参考发言列表（3-5 个选项）
    """
    try:
        # 构建 LLM prompt
        event_desc = ""
        event_fact = ""
        if historical_event:
            event_desc = historical_event.get('description', '')
            event_fact = historical_event.get('historical_fact', '')
        
        # 分析之前的对话历史，确保选项不重复
        previous_speeches = []
        if message_history:
            for msg in message_history[-5:]:  # 只看最近 5 条
                if msg.get('role') == 'user':
                    previous_speeches.append(msg.get('content', ''))
        
        prompt = f"""
你是一位精通三国历史和古代表演艺术的专家。请为以下角色生成 4-5 个符合其身份、性格和历史背景的发言选项。

【角色信息】
- 姓名：{char_name}
- 身份：{char_role}
- 目标：{char_target}

【当前局势】
- 事件：{event_name}
- 描述：{event_desc[:200] if event_desc else '未知'}
- 历史事实：{event_fact[:100] if event_fact else '未知'}
- 回合数：第{turn_count}回合

【要求】
1. 每个发言 10-30 字，符合三国时期的语言风格
2. 发言要体现角色的身份、立场和性格
3. 提供不同类型的选项：
   - 激进型（主张行动）
   - 稳健型（谨慎行事）
   - 谋略型（出谋划策）
   - 情感型（表达情绪）
4. 发言要与当前历史事件相关
5. 使用半文半白的语言，但要易懂
6. **重要**：避免与之前重复，以下是主公之前的发言（如有）：
   {chr(10).join(f'   - "{s}"' for s in previous_speeches) if previous_speeches else '   无'}

【输出格式】
直接返回 4-5 个发言选项，每行一个，不要编号，不要额外说明。
例如：
末将愿领兵出征，为大人效犬马之劳！
此事非同小可，还望三思而后行。
在下有一计，可解此困局...
"""
        
        # 调用 LLM 生成发言
        response = call_llm(prompt=prompt)
        
        # 解析响应，提取发言选项
        speeches = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # 移除可能的编号和标记
            if line and len(line) >= 5:  # 至少 5 个字符
                # 清理行首的编号、符号等
                import re
                cleaned = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                if cleaned and len(cleaned) >= 5 and cleaned not in previous_speeches:
                    speeches.append(cleaned)
        
        # 确保至少有 3 个发言
        if len(speeches) < 3:
            # 如果 LLM 生成的不够，使用备用方案
            fallback_speeches = get_fallback_speeches(char_name, char_role, event_name)
            speeches.extend([s for s in fallback_speeches if s not in previous_speeches])
        
        return speeches[:5]  # 最多返回 5 个
        
    except Exception as e:
        print(f"⚠️  LLM 生成参考发言失败，使用备用方案：{e}")
        # 降级到硬编码的备用方案
        return get_fallback_speeches(char_name, char_role, event_name)


def get_fallback_speeches(char_name: str, char_role: str, event_name: str) -> list:
    """
    备用的硬编码发言生成（当 LLM 失败时使用）
    
    Args:
        char_name: 角色名称
        char_role: 角色身份
        event_name: 当前历史事件
        
    Returns:
        list: 参考发言列表
    """
    speeches = []
    
    # 基于角色类型生成符合身份的发言
    if "武将" in char_role or "将军" in char_role or "武" in char_role:
        speeches.extend([
            f"末将愿领兵出征，为{char_name}效犬马之劳！",
            f"某家不才，愿为先锋，直取敌将首级！",
            f"主公勿忧，此战交予我等便是！"
        ])
    elif "谋士" in char_role or "军师" in char_role or "谋" in char_role:
        speeches.extend([
            f"在下有一计，可解此困局...",
            f"观天象、察人事，此事当从长计议。",
            f"兵法云：知己知彼，百战不殆。容臣细细分析。"
        ])
    elif "君主" in char_role or "主公" in char_role or "帝" in char_role:
        speeches.extend([
            f"孤意已决，诸公不必再谏。",
            f"此事关乎社稷安危，望诸卿共商大计。",
            f"天命所归，孤当顺天应人。"
        ])
    elif "商人" in char_role or "商" in char_role:
        speeches.extend([
            f"小人愿倾尽家财，助大人成就大业。",
            f"商道即人道，此事可从长计议。",
            f"有利可图，自当效劳。"
        ])
    elif "官员" in char_role or "臣" in char_role:
        speeches.extend([
            f"臣有一言，望陛下纳之。",
            f"为国为民，臣万死不辞。",
            f"朝堂之上，岂容小人猖狂。"
        ])
    else:
        # 通用发言，符合三国时期语言风格
        speeches.extend([
            f"此事非同小可，还望三思而后行。",
            f"天下大势，分久必合，合久必分。",
            f"时势造英雄，英雄亦适时。",
            f"某虽不才，愿效绵薄之力。"
        ])
    
    # 根据历史事件调整发言
    if "起义" in event_name or "战乱" in event_name or "黄巾" in event_name:
        speeches.extend([
            "乱世之中，唯有强者方能生存。",
            "百姓何辜，遭此战乱..."
        ])
    elif "刺杀" in event_name or "阴谋" in event_name or "诛" in event_name:
        speeches.extend([
            "此事蹊跷，恐有阴谋。",
            "防人之心不可无，还望小心行事。"
        ])
    elif "称帝" in event_name or "登基" in event_name:
        speeches.extend([
            "天命所归，当顺天应人。",
            "名不正则言不顺，此事还需从长计议。"
        ])
    elif "讨董" in event_name or "董卓" in event_name:
        speeches.extend([
            "董贼残暴，天下共愤，当起兵讨之！",
            "宁为玉碎，不为瓦全，誓与董贼不两立！"
        ])
    
    # 去重并返回前 5 个发言
    unique_speeches = list(dict.fromkeys(speeches))
    return unique_speeches[:5]


def generate_reference_speeches_from_options(char_name: str, char_role: str, char_target: str, narrator_message: str, historical_event: dict = None, turn_count: int = 0, message_history: list = None) -> list:
    """
    根据主持人给出的选项生成对应的参考策略
    
    Args:
        char_name: 角色名称
        char_role: 角色身份
        char_target: 角色目标
        narrator_message: 主持人的消息（包含选项）
        historical_event: 完整的历史事件信息（可选）
        turn_count: 当前回合数
        message_history: 消息历史记录（可选）
        
    Returns:
        list: 参考策略列表（与主持人选项对齐）
    """
    try:
        # 从主持人消息中提取选项内容
        import re
        options = []
        
        # 匹配 "选项一"、"选项二" 等模式
        if not options:
            chinese_option_pattern = r'选项\s*[一二三四五 1-5][:\s,，]*([^\n]+)'
        matches = re.findall(chinese_option_pattern, narrator_message)
        if matches:
            # matches 现在是元组列表 [(' ', '派遣斥候...'), ('：', '亲自带领...')]
            # 我们只需要第二个分组（选项内容）
            options = [m.strip() for m in matches[:5]]



        # 如果没有找到 "选项 X" 模式，尝试找 "你可以..."
        if not options:
            can_do_pattern = r'你可以 ([^。！？]+)[。！？]'
            matches = re.findall(can_do_pattern, narrator_message)
            if matches:
                options = [m.strip() for m in matches[:5]]
        
        # 如果还是找不到，直接使用原始方法
        if not options:
          return generate_reference_speeches(
                char_name, char_role, char_target,
                historical_event.get('event', '三国事件') if historical_event else '三国事件',
                historical_event, turn_count, message_history
            )
        
        # 为每个选项生成对应的参考策略
        speeches = []
        for opt in options:
            # 简化选项描述，变成玩家可用的策略
            # strategy = f"针对：{opt[:50]}"
            speeches.append(opt)
        
        return speeches if speeches else generate_reference_speeches(
            char_name, char_role, char_target,
            historical_event.get('event', '三国事件') if historical_event else '三国事件',
            historical_event, turn_count, message_history
        )
        
    except Exception as e:
        print(f"⚠️  从选项生成参考策略失败：{e}")
        # 降级到原始方法
        return generate_reference_speeches(
            char_name, char_role, char_target,
            historical_event.get('event', '三国事件') if historical_event else '三国事件',
            historical_event, turn_count, message_history
        )


# ==================== 辅助函数 ====================
def parse_character_selection(response_text: str, selection_index: int) -> Dict[str, Any]:
    """
    解析 LLM 生成的角色选择，提取玩家选择的角色信息
    
    Args:
        response_text: LLM 生成的响应文本
        selection_index: 玩家选择的角色编号 (1-4)
        
    Returns:
        Dict[str, Any]: 解析后的角色信息
    """
    try:
        # 使用正则表达式提取角色信息
        # 匹配模式：数字。姓名 - 身份，背景简介
        pattern = r'(\d+)\.\s*([^\-]+)\s*-\s*([^，,]+)[，,]?([^\n]+)?'
        matches = re.findall(pattern, response_text)
        
        if matches and len(matches) >= selection_index:
            selected = matches[selection_index - 1]
            return {
                'index': int(selected[0]),
                'name': selected[1].strip(),
                'role_type': selected[2].strip(),
                'background': selected[3].strip() if selected[3] else '未知',
                'target': '完成角色目标'  # 默认目标
            }
        else:
            # 如果解析失败，返回默认角色
            return {
                'index': selection_index,
                'name': f'角色{selection_index}',
                'role_type': '江湖人士',
                'background': '乱世中的普通人',
                'target': '生存下去'
            }
    except Exception as e:
        print(f"解析角色选择失败：{e}")
        return {
            'index': selection_index,
            'name': f'角色{selection_index}',
            'role_type': '江湖人士',
            'background': '乱世中的普通人',
            'target': '生存下去'
        }


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
        print("\n" + "="*60)
        print("🎭 [主持人 Node] 开始执行")
        print("="*60)
        
        # .get() 方法安全获取字典值，类似 Java 的 map.getOrDefault(key, default)
        # state.get("current_phase", "intro") 如果键不存在，返回默认值 "intro"
        current_phase = state.get("current_phase", "intro")
        print(f"📌 当前阶段：{current_phase}")
        print(f"📌 游戏 ID: {state.get('game_id')}")
        print(f"📌 当前回合数：{state.get('turn_count', 0)}")
        print(f"📌 已收集线索：{len(state.get('collected_clues', []))} 条")
        
        # f-string 格式化字符串 (Python 3.6+)
        # 类似 Java 的 String.format() 或 "文本 " + variable
        
        # 检查是否需要角色选择（第一次执行时）
        characters = state.get('characters', {})
        active_char = state.get('active_character')
        is_alive = state.get('is_alive', True)
        game_over = state.get('game_over', False)
        
        # 如果游戏已结束或玩家已死亡，直接返回
        if game_over or not is_alive:
            death_reason = state.get('death_reason', '未知')
            print(f"\n💀 游戏已结束：{death_reason}")
            return {}
        
        if not characters and current_phase == 'intro':
            # ========== RAG 驱动的角色生成 ==========
            print("\n🔍 [RAG] 正在从三国历史中检索事件和角色...")
            
            # 初始化 response 和 result，避免未定义错误
            response = ""
            result = {}
            
            try:
                # 1. 创建 RAG 生成器
                generator = RAGCharacterGenerator()
                
                # 2. 选择一个历史事件作为游戏背景
                query = "三国 谋杀 阴谋 刺杀"  # 根据游戏类型调整
                historical_event = generator.select_historical_event(query)
                
                if historical_event:
                    print(f"✅ [RAG] 选中历史事件：{historical_event['event']}")
                    
                    # 3. 从事件中提取相关人物生成角色
                    rag_characters = generator.extract_characters_from_event(
                        historical_event, 
                        num_characters=4
                    )
                    
                    # 4. 构建角色选择提示
                    background_text = f"""
                    【历史背景】{historical_event.get('background', '')}
                    【事件描述】{historical_event.get('description', '')}
                    【历史事实】{historical_event.get('historical_fact', '')}
                    
                    基于以上真实历史事件，为你呈现以下角色：
                    """
                    
                    character_list = "\n".join([
                        f"{i}. {char['name']} - {char['role_type']}\n   {char['background']}\n   目标：{char['target']}" 
                        for i, char in enumerate(rag_characters, 1)
                    ])
                    
                    response = f"{background_text}\n\n【可选角色】\n{character_list}"
                    
                    # 5. 存储 RAG 上下文到状态
                    result = {
                        'rag_context': [historical_event],
                        'historical_event': historical_event,
                        'rag_characters': rag_characters
                    }
                else:
                    # RAG 检索失败，回退到通用模式
                    print("⚠️  [RAG] 检索失败，使用通用角色生成")
                    response = "通用角色生成文本..."
                    result = {}
                    
            except Exception as e:
                print(f"❌ [RAG] 生成失败：{e}")
                import traceback
                traceback.print_exc()
                response = "RAG 系统异常，使用备用方案..."
                result = {}
        else:
            # 游戏进行中 - 生成动态事件
            historical_event = state.get('historical_event', {})
            player_input = state.get('player_input')
            turn_count = state.get('turn_count', 0)
            
            # 获取或创建事件生成器
            if 'event_generator' not in state.get('metadata', {}):
                # 初始化事件生成器
                event_gen = DynamicEventGenerator()
                
                # 设置时空背景
                time_period = f"公元{184 + (turn_count // 2)}年"  # 每 2 回合过 1 年
                location = "冀州"  # 初始地点，可以根据角色调整
                
                if historical_event:
                    # 从历史事件中推断地点
                    desc = historical_event.get('description', '')
                    for loc in ['冀州', '荆州', '益州', '扬州', '徐州', '豫州', '兖州', '青州', '并州', '凉州']:
                        if loc in desc:
                            location = loc
                            break
                
                event_gen.set_context(time_period, location)
                
                # 如果是第一次，生成初始事件
                if turn_count <= 1 and historical_event:
                    initial_evt = event_gen.generate_initial_event(historical_event)
                    print(f"🎬 初始事件：{initial_evt['name']}")
            
            # 获取当前角色信息（用于生成与角色相关的剧情）
            characters = state.get('characters', {})
            active_char = state.get('active_character')
            char_data = characters.get(active_char, {}) if characters else {}
            char_name = char_data.get('name', '主公')
            char_role = char_data.get('role_type', '英雄')
            
            prompt = f"""
            你是一位专业的 RPG 叙事者。当前是三国乱世。
                      
            【玩家角色】
            - 姓名：{char_name}
            - 身份：{char_role}
                      
            请根据以下信息推进剧情：
            - 游戏 ID: {state.get('game_id')}
            - 回合数：第{state.get('turn_count', 0)}回合
            - 已发生事件：{len(state.get('player_choices', []))} 次玩家行动
                      
            【重要要求】
           1. **以玩家角色为中心**：剧情必须围绕{char_name}展开，描述{char_name}的行动和遭遇，不要描述其他人物（如刘备、曹操等）的独立行动
            2. **提供互动选项**：给{char_name}提供 2-3 个具体的行动选项
            3. **符合角色身份**：选项要符合{char_role}的身份和能力范围
            4. **事件要有因果关系**：与上一回合的事件相衔接
                      
            【互动选项格式要求】
            - **必须**使用统一格式：
            - 每个选项独立一行
            - 示例：
              选项一 派遣斥候侦查宛城的防御情况...
              选项二 亲自带领精锐部队夜袭宛城...
              选项三 与部下商议制定更周全的计划...
                      
            请描述当前的局势变化、新发生的事件，并引导{char_name}进行下一步行动。
            """
        
        print("🤖 正在调用 LLM...")
        # await 等待异步函数完成
        # 类似 Java 的 CompletableFuture.get() 或 reactor 的 block()
        response = call_llm(prompt=prompt)
        print(f"✅ LLM 响应完成，生成了 {len(response)} 字符的内容")
        
        # 如果是角色选择阶段，提示玩家输入
        if not characters and current_phase == 'intro':
            print("\n" + "="*60)
            print("🎭 请选择您的角色（输入数字 1-4）:")
            print("="*60)
            print(response)
            print("\n💬 请在控制台输入您选择的角色编号...")
            # 实际游戏中这里应该有 input() 获取用户输入
            # 暂时跳过，后续通过 API 实现
        
        # 创建 Message 对象 (实际是字典)
        # TypedDict 在运行时还是 dict，可以直接创建
        new_message = Message(
            role="assistant",
            content=response,
            agent_type=AgentType.NARRATOR
        )
        
        print(f"📝 生成新消息 ({len(response)} 字符):")
        # 显示完整内容，不再截断
        print("-" * 60)
        print(response)
        print("-" * 60)
        
        # 返回增量更新
        # 类似 Java 的 Map.of("key", value)
        result = {
            # 追加新消息到消息列表
            # state["messages"] + [new_message] 类似 Java 的 list.add()
            "messages": state["messages"] + [new_message],
            
            # 回合数 +1
            "turn_count": state.get("turn_count", 0) + 1,
            
            # 初始化存活状态和游戏结束状态
            "is_alive": True,
            "game_over": False,
            "death_reason": None,
            
            # 保存元数据（包含事件生成器状态）
            "metadata": state.get("metadata", {}),
            
            # 更新阶段为 gameplay（如果不是 intro）
            "current_phase": "gameplay" if characters else "intro"
        }
        
        # 如果是角色选择阶段且有玩家输入，解析并创建角色
        if not characters and current_phase == 'intro':
            player_input = state.get('player_input')
            if player_input:
                try:
                    selection_index = int(player_input.strip())
                    if 1 <= selection_index <= 4:
                        # ========== 使用 RAG 生成的角色数据 ==========
                        rag_characters = state.get('rag_characters', [])
                        
                        if rag_characters and len(rag_characters) >= selection_index:
                            # 从 RAG 角色中选择
                            char_info = rag_characters[selection_index - 1]
                            print(f"✅ [RAG] 玩家选择了：{char_info['name']}")
                        else:
                            # 回退到旧版解析
                            char_info = parse_character_selection(response, selection_index)
                        
                        # 创建角色对象
                        character_id = f"char_{selection_index}"
                        character_data = {
                            'character_id': character_id,
                            'name': char_info['name'],
                            'background': char_info.get('background', ''),
                            'role_type': char_info.get('role_type', ''),
                            'relationships': char_info.get('relationships', {}),
                            'secrets': char_info.get('secrets', []),
                            'target': char_info.get('target', ''),
                            'alibi': None,
                            'historical_basis': char_info.get('historical_basis', '')
                        }
                        
                        # 添加到角色列表
                        characters_dict = {character_id: character_data}
                        result['characters'] = characters_dict
                        result['active_character'] = character_id
                        
                        print(f"\n✅ 角色创建成功！")
                        print(f"  - 角色姓名：{char_info['name']}")
                        print(f"  - 角色身份：{char_info.get('role_type', '未知')}")
                        print(f"  - 角色 ID: {character_id}")
                        print(f"  - 历史背景：{char_info.get('historical_basis', '未知')}")
                        
                        # 记录玩家选择
                        current_choices = state.get('player_choices', [])
                        current_choices.append({
                            'type': 'character_selection',
                            'selection': char_info,
                            'turn': state.get('turn_count', 0)
                        })
                        result['player_choices'] = current_choices
                        
                        # 清除 player_input，避免被后续节点误用
                        result['player_input'] = None
                except (ValueError, IndexError) as e:
                    print(f"⚠️  无效的角色选择：{e}")
        
        print(f"✅ [主持人 Node] 执行完成，返回更新：{list(result.keys())}")
        
        # 如果有角色且不是角色选择阶段，提示玩家发言
        if characters and active_char and current_phase != 'intro':
            char_data = characters.get(active_char, {})
            char_name = char_data.get('name', '未知角色')
            char_role = char_data.get('role_type', '未知')
            char_target = char_data.get('target', '')
            historical_event = state.get('historical_event', {})
            event_name = historical_event.get('event', '三国事件')
            
            print("\n" + "="*60)
            print(f"⚔️  {char_name}，轮到您行动了！")
            print("="*60)
            print(f"\n您的身份：{char_role}")
            print(f"当前局势：{event_name}")
            print(f"\n您可以：")
            print("  1. 询问其他人物")
            print("  2. 探查线索")
            print("  3. 表达立场")
            print("  4. 做出决策")
            
            # 获取最新的主持人消息（包含当前选项）
            latest_narrator_msg = None
            for msg in reversed(state.get('messages', [])):
                if msg.get('agent_type') == AgentType.NARRATOR and msg.get('role') == 'assistant':
                    latest_narrator_msg = msg.get('content', '')
                    break
            
            # 生成符合角色性格的参考发言（与主持人选项对齐）
            print(f"\n📜 参考策略 (符合{char_name}的性格和立场):")
            print("-" * 60)
            
            # 根据主持人给出的选项生成对应的参考策略
            historical_event = state.get('historical_event', {})
            message_history = state.get('messages', [])
            turn_count = state.get('turn_count', 0)
            
            # 如果有主持人的选项，基于选项生成参考策略
            if latest_narrator_msg and ('选项一' in latest_narrator_msg or '选项二' in latest_narrator_msg or '选项三' in latest_narrator_msg or '你可以' in latest_narrator_msg):
                # 从主持人消息中提取选项内容，生成对应的参考策略
              reference_speeches = generate_reference_speeches_from_options(
                    char_name, 
                    char_role, 
                    char_target, 
                    latest_narrator_msg,  # 传递主持人的消息
                    historical_event,
                    turn_count,
                    message_history
                )
            else:
                # 否则使用常规方式生成
             reference_speeches = generate_reference_speeches(
                    char_name, 
                    char_role, 
                    char_target, 
                    event_name,
                    historical_event,
                    turn_count,
                    message_history
                )

            for i, speech in enumerate(reference_speeches, 1):
                print(f"  [{i}] {speech}")
            
            print("-" * 60)
            print(f"\n请输入您的行动:")
            print(f"  - 直接输入文字使用自定义行动 (推荐)")
            print(f"  - 输入数字 1-{len(reference_speeches)} 选择参考策略")
            print(f"  - 关键剧情时请使用自定义行动，需符合历史背景")
            
            # 获取玩家输入
            try:
                player_input_raw = input(">>> ").strip()
                
                if not player_input_raw:
                    print("⚠️  阁下未做决断，暂且按兵不动...\n")
                else:
                    # 检查是否是数字选择
                    player_speech = ""
                    is_custom_input = False
                    if player_input_raw.isdigit() and 1 <= int(player_input_raw) <= len(reference_speeches):
                        # 玩家选择了参考发言
                        selected_index = int(player_input_raw) - 1
                        player_speech = reference_speeches[selected_index]
                        print(f"\n📝 您选择了：{player_speech}")
                    else:
                        # 玩家自定义发言
                        player_speech = player_input_raw
                        is_custom_input = True
                        print(f"\n📝 您的行动：{player_speech}")
                        
                        # 关键剧情点验证（如果是自定义输入）
                        if turn_count in [2, 3]:  # 第 2、3 回合为关键剧情点
                            print("\n⚠️  当前为关键剧情点，正在验证您的决策是否符合历史背景...")
                            validation_prompt = f"""
                            请判断以下主公的决策是否符合三国时期的历史背景和 RAG 史料：
                            
                            【当前历史事件】{event_name}
                            【历史事实】{historical_event.get('historical_fact', '未知')[:100]}
                            【主公决策】{player_speech}
                            
                            如果符合历史背景（即使试图改变历史但策略合理），返回：VALID
                            如果严重违背历史（如出现现代科技、超自然力量等），返回：INVALID
                            并简要说明理由。
                            
                            返回格式：VALID/INVALID + 理由
                            """
                            validation_result = call_llm(prompt=validation_prompt)
                            
                            if "INVALID" in validation_result.upper():
                                print(f"❌ 您的决策不符合历史背景：{validation_result}")
                                print("💡 请重新考虑您的决策，使其符合三国时期的历史条件。")
                                print("\n是否坚持此决策？(y/n):")
                                confirm = input(">>> ").strip().lower()
                                if confirm != 'y':
                                    print("⚠️  请重新输入您的行动:")
                                    player_speech = input(">>> ").strip()
                                    is_custom_input = True
                            else:
                                print(f"✅ 您的决策通过历史验证：{validation_result}")
                    
                    # 将玩家发言添加到消息历史
                    player_message = Message(
                        role="user",
                        content=player_speech,
                        agent_type=None
                    )
                    result["messages"] = result.get("messages", state["messages"]) + [player_message]
                    
                    # 存储玩家输入到状态
                    result["player_input"] = player_speech
                    
                    # 记录玩家选择历史
                    current_choices = state.get('player_choices', [])
                    current_choices.append({
                        'type': 'speech',
                        'content': player_speech,
                        'turn': state.get('turn_count', 0),
                        'character': active_char,
                        'is_reference': not is_custom_input,  # 标记是否使用了参考发言
                        'is_custom': is_custom_input,         # 标记是否为自定义输入
                        'validated': is_custom_input          # 自定义输入已验证
                    })
                    result['player_choices'] = current_choices
                    
                    print("✅ 您的行动已记录，将影响天下大势...\n")
                    
                    # ========== 动态事件推进 ==========
                    # 根据玩家行动生成/推进事件
                    historical_event = state.get('historical_event', {})
                    turn_count = state.get('turn_count', 0)
                    
                    # 尝试从 metadata 中获取事件生成器
                    metadata = state.get('metadata', {})
                    event_gen = metadata.get('event_generator')
                    
                    if not event_gen:
                        # 创建新的事件生成器
                        event_gen = DynamicEventGenerator()
                        
                        # 设置时空背景
                        time_period = f"公元{184 + (turn_count // 2)}年"
                        location = "冀州"
                        
                        if historical_event:
                            desc = historical_event.get('description', '')
                            for loc in ['冀州', '荆州', '益州', '扬州', '徐州', '豫州', '兖州', '青州', '并州', '凉州']:
                                if loc in desc:
                                    location = loc
                                    break
                        
                        event_gen.set_context(time_period, location)
                        
                        # 如果是早期回合，生成初始事件
                        if turn_count <= 2 and historical_event:
                            event_gen.generate_initial_event(historical_event)
                    
                    # 推进事件（根据玩家行动）
                    new_event = event_gen.advance_turn(player_speech, turn_count)
                    
                    # 保存事件生成器状态到 metadata
                    metadata['event_generator'] = event_gen
                    metadata['current_event'] = new_event
                    result['metadata'] = metadata
                    
                    # 如果有新事件发生，添加到消息历史
                    if new_event and new_event.get('type') == 'dynamic':
                        event_message = f"""
📜 **新事件发生**

【事件名称】{new_event.get('name', '未知事件')}
【发生地点】{new_event.get('location', '某地')}
【时间】{new_event.get('time', time_period)}

【事件描述】
{new_event.get('description', '')}

【涉及人物】{', '.join(new_event.get('participants', ['不明']))}
【局势影响】{new_event.get('consequences', '待观察')}
【游戏效果】{new_event.get('game_effect', '无明显变化')}

你需要对此做出决策...
                        """
                        result["messages"] = result["messages"] + [Message(
                            role="assistant",
                            content=event_message,
                            agent_type=AgentType.NARRATOR
                        )]
                        print(f"\n📜 新事件：{new_event.get('name')}")
                    
                    # ========== 生死判定逻辑 ==========
                    # 检查玩家的决策是否会导致死亡
                    if is_custom_input and turn_count >= 2:  # 第 2 回合后才开始判定
                        death_check_result = self._check_death_risk(
                            player_speech, 
                            historical_event,
                            char_data,
                            turn_count
                        )
                        
                        if death_check_result['should_die']:
                            # 玩家死亡，游戏结束
                            result['is_alive'] = False
                            result['game_over'] = True
                            result['death_reason'] = death_check_result['reason']
                            
                            # 添加死亡消息
                            death_message = f"""
💀 **你的结局**

{death_check_result['death_description']}

【死亡原因】{death_check_result['reason']}

【历史评价】{death_check_result['historical_judgment']}

游戏已结束。感谢游玩！
                            """
                            result["messages"] = result["messages"] + [Message(
                                role="assistant",
                                content=death_message,
                                agent_type=AgentType.NARRATOR
                            )]
                            
                            print(f"\n💀 玩家死亡：{death_check_result['reason']}")
            except Exception as e:
                print(f"⚠️  获取玩家输入失败：{e}\n")
        
        return result
        
    except Exception as e:
        # 异常处理
        # str(e) 将异常转为字符串，类似 e.getMessage()
        error_msg = f"Narrator error: {str(e)}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"errors": state.get("errors", []) + [error_msg]}


def _check_death_risk(player_speech: str, historical_event: dict, char_data: dict, turn_count: int) -> dict:
    """
    检查玩家的决策是否有死亡风险
    
    Args:
        player_speech: 玩家的决策/发言
        historical_event: 当前历史事件
        char_data: 角色数据
        turn_count: 当前回合数
        
    Returns:
        dict: 死亡判定结果
        {
            'should_die': bool,      # 是否应该死亡
            'reason': str,           # 死亡原因
            'death_description': str, # 死亡描述
            'historical_judgment': str # 历史评价
        }
    """
    try:
        # 构建 prompt 让 LLM 判断是否会导致死亡
        event_name = historical_event.get('event', '三国事件')
        event_fact = historical_event.get('historical_fact', '')
        char_name = char_data.get('name', '未知角色')
        char_role = char_data.get('role_type', '未知')
        
        prompt = f"""
你是一位熟悉三国历史的裁判。请判断以下主公的决策是否会导致其死亡。

【当前局势】
- 历史事件：{event_name}
- 历史事实：{event_fact[:150] if event_fact else '未知'}
- 角色姓名：{char_name}
- 角色身份：{char_role}
- 当前回合：第{turn_count}回合

【主公决策】
{player_speech}

【判定标准】
以下情况会导致死亡（任选其一即可）：
1. **战场冒险**：主动要求单挑、冲锋陷阵且成功率低
2. **政治错误**：公开反对权臣、揭露阴谋被反咬
3. **背叛行为**：背叛强大势力投靠弱小势力
4. **违抗军令**：拒绝执行必死的命令
5. **复仇冲动**：独自挑战实力远超自己的敌人
6. **天灾人祸**：身处必死之地（如城破被围）

【输出格式】
返回 JSON 格式：
{{
    "should_die": true/false,
    "reason": "死亡原因（一句话概括）",
    "death_description": "详细的死亡场景描述（50-100 字，半文半白风格）",
    "historical_judgment": "历史评价（如'勇烈可嘉，然智谋不足'）"
}}

注意：
- 大部分作死行为应该导致死亡（70% 概率）
- 但如果策略高明、符合历史趋势，可以不死
- 描述要符合三国时期的语言风格
"""
        
        # 调用 LLM
        response = call_llm(prompt=prompt)
        
        # 解析 JSON
        import json
        import re
        
        # 提取 JSON 部分
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'should_die': result.get('should_die', False),
                'reason': result.get('reason', '不明原因'),
                'death_description': result.get('death_description', ''),
                'historical_judgment': result.get('historical_judgment', '')
            }
        else:
            # 解析失败，默认不死亡
            return {
                'should_die': False,
                'reason': '',
                'death_description': '',
                'historical_judgment': ''
            }
            
    except Exception as e:
        print(f"⚠️  生死判定失败：{e}")
        return {
            'should_die': False,
            'reason': '',
            'death_description': '',
            'historical_judgment': ''
        }


# ==================== 侦探 Node ====================
async def detective_node(state: GameState) -> Dict[str, Any]:
    """
    侦探 Agent - 负责推理分析和线索整合
    
    @Java 程序员提示:
    - 结构与 narrator_node 类似
    - 每个 Node 专注于特定职责，类似单一职责原则
    """
    try:
        print("\n" + "="*60)
        print("🔍 [侦探 Node] 开始执行")
        print("="*60)
        
        # 获取线索信息
        clues = state.get("clues", [])  # 所有线索
        collected = state.get("collected_clues", [])  # 已收集线索
        historical_event = state.get("historical_event")  # RAG 选中的历史事件
        player_input = state.get("player_input")  # 主公最新指示
        
        print(f"📌 可用情报数量：{len(clues)}")
        print(f"📌 已收集情报：{len(collected)}")
        if historical_event:
            print(f"📌 当前历史事件：{historical_event['event']}")
        if player_input:
            print(f"💬 主公指示：{player_input[:50]}..." if len(player_input) > 50 else f"💬 主公指示：{player_input}")
        
        # 构建 prompt，确保始终有值
        event_name = historical_event['event'] if historical_event else '三国时期'
        event_desc = historical_event['description'] if historical_event else '未知'
        
        # 如果有玩家发言，加入到 prompt 中
        player_context = ""
        if player_input:
            # 检查玩家是否使用了参考发言
            player_choices = state.get('player_choices', [])
            is_reference = False
            if player_choices and player_choices[-1].get('is_reference'):
                is_reference = True
            
            # 构建对主公指示的回应上下文
            player_context = f"""
            
            【主公最新决策】{player_input}
            {"(此乃符合时局的谋略)" if is_reference else "(此乃主公的决断)"}
            
            【重要】作为谋士，你需要：
            1. 首先对主公的决策表示敬意并做出符合你身份的回应
            2. 然后结合情报进行推理分析
            3. 评估此决策对当前局势的影响
            4. 提出具体的执行建议或补充策略
            
            【历史走向原则】
            - 当前历史大势是：{event_name}
            - 若主公决策顺应历史趋势，可详述如何借势推进
            - 若主公试图逆转重大历史事件，需谨慎回应：
              * 可以表示理解但指出实际困难重重
              * 或委婉建议更可行的替代方案
              * 体现历史的必然性，但不完全否定可能性
            
            【回应风格】
            - 使用三国时期的语言风格（半文半白）
            - 体现谋士的身份和智慧
            - 避免现代用语或超自然内容
            - 字数控制在 50-100 字左右
            """
        
        prompt = f"""
        你是一位经验丰富的谋士，正在辅佐主公成就大业。
        
        【当前局势】
        历史背景：{event_name}
        事件描述：{event_desc}
        {player_context}
        
        【可用情报】
        情报总数：{len(clues)}
        已知情报：{collected}
        
        【你的任务】
        请结合历史事实和现有情报进行推理分析：
        1. 分析当前局势的关键点
        2. 评估各方势力的动向
        3. 提出可行的策略建议
        {"4. 特别要对主公的决策给出详细的分析和执行建议" if player_input else ""}
        
        注意：你的分析必须符合三国时期的历史逻辑，体现谋士的智慧。
        """
        
        print("🤖 正在调用 LLM...")
        response = call_llm(prompt=prompt)
        print(f"✅ LLM 响应完成，生成了 {len(response)} 字符的内容")
        
        new_message = Message(
            role="assistant",
            content=response,
            agent_type=AgentType.DETECTIVE
        )
        
        print(f"📝 生成新消息 ({len(response)} 字符):")
        # 显示完整内容，不再截断
        print("-" * 60)
        print(response)
        print("-" * 60)
        
        # 只更新消息列表
        result = {
            "messages": state["messages"] + [new_message],
            # 清空 player_input，避免被后续节点误用
            "player_input": None,
            # 继承存活状态
            "is_alive": state.get("is_alive", True),
            "game_over": state.get("game_over", False),
            "death_reason": state.get("death_reason"),
            # 继承元数据（事件生成器状态）
            "metadata": state.get("metadata", {})
        }
        
        print(f"✅ [侦探 Node] 执行完成，返回更新：{list(result.keys())}")
        return result
        
    except Exception as e:
        error_msg = f"Detective error: {str(e)}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"errors": state.get("errors", []) + [error_msg]}


# ==================== 嫌疑人 Node ====================
async def suspect_node(state: GameState) -> Dict[str, Any]:
    """
    嫌疑人 Agent - 扮演被询问的嫌疑人角色
    
    @Java 程序员提示:
    - 这个角色会"扮演"特定角色，类似角色扮演
    - 根据角色设定回应，保持角色一致性
    """
    try:
        print("\n" + "="*60)
        print("👤 [嫌疑人 Node] 开始执行")
        print("="*60)
        
        # 获取当前活跃角色
        active_char = state.get("active_character")
        characters = state.get("characters", {})
        historical_event = state.get("historical_event")  # RAG 选中的历史事件
        
        print(f"📌 当前活跃角色：{active_char}")
        print(f"📌 角色信息：{list(characters.keys()) if characters else '无'}")
        if historical_event:
            print(f"📌 当前历史事件：{historical_event['event']}")
        
        char_data = characters.get(active_char, {}) if characters else {}
        
        # 构建 prompt，确保始终有值
        char_name = char_data.get('name', active_char or '未知角色')
        char_role = char_data.get('role_type', '未知')
        char_background = char_data.get('background', '未知')
        char_target = char_data.get('target', '未知')
        char_secrets = ', '.join(char_data.get('secrets', [])) or '无'
        char_history = char_data.get('historical_basis', '三国时期')
        
        event_name = historical_event['event'] if historical_event else '三国时期的某个事件'
        event_desc = historical_event['description'] if historical_event else '未知'
        
        prompt = f"""
        你现在扮演嫌疑人角色：{char_name}
        
        角色设定：
        - 身份：{char_role}
        - 背景：{char_background}
        - 目标：{char_target}
        - 秘密：{char_secrets}
        - 历史背景：{char_history}
        
        当前历史事件：{event_name}
        事件描述：{event_desc}
        
        请根据角色设定和历史背景回应询问，保持角色一致性。
        你的回答必须符合三国时期的历史逻辑和人物关系。
        """
        
        print("🤖 正在调用 LLM...")
        response = call_llm(prompt=prompt)
        print(f"✅ LLM 响应完成，生成了 {len(response)} 字符的内容")
        
        new_message = Message(
            role="assistant",
            content=response,
            agent_type=AgentType.SUSPECT
        )
        
        print(f"📝 生成新消息 ({len(response)} 字符):")
        # 显示完整内容，不再截断
        print("-" * 60)
        print(response)
        print("-" * 60)
        
        result = {
            "messages": state["messages"] + [new_message],
            # 继承存活状态
            "is_alive": state.get("is_alive", True),
            "game_over": state.get("game_over", False),
            "death_reason": state.get("death_reason"),
            # 继承元数据
            "metadata": state.get("metadata", {})
        }
        
        print(f"✅ [嫌疑人 Node] 执行完成，返回更新：{list(result.keys())}")
        return result
        
    except Exception as e:
        error_msg = f"Suspect error: {str(e)}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"errors": state.get("errors", []) + [error_msg]}


# ==================== 证人 Node ====================
async def witness_node(state: GameState) -> Dict[str, Any]:
    """
    证人 Agent - RPG 中的 NPC 互动，提供情报、任务或线索
    
    @Java 程序员提示:
    - 这个 Node 会使用 RAG 检索结果
    - retrieved_contexts 是从向量数据库检索的相关上下文
    - 类似 RPG 游戏中的 NPC 对话系统
    """
    try:
        print("\n" + "="*60)
        print("👁️  [证人 Node] 开始执行")
        print("="*60)
        
        # 获取 RAG 检索结果
        retrieved = state.get("retrieved_contexts", [])
        historical_event = state.get("historical_event")  # RAG 选中的历史事件
        
        # 获取当前角色信息
        characters = state.get('characters', {})
        active_char = state.get('active_character')
        char_data = characters.get(active_char, {}) if characters else {}
        char_name = char_data.get('name', '主公')
        char_role = char_data.get('role_type', '英雄')
        
        print(f"📌 检索到的上下文数量：{len(retrieved)}")
        if historical_event:
            print(f"📌 当前历史事件：{historical_event['event']}")
        print(f"📌 玩家角色：{char_name} ({char_role})")
        
        # 条件判断和列表切片
        # retrieved[:3] 取前 3 个元素，类似 Java 的 list.subList(0, 3)
        if retrieved:
            context_text = "\n".join([ctx.get("content", "") for ctx in retrieved[:3]])
            print(f"📌 使用前 {min(3, len(retrieved))} 条上下文")
        else:
            context_text = "暂无相关证据"
            print("⚠️  无检索结果，使用默认文本")
        
        # 构建 prompt，确保始终有值
        event_name = historical_event['event'] if historical_event else '三国时期'
        event_fact = historical_event['historical_fact'] if historical_event else '未知'
        
        prompt = f"""
        你是三国时期的一位 NPC（可能是百姓、商人、士兵、官员等），正在与玩家{char_name}对话。
        
        【你的身份】随机生成一个符合当前场景的身份（如：冀州的老农、许昌的商人、洛阳的退伍老兵等）
        
        【当前背景】
        - 时代：{event_name}
        - 历史事实：{event_fact[:100] if event_fact else '未知'}
        
        【可用情报】（来自 RAG 检索）
        {context_text[:300] if context_text != "暂无相关证据" else "你听到了一些关于时局的传闻"}
        
        【你的任务】
        请以 NPC 的身份与{char_name}互动，可以：
      1. **提供情报**：告诉{char_name}一些关于当前局势的信息（基于 RAG 检索内容）
       2. **给予任务**：给{char_name}一个小型任务或请求帮助（如护送、侦查、讨伐等）
        3. **分享线索**：透露一些有用的线索或秘密
        4. **表达立场**：表明你对当前局势的看法和对{char_name}的态度
        
        【重要要求】
        请在回复的**最后**使用以下格式标记事件效果（如果有）：
        
        [GAME_EFFECT]
        类型：task/reward/info/battle
        名称：简短的任务或事件名称
        描述：具体的效果描述（如：获得 100 兵力、发现黄巾军营地、获得情报等）
        数值变化：{{"兵力": +100, "粮草": -50}} 或其他资源变化
        [/GAME_EFFECT]
        
        【要求】
        - 使用三国时期的语言风格（半文半白）
        - 符合你的 NPC 身份
        - 称呼对方为{char_name}或{char_role}
        - 字数 80-150 字
        - 结尾给{char_name}留下回应或选择的空间
        
        请直接开始对话，不要有任何额外说明。
        """
        
        print("🤖 正在调用 LLM...")
        response = call_llm(prompt=prompt)
        print(f"✅ LLM 响应完成，生成了 {len(response)} 字符的内容")
        
        new_message = Message(
            role="assistant",
            content=response,
            agent_type=AgentType.WITNESS
        )
        
        print(f"📝 生成新消息 ({len(response)} 字符):")
        # 显示完整内容，不再截断
        print("-" * 60)
        print(response)
        print("-" * 60)
        
        # 返回更新，包括检索结果
        result = {
            "messages": state["messages"] + [new_message],
            "retrieved_contexts": retrieved,
            # 继承存活状态
            "is_alive": state.get("is_alive", True),
            "game_over": state.get("game_over", False),
            "death_reason": state.get("death_reason"),
            # 继承元数据
            "metadata": state.get("metadata", {})
        }
        
        print(f"✅ [证人 Node] 执行完成，返回更新：{list(result.keys())}")
        return result
        
    except Exception as e:
        error_msg = f"Witness error: {str(e)}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"errors": state.get("errors", []) + [error_msg]}


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
