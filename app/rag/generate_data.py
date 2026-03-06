"""
数据生成脚本 - 基于 truehis_sources.txt 生成 romance_three_kingdoms.json
使用智谱 AI 大模型补全每条数据

@Java 程序员提示:
- 这是 ETL (Extract-Transform-Load) 脚本
- 从 txt 文件提取数据 → 调用 AI 处理 → 加载到 JSON 文件
- 类似 Java 的批处理程序
"""
import json  # JSON 处理
import re  # 正则表达式
import os  # 操作系统接口
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv  # 加载环境变量

# 导入智谱 AI 客户端
import sys

from zhipuai import ZhipuAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.llm_client import call_llm

# 加载环境变量
load_dotenv()

# 验证 API Key 是否已设置
def validate_api_key():
    """
    验证智谱 AI API Key 是否已配置
    
    Returns:
        bool: API Key 是否有效
    """
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("❌ 错误：ZHIPU_API_KEY 环境变量未设置")
        print("\n请按照以下步骤配置:")
        print("1. 在项目根目录的 .env 文件中添加:")
        print("   ZHIPU_API_KEY=your_api_key_here")
        print("\n2. 从智谱 AI 开放平台获取 API Key:")
        print("   https://open.bigmodel.cn/")
        print("\n3. 重新运行此程序")
        return False
    
    # 简单的格式验证
    if len(api_key) < 10 or '.' not in api_key:
        print("⚠️  警告：API Key 格式可能不正确")
        print(f"   当前 API Key: {api_key[:5]}...{api_key[-5:]}")
        print("   请确认 .env 文件中的 ZHIPU_API_KEY 配置正确")
        return False
    
    print(f"✅ API Key 验证通过：{api_key[:5]}...{api_key[-5:]}")
    return True



class DataGenerator:
    """
    数据生成器 - 从 txt 文件生成 JSON 数据
    
    @Java 程序员提示:
    - 这是数据处理类
    - 封装了整个生成流程
    - 类似 Java 的 Builder 模式
    """
    
    def __init__(self, input_file: str, output_file: str):
        """
        构造方法
        
        Args:
            input_file: 输入文件路径 (truehis_sources.txt)
            output_file: 输出文件路径 (romance_three_kingdoms.json)
        """
        self.input_file = input_file
        self.output_file = output_file
        self.data_list = []  # 存储生成的数据
    
    def parse_txt_file(self) -> List[Dict[str, str]]:
        """
        解析 txt 文件，提取原始数据
            
        Returns:
            List[Dict[str, str]]: 提取的原始数据列表
            
        @Java 程序员提示:
        - 这是 ETL 的 Extract 阶段
        - 使用正则表达式解析文本
        - 类似 Java 的 Pattern + Matcher
        """
        print(f"正在读取文件：{self.input_file}")
            
        # 读取文件内容
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 只使用正则匹配【正史】和【演义】标记，提取完整文本块
        zhengshi_pattern = r'【正史】(.*?)(?=\n【|$)'
        yanyi_pattern = r'【演义】(.*?)(?=\n【|$)'
            
        zhengshi_matches = re.findall(zhengshi_pattern, content, re.DOTALL)
        yanyi_matches = re.findall(yanyi_pattern, content, re.DOTALL)
            
        print(f"匹配到正史 {len(zhengshi_matches)} 条")
        print(f"匹配到演义 {len(yanyi_matches)} 条")
            
        raw_data = []
            
        # 处理正史数据 - 只保留完整文本，不提取任何字段
        for i, full_text in enumerate(zhengshi_matches):
            raw_data.append({
                "source_type": "正史",
                "full_text": full_text.strip(),
                "index": i  # 记录原始索引
            })
            
        # 处理演义数据 - 只保留完整文本，不提取任何字段
        for i, full_text in enumerate(yanyi_matches):
            raw_data.append({
                "source_type": "演义",
                "full_text": full_text.strip(),
                "index": i  # 记录原始索引
            })
            
        print(f"成功解析 {len(raw_data)} 条原始数据")
        if raw_data:
            print(f"第一条示例 (前 100 字): {raw_data[0]['full_text'][:100]}...")
        return raw_data
    
    def _extract_event(self, event_line: str) -> str:
        """
        从事件行中提取事件名称（已废弃，改用大模型）
            
        @Java 程序员提示:
        - 此方法已废弃，现在所有字段都交给大模型总结
        """
        return ""
    
    def _extract_theme_from_event(self, event: str) -> str:
        """
        从 event 字段中提取 theme：判断是人物、群体、地理位置还是历史事件
        
        Args:
            event: 事件名称
            
        Returns:
            str: theme（人物/群体/地理位置/历史事件等）
        
        @Java 程序员提示:
        - 使用大模型判断 event 的语义类型
        - 类似 Java 的分类器
        """
        try:
            # 构建专门的 prompt 来判断 event 的类型
            theme_prompt = f"""
请分析以下三国事件名称，判断它主要描述的是什么类型的主体。

【事件名称】
{event}

【分类要求】
请从以下类别中选择一个最合适的：
1. **人物**：具体的历史人物（如"诸葛亮"、"曹操"）
2. **群体**：组织、军队、势力（如"黄巾军"、"蜀汉"）
3. **地理位置**：城市、地点（如"洛阳"、"荆州"）
4. **历史事件**：战役、政变等（如"赤壁之战"、"黄巾起义"）
5. **制度政策**：政治制度、政策（如"屯田制"）
6. **其他**：不属于以上类别的事物

【输出格式】
只输出类别名称（人物/群体/地理位置/历史事件/制度政策/其他），不要有任何额外文字。
"""
            
            # 调用大模型
            response = call_llm(
                prompt=theme_prompt,
                model="glm-4-flash",
                temperature=0.3  # 降低温度，使输出更稳定
            )
            # 提取响应中的关键词
            theme = response.strip()
            # 去除可能的标点符号
            theme = theme.rstrip('.,.!?.')
            
            print(f"    [AI 原始返回]: {response.strip()}")
            print(f"    [处理后 theme]: {theme}")
            
            # 如果 AI 返回的内容不在预期范围内，使用默认值
            valid_themes = ['人物', '群体', '地理位置', '历史事件', '制度政策', '其他']
            if theme not in valid_themes:
                # 尝试匹配第一个词
                for valid_theme in valid_themes:
                    if valid_theme in theme:
                        theme = valid_theme
                        break
                else:
                    theme = '其他'  # 默认值
            
            return theme
            
        except Exception as e:
            print(f"  ⚠️  theme 提取失败：{e}，使用默认值")
            return '其他'
    
    def enrich_with_ai(self, raw_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        使用 AI 大模型补全数据
        
        Args:
            raw_data: 原始数据列表
            
        Returns:
            List[Dict[str, Any]]: 补全后的数据列表
        
        @Java 程序员提示:
        - 这是 ETL 的 Transform 阶段
        - 调用外部 API (智谱 AI)
        - 类似 Java 调用 REST 服务
        - 异步操作提高性能
        """
        print(f"正在使用 AI 补全 {len(raw_data)} 条数据...")
        
        enriched_data = []
        
        for i, item in enumerate(raw_data, 1):
            # 从完整文本中提取第一行用于显示
            first_line = item['full_text'].split('\n')[0][:50] if '\n' in item['full_text'] else item['full_text'][:50]
            print(f"\n处理第 {i}/{len(raw_data)} 条：{first_line}...")
            
            # 构建提示词
            prompt = self._build_enrich_prompt(item)
            
            try:
                # 验证 API Key
                if not validate_api_key():
                    print(f"✗ 第 {i}/{len(raw_data)} 条数据处理失败：API Key 未配置")
                    enriched_item = self._create_default_item(item, i)
                    enriched_data.append(enriched_item)
                    continue
                
                # 调用智谱 AI
                response = call_llm(
                    prompt=prompt,
                    model="glm-4-flash",
                    temperature=0.7
                )
                # 检查 API 调用是否成功
                if response.startswith("[错误]"):
                    print(f"✗ AI 调用失败：{response}")
                    enriched_item = self._create_default_item(item, i)
                    enriched_data.append(enriched_item)
                    continue
                
                # 解析 AI 返回的 JSON
                enriched_item = self._parse_ai_response(response, item)
                                
                # 强制修正 id 为递增序号（确保顺序）
                enriched_item['id'] = f"tk_{i:03d}"
                                
                # 从 event 字段总结 theme 字段：判断是人物、群体、地理位置还是历史事件
                if 'event' in enriched_item and enriched_item['event']:
                    theme = self._extract_theme_from_event(enriched_item['event'])
                    enriched_item['theme'] = theme
                    # 埋点：输出 theme 判断结果
                    print(f"  📊 event: {enriched_item['event']} → theme: {theme}")
                
                # 清理 description 字段：删除开头的序号（如"365."等）
                if 'description' in enriched_item:
                    enriched_item['description'] = re.sub(r'^\d+[\.、]\s*', '', enriched_item['description'])
                
                enriched_data.append(enriched_item)
                
                print(f"✓ 补全成功")
                
            except Exception as e:
                print(f"✗ 第 {i}/{len(raw_data)} 条处理失败：{e}")
                # 使用默认值
                enriched_item = self._create_default_item(item, i)
                enriched_data.append(enriched_item)
        
        return enriched_data
    
    def _build_enrich_prompt(self, item: Dict[str, str]) -> str:
        """
        构建 AI 补全提示词
        
        Args:
            item: 原始数据项
            
        Returns:
            str: 提示词
        
        @Java 程序员提示:
        - Prompt Engineering (提示词工程)
        - 类似 Java 的模板字符串
        - 指导 AI 生成所需格式
        """
        return f"""
请根据以下三国历史事件的完整文本，总结并生成符合 JSON 格式的数据。

【原始文本】（来源类型：{item['source_type']}）
{item['full_text']}

【输出要求】
请生成一个 JSON 对象，包含以下字段：
1. "id": 格式为 "tk_XXX"，XXX 为三位数字编号（从 001 开始）
2. "theme": **关键**：从事件中提取主要人物、地点或事物名称。可能是：
   - 人名：如"诸葛亮"、"曹操"、"关羽"
   - 地名：如"下邳"、"洛阳"、"荆州"
   - 事件名：如"黄巾起义"、"赤壁之战"
   - 制度/政策：如"屯田制"、"九品中正制"
   - 其他：根据事件内容判断最核心的主体
3. "event": 事件名称（简洁，10 字以内，去除开头的序号如"365."等）
4. "source_type": "{item['source_type']}"
5. "description": **包含背景说明 + 事件描述 + 社会影响的全部内容**（100-200 字，从原文中总结）
   - 背景说明：事件发生的历史背景、原因
   - 事件描述：具体发生了什么
   - 社会影响：事件的结果、影响、意义
6. "game_effect": **游戏效果字段**，从原文或 description 中提取"游戏效果："或"游戏/叙事效果："后面的内容（如有），没有则为从description总结设计最后都加上合理的影响数值或者百分比
7. "historical_fact": 从原文中提取出处文献的真实考证（如《后汉书·皇甫嵩传》记载了什么，或说明虚构）
8. "dramatic_value": 戏剧价值评级，从原文中的效果描述判断：
   - "very_high": 效果非常显著（如全境影响、数值变化>50%）
   - "high": 效果显著（如地区影响、数值变化 30-50%）
   - "medium": 效果一般（如局部影响、数值变化 10-30%）
   - "low": 效果轻微（如单一事件、数值变化<10%）
9. "tags": 标签数组（3-5 个关键词，如人物、地点、战役、计谋、制度等）

【输出格式】
直接输出 JSON 对象，不要有任何额外文字，不要使用 Markdown 代码块。

示例格式:
{{
  "id": "tk_001",
  "theme": "诸葛亮",
  "event": "草船借箭",
  "source_type": "演义",
  "description": "背景说明：黄巾之乱后，西北凉州地区因长期民族矛盾与苛政，再度爆发大规模叛乱。事件描述：羌人与汉人豪强边章、韩遂等人联合，聚众十余万，攻掠三辅（京兆尹、左冯翊、右扶风），兵锋直指长安。社会影响：严重威胁东汉西部统治，朝廷被迫调集重兵镇压，加剧了东汉王朝的衰落。",
  "game_effect": "玩家可在此区域进行平叛任务，获得声望奖励",
  "historical_fact": "《后汉书·皇甫嵩传》记载了此次叛乱的详细经过",
  "dramatic_value": "very_high",
  "tags": ["叛乱", "凉州", "东汉末年", "民族矛盾"]
}}
"""
    
    def _parse_ai_response(self, response: str, original_item: Dict[str, str]) -> Dict[str, Any]:
        """
        解析 AI 返回的 JSON 响应
        
        Args:
            response: AI 返回的字符串
            original_item: 原始数据项
            
        Returns:
            Dict[str, Any]: 解析后的数据项
        
        @Java 程序员提示:
        - JSON 解析
        - 类似 Java 的 Jackson ObjectMapper
        - 错误处理很重要
        """
        try:
            # 尝试从响应中提取 JSON
            # AI 可能返回 Markdown 代码块
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # 确保必需字段存在
                required_fields = ["id", "theme", "event", "source_type",
                                   "description", "game_effect", "historical_fact", "dramatic_value", "tags"]
                
                for field in required_fields:
                    if field not in data:
                        data[field] = original_item.get(field, "")
                
                return data
            else:
                raise ValueError("未找到 JSON 内容")
        
        except Exception as e:
            print(f"JSON 解析失败：{e}，使用默认值")
            return self._create_default_item(original_item)
    
    def _create_default_item(self, item: Dict[str, str], index: int = 0) -> Dict[str, Any]:
        """
        创建默认数据项 (AI 失败时使用)
        
        Args:
            item: 原始数据项
            index: 索引编号
            
        Returns:
            Dict[str, Any]: 默认数据项
        """
        full_text = item.get('full_text', '')
        
        # 简单规则：从完整文本中提取第一行作为 event
        first_line = full_text.split('\n')[0] if '\n' in full_text else full_text[:50]
        # 去掉序号
        import re
        first_line = re.sub(r'^\d+[\.、]\s*', '', first_line).strip()
        
        return {
            "id": f"tk_{index:03d}",
            "theme": "未知",
            "event": first_line[:20],
            "source_type": item["source_type"],
            "description": full_text[:200],
            "game_effect": "",  # 默认空值
            "historical_fact": f"详见{item['source_type']}记载",
            "dramatic_value": "low",
            "tags": [item["source_type"], "三国"]
        }
    
    def save_to_json(self, data: List[Dict[str, Any]]):
        """
        保存数据到 JSON 文件
        
        Args:
            data: 要保存的数据列表
        
        @Java 程序员提示:
        - 这是 ETL 的 Load 阶段
        - 文件 I/O 操作
        - 类似 Java 的 FileWriter + Jackson
        """
        # 确保目录存在
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存 JSON
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 数据已保存到：{self.output_file}")
        print(f"  共 {len(data)} 条记录")
    
    def generate(self, use_ai: bool = True, test_mode: bool = False, test_count: int = 10):
        """
        执行完整的生成流程
        
        Args:
            use_ai: 是否使用 AI 补全
            test_mode: 是否启用测试模式（只处理部分数据）
            test_count: 测试模式下处理的数据条数
        
        @Java 程序员提示:
        - 门面方法 (Facade)
        - 封装整个流程
        - 类似 Java 的模板方法模式
        """
        print("=" * 60)
        print("开始生成 romance_three_kingdoms.json")
        print("=" * 60)
        
        # 预先验证 API Key（如果使用 AI）
        if use_ai:
            print("\n🔍 正在验证 API Key 配置...")
            if not validate_api_key():
                print("\n⚠️  由于 API Key 未配置，自动切换到默认模式")
                use_ai = False
            else:
                print("✅ API Key 验证通过，将使用 AI 补全数据\n")
        
        # 步骤 1: 解析 txt 文件
        raw_data = self.parse_txt_file()
        
        if not raw_data:
            print("✗ 未找到任何数据")
            return
        
        # 测试模式：均匀抽样
        if test_mode and len(raw_data) > test_count:
            print(f"\n📊 测试模式：从 {len(raw_data)} 条数据中均匀抽取 {test_count} 条")
            import random
            # 计算抽样间隔
            step = len(raw_data) // test_count
            # 均匀选取索引
            selected_indices = [i * step for i in range(test_count)]
            # 确保不超过列表长度
            selected_indices = [min(idx, len(raw_data) - 1) for idx in selected_indices]
            # 按索引选取数据
            raw_data = [raw_data[i] for i in selected_indices]
            print(f"✓ 已选取 {len(raw_data)} 条数据，索引分布：{selected_indices}")
        
        # 步骤 2: 补全数据 (AI 或默认)
        if use_ai:
            enriched_data = self.enrich_with_ai(raw_data)
        else:
            print("跳过 AI 补全，使用默认值")
            enriched_data = [self._create_default_item(item, i) 
                           for i, item in enumerate(raw_data, 1)]
        
        # 步骤 3: 保存到 JSON
        self.save_to_json(enriched_data)
        
        print("\n" + "=" * 60)
        print("✓ 数据生成完成！")
        print("=" * 60)


def main():
    """
    主函数 - 程序入口
    
    @Java 程序员提示:
    - 类似 Java 的 public static void main(String[] args)
    - 程序执行的起点
    """
    # 文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_file = os.path.join(script_dir, "truehis_sources.txt")
    output_file = os.path.join(project_root, "data", "romance_three_kingdoms.json")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"✗ 输入文件不存在：{input_file}")
        return
    
    # 询问是否使用 AI
    use_ai_input = input("是否使用 AI 补全数据？(y/n，默认 y): ").strip().lower()
    use_ai = use_ai_input not in ['n', 'no']
    
    # 如果选择使用 AI，先验证 API Key
    if use_ai:
        print("\n🔍 正在验证 API Key 配置...")
        if not validate_api_key():
            print("\n⚠️  API Key 未配置，将使用默认值生成数据")
            print("💡 提示：配置 API Key 后可以获得更高质量的数据\n")
            use_ai = False
        else:
            print("✅ API Key 验证通过\n")
    
    # 询问是否启用测试模式
    test_mode_input = input("是否启用测试模式（只处理 10 条均匀分布的数据）？(y/n，默认 n): ").strip().lower()
    test_mode = test_mode_input in ['y', 'yes']
    if test_mode:
        print("\n📊 测试模式：将从文档中均匀抽取 10 条数据")
    else:
        test_mode = False
    
    # 创建生成器并执行
    generator = DataGenerator(input_file, output_file)
    
    # 执行生成
    generator.generate(use_ai=use_ai, test_mode=test_mode, test_count=10)


if __name__ == "__main__":
    main()
