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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.llm_client import call_llm

# 加载环境变量
load_dotenv()


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
        
        # 使用两个单独的正则表达式分别匹配【正史】和【演义】
        zhengshi_pattern = r'【正史】([^\n]+)\n(.*?)(?=\n【|$)'
        yanyi_pattern = r'【演义】([^\n]+)\n(.*?)(?=\n【|$)'
        
        zhengshi_matches = re.findall(zhengshi_pattern, content, re.DOTALL)
        yanyi_matches = re.findall(yanyi_pattern, content, re.DOTALL)
        
        print(f"匹配到正史 {len(zhengshi_matches)} 条")
        print(f"匹配到演义 {len(yanyi_matches)} 条")
        
        raw_data = []
        
        # 处理正史数据
        for match in zhengshi_matches:
            event_line = match[0]  # 事件行
            description = match[1]  # 描述内容
            character = self._extract_character(event_line)
            event = self._extract_event(event_line)
            
            raw_data.append({
                "source_type": "正史",
                "character": character,
                "event": event,
                "description": description.strip()
            })
        
        # 处理演义数据
        for match in yanyi_matches:
            event_line = match[0]  # 事件行
            description = match[1]  # 描述内容
            character = self._extract_character(event_line)
            event = self._extract_event(event_line)
            
            raw_data.append({
                "source_type": "演义",
                "character": character,
                "event": event,
                "description": description.strip()
            })
        
        print(f"成功解析 {len(raw_data)} 条原始数据")
        return raw_data
    
    def _extract_character(self, event_line: str) -> str:
        """
        从事件行中提取人物名称
        
        Args:
            event_line: 事件描述行
            
        Returns:
            str: 人物名称
        
        @Java 程序员提示:
        - 私有方法 (_开头)
        - 类似 Java 的 private 方法
        - 使用简单规则提取人名
        """
        # 简单规则：查找常见人名
        # 实际可以更智能
        common_names = [
            "诸葛亮", "关羽", "张飞", "赵云", "马超", "黄忠",
            "曹操", "刘备", "孙权", "周瑜", "司马懿", "吕布",
            "董卓", "袁绍", "袁术", "刘表", "刘璋", "张鲁",
            "马超", "姜维", "邓艾", "钟会", "孙坚", "孙策",
            "陆逊", "吕蒙", "鲁肃", "张辽", "夏侯惇", "夏侯渊",
            "曹仁", "曹丕", "曹植", "荀彧", "郭嘉", "贾诩",
            "许褚", "典韦", "徐晃", "张郃", "乐进", "李典",
            "甘宁", "太史慈", "魏延", "庞统", "徐庶"
        ]
        
        for name in common_names:
            if name in event_line:
                return name
        
        # 如果没找到，返回"未知"
        return "未知人物"
    
    def _extract_event(self, event_line: str) -> str:
        """
        从事件行中提取事件名称
        
        Args:
            event_line: 事件描述行
            
        Returns:
            str: 事件名称
        
        @Java 程序员提示:
        - 提取事件标题
        - 类似 Java 的 substring 操作
        """
        # 简单规则：取事件行的前 20 个字符
        # 实际可以更智能
        event = event_line.split('：')[0] if '：' in event_line else event_line[:20]
        return event.strip()
    
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
            print(f"\n处理第 {i}/{len(raw_data)} 条：{item['event']}")
            
            # 构建提示词
            prompt = self._build_enrich_prompt(item)
            
            try:
                # 调用智谱 AI
                response = call_llm(
                    prompt=prompt,
                    model="glm-4-flash",
                    temperature=0.7
                )
                
                # 解析 AI 返回的 JSON
                enriched_item = self._parse_ai_response(response, item)
                enriched_data.append(enriched_item)
                
                print(f"✓ 补全成功")
                
            except Exception as e:
                print(f"✗ AI 调用失败：{e}")
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
请根据以下历史事件信息，生成符合 JSON 格式的数据。

【输入信息】
- 来源类型：{item['source_type']}
- 人物：{item['character']}
- 事件：{item['event']}
- 描述：{item['description']}

【输出要求】
请生成一个 JSON 对象，包含以下字段：
1. "id": 格式为 "tk_XXX"，XXX 为三位数字编号
2. "character": 人物姓名
3. "event": 事件名称
4. "source_type": "正史"或"演义"
5. "description": 事件描述 (100 字以内)
6. "historical_fact": 历史事实考证 (引用史书原文或说明虚构)
7. "dramatic_value": 戏剧价值评级 ("very_high", "high", "medium", "low")
8. "tags": 标签数组 (3-5 个关键词)

【输出格式】
直接输出 JSON 对象，不要有任何额外文字。

示例格式:
{{
  "id": "tk_001",
  "character": "诸葛亮",
  "event": "草船借箭",
  "source_type": "演义",
  "description": "诸葛亮利用大雾天气，以草船从曹操处借得十万支箭。",
  "historical_fact": "《三国演义》第四十六回虚构情节。历史上孙权曾有类似事迹，但非诸葛亮所为。",
  "dramatic_value": "very_high",
  "tags": ["智谋", "虚构", "经典桥段", "赤壁之战"]
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
                required_fields = ["id", "character", "event", "source_type", 
                                   "description", "historical_fact", "dramatic_value", "tags"]
                
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
        return {
            "id": f"tk_{index:03d}",
            "character": item["character"],
            "event": item["event"],
            "source_type": item["source_type"],
            "description": item["description"][:100] if len(item["description"]) > 100 else item["description"],
            "historical_fact": f"详见{item['source_type']}记载",
            "dramatic_value": "high",
            "tags": [item["source_type"], "三国", item["character"]]
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
    
    def generate(self, use_ai: bool = True):
        """
        执行完整的生成流程
        
        Args:
            use_ai: 是否使用 AI 补全
        
        @Java 程序员提示:
        - 门面方法 (Facade)
        - 封装整个流程
        - 类似 Java 的模板方法模式
        """
        print("=" * 60)
        print("开始生成 romance_three_kingdoms.json")
        print("=" * 60)
        
        # 步骤 1: 解析 txt 文件
        raw_data = self.parse_txt_file()
        
        if not raw_data:
            print("✗ 未找到任何数据")
            return
        
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
    
    # 创建生成器并执行
    generator = DataGenerator(input_file, output_file)
    
    # 询问是否使用 AI
    use_ai = input("是否使用 AI 补全数据？(y/n，默认 y): ").strip().lower()
    if use_ai not in ['n', 'no']:
        use_ai = True
    
    # 执行生成
    generator.generate(use_ai=use_ai)


if __name__ == "__main__":
    main()
