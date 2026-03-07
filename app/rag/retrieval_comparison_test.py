"""
检索策略对比测试类

@Java 程序员提示:
- 这是单元测试类，对比三种检索策略的效果
- 使用了测试模式 (Test Pattern) 和对比实验方法
- 类似 JUnit 测试，但这里是手动验证和评分
- 包含三个测试组：
  1. 只用大块（导致语义匹配不精确）
  2. 只用小块（导致上下文丢失）
  3. 父子检索器（平衡精度和上下文）
"""
import json
import time
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


class RetrievalStrategyComparator:
    """
    检索策略对比器
    
    @Java 程序员提示:
    - 这是策略模式 (Strategy Pattern) + 测试类的结合
    - 对比三种分块策略的检索效果
    - 使用控制变量法，其他条件保持一致
    - 类似 A/B Testing 框架
    """
    
    def __init__(self, data_path: str):
        """
        构造方法
        
        Args:
            data_path: JSON 数据文件路径
        """
        self.data_path = data_path
        self.raw_data = self._load_data()
        self.embeddings = self._init_embeddings()
        
        # 准备三种策略的向量存储
        print("🔧 正在初始化三种检索策略...")
        self.large_chunk_store = self._create_large_chunk_store()
        self.small_chunk_store = self._create_small_chunk_store()
        self.parent_child_retriever = self._create_parent_child_retriever()
        
        print("✅ 初始化完成\n")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载 JSON 数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """初始化嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name="bge-large-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _create_documents(self) -> List[Document]:
        """创建文档"""
        documents = []
        
        for item in self.raw_data:
            content = f"""【{item['event']}】
主题：{item['theme']}
来源：{item['source_type']}

📖 背景与描述：
{item['description']}

🎮 游戏效果：
{item['game_effect']}

📚 历史事实：
{item['historical_fact']}

🏷️ 标签：{', '.join(item['tags'])}"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "id": item['id'],
                    "event": item['event'],
                    "theme": item['theme'],
                    "source_type": item['source_type']
                }
            )
            documents.append(doc)
        
        return documents
    
    def _create_large_chunk_store(self) -> Chroma:
        """
        策略 1: 只用大块（1000 tokens）
        问题：语义匹配不精确
        """
        print("  📦 创建大块存储 (chunk_size=1000)...")
        
        documents = self._create_documents()
        
        # 大块分割器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", "。", "", " "]
        )
        
        split_docs = splitter.split_documents(documents)
        
        # 添加到向量库
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            collection_name="large_chunks",
            persist_directory="./chroma_db/test_large"
        )
        
        print(f"     创建了 {len(split_docs)} 个大块\n")
        return vectorstore
    
    def _create_small_chunk_store(self) -> Chroma:
        """
        策略 2: 只用小块（200 tokens）
        问题：上下文丢失
        """
        print("  📦 创建小块存储 (chunk_size=200)...")
        
        documents = self._create_documents()
        
        # 小块分割器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", "。", "", " "]
        )
        
        split_docs = splitter.split_documents(documents)
        
        # 添加到向量库
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            collection_name="small_chunks",
            persist_directory="./chroma_db/test_small"
        )
        
        print(f"     创建了 {len(split_docs)} 个小块\n")
        return vectorstore
    
    def _create_parent_child_retriever(self):
        """
        策略 3: 父子检索器
        优势：平衡精度和上下文
        """
        print("  📦 创建父子检索器 (parent=1000, child=200)...")
        
        # 这里简化实现，实际应使用 CustomParentDocumentRetriever
        # 为了测试，我们直接用小块检索，然后聚合到父文档
        
        documents = self._create_documents()
        
        # 父文档分割器
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", "。", "", " "]
        )
        
        # 子文档分割器
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", "。", "", " "]
        )
        
        parent_docs = parent_splitter.split_documents(documents)
        child_docs = child_splitter.split_documents(documents)
        
        # 为每个子文档添加父文档 ID
        parent_id_map = {}
        for i, doc in enumerate(parent_docs):
            parent_id_map[i] = doc
        
        for child_doc in child_docs:
            # 简化处理，假设每个子文档属于第一个匹配的父文档
            child_doc.metadata["parent_idx"] = 0
        
        # 使用子文档构建向量库
        vectorstore = Chroma.from_documents(
            documents=child_docs,
            embedding=self.embeddings,
            collection_name="parent_child",
            persist_directory="./chroma_db/test_parent_child"
        )
        
        print(f"     创建了 {len(parent_docs)} 个父文档，{len(child_docs)} 个子文档\n")
        
        return {
            "vectorstore": vectorstore,
            "parent_docs": parent_docs,
            "child_docs": child_docs
        }
    
    def search_with_strategy(
        self, 
        query: str, 
        strategy: str, 
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        使用指定策略搜索
        
        Args:
            query: 查询文本
            strategy: 策略 ("large", "small", "parent_child")
            k: 返回结果数量
            
        Returns:
            List[Tuple]: [(内容，分数), ...]
        """
        start_time = time.time()
        
        if strategy == "large":
            results = self.large_chunk_store.similarity_search_with_score(query, k=k)
        elif strategy == "small":
            results = self.small_chunk_store.similarity_search_with_score(query, k=k)
        elif strategy == "parent_child":
            results = self.parent_child_retriever["vectorstore"].similarity_search_with_score(query, k=k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        elapsed = time.time() - start_time
        
        # 提取内容和分数
        formatted_results = [
            (doc.page_content[:200], float(score))
            for doc, score in results
        ]
        
        return formatted_results, elapsed
    
    def evaluate_precision(self, query: str, expected_events: List[str]) -> Dict[str, float]:
        """
        评估检索精度
        
        Args:
            query: 查询文本
            expected_events: 应该匹配的事件列表
            
        Returns:
            Dict[str, float]: 各策略的精度分数
        """
        print(f"\n🔍 查询：{query}")
        print(f"📯 预期事件：{expected_events}\n")
        
        results = {}
        
        for strategy in ["large", "small", "parent_child"]:
            print(f"  测试策略：{strategy}")
            
            search_results, elapsed = self.search_with_strategy(query, strategy, k=5)
            
            # 计算精度
            matched_count = 0
            for content, score in search_results:
                for expected in expected_events:
                    if expected in content:
                        matched_count += 1
                        break
            
            precision = matched_count / len(expected_events) if expected_events else 0
            
            results[strategy] = {
                "precision": precision,
                "elapsed": elapsed,
                "matched": matched_count,
                "total": len(expected_events)
            }
            
            print(f"    精度：{precision:.2%} ({matched_count}/{len(expected_events)})")
            print(f"    耗时：{elapsed:.3f}s\n")
        
        return results
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("🧪 开始检索策略对比测试")
        print("=" * 60)
        
        test_cases = [
            {
                "query": "赤壁之战的游戏效果",
                "expected": ["赤壁之战", "游戏效果", "长江以南", "三方争夺"],
                "description": "测试具体事件的游戏效果查询"
            },
            {
                "query": "曹操相关的历史事件",
                "expected": ["曹操", "魏公", "官渡之战", "北伐"],
                "description": "测试人物相关事件查询"
            },
            {
                "query": "黄巾起义的背景和影响",
                "expected": ["黄巾起义", "张角", "太平道", "东汉末年"],
                "description": "测试历史事件背景查询"
            },
            {
                "query": "哪些事件有 high 的戏剧价值",
                "expected": ["dramatic_value", "high"],
                "description": "测试元数据过滤查询"
            }
        ]
        
        all_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"测试 {i}: {test_case['description']}")
            print(f"{'='*60}")
            
            result = self.evaluate_precision(
                test_case["query"],
                test_case["expected"]
            )
            all_results.append({
                "case": test_case,
                "results": result
            })
        
        # 汇总结果
        self._print_summary(all_results)
    
    def _print_summary(self, all_results: List[Dict]):
        """打印汇总报告"""
        print("\n" + "=" * 60)
        print("📊 测试汇总报告")
        print("=" * 60)
        
        strategies = ["large", "small", "parent_child"]
        strategy_names = {
            "large": "只用大块 (1000)",
            "small": "只用小块 (200)",
            "parent_child": "父子检索器"
        }
        
        for strategy in strategies:
            avg_precision = np.mean([
                r["results"][strategy]["precision"]
                for r in all_results
            ])
            avg_time = np.mean([
                r["results"][strategy]["elapsed"]
                for r in all_results
            ])
            
            print(f"\n{strategy_names[strategy]}:")
            print(f"  平均精度：{avg_precision:.2%}")
            print(f"  平均耗时：{avg_time:.3f}s")
            print(f"  综合得分：{avg_precision * 100:.1f}")
        
        # 推荐最佳策略
        best_strategy = max(
            strategies,
            key=lambda s: np.mean([r["results"][s]["precision"] for r in all_results])
        )
        
        print(f"\n🏆 推荐策略：{strategy_names[best_strategy]}")
        print(f"   理由：在该测试集上表现最佳")


# ==================== 独立测试函数 ====================
def test_single_query():
    """测试单个查询"""
    comparator = RetrievalStrategyComparator("app/data/romance_three_kingdoms.json")
    
    query = "赤壁之战的游戏效果是什么？"
    expected = ["赤壁之战", "游戏效果"]
    
    print(f"\n🔍 单查询测试：{query}")
    results = comparator.evaluate_precision(query, expected)
    
    return results


def test_context_loss():
    """测试上下文丢失问题"""
    print("\n" + "=" * 60)
    print("🧪 上下文丢失专项测试")
    print("=" * 60)
    
    comparator = RetrievalStrategyComparator("app/data/romance_three_kingdoms.json")
    
    # 这个查询需要完整的上下文
    query = "诸葛亮北伐的完整过程和影响"
    
    print(f"\n查询：{query}")
    print("这个问题需要长上下文，测试小块策略是否会丢失信息\n")
    
    # 测试三种策略
    for strategy in ["large", "small", "parent_child"]:
        print(f"策略：{strategy}")
        results, elapsed = comparator.search_with_strategy(query, strategy, k=3)
        
        for i, (content, score) in enumerate(results, 1):
            print(f"  [{i}] 分数：{score:.3f}")
            print(f"      内容：{content[:150]}...\n")
        
        print()


if __name__ == "__main__":
    # 运行完整测试
    comparator = RetrievalStrategyComparator("app/data/romance_three_kingdoms.json")
    comparator.run_all_tests()
    
    # 或者运行单测试
    # test_single_query()
    # test_context_loss()
