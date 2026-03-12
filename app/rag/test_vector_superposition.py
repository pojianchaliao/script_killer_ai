"""
向量叠加数学直觉测试 - RAG 中的语义合成与剥离
验证：v⃗_target ≈ v⃗_A + v⃗_B - v⃗_C

@Java 程序员提示:
- 这个测试展示了向量空间的线性性质在 RAG 中的应用
- 类似 Word2Vec 的经典例子：King - Man + Woman = Queen
- 核心思想：语义可以在 Embedding 空间中叠加和排除
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from colorama import Fore, Style, init

# 初始化 colorama
init(autoreset=True)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 尝试使用 langchain_huggingface，如果失败则使用 langchain_community
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# ==================== 向量叠加检索器 ====================
class VectorCombinationRetriever:
    """
    支持向量叠加的检索器
    
    @Java 程序员提示:
    - 这是核心类，实现向量组合检索
    - 利用 Embedding 空间的线性性质
    - 类似 Java 的策略模式 + 外观模式
    """
    
    def __init__(self, data_path: str = None, use_existing_vectorstore: bool = True):
        """
        构造方法
        
        Args:
            data_path: JSON 数据文件路径
            use_existing_vectorstore: 是否使用项目已有的 ChromaDB 向量库
        """
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN} 正在初始化向量叠加检索器{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        # 保存数据路径（用于加载文档信息）
        self.data_path = data_path
        
        # 加载数据（用于显示详细信息）
        self.documents = []
        if data_path and os.path.exists(data_path):
            self.documents = self._load_data()
            print(f"   加载了 {len(self.documents)} 条记录")
        else:
            print(f"  {Fore.YELLOW}  未提供数据文件{Style.RESET_ALL}")
        
        # 初始化嵌入模型
        print(f"{Fore.YELLOW} 步骤 1: 初始化 Embedding 模型...{Style.RESET_ALL}")
        self.embeddings = self._init_embeddings()
        print(f"   模型加载完成\n")
        
        # 使用项目已有的 ChromaDB 向量库
        if use_existing_vectorstore:
            print(f"{Fore.YELLOW} 步骤 2: 连接到现有 ChromaDB 向量库...{Style.RESET_ALL}")
            try:
                # 使用新的 langchain-chroma 包（避免弃用警告）
                try:
                    from langchain_chroma import Chroma
                except ImportError:
                    # 如果没有安装 langchain-chroma，回退到旧版本
                    from langchain_community.vectorstores import Chroma
                
                # 使用绝对路径
                vector_store_dir = Path(__file__).parent.parent.parent / "chroma_db" / "three_kingdoms"
                print(f"   向量库路径：{vector_store_dir}")
                
                self.vectorstore = Chroma(
                    collection_name="three_kingdoms_parent_child",
                    embedding_function=self.embeddings,
                    persist_directory=str(vector_store_dir)
                )
                print(f"   成功连接到向量库：{vector_store_dir}")
                print(f"   集合名称：three_kingdoms_parent_child")
                
                # 检查向量库是否为空
                try:
                    collection = self.vectorstore._collection
                    count = collection.count()
                    print(f"  ℹ  向量库中文档总数：{count}")
                    
                    if count == 0:
                        print(f"\n  {Fore.YELLOW}  向量库为空！将使用内存中的文档向量进行测试{Style.RESET_ALL}")
                        print(f"   提示：请先运行数据入库脚本：python app/rag/ingest_new.py\n")
                        self.use_vectorstore = False
                    else:
                        self.use_vectorstore = True
                        print(f"  {Fore.GREEN} 已启用 ChromaDB 向量库检索模式{Style.RESET_ALL}\n")
                        
                except Exception as e:
                    print(f"  {Fore.RED} 检查向量库失败：{e}{Style.RESET_ALL}")
                    print(f"  {Fore.YELLOW}  将使用内存中的文档向量进行测试{Style.RESET_ALL}")
                    import traceback
                    traceback.print_exc()
                    self.use_vectorstore = False
                    
            except Exception as e:
                print(f"  {Fore.RED} 连接向量库失败：{e}{Style.RESET_ALL}")
                print(f"  {Fore.YELLOW}  将使用简化模式（不依赖向量库）{Style.RESET_ALL}")
                import traceback
                traceback.print_exc()
                self.use_vectorstore = False
        else:
            self.use_vectorstore = False
        if not self.use_vectorstore:
            # 准备文档向量（如果有数据）
            if self.documents:
                print(f"{Fore.YELLOW} 步骤 2: 生成文档向量（内存模式）...{Style.RESET_ALL}")
                self.doc_texts, self.doc_vectors, self.doc_metadata = self._prepare_documents()
                print(f"   生成了 {len(self.doc_vectors)} 个文档向量\n")
            else:
                print(f"  {Fore.YELLOW}  无文档数据{Style.RESET_ALL}\n")
                self.doc_texts = []
                self.doc_vectors = None
                self.doc_metadata = []
        
        print(f"{Fore.GREEN} 检索器初始化完成{Style.RESET_ALL}\n")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载 JSON 数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在：{self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   加载了 {len(data)} 条记录")
        return data
    
    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """初始化嵌入模型（使用项目配置的 embedding 模块）"""
        # 直接使用项目中的 embedding 模块，避免重复加载
        try:
            from app.rag.embedding import embedding_model
            print(f"   使用项目已有的 Embedding 单例")
            return embedding_model.embeddings if hasattr(embedding_model, 'embeddings') else embedding_model
        except Exception:
            # 如果失败，创建新的实例
            print(f"  {Fore.YELLOW}  创建新的 Embedding 实例...{Style.RESET_ALL}")
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-zh-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def _prepare_documents(self) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """
        准备文档向量
        
        Returns:
            Tuple: (文本列表，向量数组，元数据列表)
        """
        texts = []
        metadata = []
        
        for i, doc in enumerate(self.documents[:50]):  # 只处理前 50 条用于演示
            # 拼接文档内容
            text = f"{doc.get('theme', '')} {doc.get('event', '')} {doc.get('description', '')}"
            texts.append(text)
            
            metadata.append({
                "index": i,
                "id": doc.get('id', ''),
                "event": doc.get('event', ''),
                "theme": doc.get('theme', ''),
                "source_type": doc.get('source_type', '')
            })
        
        # 批量生成向量
        vectors = self.embeddings.embed_documents(texts)
        vectors = np.array(vectors)
        
        return texts, vectors, metadata
    
    def vector_combine_search(
        self,
        positive_concepts: List[str],
        negative_concepts: List[str] = None,
        top_k: int = 3,
        use_weights: bool = False  # 新增参数：是否使用权重
    ) -> List[Dict[str, Any]]:
        """
        向量组合检索：v⃗_query = Σv⃗_positive - Σv⃗_negative
        
        Args:
            positive_concepts: 正向概念列表（需要保留的特征）
            negative_concepts: 负向概念列表（需要排除的特征）
            top_k: 返回前 k 个结果
            
        Returns:
            List[Dict[str, Any]]: 检索结果
        """
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN} 向量叠加检索实验{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        # 步骤 1: 获取所有概念的向量
        print(f"{Fore.YELLOW} 步骤 1: 计算各概念的向量表示{Style.RESET_ALL}")
        
        all_concepts = positive_concepts + (negative_concepts or [])
        concept_vectors = {}
        
        for concept in all_concepts:
            vector = self.embeddings.embed_query(concept)
            vector = np.array(vector)
            concept_vectors[concept] = vector
            print(f"   {concept}: 向量维度 = {vector.shape}, 前 5 维 = {vector[:5].round(3)}")
        
        # 步骤 2: 向量叠加（带权重优化）
        print(f"\n{Fore.YELLOW} 步骤 2: 向量叠加运算{Style.RESET_ALL}")
        if use_weights:
            print(f"  {Fore.CYAN}使用加权策略：主要概念权重更高{Style.RESET_ALL}")
        
        print(f"  公式：v⃗_query = {' + '.join(positive_concepts)}")
        if negative_concepts:
            print(f"         v⃗_query = v⃗_query - {' - '.join(negative_concepts)}")
        
        combined_vector = np.zeros_like(list(concept_vectors.values())[0])
        
        # 加上正向概念（带权重）
        for i, concept in enumerate(positive_concepts):
            if use_weights:
                # 第一个概念权重为 1.0，后续递减
                weight = 1.0 / (i + 1)
            else:
                weight = 1.0
            combined_vector += weight * concept_vectors[concept]
            print(f"  {Fore.GREEN}+{Style.RESET_ALL} 加上 '{concept}' (权重：{weight:.2f})")
        
        # 减去负向概念（带权重）
        if negative_concepts:
            for i, concept in enumerate(negative_concepts):
                if use_weights:
                    # 负向概念权重减半，避免过度排除
                    weight = 0.5 / (i + 1)
                else:
                    weight = 1.0
                combined_vector -= weight * concept_vectors[concept]
                print(f"  {Fore.RED}-{Style.RESET_ALL} 减去 '{concept}' (权重：{weight:.2f})")
        
        # 归一化组合向量（使其长度为 1，便于余弦相似度计算）
        norm = np.linalg.norm(combined_vector)
        if norm > 0:
            combined_vector = combined_vector / (norm + 1e-8)
        
        print(f"\n  {Fore.GREEN} 组合向量生成完成{Style.RESET_ALL}")
        print(f"     维度：{combined_vector.shape}")
        print(f"     前 5 维：{combined_vector[:5].round(3)}")
        print(f"     向量长度：{np.linalg.norm(combined_vector):.4f}")
        
        # 步骤 3: 使用组合向量检索
        print(f"\n{Fore.YELLOW} 步骤 3: 使用组合向量检索{Style.RESET_ALL}")
        
        if self.use_vectorstore:
            # 在实际的 ChromaDB 向量库中搜索
            print(f"  {Fore.GREEN} 在 ChromaDB 向量库中执行相似度搜索...{Style.RESET_ALL}")
            
            try:
                #  关键修复：直接获取 collection 并使用向量搜索
                # 不要使用 similarity_search_with_score（它期望文本输入）
                collection = self.vectorstore._collection
                
                # 使用向量的原生搜索 API
                results = collection.query(
                    query_embeddings=[combined_vector.tolist()],
                    n_results=top_k * 2,
                    include=['documents', 'metadatas', 'distances']
                )
                
                # 处理结果，去重父文档
                seen_ids = set()
                unique_results = []

                if results and len(results['ids']) > 0:
                    #  调试信息：显示原始距离值
                    print(f"\n  {Fore.CYAN}[调试] ChromaDB 返回的原始距离值:{Style.RESET_ALL}")
                    for i, dist in enumerate(results['distances'][0][:3], 1):
                        print(f"    第{i}名：distance={dist:.4f} → similarity={1.0-dist:.4f}")
                    
                    for i, doc_id in enumerate(results['ids'][0]):
                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                        doc_content = results['documents'][0][i] if results['documents'] else ''
                        distance = results['distances'][0][i] if results['distances'] else 0.0
                        
                        # 将距离转换为相似度分数（ChromaDB 使用余弦距离）
                        score = 1.0 - distance
                        
                        doc_id_from_meta = metadata.get("id") or metadata.get("parent_id")
                        if doc_id_from_meta and doc_id_from_meta not in seen_ids:
                            seen_ids.add(doc_id_from_meta)
                            unique_results.append({
                                "document_id": doc_id,
                                "content": doc_content,
                                "similarity_score": float(score),
                                "metadata": metadata
                            })
                            if len(unique_results) >= top_k:
                                break
                
                # 显示结果
                print(f"\n{Fore.GREEN} 检索结果:{Style.RESET_ALL}")
                for i, result in enumerate(unique_results, 1):
                    metadata = result["metadata"]
                    score = result["similarity_score"]
                    content_preview = result["content"][:150].replace('\n', ' ') if result["content"] else ''
                    
                    event = metadata.get('event', 'Unknown')
                    theme = metadata.get('theme', 'Unknown')
                    source_type = metadata.get('source_type', 'Unknown')
                    
                    print(f"\n  {i}. {event} (相似度：{score:.4f})")
                    print(f"     主题：{theme}")
                    print(f"     来源：{source_type}")
                    print(f"     内容：{content_preview}...")
                
                return unique_results
                
            except Exception as e:
                print(f"  {Fore.RED} 检索失败：{e}{Style.RESET_ALL}")
                import traceback
                traceback.print_exc()
                return []
        elif self.doc_vectors is not None and len(self.doc_vectors) > 0:
            # 计算与所有文档的余弦相似度
            # 因为向量已归一化，点积 = 余弦相似度
            similarities = np.dot(self.doc_vectors, combined_vector)
            
            # 获取 top_k 个最相似的文档
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "document": self.documents[idx],
                    "text": self.doc_texts[idx],
                    "similarity_score": float(similarities[idx]),
                    "metadata": self.doc_metadata[idx]
                })
            
            # 显示结果
            print(f"\n{Fore.GREEN} 检索结果:{Style.RESET_ALL}")
            for i, result in enumerate(results, 1):
                doc = result["document"]
                score = result["similarity_score"]
                print(f"\n  {i}. {doc.get('event', 'Unknown')} (相似度：{score:.4f})")
                print(f"     主题：{doc.get('theme', 'Unknown')}")
                print(f"     来源：{doc.get('source_type', 'Unknown')}")
                print(f"     描述：{doc.get('description', '')[:100]}...")
            
            return results
        else:
            print(f"\n  {Fore.YELLOW}  无向量库数据，跳过实际检索{Style.RESET_ALL}")
            print(f"   提示：请确保已运行数据入库脚本")
        
        return []
    
    def analyze_vector_geometry(self, concept_pairs: List[Tuple[str, str]]):
        """
        分析向量空间的几何关系
        
        Args:
            concept_pairs: 概念对列表，例如 [("国王", "君主"), ("将军", "士兵")]
        """
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN} 向量空间几何分析{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        for i, (concept_a, concept_b) in enumerate(concept_pairs, 1):
            vector_a = np.array(self.embeddings.embed_query(concept_a))
            vector_b = np.array(self.embeddings.embed_query(concept_b))
            
            # 计算余弦相似度
            cosine_sim = np.dot(vector_a, vector_b) / (
                np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            )
            
            # 计算欧氏距离
            euclidean_dist = np.linalg.norm(vector_a - vector_b)
            
            print(f"{Fore.YELLOW}[对比{i}] '{concept_a}' vs '{concept_b}'{Style.RESET_ALL}")
            print(f"  余弦相似度：{cosine_sim:.4f} (越接近 1 越相似)")
            print(f"  欧氏距离：{euclidean_dist:.4f} (越小越接近)")
            
            if cosine_sim > 0.9:
                print(f"  {Fore.GREEN} 语义非常接近{Style.RESET_ALL}")
            elif cosine_sim > 0.7:
                print(f"  {Fore.GREEN} 语义比较接近{Style.RESET_ALL}")
            else:
                print(f"  {Fore.YELLOW} 语义差异较大{Style.RESET_ALL}")
            print()


# ==================== 测试用例 ====================
def test_basic_vector_combination(retriever: VectorCombinationRetriever):
    """
    基础测试：简单的向量叠加
    
    场景：查找"谋略相关的计策"
    公式：v⃗ = v⃗_谋略 + v⃗_计策 + v⃗_智慧
    """
    print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA} 测试 1: 基础向量叠加 - '谋略与计策'{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    
    results = retriever.vector_combine_search(
        positive_concepts=["谋略", "计策", "智慧"],
        negative_concepts=None,
        top_k=3,
        use_weights=True  # 使用权重优化
    )
    
    return results


def test_negative_constraint(retriever: VectorCombinationRetriever):
    """
    负向约束测试：如何排除不需要的特征
    
    场景：查找"聪明的角色，但不要过于狡猾"
    公式：v⃗ = v⃗_聪明 + v⃗_智慧 - v⃗_狡猾
    """
    print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA} 测试 2: 负向约束 - '聪明但不狡猾'{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW} 面试防御话术演示:{Style.RESET_ALL}")
    print(f"""
    {Fore.CYAN}面试官:{Style.RESET_ALL} "你提到的数学直觉具体怎么用在 RAG 里？"
    
    {Fore.GREEN}你:{Style.RESET_ALL} "具体的公式推导我可能没法立刻手推，但我对向量空间的线性性质有很深的直觉。
    
    比如在这个测试中，用户问'{Fore.YELLOW}聪明的角色，但不要过于狡猾{Style.RESET_ALL}'，
    我会构造这样的组合向量：
    
    {Fore.YELLOW}v⃗_query = v⃗_聪明 + v⃗_智慧 - v⃗_狡猾{Style.RESET_ALL}
    
    我理解这背后的原理是：
     {Fore.GREEN}高维空间中的方向代表了语义特征{Style.RESET_ALL}
     {Fore.GREEN}向量的加减就是特征的合成与剥离{Style.RESET_ALL}
    
    基于这个直觉，我在调试 RAG 时：
    1. 会特别关注{Fore.RED}负向约束{Style.RESET_ALL}在向量空间的表现
    2. 不仅仅是调阈值，而是从{Fore.BLUE}语义空间结构{Style.RESET_ALL}入手
    3. 这让我能更灵活地处理复杂的业务查询
    4. 而不是死板地依赖单点检索"
    """)
    
    results = retriever.vector_combine_search(
        positive_concepts=["聪明", "智慧", "机智"],
        negative_concepts=["狡猾", "奸诈", "诡计多端"],
        top_k=3,
        use_weights=False  # 使用权重优化
    )

    results = retriever.vector_combine_search(
        positive_concepts=["聪明", "智慧", "机智","狡猾", "奸诈", "诡计多端"],
        negative_concepts=[],
        top_k=3,
        use_weights=False  # 使用权重优化
    )
    return results


def test_rag_game_scenario(retriever: VectorCombinationRetriever):
    """
    RAG 游戏场景测试：剧本杀角色检索
    
    场景：查找"有动机的反派角色，但不是主谋"
    公式：v⃗ = v⃗_动机 + v⃗_反派 + v⃗_三国 - v⃗_主谋
    """
    print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA} 测试 3: RAG 剧本杀场景 - '有动机的反派但不是主谋'{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    
    results = retriever.vector_combine_search(
        positive_concepts=["谋杀动机", "反派角色", "三国演义", "阴谋"],
        negative_concepts=["主谋", "真凶", "幕后黑手"],
        top_k=3,
        use_weights=True  # 使用权重优化
    )
    
    return results


def test_vector_analogy(retriever: VectorCombinationRetriever):
    """
    向量类比测试：验证向量空间的线性性质
    
    经典测试：king - man + woman = queen
    三国测试：诸葛亮 - 蜀汉 + 曹魏 = ?
    """
    print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA} 测试 4: 向量类比推理 - 验证线性性质{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    
    # 测试几组概念对的相似度
    retriever.analyze_vector_geometry([
        ("诸葛亮", "谋士"),
        ("曹操", "君主"),
        ("关羽", "武将"),
        ("赤壁之战", "战役"),
        ("战争", "和平"),
    ])


def test_complex_query(retriever: VectorCombinationRetriever):
    """
    复杂测试：多概念组合
    
    场景：查找"诸葛亮类型的谋士，但不要那么忠诚"
    公式：v⃗ = v⃗_诸葛亮 + v⃗_谋略 + v⃗_智慧 - v⃗_忠诚
    """
    print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA} 测试 5: 复杂语义组合 - '诸葛亮式谋士但不忠诚'{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    
    results = retriever.vector_combine_search(
        positive_concepts=["诸葛亮", "谋略", "智慧", "军事家"],
        negative_concepts=["忠诚", "鞠躬尽瘁"],
        top_k=3,
        use_weights=True  # 使用权重优化
    )
    
    return results


# ==================== 主函数 ====================
def main():
    """
    运行所有测试
    
    @Java 程序员提示:
    - 类似 JUnit 的测试套件
    - 展示向量叠加在 RAG 中的应用
    """
    print(f"\n{Fore.RED}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.RED}🚀 RAG 向量叠加数学直觉测试{Style.RESET_ALL}")
    print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}")
    
    # 数据文件路径 - 使用绝对路径
    data_file = str(Path(__file__).parent.parent / "data" / "romance_three_kingdoms.json")
    print(f"{Fore.CYAN} 数据文件：{data_file}{Style.RESET_ALL}\n")
    
    try:
        # 创建检索器（使用项目已有的 ChromaDB 向量库）
        print(f"{Fore.CYAN} 将连接到项目现有的 ChromaDB 向量库{Style.RESET_ALL}")
        print(f"   向量库路径：./chroma_db/three_kingdoms")
        print(f"   集合名称：three_kingdoms_parent_child\n")
        
        retriever = VectorCombinationRetriever(
            data_path=data_file,
            use_existing_vectorstore=True
        )
        
        # 运行所有测试
        test_basic_vector_combination(retriever)
        test_negative_constraint(retriever)
        test_rag_game_scenario(retriever)
        test_vector_analogy(retriever)
        test_complex_query(retriever)
        
        # 总结
        print(f"\n{Fore.RED}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.RED} 测试总结{Style.RESET_ALL}")
        print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}\n")
        
        print(f"""{Fore.CYAN}核心思想:{Style.RESET_ALL}
1. 向量空间中的{Fore.YELLOW}方向{Style.RESET_ALL}代表语义特征
2. 向量的{Fore.YELLOW}加法{Style.RESET_ALL} = 特征合成 (如："苹果" + "公司" = "Apple Inc")
3. 向量的{Fore.YELLOW}减法{Style.RESET_ALL} = 特征剥离 (如："电动车" - "特斯拉" = "平价电动车")
4. {Fore.GREEN}负向约束{Style.RESET_ALL}在向量空间中表现为特定方向的抑制

{Fore.CYAN}实际应用价值:{Style.RESET_ALL}
 处理复杂查询（多条件组合）
 实现负向约束（排除特定特征）
 提高检索灵活性（不依赖单一阈值）
 更好的业务查询支持（语义级别的操控）

{Fore.GREEN} 所有测试完成！{Style.RESET_ALL}
""")
        
    except Exception as e:
        print(f"\n{Fore.RED}❌ 测试失败：{e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
