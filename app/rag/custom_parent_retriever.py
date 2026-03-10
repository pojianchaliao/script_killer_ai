"""
自定义 ParentDocumentRetriever - 针对三国数据结构优化

@Java 程序员提示:
- 这是 LangChain ParentDocumentRetriever 的自定义实现
- 核心思想：将父文档和子文档都存入向量库
- 检索时同时检索父文档和子文档，返回匹配的父文档
- 解决：大块语义不精确 vs 小块丢失上下文的矛盾
"""
import warnings

# 抑制 LangChain 弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==================== HuggingFace Token 配置 ====================
import os
from pathlib import Path as PathLib

def get_hf_token():
    """获取 HuggingFace Token（优先级：环境变量 > .env > ~/.huggingface_token）"""
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    
    try:
        project_root = PathLib(__file__).parent.parent.parent
        env_file = project_root / ".env"
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    
    try:
        home_dir = PathLib.home()
        token_file = home_dir / ".huggingface_token"
        if token_file.exists():
            with open(token_file, 'r', encoding='utf-8') as f:
                token = f.read().strip()
                if token:
                    return token
    except Exception:
        pass
    
    return None

hf_token = get_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    print("⚠️  未设置 HF_TOKEN，下载速度可能较慢")
    print("   获取 token: https://huggingface.co/settings/tokens\n")

from typing import List, Dict, Any, Optional, Tuple, Set
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 尝试使用新包，如果失败则使用旧包
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


# ==================== 自定义文档处理器 ====================
class ThreeKingdomsDocumentProcessor:
    """
    三国数据文档处理器
    
    @Java 程序员提示:
    - 负责将 JSON 数据转换为适合 RAG 的文档格式
    - 类似 DAO 层的数据转换
    - 支持多种分块策略
    """
    
    def __init__(self, data_path: str):
        """
        构造方法
        
        Args:
            data_path: JSON 数据文件路径
        """
        self.data_path = data_path
        self.raw_data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载 JSON 数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_full_documents(self) -> List[Document]:
        """
        创建完整文档（父文档）
        
        Returns:
            List[Document]: 包含所有字段的完整文档列表
        """
        documents = []
        
        for item in self.raw_data:
            # 格式化完整内容
            content = self._format_full_content(item)
            
            # 创建文档对象
            doc = Document(
                page_content=content,
                metadata={
                    "id": item['id'],
                    "event": item['event'],
                    "theme": item['theme'],
                    "source_type": item['source_type'],
                    "dramatic_value": item.get('dramatic_value', 'unknown'),
                    "tags": item['tags'],
                    "doc_type": "parent"
                }
            )
            documents.append(doc)
        
        return documents
    
    def create_child_documents(self) -> List[Document]:
        """
        创建子文档（按字段拆分）
        
        Returns:
            List[Document]: 按字段拆分的子文档列表
        """
        documents = []
        
        for item in self.raw_data:
            parent_id = item['id']
            
            # 1. description 子文档
            desc_doc = Document(
                page_content=f"{item['event']} - {item['theme']}\n{item['description']}",
                metadata={
                    "parent_id": parent_id,
                    "field": "description",
                    "event": item['event'],
                    "theme": item['theme']
                }
            )
            documents.append(desc_doc)
            
            # 2. game_effect 子文档
            effect_doc = Document(
                page_content=f"{item['event']}\n游戏效果：{item['game_effect']}",
                metadata={
                    "parent_id": parent_id,
                    "field": "game_effect",
                    "event": item['event']
                }
            )
            documents.append(effect_doc)
            
            # 3. historical_fact 子文档
            fact_doc = Document(
                page_content=f"{item['event']}\n历史事实：{item['historical_fact']}",
                metadata={
                    "parent_id": parent_id,
                    "field": "historical_fact",
                    "event": item['event']
                }
            )
            documents.append(fact_doc)
            
            # 4. tags 子文档
            tags_doc = Document(
                page_content=f"{item['event']}\n标签：{', '.join(item['tags'])}",
                metadata={
                    "parent_id": parent_id,
                    "field": "tags",
                    "event": item['event'],
                    "tags": item['tags']
                }
            )
            documents.append(tags_doc)
        
        return documents
    
    def _format_full_content(self, item: Dict[str, Any]) -> str:
        """
        格式化完整内容
        
        Args:
            item: JSON 对象
            
        Returns:
            str: 格式化后的文本内容
        """
        return f"""【{item['event']}】
主题：{item['theme']}
来源：{item['source_type']}
戏剧价值：{item.get('dramatic_value', 'unknown')}

📖 背景与描述：
{item['description']}

🎮 游戏效果：
{item['game_effect']}

📚 历史事实：
{item['historical_fact']}

🏷️ 标签：{', '.join(item['tags'])}"""


# ==================== 自定义 ParentDocumentRetriever ====================
class CustomParentDocumentRetriever:
    """
    自定义 ParentDocumentRetriever
    
    @Java 程序员提示:
    - 将父文档和子文档都存入向量库
    - 检索时同时检索父文档和子文档
    - 返回匹配到的父文档（去重）
    - 不再使用 LangChain 的 ParentDocumentRetriever
    """
    
    def __init__(
        self,
        data_path: str,
        parent_chunk_size: int = 1000,
        child_chunk_size: int = 200,
        chunk_overlap: int = 20,
        search_k: int = 5,
        persist_directory: Optional[str] = None
    ):
        """
        构造方法
        
        Args:
            data_path: JSON 数据文件路径
            parent_chunk_size: 父文档大小 (tokens)
            child_chunk_size: 子文档大小 (tokens)
            chunk_overlap: 重叠部分大小
            search_k: 检索返回的文档数量
            persist_directory: 向量存储持久化目录
        """
        self.data_path = data_path
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_k = search_k
        self.persist_directory = persist_directory or "./chroma_db/three_kingdoms"
        
        # 初始化组件
        self.processor = ThreeKingdomsDocumentProcessor(data_path)
        self.embeddings = self._init_embeddings()
        self.vectorstore = self._init_vectorstore()
    
    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """初始化嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _init_vectorstore(self) -> Chroma:
        """初始化向量存储"""
        return Chroma(
            collection_name="three_kingdoms_parent_child",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def add_documents(self):
        """添加文档到检索器"""
        # 获取父文档和子文档
        parent_docs = self.processor.create_full_documents()
        child_docs = self.processor.create_child_documents()
        
        # 将所有文档添加到向量库
        all_docs = parent_docs + child_docs
        self.vectorstore.add_documents(all_docs)
        
        print(f"✅ 已添加 {len(parent_docs)} 个父文档和 {len(child_docs)} 个子文档")
        print(f"   总计：{len(all_docs)} 个文档")
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            k: 返回结果数量（可选，覆盖默认值）
            
        Returns:
            List[Document]: 检索到的父文档列表（去重）
        """
        search_k = k if k is not None else self.search_k
        
        # 在向量库中搜索所有文档
        results = self.vectorstore.similarity_search(query, k=search_k * 2)  # 扩大搜索范围以确保找到足够的父文档
        
        # 提取唯一的父文档 ID
        parent_ids: Set[str] = set()
        parent_docs_map: Dict[str, Document] = {}
        
        for doc in results:
            doc_type = doc.metadata.get("doc_type")
            
            # 如果是父文档，直接加入结果
            if doc_type == "parent":
                doc_id = doc.metadata.get("id")
                if doc_id and doc_id not in parent_ids:
                    parent_ids.add(doc_id)
                    parent_docs_map[doc_id] = doc
            else:
                # 如果是子文档，通过 parent_id 找到对应的父文档
                parent_id = doc.metadata.get("parent_id")
                if parent_id and parent_id not in parent_ids:
                    # 需要从原始数据中构建父文档
                    parent_doc = self._get_parent_doc_by_id(parent_id)
                    if parent_doc:
                        parent_ids.add(parent_id)
                        parent_docs_map[parent_id] = parent_doc
        
        # 返回父文档列表
        return list(parent_docs_map.values())[:search_k]
    
    def _get_parent_doc_by_id(self, parent_id: str) -> Optional[Document]:
        """
        根据 ID 获取父文档
        
        Args:
            parent_id: 父文档 ID
            
        Returns:
            Optional[Document]: 父文档对象，如果不存在则返回 None
        """
        # 从原始数据中查找
        for item in self.processor.raw_data:
            if item['id'] == parent_id:
                content = self.processor._format_full_content(item)
                return Document(
                    page_content=content,
                    metadata={
                        "id": item['id'],
                        "event": item['event'],
                        "theme": item['theme'],
                        "source_type": item['source_type'],
                        "dramatic_value": item.get('dramatic_value', 'unknown'),
                        "tags": item['tags'],
                        "doc_type": "parent"
                    }
                )
        return None
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        检索并返回带分数的文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Dict]: 包含文档内容和分数的字典列表
        """
        # 在向量库中搜索带分数的文档
        child_results = self.vectorstore.similarity_search_with_score(query, k=k * 2)
        
        # 提取父文档 ID 并去重
        parent_ids: Set[str] = set()
        parent_docs_map: Dict[str, Tuple[Document, float]] = {}
        
        for doc, score in child_results:
            doc_type = doc.metadata.get("doc_type")
            
            if doc_type == "parent":
                doc_id = doc.metadata.get("id")
                if doc_id and doc_id not in parent_ids:
                    parent_ids.add(doc_id)
                    parent_docs_map[doc_id] = (doc, score)
            else:
                parent_id = doc.metadata.get("parent_id")
                if parent_id and parent_id not in parent_ids:
                    parent_doc = self._get_parent_doc_by_id(parent_id)
                    if parent_doc:
                        parent_ids.add(parent_id)
                        # 使用子文档的分数
                        parent_docs_map[parent_id] = (parent_doc, score)
        
        # 返回带分数的父文档
        results = []
        for parent_id, (doc, score) in parent_docs_map.items():
            results.append({
                "document": doc,
                "metadata": doc.metadata,
                "content": doc.page_content,
                "score": float(score)
            })
        
        return results[:k]


# ==================== 工厂函数 ====================
def create_three_kingdoms_retriever(
    data_path: str = "app/data/romance_three_kingdoms.json",
    parent_chunk_size: int = 1000,
    child_chunk_size: int = 200,
    chunk_overlap: int = 20,
    search_k: int = 5
) -> CustomParentDocumentRetriever:
    """
    创建三国数据检索器的工厂函数
    
    Args:
        data_path: JSON 数据路径
        parent_chunk_size: 父文档大小
        child_chunk_size: 子文档大小
        chunk_overlap: 重叠大小
        search_k: 检索结果数
        
    Returns:
        CustomParentDocumentRetriever: 配置好的检索器
    """
    return CustomParentDocumentRetriever(
        data_path=data_path,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
        search_k=search_k
    )
