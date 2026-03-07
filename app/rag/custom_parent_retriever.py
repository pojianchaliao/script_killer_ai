"""
自定义 ParentDocumentRetriever - 针对三国数据结构优化

@Java 程序员提示:
- 这是 LangChain ParentDocumentRetriever 的自定义实现
- 核心思想：用小 chunk 检索，返回大 chunk（保留上下文）
- 类似：先找到精确匹配的段落，再返回完整的章节
- 解决：大块语义不精确 vs 小块丢失上下文的矛盾
"""
from typing import List, Dict, Any, Optional, Tuple
import json
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.retrievers import ParentDocumentRetriever as LangChainParentRetriever
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
    - 这是 LangChain ParentDocumentRetriever 的封装
    - 使用小的子文档进行检索，返回大的父文档
    - 解决了大块语义不精确和小块丢失上下文的矛盾
    - 使用了装饰器模式 (Decorator Pattern)
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
            search_k: 检索返回的子文档数量
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
        self.docstore = self._init_docstore()
        self.retriever = self._init_retriever()
    
    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """初始化嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name="bge-large-zh-v1.5",
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
    
    def _init_docstore(self) -> InMemoryStore:
        """初始化文档存储"""
        return InMemoryStore()
    
    def _init_retriever(self) -> LangChainParentRetriever:
        """初始化 LangChain ParentDocumentRetriever"""
        # 父文档分割器 - 保持完整事件
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "", " "]
        )
        
        # 子文档分割器 - 按字段切分
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "", " "]
        )
        
        # 创建 LangChain ParentDocumentRetriever
        retriever = LangChainParentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": self.search_k}
        )
        
        return retriever
    
    def add_documents(self):
        """添加文档到检索器"""
        # 获取父文档
        parent_docs = self.processor.create_full_documents()
        
        # 添加到检索器（会自动处理父子关系）
        self.retriever.add_documents(parent_docs)
        
        print(f"✅ 已添加 {len(parent_docs)} 个父文档")
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            k: 返回结果数量（可选，覆盖默认值）
            
        Returns:
            List[Document]: 检索到的父文档列表
        """
        if k is not None:
            # 临时修改 search_k
            original_k = self.search_k
            self.retriever.search_kwargs["k"] = k
            results = self.retriever.invoke(query)
            self.retriever.search_kwargs["k"] = original_k
        else:
            results = self.retriever.invoke(query)
        
        return results
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        检索并返回带分数的文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Dict]: 包含文档内容和分数的字典列表
        """
        # 先在子文档中搜索（带分数）
        child_results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # 提取父文档 ID
        parent_ids = list(set([
            doc.metadata.get("parent_id") or doc.metadata.get("id")
            for doc, _ in child_results
        ]))
        
        # 返回父文档
        results = []
        for parent_id in parent_ids[:k]:
            # 从 docstore 获取父文档
            parent_doc = self.docstore.mget([parent_id])[0]
            
            if parent_doc:
                results.append({
                    "document": parent_doc,
                    "metadata": parent_doc.metadata,
                    "content": parent_doc.page_content
                })
        
        return results


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
