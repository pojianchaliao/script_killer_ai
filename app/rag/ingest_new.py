"""
RAG Ingest - 数据入库脚本
支持普通文档和父子文档检索策略

@Java 程序员提示:
- Ingestor 是 ETL (Extract-Transform-Load) 工具
- 类似 Java 的批处理：读取文件 → 分块 → 向量化 → 存储
- 使用 Path 对象处理文件路径 (比字符串更安全)
- 支持多种分块策略：recursive, sentence, simple, parent_child
"""
import warnings

# 抑制 LangChain 弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
from pathlib import Path as PathLib, Path

# 添加项目根目录到 Python 路径
project_root = PathLib(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List, Dict, Any, Optional, Tuple  # 类型注解
from pathlib import Path as PathLib
project_root = PathLib(__file__).parent.parent.parent
data_dir=str(project_root / "app" / "data")
json_file_path=str(project_root / "app" / "data" / "romance_three_kingdoms.json")
import json  # JSON 处理
from app.config import settings  # 导入配置
from app.rag.embedding import get_embeddings  # 导入嵌入函数
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 尝试使用新包，如果失败则使用旧包
from langchain_chroma import Chroma



# ==================== 文档入库工具类 ====================
class DocumentIngestor:
    """
    文档入库工具 - 将文档转换为向量并存储
    
    @Java 程序员提示:
    - 这是数据处理管道 (Pipeline)
    - 流程：加载 → 分块 → 向量化 → 存储
    - 类似 Spring Batch 的 ItemProcessor
    - Path 是面向对象的路径处理，类似 Java 的 java.nio.file.Path
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        vector_store_path: str = None
    ):
        """
        构造方法
        
        Args:
            data_dir: 数据目录，存放待处理的文档
            vector_store_path: 向量存储路径
        
        @Java 程序员提示:
        - Path(data_dir) 将字符串转为 Path 对象
        - Path 提供更安全的路径操作
        - 类似 Java: Paths.get(dataDir)
        """
        self.data_dir = Path(data_dir)  # 转换为 Path 对象
        self.vector_store_path = vector_store_path or settings.VECTOR_STORE_PATH
        
        # 从配置读取分块参数
        self.chunk_size = settings.CHUNK_SIZE  # 每块大小
        self.chunk_overlap = settings.CHUNK_OVERLAP  # 块之间重叠
        
        # 父子文档专用参数
        self.parent_chunk_size = 1000  # 父文档大小
        self.child_chunk_size = 200    # 子文档大小
        self.parent_child_overlap = 20  # 父子文档重叠
    
    def load_documents(
        self,
        file_pattern: str = "*.txt",
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        加载文档
        
        Args:
            file_pattern: 文件匹配模式，如 "*.txt", "*.md", "*.json"
            recursive: 是否递归搜索子目录
            
        Returns:
            List[Dict[str, Any]]: 文档列表，每个包含内容和元数据
        
        @Java 程序员提示:
        - rglob 是递归 glob，类似 Java 的 Files.walk + PathMatcher
        - glob 是非递归的，类似 Java 的 Files.list + PathMatcher
        - 类似 Spring 的 ResourcePatternResolver
        """
        documents = []  # 存储加载的文档
        
        # 根据 recursive 选择搜索方式
        if recursive:
            # 递归搜索所有子目录
            # 类似 Java: Files.walk(dataDir).filter(path -> path.matches("*.txt"))
            files = list(self.data_dir.rglob(file_pattern))
        else:
            # 只搜索当前目录
            files = list(self.data_dir.glob(file_pattern))
        
        # 遍历所有匹配的文件
        for file_path in files:
            try:
                # 打开并读取文件
                # 'r' 表示只读，encoding='utf-8' 指定编码
                # 类似 Java: Files.readString(filePath, StandardCharsets.UTF_8)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 构建文档对象
                # 类似 Java 的 Map.of()
                documents.append({
                    "content": content,  # 文件内容
                    "metadata": {
                        "source": str(file_path),  # 文件路径
                        "filename": file_path.name  # 文件名
                    }
                })
            
            except Exception as e:
                # 异常处理
                print(f"加载文件失败 {file_path}: {e}")
        
        print(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def load_json_documents(
        self,
        json_file_path: str
    ) -> List[Dict[str, Any]]:
        """
        加载 JSON 格式的文档（针对三国数据结构）
        
        Args:
            json_file_path: JSON 文件路径
            
        Returns:
            List[Dict[str, Any]]: 文档列表，每个包含格式化后的内容和元数据
        
        @Java 程序员提示:
        - 专门处理 JSON 数组格式的数据
        - 类似 Java 的 Jackson/Gson 解析
        - 将 JSON 对象转换为统一的文档格式
        """
        documents = []
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 遍历 JSON 数组中的每个对象
            for item in json_data:
                # 格式化完整内容
                content = self._format_json_content(item)
                
                # 构建文档对象
                documents.append({
                    "content": content,
                    "metadata": {
                        "id": item.get('id', ''),
                        "event": item.get('event', ''),
                        "theme": item.get('theme', ''),
                        "source_type": item.get('source_type', ''),
                        "dramatic_value": item.get('dramatic_value', 'unknown'),
                        "tags": item.get('tags', []),
                        "source": json_file_path,
                        "doc_type": "json_parent"
                    }
                })
            
            print(f"成功加载 {len(documents)} 条 JSON 记录")
            
        except Exception as e:
            print(f"加载 JSON 文件失败 {json_file_path}: {e}")
        
        return documents
    
    def _format_json_content(self, item: Dict[str, Any]) -> str:
        """
        格式化 JSON 对象为文本内容
        
        Args:
            item: JSON 对象
            
        Returns:
            str: 格式化后的文本
        """
        tags_str = ', '.join(item.get('tags', []))
        
        return f"""【{item.get('event', 'Unknown')}】
主题：{item.get('theme', 'Unknown')}
来源：{item.get('source_type', 'Unknown')}
戏剧价值：{item.get('dramatic_value', 'unknown')}

📖 背景与描述：
{item.get('description', '')}

🎮 游戏效果：
{item.get('game_effect', '')}

📚 历史事实：
{item.get('historical_fact', '')}

🏷️ 标签：{tags_str}"""
    
    def split_documents(
        self,
        documents: List[Dict[str, Any]],
        strategy: str = "recursive"
    ) -> List[Dict[str, Any]]:
        """
        文档分块 - 将长文档切分为小块
        
        Args:
            documents: 文档列表
            strategy: 分块策略 ("recursive", "sentence", "simple", "parent_child")
            
        Returns:
            List[Dict[str, Any]]: 分块后的文档列表
        
        @Java 程序员提示:
        - 分块是为了适应 LLM 的上下文限制
        - 类似文本分割器 (Text Splitter)
        - 策略模式：支持多种分块方式
        - parent_child 策略会创建父子文档关系
        """
        chunks = []  # 存储所有分块
        
        # 遍历每个文档
        for doc in documents:
            content = doc["content"]  # 获取内容
            metadata = doc["metadata"]  # 获取元数据
            
            # 根据策略选择分块方法
            if strategy == "recursive":
                # 递归字符分块
                text_chunks = self._recursive_split(
                    content,
                    self.chunk_size,
                    self.chunk_overlap
                )
            
            elif strategy == "sentence":
                # 句子分块
                text_chunks = self._sentence_split(content)
            
            elif strategy == "parent_child":
                # 父子文档分块：创建父文档和子文档
                parent_chunks, child_chunks = self._create_parent_child_chunks(
                    content,
                    metadata
                )
                
                # 添加父文档
                for i, chunk in enumerate(parent_chunks):
                    chunk_doc = {
                        "content": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_index": i,
                            "total_chunks": len(parent_chunks),
                            "doc_type": "parent"
                        }
                    }
                    chunks.append(chunk_doc)
                
                # 添加子文档
                for i, chunk in enumerate(child_chunks):
                    chunk_doc = {
                        "content": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_index": i,
                            "total_chunks": len(child_chunks),
                            "doc_type": "child",
                            "parent_id": f"{metadata.get('id', '')}_parent"
                        }
                    }
                    chunks.append(chunk_doc)
                
                continue  # 跳过后续的普通处理
            
            else:
                # 简单分块：固定大小切分
                text_chunks = [
                    content[i:i+self.chunk_size]
                    for i in range(0, len(content), self.chunk_size - self.chunk_overlap)
                ]
            
            # 为每个分块添加元数据
            for i, chunk in enumerate(text_chunks):
                chunk_doc = {
                    "content": chunk,  # 分块内容
                    "metadata": {
                        # 复制原文档的元数据
                        **metadata,
                        # 添加分块信息
                        "chunk_index": i,  # 当前是第几块
                        "total_chunks": len(text_chunks)  # 总共多少块
                    }
                }
                chunks.append(chunk_doc)
        
        print(f"分块完成：{len(chunks)} 个片段")
        return chunks
    
    def _create_parent_child_chunks(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        创建父子文档块
        
        Args:
            content: 文档内容
            metadata: 元数据
            
        Returns:
            Tuple[List[str], List[str]]: (父文档块列表，子文档块列表)
        
        @Java 程序员提示:
        - 这是父子检索的核心方法
        - 父文档用于返回完整上下文
        - 子文档用于精确检索
        - 类似数据库的主外键关系
        """
        # 父文档分割器
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_child_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "", " "]
        )
        
        # 子文档分割器
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.parent_child_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "", " "]
        )
        
        # 创建文档对象
        doc = Document(page_content=content, metadata=metadata)
        
        # 分割
        parent_docs = parent_splitter.split_documents([doc])
        child_docs = child_splitter.split_documents([doc])
        
        # 提取内容
        parent_chunks = [d.page_content for d in parent_docs]
        child_chunks = [d.page_content for d in child_docs]
        
        return parent_chunks, child_chunks
    
    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        递归字符分块
        
        Args:
            text: 待分块的文本
            chunk_size: 每块大小
            overlap: 重叠大小
            
        Returns:
            List[str]: 分块后的文本列表
        
        @Java 程序员提示:
        - 这是私有方法 (_开头)
        - 类似 Java 的 private 方法
        - 递归分块：尽量在句子边界切分
        """
        chunks = []  # 存储分块
        start = 0  # 起始位置
        
        # 循环切分，直到处理完所有文本
        while start < len(text):
            # 计算结束位置
            end = start + chunk_size
            
            # 截取文本块
            chunk = text[start:end]
            chunks.append(chunk)
            
            # 移动起始位置 (减去重叠部分)
            # 类似 Java: start = end - overlap
            start = end - overlap
        
        return chunks
    
    def _sentence_split(self, text: str) -> List[str]:
        """
        句子分块 - 按句子边界切分
        
        Args:
            text: 待分块的文本
            
        Returns:
            List[str]: 按句子切分的文本列表
        
        @Java 程序员提示:
        - 使用正则表达式分割句子
        - 类似 Java 的 String.split()
        - re.split() 类似 Pattern.split()
        """
        import re  # 导入正则表达式模块
        
        # 按句子结束符分割：. ! ?
        # r'[.!?.]' 是正则表达式，匹配任意一个标点
        # 类似 Java: text.split("[.!?.]")
        sentences = re.split(r'[.!?.]', text)
        
        # 去除空白并过滤空字符串
        # 列表推导式，类似 Java Stream 的 filter + map
        return [s.strip() for s in sentences if s.strip()]
    
    def embed_and_store(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32,
        collection_name: str = "documents"
    ):
        """
        向量化并存储
        
        Args:
            chunks: 分块列表
            batch_size: 批次大小 (一次处理多少块)
            collection_name: ChromaDB 集合名称
        
        @Java 程序员提示:
        - 批量处理提高性能
        - 类似 Java 的 BatchProcessor
        - 分批避免内存溢出
        - ChromaDB 是向量数据库
        """
        print(f"开始向量化 {len(chunks)} 个片段...")
        
        # 初始化 ChromaDB
        # 尝试使用新包，如果失败则使用旧包
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # 使用正确的模型名称：BAAI/bge-large-zh-v1.5
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=self.vector_store_path
        )
        
        # 分批处理
        # range(0, len(chunks), batch_size) 生成 0, 32, 64, ...
        # 类似 Java: for (int i = 0; i < chunks.size(); i += batch_size)
        for i in range(0, len(chunks), batch_size):
            # 获取当前批次
            # 切片操作，类似 Java 的 subList
            batch = chunks[i:i+batch_size]
            
            # 提取内容列表
            # 列表推导式，类似 Java Stream 的 map
            contents = [chunk["content"] for chunk in batch]
            
            # 提取元数据列表
            metadatas = [chunk["metadata"] for chunk in batch]
            
            # 调用嵌入函数，获取向量
            # get_embeddings 返回二维数组
            # embeddings_batch = get_embeddings(contents)
            
            # 添加到向量库
            # ChromaDB 会自动处理向量化
            vectorstore.add_texts(
                texts=contents,
                metadatas=metadatas
            )
            
            # 打印进度
            total_batches = (len(chunks) - 1) // batch_size + 1
            print(f"处理批次 {i//batch_size + 1}/{total_batches}")
        
        print("向量化完成！")
    
    def ingest_all(
        self,
        file_pattern: str = "*.txt",
        split_strategy: str = "recursive",
        is_json: bool = False,
        collection_name: str = "documents"
    ):
        """
        完整的入库流程
        
        Args:
            file_pattern: 文件匹配模式
            split_strategy: 分块策略
            is_json: 是否是 JSON 文件
            collection_name: ChromaDB 集合名称
        
        @Java 程序员提示:
        - 这是门面方法 (Facade)
        - 封装了整个 ETL 流程
        - 类似 Java 的批处理作业
        """
        # 步骤 1: 加载文档
        if is_json:
            # 加载 JSON 文件
            json_files = list(self.data_dir.rglob(file_pattern))
            all_documents = []
            for json_file in json_files:
                docs = self.load_json_documents(str(json_file))
                all_documents.extend(docs)
        else:
            # 加载普通文本文件
            all_documents = self.load_documents(file_pattern)
        
        # 步骤 2: 分块
        chunks = self.split_documents(all_documents, split_strategy)
        
        # 步骤 3: 向量化并存储
        self.embed_and_store(chunks, collection_name=collection_name)
        
        print("✓ 数据入库完成")
    
    def ingest_json_with_parent_child(
        self,
        json_file_path: str,
        collection_name: str = "three_kingdoms_parent_child"
    ):
        """
        专门用于 JSON 数据的父子文档入库
        
        Args:
            json_file_path: JSON 文件路径
            collection_name: ChromaDB 集合名称
        
        @Java 程序员提示:
        - 这是便捷方法，专门处理父子文档
        - 自动使用 parent_child 分块策略
        - 一步完成整个流程
        """
        print(f"📥 正在处理 JSON 文件：{json_file_path}")
        
        # 步骤 1: 加载 JSON 数据
        documents = self.load_json_documents(json_file_path)
        
        # 步骤 2: 创建父子文档块
        chunks = self.split_documents(documents, strategy="parent_child")
        
        # 统计父子文档数量
        parent_count = sum(1 for c in chunks if c["metadata"].get("doc_type") == "parent")
        child_count = sum(1 for c in chunks if c["metadata"].get("doc_type") == "child")
        
        print(f"📊 创建了 {parent_count} 个父文档，{child_count} 个子文档")
        
        # 步骤 3: 向量化并存储
        self.embed_and_store(chunks, collection_name=collection_name)
        
        print(f"✅ JSON 父子文档入库完成！集合名称：{collection_name}")


# ==================== 命令行入口 ====================
def main():
    """
    命令行入口函数
    
    @Java 程序员提示:
    - 类似 Java 的 public static void main(String[] args)
    - 可以直接运行这个脚本执行入库操作
    - 也可以作为模块被其他代码导入使用
    """
    # 创建入库工具实例
    ingestor = DocumentIngestor(
        data_dir="./data",  # 数据目录
        vector_store_path=settings.VECTOR_STORE_PATH  # 向量存储路径
    )
    
    # 示例 1: 处理普通 txt 文件
    # ingestor.ingest_all(
    #     file_pattern="*.txt",  # 匹配 txt 文件
    #     split_strategy="recursive"  # 使用递归分块
    # )
    
    # 示例 2: 处理 JSON 数据并使用父子文档策略
    ingestor.ingest_json_with_parent_child(
        json_file_path="./data/romance_three_kingdoms.json",
        collection_name="three_kingdoms_parent_child"
    )


# ==================== 脚本入口判断 ====================
# if __name__ == "__main__": 类似 Java 的 main 方法判断
# 只有直接运行脚本时才会执行，import 时不会执行
if __name__ == "__main__":
    main()


# ==================== 使用示例 (注释) ====================
# @Java 程序员提示:
# 
# 使用方式 1: 命令行运行（默认处理 JSON 父子文档）
# python ingest.py
#
# 使用方式 2: 作为模块导入 - 处理普通文件
# from app.rag.ingest import DocumentIngestor
# ingestor = DocumentIngestor()
# ingestor.ingest_all("*.md", split_strategy="sentence")
#
# 使用方式 3: 处理 JSON 数据
# from app.rag.ingest import DocumentIngestor
# ingestor = DocumentIngestor()
# ingestor.ingest_json_with_parent_child(
#     json_file_path="./data/romance_three_kingdoms.json"
# )
#
# 使用方式 4: 自定义参数
# ingestor = DocumentIngestor(
#     data_dir="./custom_data",
#     vector_store_path="./custom_vector_store"
# )
# ingestor.ingest_all(
#     file_pattern="*.json",
#     split_strategy="parent_child",
#     is_json=True,
#     collection_name="custom_collection"
# )
#
# 完整流程:
# 1. 从 ./data 目录读取 romance_three_kingdoms.json
# 2. 解析 JSON 数组，格式化每个对象
# 3. 创建父子文档块：
#    - 父文档：1000 tokens，保留完整上下文
#    - 子文档：200 tokens，用于精确检索
#    - 重叠：20 tokens
# 4. 批量向量化 (每批 32 块)
# 5. 存储到 ChromaDB 向量数据库
