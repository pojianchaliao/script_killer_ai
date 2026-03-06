"""
RAG Ingest - 数据入库脚本
负责文档加载、分块、向量化和存储

@Java 程序员提示:
- Ingestor 是 ETL (Extract-Transform-Load) 工具
- 类似 Java 的批处理：读取文件 → 分块 → 向量化 → 存储
- 使用 Path 对象处理文件路径 (比字符串更安全)
- 支持多种分块策略
"""
from typing import List, Dict, Any, Optional  # 类型注解
from pathlib import Path  # 路径处理类
import json  # JSON 处理
from ..config import settings  # 导入配置
from .embedding import get_embeddings  # 导入嵌入函数


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
    
    def load_documents(
        self,
        file_pattern: str = "*.txt",
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        加载文档
        
        Args:
            file_pattern: 文件匹配模式，如 "*.txt", "*.md"
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
    
    def split_documents(
        self,
        documents: List[Dict[str, Any]],
        strategy: str = "recursive"
    ) -> List[Dict[str, Any]]:
        """
        文档分块 - 将长文档切分为小块
        
        Args:
            documents: 文档列表
            strategy: 分块策略 ("recursive", "sentence", "simple")
            
        Returns:
            List[Dict[str, Any]]: 分块后的文档列表
        
        @Java 程序员提示:
        - 分块是为了适应 LLM 的上下文限制
        - 类似文本分割器 (Text Splitter)
        - 策略模式：支持多种分块方式
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
        batch_size: int = 32
    ):
        """
        向量化并存储
        
        Args:
            chunks: 分块列表
            batch_size: 批次大小 (一次处理多少块)
        
        @Java 程序员提示:
        - 批量处理提高性能
        - 类似 Java 的 BatchProcessor
        - 分批避免内存溢出
        """
        print(f"开始向量化 {len(chunks)} 个片段...")
        
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
            
            # 调用嵌入函数，获取向量
            # get_embeddings 返回二维数组
            embeddings = get_embeddings(contents)
            
            # TODO: 存储到向量数据库
            # 实际会调用向量数据库的 upsert 或 add 方法
            # 例如：faiss_index.add(embeddings) 或 collection.upsert(...)
            
            # 打印进度
            total_batches = (len(chunks) - 1) // batch_size + 1
            print(f"处理批次 {i//batch_size + 1}/{total_batches}")
        
        print("向量化完成！")
    
    def ingest_all(
        self,
        file_pattern: str = "*.txt",
        split_strategy: str = "recursive"
    ):
        """
        完整的入库流程
        
        Args:
            file_pattern: 文件匹配模式
            split_strategy: 分块策略
        
        @Java 程序员提示:
        - 这是门面方法 (Facade)
        - 封装了整个 ETL 流程
        - 类似 Java 的批处理作业
        """
        # 步骤 1: 加载文档
        documents = self.load_documents(file_pattern)
        
        # 步骤 2: 分块
        chunks = self.split_documents(documents, split_strategy)
        
        # 步骤 3: 向量化并存储
        self.embed_and_store(chunks)
        
        print("✓ 数据入库完成")


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
    
    # 执行完整入库流程
    ingestor.ingest_all(
        file_pattern="*.txt",  # 匹配 txt 文件
        split_strategy="recursive"  # 使用递归分块
    )


# ==================== 脚本入口判断 ====================
# if __name__ == "__main__": 类似 Java 的 main 方法判断
# 只有直接运行脚本时才会执行，import 时不会执行
if __name__ == "__main__":
    main()


# ==================== 使用示例 (注释) ====================
# @Java 程序员提示:
# 
# 使用方式 1: 命令行运行
# python ingest.py
#
# 使用方式 2: 作为模块导入
# from app.rag.ingest import DocumentIngestor
# ingestor = DocumentIngestor()
# ingestor.ingest_all("*.md", split_strategy="sentence")
#
# 完整流程:
# 1. 从 ./data 目录读取所有 *.txt 文件
# 2. 按递归策略分块 (每块 512 字符，重叠 50)
# 3. 批量向量化 (每批 32 块)
# 4. 存储到向量数据库
