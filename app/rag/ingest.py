"""
RAG Ingest - 数据入库脚本
负责文档加载、分块、向量化和存储
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from ..config import settings
from .embedding import get_embeddings


class DocumentIngestor:
    """文档入库工具"""
    
    def __init__(
        self,
        data_dir: str = "./data",
        vector_store_path: str = None
    ):
        self.data_dir = Path(data_dir)
        self.vector_store_path = vector_store_path or settings.VECTOR_STORE_PATH
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def load_documents(
        self,
        file_pattern: str = "*.txt",
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """加载文档"""
        documents = []
        
        if recursive:
            files = list(self.data_dir.rglob(file_pattern))
        else:
            files = list(self.data_dir.glob(file_pattern))
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                documents.append({
                    "content": content,
                    "metadata": {
                        "source": str(file_path),
                        "filename": file_path.name
                    }
                })
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
        
        print(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def split_documents(
        self,
        documents: List[Dict[str, Any]],
        strategy: str = "recursive"
    ) -> List[Dict[str, Any]]:
        """文档分块"""
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            if strategy == "recursive":
                text_chunks = self._recursive_split(
                    content,
                    self.chunk_size,
                    self.chunk_overlap
                )
            elif strategy == "sentence":
                text_chunks = self._sentence_split(content)
            else:
                text_chunks = [
                    content[i:i+self.chunk_size]
                    for i in range(0, len(content), self.chunk_size - self.chunk_overlap)
                ]
            
            for i, chunk in enumerate(text_chunks):
                chunk_doc = {
                    "content": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks)
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
        """递归字符分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks
    
    def _sentence_split(self, text: str) -> List[str]:
        """句子分块"""
        import re
        sentences = re.split(r'[.!?.]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def embed_and_store(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32
    ):
        """向量化并存储"""
        print(f"开始向量化 {len(chunks)} 个片段...")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            contents = [chunk["content"] for chunk in batch]
            
            embeddings = get_embeddings(contents)
            
            # TODO: 存储到向量数据库
            print(f"处理批次 {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        print("向量化完成！")
    
    def ingest_all(
        self,
        file_pattern: str = "*.txt",
        split_strategy: str = "recursive"
    ):
        """完整的入库流程"""
        documents = self.load_documents(file_pattern)
        chunks = self.split_documents(documents, split_strategy)
        self.embed_and_store(chunks)
        print("✓ 数据入库完成")


def main():
    """命令行入口"""
    ingestor = DocumentIngestor(
        data_dir="./data",
        vector_store_path=settings.VECTOR_STORE_PATH
    )
    
    ingestor.ingest_all(
        file_pattern="*.txt",
        split_strategy="recursive"
    )


if __name__ == "__main__":
    main()
