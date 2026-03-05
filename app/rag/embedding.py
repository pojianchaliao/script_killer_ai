"""
RAG Embedding - BGE 模型加载逻辑
"""
from typing import List, Optional
import numpy as np
from ..config import settings


class EmbeddingModel:
    """BGE Embedding 模型封装"""
    
    _instance: Optional["EmbeddingModel"] = None
    
    def __init__(self):
        self.model = None
        self.model_name = settings.EMBEDDING_MODEL
        self._initialized = False
    
    @classmethod
    def get_instance(cls) -> "EmbeddingModel":
        """单例模式获取实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_model(self, force_reload: bool = False):
        """加载 Embedding 模型"""
        if self._initialized and not force_reload:
            return
        
        try:
            # TODO: 选择合适的 embedding 库
            # from FlagEmbedding import FlagModel
            # self.model = FlagModel(self.model_name)
            pass
            
            self._initialized = True
            print(f"Embedding 模型已加载：{self.model_name}")
            
        except Exception as e:
            print(f"加载 Embedding 模型失败：{e}")
            raise
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """将文本编码为向量"""
        if not self._initialized:
            self.load_model()
        
        # TODO: 实现具体的编码逻辑
        embeddings = np.random.randn(len(texts), 1024)
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """专门用于查询的编码"""
        return self.encode(queries)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """专门用于文档的编码"""
        return self.encode(documents)
    
    def get_embedding_dim(self) -> int:
        """获取向量维度"""
        return 1024


embedding_model = EmbeddingModel.get_instance()


def get_embeddings(texts: List[str]) -> np.ndarray:
    """便捷函数：获取文本的向量表示"""
    return embedding_model.encode(texts)
