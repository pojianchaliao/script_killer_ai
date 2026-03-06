"""
RAG Embedding - BGE 模型加载逻辑
将文本转换为向量表示

@Java 程序员提示:
- Embedding 是将文本转换为向量的过程
- BGE 是智源研究院的嵌入模型，类似 OpenAI 的 text-embedding
- 单例模式确保模型只加载一次，节省内存
- numpy 是 Python 的科学计算库，类似 Java 的 ND4J
"""
from typing import List, Optional  # 类型注解
import numpy as np  # numpy 库，用于数值计算
from ..config import settings  # 导入配置


# ==================== Embedding 模型类 ====================
class EmbeddingModel:
    """
    BGE Embedding 模型封装
    
    @Java 程序员提示:
    - 这是单例模式实现
    - _instance 是类变量 (类似 Java 的 static 字段)
    - get_instance() 是静态方法 (类似 Java 的 static 方法)
    - 确保全局只有一个模型实例，节省内存
    """
    
    # 类变量 (类似 Java 的 static 字段)
    # _instance 存储唯一的实例
    _instance: Optional["EmbeddingModel"] = None
    
    def __init__(self):
        """
        构造方法
        
        @Java 程序员提示:
        - __init__ 是 Python 的构造器
        - self 类似 Java 的 this
        - 字段不需要预先声明类型
        """
        self.model = None  # 模型对象，初始为 None
        self.model_name = settings.EMBEDDING_MODEL  # 从配置读取模型名称
        self._initialized = False  # 初始化标志
    
    @classmethod
    def get_instance(cls) -> "EmbeddingModel":
        """
        获取单例实例的静态方法
        
        Returns:
            EmbeddingModel: 单例对象
        
        @Java 程序员提示:
        - @classmethod 是类方法装饰器
        - cls 参数类似 Java 的 Class 对象
        - 类似 Java: public static EmbeddingModel getInstance()
        - 第一次调用时创建实例，之后直接返回
        """
        # 如果实例不存在，创建新实例
        if cls._instance is None:
            cls._instance = cls()  # 调用构造方法
        
        return cls._instance
    
    def load_model(self, force_reload: bool = False):
        """
        加载 Embedding 模型
        
        Args:
            force_reload: 是否强制重新加载
        
        @Java 程序员提示:
        - 类似 Java 的 Bean 初始化方法
        - force_reload 类似刷新标志
        - 使用懒加载 (lazy loading)：首次使用时才加载
        """
        # 如果已初始化且不强制重载，直接返回
        # 类似 Java: if (initialized && !forceReload) return;
        if self._initialized and not force_reload:
            return
        
        try:
            # TODO: 选择合适的 embedding 库
            # 实际实现会使用 FlagEmbedding 或 sentence-transformers
            
            # 示例代码 (注释):
            # from FlagEmbedding import FlagModel
            # self.model = FlagModel(self.model_name)
            
            # 这里只是占位符
            pass
            
            # 设置初始化标志
            self._initialized = True
            print(f"Embedding 模型已加载：{self.model_name}")
            
        except Exception as e:
            # 异常处理
            print(f"加载 Embedding 模型失败：{e}")
            raise  # 重新抛出异常
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            texts: 文本列表，类似 Java 的 List<String>
            batch_size: 批次大小，一次处理多少文本
            normalize: 是否归一化向量
            
        Returns:
            np.ndarray: numpy 数组，形状为 (文本数量，向量维度)
        
        @Java 程序员提示:
        - np.ndarray 类似 Java 的多维数组 double[][]
        - 返回的向量用于语义相似度计算
        - normalize 使向量长度为 1，便于余弦相似度计算
        """
        # 如果未初始化，先加载模型
        if not self._initialized:
            self.load_model()
        
        # TODO: 实现具体的编码逻辑
        # 这里生成随机向量作为示例
        # 实际会调用模型进行编码
        
        # np.random.randn 生成标准正态分布的随机数
        # (len(texts), 1024) 是形状：文本数量 × 1024 维
        embeddings = np.random.randn(len(texts), 1024)
        
        # 归一化处理
        if normalize:
            # np.linalg.norm 计算范数 (长度)
            # axis=1 表示按行计算
            # keepdims=True 保持维度
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # 归一化：向量 / 长度
            # 类似 Java: for each vector: vector = vector / norm
            embeddings = embeddings / (norms + 1e-8)  # 加小值避免除零
        
        return embeddings
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        专门用于查询的编码
        
        Args:
            queries: 查询文本列表
            
        Returns:
            np.ndarray: 查询向量
        
        @Java 程序员提示:
        - 有些 Embedding 模型对查询和文档使用不同编码
        - 类似 Java 的重载方法
        """
        # 直接调用基础 encode 方法
        return self.encode(queries)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        专门用于文档的编码
        
        Args:
            documents: 文档文本列表
            
        Returns:
            np.ndarray: 文档向量
        
        @Java 程序员提示:
        - 文档编码可能与查询编码不同
        - 这是为了优化检索效果
        """
        return self.encode(documents)
    
    def get_embedding_dim(self) -> int:
        """
        获取向量维度
        
        Returns:
            int: 向量维度 (例如 1024)
        
        @Java 程序员提示:
        - 类似 Java 的 getter 方法
        - 返回向量的维度
        """
        return 1024  # BGE 模型的默认维度


# ==================== 全局单例 ====================
# 创建全局可访问的单例对象
# 类似 Java: public static final EmbeddingModel embeddingModel = EmbeddingModel.getInstance();
embedding_model = EmbeddingModel.get_instance()


# ==================== 便捷函数 ====================
def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    便捷函数：获取文本的向量表示
    
    Args:
        texts: 文本列表
        
    Returns:
        np.ndarray: 向量数组
    
    @Java 程序员提示:
    - 这是模块级别的工具函数
    - 类似 Java 的静态工具方法：EmbeddingUtils.getEmbeddings(texts)
    - 直接调用全局单例，简化使用
    """
    return embedding_model.encode(texts)


# ==================== 使用示例 (注释) ====================
# @Java 程序员提示:
# 
# 使用方式 1: 使用单例
# model = EmbeddingModel.get_instance()
# vectors = model.encode(["你好", "世界"])
#
# 使用方式 2: 使用便捷函数
# vectors = get_embeddings(["你好", "世界"])
#
# 向量相似度计算:
# from sklearn.metrics.pairwise import cosine_similarity
# similarity = cosine_similarity(vector1, vector2)
