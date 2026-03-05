"""
应用配置管理 - 基于 Pydantic BaseSettings
从 .env 文件读取环境变量
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    APP_NAME: str = "Script Killer AI"
    APP_ENV: str = "development"
    DEBUG: bool = True
    
    # 智谱 AI 配置
    ZHIPU_API_KEY: str
    ZHIPU_MODEL: str = "glm-4"
    ZHIPU_BASE_URL: Optional[str] = None
    
    # RAG 配置
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    VECTOR_STORE_PATH: str = "./data/vector_store"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # 数据库配置
    DATABASE_URL: Optional[str] = None
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置单例"""
    return settings
