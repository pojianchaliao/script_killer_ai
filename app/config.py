"""
应用配置管理 - 基于 Pydantic BaseSettings
从 .env 文件读取环境变量

@Java 程序员提示:
- Pydantic 类似 Java 的 Bean Validation + Lombok
- BaseSettings 自动从环境变量读取配置
- 类型注解强制类型检查，类似 Java 的字段类型声明
"""
import os

from dotenv import load_dotenv  # 添加这行
from pathlib import Path  # 添加这行


# ==================== 手动加载 .env 文件 ====================
def load_env_file():
    project_root = Path(__file__).parent.parent
    env_file_path = project_root / ".env"

    if env_file_path.exists():
        load_dotenv(dotenv_path=env_file_path)
        print(f"✓ 已加载 .env 文件：{env_file_path}")
        return True
    else:
        print(f"⚠️  警告：.env 文件不存在：{env_file_path}")
        return False


# 立即执行加载函数
load_env_file()

from pydantic_settings import BaseSettings  # Pydantic 的配置基类
from typing import Optional  # Optional 表示字段可以为 None


class Settings(BaseSettings):
    """
    应用配置类 - 所有配置项的单一点
    
    @Java 程序员提示:
    - class 定义类，类似 Java
    - 字段不需要声明类型 (Python 是动态类型)，但建议加类型注解
    - 字段默认值使用 = 赋值
    - 没有显式的 getter/setter，直接通过字段名访问
    """
    
    # ==================== 应用基础配置 ====================
    # 字段类型：str (字符串)
    # 默认值："Script Killer AI"
    # 类似 Java: private String APP_NAME = "Script Killer AI";
    APP_NAME: str = "Script Killer AI"
    
    # 运行环境：development, production, testing
    APP_ENV: str = "development"
    
    # 调试模式：True/False (布尔类型)
    # Python 的布尔值是 True/False，类似 Java 的 true/false
    DEBUG: bool = True
    
    # ==================== 智谱 AI 配置 ====================
    # 必填字段 (没有默认值)
    # 必须从环境变量或.env 文件读取
    # 类似 Java: private String ZHIPU_API_KEY; (没有初始值)
    ZHIPU_API_KEY: str
    
    # AI 模型名称，默认使用 glm-4
    ZHIPU_MODEL: str = "glm-4"
    
    # API 基础 URL，可以为 None (空值)
    # Optional[str] 类似 Java 的 @Nullable String
    # None 类似 Java 的 null
    ZHIPU_BASE_URL: Optional[str] = None
    
    # ==================== RAG 配置 ====================
    # 嵌入模型：用于文本向量化
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    
    # 向量存储路径
    VECTOR_STORE_PATH: str = "./data/vector_store"
    
    # 文本分块大小 (整数类型)
    CHUNK_SIZE: int = 512
    
    # 文本分块重叠大小
    CHUNK_OVERLAP: int = 50
    
    # ==================== 数据库配置 ====================
    # 数据库连接 URL，可以为 None
    DATABASE_URL: Optional[str] = None
    
    # ==================== 日志配置 ====================
    # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL: str = "INFO"
    
    # ==================== 内部配置类 ====================
    # 嵌套类，用于配置 Pydantic 的行为
    class Config:
        """
        Pydantic 配置类
        
        @Java 程序员提示:
        - 这是嵌套类，类似 Java 的内部类
        - 用于配置框架行为，不是业务配置
        """
        env_file = ".env"  # 从.env 文件读取环境变量
        env_file_encoding = "utf-8"  # 文件编码
        case_sensitive = True  # 区分大小写

class Config:
    env_file = None  # 改为 None
    env_file_encoding = "utf-8"
    case_sensitive = True

# ==================== 全局单例 ====================
# 在模块级别创建单例对象
# 类似 Java: public static final Settings settings = new Settings();
def create_settings():
    try:
        settings_instance = Settings()
        print(f"✓ 配置加载成功，环境：{settings_instance.APP_ENV}")
        return settings_instance
    except Exception as e:
        print(f"❌ 配置加载失败：{e}")
        print(f"ZHIPU_API_KEY 环境变量：{os.getenv('ZHIPU_API_KEY', '未设置')}")
        raise

settings = create_settings()


def get_settings() -> Settings:
    """
    获取配置单例的函数
    
    Returns:
        Settings: 配置对象实例
    
    @Java 程序员提示:
    - def 定义函数，类似 Java 的方法
    - -> Settings 是返回值类型注解
    - 函数名使用小写 + 下划线 (Python 命名规范)
    - 类似 Java 的 getter 方法，但更简洁
    """
    return settings
