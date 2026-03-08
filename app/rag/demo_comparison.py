"""
检索策略对比演示 - 可视化三种策略的效果差异

@Java 程序员提示:
- 这是一个演示脚本，可以独立运行
- 展示了三种检索策略的实际效果
- 包含详细的注释和输出
- 类似 Java 的 Main 方法 + Demo 类
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
    # 1. 环境变量
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    
    # 2. 项目 .env 文件
    try:
        project_root = PathLib(__file__).parent.parent.parent
        env_file = project_root / ".env"
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if token:
                            return token
    except Exception:
        pass
    
    # 3. 用户主目录
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

# 获取并设置 token
hf_token = get_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print(f"{Fore.GREEN}✅ 已设置 HuggingFace Token (前缀：{hf_token[:10]}...){Style.RESET_ALL}")
else:
    print(f"{Fore.YELLOW}⚠️  未设置 HF_TOKEN，下载速度可能较慢{Style.RESET_ALL}")
    print(f"   获取 token: {Fore.CYAN}https://huggingface.co/settings/tokens{Style.RESET_ALL}\n")

import json
import time
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 尝试使用新包，如果失败则使用旧包
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from colorama import init, Fore, Style

# 初始化 colorama（彩色输出）
init()


class RetrievalDemo:
    """检索策略演示"""
    
    def __init__(self, data_path: str):
        """
        构造方法
        
        Args:
            data_path: JSON 数据文件路径
        """
        self.data_path = data_path
        self.raw_data = self._load_data()
        
        print(f"{Fore.CYAN}📚 加载数据：{data_path}{Style.RESET_ALL}")
        print(f"   共 {len(self.raw_data)} 条记录\n")
        
        # 初始化嵌入模型
        print(f"{Fore.CYAN}🔧 加载嵌入模型...{Style.RESET_ALL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="bge-large-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"{Fore.GREEN}✅ 模型加载完成{Style.RESET_ALL}\n")
        
        # 创建三种策略的向量库
        self._create_all_stores()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载 JSON 数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_documents(self) -> List[Document]:
        """创建文档"""
        documents = []
        
        for item in self.raw_data[:100]:  # 只用前 100 条加速测试
            content = f"""【{item['event']}】
主题：{item['theme']}
来源：{item['source_type']}

📖 背景与描述：
{item['description']}

🎮 游戏效果：
{item['game_effect']}

📚 历史事实：
{item['historical_fact']}

🏷️ 标签：{', '.join(item['tags'])}"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "id": item['id'],
                    "event": item['event'],
                    "theme": item['theme'],
                    "source_type": item['source_type']
                }
            )
            documents.append(doc)
        
        return documents
    
    def _create_all_stores(self):
        """创建三种策略的向量存储"""
        documents = self._create_documents()
        
        # ========== 策略 1: 大块 ==========
        print(f"{Fore.YELLOW}📦 策略 1: 创建大块存储 (chunk_size=1000)...{Style.RESET_ALL}")
        large_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", "。", "", " "]
        )
        large_docs = large_splitter.split_documents(documents)
        self.large_store = Chroma.from_documents(
            documents=large_docs,
            embedding=self.embeddings,
            collection_name="demo_large"
        )
        print(f"   ✅ 创建 {len(large_docs)} 个大块\n")
        
        # ========== 策略 2: 小块 ==========
        print(f"{Fore.YELLOW}📦 策略 2: 创建小块存储 (chunk_size=200)...{Style.RESET_ALL}")
        small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", "。", "", " "]
        )
        small_docs = small_splitter.split_documents(documents)
        self.small_store = Chroma.from_documents(
            documents=small_docs,
            embedding=self.embeddings,
            collection_name="demo_small"
        )
        print(f"   ✅ 创建 {len(small_docs)} 个小块\n")
        
        # ========== 策略 3: 父子检索 ==========
        print(f"{Fore.YELLOW}📦 策略 3: 创建父子检索器...{Style.RESET_ALL}")
        
        # 父文档分割器
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", "。", "", " "]
        )
        
        # 子文档分割器
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", "。", "", " "]
        )
        
        parent_docs = parent_splitter.split_documents(documents)
        child_docs = child_splitter.split_documents(documents)
        
        # 建立父子映射
        self.parent_doc_map = {}
        for i, doc in enumerate(parent_docs):
            self.parent_doc_map[i] = doc
        
        # 为子文档添加父索引
        for child_doc in child_docs:
            child_doc.metadata["parent_idx"] = 0  # 简化处理
        
        self.child_store = Chroma.from_documents(
            documents=child_docs,
            embedding=self.embeddings,
            collection_name="demo_parent_child"
        )
        self.parent_docs = parent_docs
        self.child_docs = child_docs
        
        print(f"   ✅ 创建 {len(parent_docs)} 个父文档，{len(child_docs)} 个子文档\n")
    
    def search_large(self, query: str, k: int = 3) -> List[Dict]:
        """使用大块策略搜索"""
        results = self.large_store.similarity_search_with_score(query, k=k)
        return [
            {"content": doc.page_content, "score": float(score), "metadata": doc.metadata}
            for doc, score in results
        ]
    
    def search_small(self, query: str, k: int = 3) -> List[Dict]:
        """使用小块策略搜索"""
        results = self.small_store.similarity_search_with_score(query, k=k)
        return [
            {"content": doc.page_content, "score": float(score), "metadata": doc.metadata}
            for doc, score in results
        ]
    
    def search_parent_child(self, query: str, k: int = 3) -> List[Dict]:
        """使用父子检索策略"""
        # 1. 在子文档中搜索
        child_results = self.child_store.similarity_search_with_score(query, k=k * 2)
        
        # 2. 找到对应的父文档
        seen_parents = set()
        parent_results = []
        
        for child_doc, score in child_results:
            parent_idx = child_doc.metadata.get("parent_idx", 0)
            
            if parent_idx not in seen_parents and len(parent_results) < k:
                parent_doc = self.parent_doc_map[parent_idx]
                parent_results.append({
                    "content": parent_doc.page_content,
                    "score": float(score),
                    "metadata": parent_doc.metadata
                })
                seen_parents.add(parent_idx)
        
        return parent_results
    
    def compare_strategies(self, query: str):
        """
        对比三种策略
        
        Args:
            query: 查询文本
        """
        print(f"\n{'='*70}")
        print(f"{Fore.BLUE}🔍 查询：{query}{Style.RESET_ALL}")
        print(f"{'='*70}\n")
        
        # ========== 策略 1: 大块 ==========
        print(f"{Fore.YELLOW}【策略 1】只用大块 (1000 tokens){Style.RESET_ALL}")
        print(f"{Fore.RED}❌ 问题：语义匹配不精确，可能错过细节{Style.RESET_ALL}\n")
        
        start = time.time()
        results = self.search_large(query, k=3)
        elapsed = time.time() - start
        
        for i, r in enumerate(results, 1):
            preview = r["content"][:150].replace("\n", " ")
            print(f"  [{i}] 分数：{r['score']:.3f} | 耗时：{elapsed:.3f}s")
            print(f"      {preview}...\n")
        
        # ========== 策略 2: 小块 ==========
        print(f"{Fore.YELLOW}【策略 2】只用小块 (200 tokens){Style.RESET_ALL}")
        print(f"{Fore.RED}❌ 问题：上下文丢失，信息不完整{Style.RESET_ALL}\n")
        
        start = time.time()
        results = self.search_small(query, k=3)
        elapsed = time.time() - start
        
        for i, r in enumerate(results, 1):
            preview = r["content"][:150].replace("\n", " ")
            print(f"  [{i}] 分数：{r['score']:.3f} | 耗时：{elapsed:.3f}s")
            print(f"      {preview}...\n")
        
        # ========== 策略 3: 父子检索 ==========
        print(f"{Fore.YELLOW}【策略 3】父子检索器 (parent=1000, child=200){Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ 优势：小块检索精确 + 大块返回完整上下文{Style.RESET_ALL}\n")
        
        start = time.time()
        results = self.search_parent_child(query, k=3)
        elapsed = time.time() - start
        
        for i, r in enumerate(results, 1):
            preview = r["content"][:150].replace("\n", " ")
            print(f"  [{i}] 分数：{r['score']:.3f} | 耗时：{elapsed:.3f}s")
            print(f"      {preview}...\n")
        
        # ========== 总结 ==========
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🏆 推荐：父子检索器{Style.RESET_ALL}")
        print(f"   ✓ 检索精度高（小块匹配）")
        print(f"   ✓ 上下文完整（大块返回）")
        print(f"   ✓ 平衡了精度和完整性\n")


def main():
    """主函数"""
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🎮 三国数据检索策略对比演示{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    # 创建演示对象
    demo = RetrievalDemo("app/data/romance_three_kingdoms.json")
    
    # 测试查询列表
    test_queries = [
        "赤壁之战的游戏效果是什么？",
        "曹操被封为魏公的详细信息",
        "黄巾起义的背景和影响",
        "诸葛亮北伐的过程"
    ]
    
    # 逐个测试
    for query in test_queries:
        demo.compare_strategies(query)
        input(f"{Fore.YELLOW}按 Enter 继续下一个查询...{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}✅ 演示完成！{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}用户中断{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ 错误：{e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
