"""
快速入库脚本 - 将 romance_three_kingdoms.json 加载到 ChromaDB 向量库
专为演示向量叠加测试准备数据
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import sys
import time
from pathlib import Path
from colorama import Fore, Style, init
from tqdm import tqdm

# 初始化 colorama
init(autoreset=True)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
print(f"{Fore.CYAN}三国数据入库脚本{Style.RESET_ALL}")
print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

# 文件路径
json_file_path = Path(__file__).parent.parent / "data" / "romance_three_kingdoms.json"
vector_store_path = Path(__file__).parent.parent.parent / "chroma_db" / "three_kingdoms"

print(f"{Fore.YELLOW}数据文件：{json_file_path}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}向量库路径：{vector_store_path}{Style.RESET_ALL}\n")

# 检查文件是否存在
if not json_file_path.exists():
    print(f"{Fore.RED}错误：JSON 文件不存在：{json_file_path}{Style.RESET_ALL}")
    sys.exit(1)

# 步骤 1: 加载 JSON 数据
print(f"{Fore.YELLOW}步骤 1: 加载 JSON 数据...{Style.RESET_ALL}")
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"  [OK] 加载了 {len(data)} 条记录\n")

# 步骤 2: 初始化嵌入模型
print(f"{Fore.YELLOW}步骤 2: 初始化 Embedding 模型...{Style.RESET_ALL}")
try:
    # 尝试使用新的 langchain-huggingface，如果失败则使用 community
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"  [OK] 模型加载完成 (BGE-Large-ZH-v1.5)\n")
except Exception as e:
    print(f"{Fore.RED}错误：模型加载失败：{e}{Style.RESET_ALL}")
    sys.exit(1)

# 步骤 3: 格式化文档（分批处理）
print(f"{Fore.YELLOW}步骤 3: 格式化文档内容...{Style.RESET_ALL}")
from langchain_core.documents import Document

batch_size = 100  # 每批处理 100 条
documents = []
total_records = len(data)

for batch_start in tqdm(range(0, total_records, batch_size), desc="格式化文档", unit="批"):
    batch_end = min(batch_start + batch_size, total_records)
    batch_data = data[batch_start:batch_end]
    
    for item in batch_data:
        # 格式化内容
        tags_str = ', '.join(item.get('tags', []))
        content = f"""【{item.get('event', 'Unknown')}】
主题：{item.get('theme', 'Unknown')}
来源：{item.get('source_type', 'Unknown')}
戏剧价值：{item.get('dramatic_value', 'unknown')}

背景与描述：
{item.get('description', '')}

游戏效果：
{item.get('game_effect', '')}

历史事实：
{item.get('historical_fact', '')}

标签：{tags_str}"""
        
        # 创建文档对象
        doc = Document(
            page_content=content,
            metadata={
                "id": item.get('id', ''),
                "event": item.get('event', ''),
                "theme": item.get('theme', ''),
                "source_type": item.get('source_type', ''),
                "dramatic_value": item.get('dramatic_value', 'unknown'),
                "tags": item.get('tags', []),
                "doc_type": "json_document"
            }
        )
        documents.append(doc)
    
    # 每批处理后暂停一小段时间，让进度条可见
    time.sleep(0.05)

print(f"{Fore.GREEN}  [OK] 格式化了 {len(documents)} 个文档{Style.RESET_ALL}\n")

# 步骤 4: 创建 ChromaDB 向量库（分批向量化）
print(f"{Fore.YELLOW}步骤 4: 创建 ChromaDB 向量库...{Style.RESET_ALL}")
try:
    from langchain_chroma import Chroma
    
    # 确保目录存在
    vector_store_path.mkdir(parents=True, exist_ok=True)
    
    # 先创建空的向量库
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="three_kingdoms_parent_child",
        persist_directory=str(vector_store_path)
    )
    
    print(f"  [OK] 成功创建向量库：{vector_store_path}")
    print(f"  [OK] 集合名称：three_kingdoms_parent_child")
    
    # 分批添加到向量库
    print(f"\n{Fore.YELLOW}正在向量化并添加到向量库...{Style.RESET_ALL}")
    
    for batch_start in tqdm(range(0, len(documents), batch_size), desc="向量化", unit="批"):
        batch_end = min(batch_start + batch_size, len(documents))
        batch_docs = documents[batch_start:batch_end]
        batch_ids = [f"doc_{i}" for i in range(batch_start, batch_end)]
        
        # 批量添加到向量库
        vectorstore.add_documents(
            documents=batch_docs,
            ids=batch_ids
        )
        
        # 更新进度
        progress = (batch_end / len(documents)) * 100
        print(f"  进度：{progress:.1f}% ({batch_end}/{len(documents)})")
        time.sleep(0.02)  # 短暂暂停，让进度可见
    
    print(f"\n{Fore.GREEN}  [OK] 文档总数：{len(documents)}{Style.RESET_ALL}\n")
    
except Exception as e:
    print(f"{Fore.RED}错误：创建向量库失败：{e}{Style.RESET_ALL}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤 5: 验证入库结果
print(f"{Fore.YELLOW}步骤 5: 验证入库结果...{Style.RESET_ALL}")
try:
    # 重新加载向量库进行验证
    verify_vectorstore = Chroma(
        collection_name="three_kingdoms_parent_child",
        embedding_function=embeddings,
        persist_directory=str(vector_store_path)
    )
    
    count = verify_vectorstore._collection.count()
    print(f"  [OK] 向量库中文档数量：{count}")
    
    # 测试查询
    test_query = "诸葛亮"
    results = verify_vectorstore.similarity_search(test_query, k=2)
    
    if results:
        print(f"  [OK] 测试查询成功：'{test_query}' 返回 {len(results)} 个结果")
        print(f"      第一个结果：{results[0].metadata.get('event', 'Unknown')}\n")
    else:
        print(f"  {Fore.YELLOW}警告：测试查询未返回结果{Style.RESET_ALL}\n")
    
except Exception as e:
    print(f"{Fore.RED}错误：验证失败：{e}{Style.RESET_ALL}\n")

# 完成
print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
print(f"{Fore.GREEN}入库完成！{Style.RESET_ALL}")
print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}\n")

print(f"{Fore.CYAN}下一步:{Style.RESET_ALL}")
print(f"  运行向量叠加测试：python app/rag/test_vector_superposition.py\n")
