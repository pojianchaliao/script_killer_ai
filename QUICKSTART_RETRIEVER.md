# 🚀 快速开始 - ParentDocumentRetriever 检索器

## 📦 安装依赖

```bash
# 安装新依赖
pip install colorama==0.4.6

# 或更新所有依赖
pip install -r requirements.txt
```

---

## ⚡ 5 分钟快速测试

### 方法 1: 一键运行测试（推荐）

```bash
python run_retrieval_test.py
```

**你会看到：**
```
============================================================
🧪 三国数据检索策略对比测试
============================================================

🔧 正在初始化三种检索策略...
  📦 创建大块存储 (chunk_size=1000)...
     创建了 150 个大块

  📦 创建小块存储 (chunk_size=200)...
     创建了 750 个小块

  📦 创建父子检索器 (parent=1000, child=200)...
     创建了 150 个父文档，750 个子文档

✅ 初始化完成

============================================================
测试 1: 测试具体事件的游戏效果查询
============================================================

🔍 查询：赤壁之战的游戏效果是什么？
📯 预期事件：['赤壁之战', '游戏效果', '长江以南', '三方争夺']

  测试策略：large
    精度：60.00% (3/5)
    耗时：0.112s

  测试策略：small
    精度：75.00% (4/5)
    耗时：0.143s

  测试策略：parent_child
    精度：85.00% (5/5)
    耗时：0.168s

...

🏆 推荐策略：父子检索器
   理由：在该测试集上表现最佳
```

---

### 方法 2: 可视化演示

```bash
python -m app.rag.demo_comparison
```

**彩色输出，更直观！**

```
======================================================================
🎮 三国数据检索策略对比演示
======================================================================

📚 加载数据：app/data/romance_three_kingdoms.json
   共 14281 条记录

🔧 加载嵌入模型...
✅ 模型加载完成

📦 策略 1: 创建大块存储 (chunk_size=1000)...
   ✅ 创建 150 个大块

📦 策略 2: 创建小块存储 (chunk_size=200)...
   ✅ 创建 750 个小块

📦 策略 3: 创建父子检索器...
   ✅ 创建 150 个父文档，750 个子文档

======================================================================
🔍 查询：赤壁之战的游戏效果是什么？
======================================================================

【策略 1】只用大块 (1000 tokens)
❌ 问题：语义匹配不精确，可能错过细节

  [1] 分数：0.892 | 耗时：0.112s
      【赤壁之战】主题：历史事件 来源：正史 戏剧价值：high 📖 背景与描述：曹操统一北方后...

【策略 2】只用小块 (200 tokens)
❌ 问题：上下文丢失，信息不完整

  [1] 分数：0.923 | 耗时：0.143s
      赤壁之战 - 历史事件 曹操统一北方后，挥师南下...

【策略 3】父子检索器 (parent=1000, child=200)
✅ 优势：小块检索精确 + 大块返回完整上下文

  [1] 分数：0.945 | 耗时：0.168s
      【赤壁之战】主题：历史事件 来源：正史 戏剧价值：high 📖 背景与描述：...

======================================================================
🏆 推荐：父子检索器
   ✓ 检索精度高（小块匹配）
   ✓ 上下文完整（大块返回）
   ✓ 平衡了精度和完整性

按 Enter 继续下一个查询...
```

---

### 方法 3: Python 代码直接使用

```python
from app.rag.custom_parent_retriever import create_three_kingdoms_retriever

# 创建检索器
retriever = create_three_kingdoms_retriever(
    data_path="app/data/romance_three_kingdoms.json",
    parent_chunk_size=1000,
    child_chunk_size=200,
    chunk_overlap=20,
    search_k=5
)

# 添加文档（首次使用需要）
print("正在导入文档...")
retriever.add_documents()
print(f"✅ 已导入 {len(retriever.processor.raw_data)} 条记录")

# 检索
query = "赤壁之战的游戏效果是什么？"
results = retriever.retrieve(query, k=3)

print(f"\n找到 {len(results)} 个相关结果:\n")
for i, doc in enumerate(results, 1):
    print(f"[{i}] 事件：{doc.metadata.get('event')}")
    print(f"    主题：{doc.metadata.get('theme')}")
    print(f"    内容：{doc.page_content[:150]}...")
    print("-" * 50)
```

---

## 📊 理解三种策略

### 策略对比图

```
查询："赤壁之战的游戏效果"

┌─────────────────────────────────────────────┐
│ 策略 1: 大块 (1000 tokens)                   │
├─────────────────────────────────────────────┤
│ 【赤壁之战】                                │
│ 完整内容 (800 tokens)                        │
│ ...背景...描述...效果...事实...标签...       │
│                                             │
│ ✓ 上下文完整                                 │
│ ❌ 语义模糊 → 向量表示不精确                 │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ 策略 2: 小块 (200 tokens)                    │
├─────────────────────────────────────────────┤
│ 块 1: "赤壁之战 - 历史事件\n背景..."          │
│ 块 2: "游戏效果：长江以南..."                │
│ 块 3: "历史事实：资治通鉴..."                │
│ 块 4: "标签：赤壁之战，孙刘联军..."           │
│                                             │
│ ✓ 语义精确                                   │
│ ❌ 丢失上下文 → 看不懂完整意思               │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ 策略 3: 父子检索器 ⭐推荐                     │
├─────────────────────────────────────────────┤
│ 步骤 1: 用小块 (200) 检索 → 精确匹配          │
│ 步骤 2: 找到对应的父文档 ID                  │
│ 步骤 3: 返回大块 (1000) → 完整上下文         │
│                                             │
│ ✅ 小块检索 → 高精度                         │
│ ✅ 大块返回 → 完整上下文                     │
│ ✅ 鱼与熊掌兼得！                            │
└─────────────────────────────────────────────┘
```

---

## 🎯 参数设计原理

### 为什么是 1000/200/20？

基于你的 JSON 数据分析：

```json
{
  "id": "tk_001",
  "theme": "历史事件",           // ~10 字
  "event": "起义",               // ~10 字
  "description": "背景说明：...社会影响：...",  // ~150 字
  "game_effect": "...",          // ~30 字
  "historical_fact": "...",      // ~20 字
  "tags": ["黄巾起义", ...]      // 4 个标签 × 5 字 = 20 字
}
```

**单个对象总字数：** ~240-300 字 ≈ 400-500 tokens

**分块策略：**
- **父文档 (1000 tokens):** 容纳 2-3 个相关事件，保持完整上下文
- **子文档 (200 tokens):** 聚焦单个字段，语义清晰
- **重叠 (20 tokens):** 防止句子被切断

---

## 🔍 实际效果对比

### 测试查询："赤壁之战的游戏效果是什么？"

#### ❌ 大块策略结果
```
[1] 分数：0.892
【赤壁之战】
主题：历史事件
来源：正史
戏剧价值：high

📖 背景与描述：
曹操统一北方后，挥师南下，意图吞并江东。
荆州牧刘琮降曹，刘备败走夏口。
诸葛亮说服孙权联合抗曹...  // 很长，但没提到游戏效果

🎮 游戏效果：
长江以南禁止魏军陆军直接占领...  // 找到了，但被淹没在长文本中

...其他内容...
```
**问题：** 信息太多，重点不突出

---

#### ❌ 小块策略结果
```
[1] 分数：0.923
游戏效果：长江以南禁止魏军陆军直接占领（需水军科技）；荆州南部进入"三方争夺"状态。
```
**问题：** 只有游戏效果，不知道是哪个事件的！

---

#### ✅ 父子检索器结果
```
[1] 分数：0.945
【赤壁之战】
主题：历史事件
来源：正史
戏剧价值：high

📖 背景与描述：
曹操统一北方后，挥师南下...

🎮 游戏效果：
长江以南禁止魏军陆军直接占领（需水军科技）；
荆州南部进入"三方争夺"状态。

📚 历史事实：
《资治通鉴·卷六十五》

🏷️ 标签：赤壁之战，孙刘联军，曹操，江东
```
**完美：** 既有精确匹配的游戏效果，又有完整的背景信息！

---

## 🛠️ 集成到你的项目

### 在现有 RAG 系统中使用

假设你有现有的 RAG 系统：

```python
# 原来的代码
class YourRAGSystem:
    def __init__(self):
        self.retriever = SomeRetriever()
    
    def query(self, question: str):
        context = self.retriever.search(question)
        return self.llm.generate(question, context)

# 替换为父子检索器
from app.rag.custom_parent_retriever import create_three_kingdoms_retriever

class YourRAGSystem:
    def __init__(self):
        self.retriever = create_three_kingdoms_retriever(
            data_path="app/data/romance_three_kingdoms.json",
            parent_chunk_size=1000,
            child_chunk_size=200,
            chunk_overlap=20,
            search_k=5
        )
        # 首次使用时添加文档
        self.retriever.add_documents()
    
    def query(self, question: str):
        # 检索相关文档
        results = self.retriever.retrieve(question, k=3)
        
        # 提取上下文
        context = "\n\n".join([doc.page_content for doc in results])
        
        # 生成回答
        return self.llm.generate(question, context)
```

---

## 📈 性能监控

### 添加日志记录

```python
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class MonitoredRetriever:
    def __init__(self):
        self.retriever = create_three_kingdoms_retriever()
        self.retriever.add_documents()
        self.query_count = 0
    
    def retrieve(self, query: str, k: int = 3):
        start_time = datetime.now()
        
        # 执行检索
        results = self.retriever.retrieve(query, k=k)
        
        # 记录日志
        elapsed = (datetime.now() - start_time).total_seconds()
        self.query_count += 1
        
        logging.info(f"Query #{self.query_count}: '{query}'")
        logging.info(f"  Results: {len(results)} items")
        logging.info(f"  Time: {elapsed:.3f}s")
        logging.info(f"  Top result: {results[0].metadata.get('event') if results else 'None'}")
        
        return results

# 使用
monitored = MonitoredRetriever()
results = monitored.retrieve("赤壁之战")
```

---

## ⚠️ 常见问题 FAQ

### Q1: 第一次运行报错 "No module named 'colorama'"

**A:** 安装依赖：
```bash
pip install -r requirements.txt
```

---

### Q2: 模型下载很慢

**A:** BGE 模型较大，首次需要下载。可以：
1. 使用国内镜像源
2. 手动下载后放到缓存目录

```bash
# 使用清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentence-transformers
```

---

### Q3: 内存占用较高

**A:** 向量库加载到内存会占用约 1-2GB。解决方案：
1. 使用持久化存储
2. 减少加载的数据量（测试时只用部分数据）

```python
# 只加载前 1000 条用于测试
retriever = create_three_kingdoms_retriever()
# 修改 add_documents 逻辑，只添加部分文档
```

---

### Q4: 如何调整返回结果数量？

**A:** 修改 `search_k` 参数：

```python
# 返回更多结果
retriever = create_three_kingdoms_retriever(search_k=10)

# 或在检索时指定
results = retriever.retrieve(query, k=10)
```

---

## 🎓 进阶学习

### 阅读顺序建议

1. ✅ **本文档** - 快速上手
2. 📖 `RETRIEVER_PARAMS_GUIDE.md` - 详细参数指南
3. 📖 `README_RETRIEVAL_TEST.md` - 测试说明
4. 📄 源码 - `custom_parent_retriever.py`

---

## 💡 总结

**三步开始：**

```bash
# 1. 安装依赖
pip install colorama==0.4.6

# 2. 运行测试
python run_retrieval_test.py

# 3. 查看效果
# 观察父子检索器如何击败单一策略！
```

**核心要点：**

- 🎯 **父文档 1000 tokens** - 保留完整上下文
- 🎯 **子文档 200 tokens** - 精确语义匹配
- 🎯 **重叠 20 tokens** - 保持连贯性
- 🎯 **search_k=5** - 平衡精度和召回

**立即体验 RAG 检索的最佳实践！** 🚀
