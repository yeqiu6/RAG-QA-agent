# 🧠 RAG Agent — 企业级智能知识库检索增强生成系统

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3+-green.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-purple.svg)](https://www.langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-teal.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

从零构建的**企业级 RAG（Retrieval-Augmented Generation）智能检索 Agent**，支持多格式文档导入、语义检索、高级检索策略优化与 LLM Agent 自主推理，最终封装为 FastAPI + Streamlit 双模服务。

> 🎯 **核心价值**：解决大模型知识滞后与幻觉问题，所有回答均可追溯至原始文档。

---

## ✨ 核心特性

| 模块 | 能力 |
|------|------|
| 📄 **文档处理** | 支持 .md / .txt / .pdf 多格式自动识别、递归语义分块、元数据增强 |
| 🔍 **检索引擎** | Chroma 向量数据库 + ANN 近似最近邻检索，替代暴力余弦相似度 |
| 🎯 **高级检索** | 查询重写 → 多路召回（语义+BM25）→ LLM 重排序 → 自省式检索 |
| 🤖 **Agent 智能体** | 基于 LangGraph 的四节点有状态图，自主推理、检索自纠、智能路由 |
| 🐍 **Web 界面** | Streamlit 对话式 UI，支持流式响应与文档来源追溯 |
| 🚀 **API 服务** | FastAPI RESTful 接口 + 自动生成 Swagger 文档，3 个端点开箱即用 |
| 📊 **质量评估** | 自研三指标评估器（检索命中率、信息覆盖率、幻觉检测） |
| 🔌 **LLM 可插拔** | 默认通义千问，通过修改配置可切换 DeepSeek / OpenAI / 本地模型 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    📱 客户端层                        │
│  Streamlit Web UI  │  HTTP API  │  企业微信/钉钉机器人 │
└────────┬────────────────┬────────────────┬──────────┘
         │                │                │
         ▼                ▼                ▼
┌─────────────────────────────────────────────────────┐
│                 🚀 FastAPI 服务层                     │
│  GET /api/v1/health  │  POST /api/v1/qa (基础RAG)    │
│  POST /api/v1/agent/qa (Agent智能体)                │
└────────┬────────────────┬────────────────────────────┘
         │                │
         ▼                ▼
┌────────────────┐  ┌─────────────────────────────────┐
│  🤖 Agent 层    │  │  📚 基础 RAG 层                  │
│  LangGraph      │  │  LCEL 管道 (检索→增强→生成)      │
│  四节点状态图    │  │  Prompt 模板 + 流式输出          │
│  自纠正检索      │  └──────────────┬──────────────────┘
└───────┬─────────┘                 │
        │                           │
        ▼                           ▼
┌─────────────────────────────────────────────────────┐
│                 🔍 高级检索策略层                      │
│  查询重写 → 多路召回(语义+BM25) → LLM重排序 → 自省    │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              💾 检索引擎 & 向量库层                    │
│  Chroma Vector DB  │  DashScope Embedding           │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              📄 文档处理管线                           │
│  加载(多格式) → 递归语义分块 → 元数据增强 → 入库      │
└─────────────────────────────────────────────────────┘
```

### Agent 思考流程（LangGraph 状态图）

```
用户问题
    │
    ▼
┌──────────┐    闲聊/问候    ┌──────────┐
│  analyze │───────────────→│ generate │──→ 直接回复
│ (分析)    │                │ (生成)    │
└────┬─────┘                └──────────┘
     │ 需要检索
     ▼
┌──────────┐   ┌──────────┐   相关   ┌──────────┐
│ retrieve │──→│ evaluate │────────→│ generate │──→ 有据可查的回复
│ (检索)    │   │ (评估)    │         │ (生成)    │
└──────────┘   └────┬─────┘         └──────────┘
       ↑             │ 不相关 & 未达上限
       │             │ (换检索词重搜)
       └─────────────┘
```

---

## 🛠️ 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **LLM** | 通义千问 / DeepSeek / OpenAI | 可插拔，修改配置即切换 |
| **Embedding** | DashScope text-embedding-v2 | 千问原生向量模型 |
| **编排框架** | LangChain 1.3 + LangGraph 1.2 | 业界主流 RAG/Agent 框架 |
| **向量数据库** | Chroma 1.5 | 轻量级，零配置，数据持久化 |
| **后端** | FastAPI 0.136 + Uvicorn | 异步高性能，自动 Swagger 文档 |
| **前端** | Streamlit 1.58 | 纯 Python Web UI，快速原型 |
| **文档解析** | PyPDF 6.13 + LangChain Loaders | 多格式文档支持 |
| **依赖管理** | uv | Rust 实现，比 pip/poetry 快 10-100 倍 |
| **评估** | 自研三指标评估器 | 检索命中率 + 信息覆盖率 + 幻觉检测 |

---

## 📂 项目结构

```
rag-agent/
├── src/
│   ├── config.py              # 全局配置（API Key / 模型）
│   ├── main.py                # FastAPI 入口
│   │
│   ├── document/              # 📄 阶段二：文档处理管线
│   │   ├── loader.py          #   多格式加载（.md/.txt/.pdf）
│   │   ├── splitter.py        #   递归语义分块 + 重叠窗口
│   │   └── pipeline.py        #   一站式处理管道
│   │
│   ├── retrieval/             # 💾 阶段三：向量检索引擎
│   │   ├── embeddings.py      #   Embedding 模型封装
│   │   └── vector_store.py    #   Chroma 向量库 CRUD + 语义搜索
│   │
│   ├── rag/                   # 🔍 阶段四-五：RAG 核心
│   │   ├── chain.py           #   基础 RAG（LCEL 管道）
│   │   ├── prompts.py         #   Prompt 模板
│   │   └── advanced.py        #   高级检索策略（重写/重排/多路/自省）
│   │
│   ├── agent/                 # 🤖 阶段六：Agentic RAG
│   │   └── graph.py           #   LangGraph 四节点状态图
│   │
│   ├── eval/                  # 📊 阶段七：质量评估
│   │   ├── test_data.py       #   测试数据集
│   │   └── evaluator.py       #   三指标评估器
│   │
│   └── api/                   # 🚀 阶段八：API 服务
│       ├── routes.py          #   路由（3 个端点）
│       └── schemas.py         #   Pydantic 数据模型
│
├── app.py                     # Streamlit Web 界面
├── hello_*.py                 # 各阶段教学演示脚本
├── data/sample_docs/          # 测试文档集（考勤/IT安全/差旅报销）
└── pyproject.toml             # 项目依赖配置
```

---

## 🚀 快速开始

### 1. 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)（Python 包管理器）
- 千问 API Key（[免费注册](https://bailian.console.aliyun.com/)）

### 2. 克隆 & 安装

```bash
git clone https://github.com/your-username/rag-agent.git
cd rag-agent

# 安装依赖（uv 自动创建虚拟环境）
uv sync
```

### 3. 配置 API Key

```bash
# 创建 .env 文件
cp .env.example .env
```

编辑 `.env`，填入你的千问 API Key：

```env
QWEN_API_KEY=sk-your-key-here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> 💡 如果想用 DeepSeek 或 OpenAI，修改 `src/config.py` 即可，三行配置切换模型。

### 4. 启动服务

**方式 A：Streamlit Web 界面**

```bash
uv run streamlit run app.py
# 浏览器打开 http://localhost:8501
```

**方式 B：FastAPI 服务**

```bash
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
# Swagger 文档：http://localhost:8000/docs
```

### 5. 索引测试文档

首次启动时，系统会自动将 `data/sample_docs/` 下的文档向量化并存入 Chroma。你也可以替换成自己的企业文档。

---

## 📡 API 文档

启动 FastAPI 后访问 `http://localhost:8000/docs`，可直接在 Swagger 页面测试：

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/api/v1/health` | 健康检查 |
| `POST` | `/api/v1/qa` | 基础 RAG 问答（检索→生成） |
| `POST` | `/api/v1/agent/qa` | Agent 智能问答（分析→检索→评估→生成） |

**请求示例：**

```bash
curl -X POST http://localhost:8000/api/v1/agent/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "加班费怎么算？", "k": 3}'
```

**响应示例：**

```json
{
  "question": "加班费怎么算？",
  "answer": "根据《员工手册_考勤制度》，工作日加班按1.5倍工资计算，周末加班按2倍计算...",
  "sources": [
    {
      "doc_name": "员工手册_考勤制度",
      "preview": "# 考勤制度\n\n## 工作时间\n\n公司实行弹性工作制..."
    }
  ]
}
```

---

## 🎓 教程导航

项目附带 8 个阶段的教学演示脚本，每个 `hello_*.py` 对应一个阶段的独立可运行 Demo：

| 脚本 | 阶段 | 内容 |
|------|------|------|
| `hello_llm.py` | 阶段一 | LLM API 调用 + Prompt Engineering |
| `hello_rag.py` | 阶段一 | RAG 原理 + Embedding 入门 |
| `hello_document.py` | 阶段二 | 文档加载、分块、处理管道 |
| `hello_retrieval.py` | 阶段三 | Chroma 向量库、语义检索 |
| `hello_rag_chain.py` | 阶段四 | 基础 RAG 管道（LCEL） |
| `hello_advanced_rag.py` | 阶段五 | 高级检索策略（重写/重排/多路召回/自省） |
| `hello_agent.py` | 阶段六 | LangGraph Agentic RAG 智能体 |
| `hello_eval.py` | 阶段七 | RAG 系统质量评估 |

完整课程大纲见 [`Tutoral/01-课程大纲.md`](Tutoral/01-课程大纲.md)。

---

## 🔧 关键技术实现

### 1. 文档处理管道

```python
# 一行加载整个目录
processor = DocumentProcessor(chunk_size=600, chunk_overlap=120)
chunks = processor.process("data/sample_docs")
# → 自动识别 .md/.txt/.pdf → 递归语义分块 → 元数据增强
```

### 2. 向量检索引擎

```python
store = VectorStoreManager()
store.add_documents(chunks)               # 向量化 + 入库
docs = store.search("年假有几天？", k=3)   # 语义搜索（O(log n)）
```

### 3. 高级检索策略

```python
# 集成查询重写 + 多路召回 + 重排序 + 自省的四合一检索器
advanced = AdvancedRAGChain(k=5, rerank_top=3)
result = advanced.ask("加班给多少钱？要提前申请吗？")
```

### 4. Agentic RAG 智能体

```python
agent = AgenticRAG(k=3, max_iterations=3)

# 聊天类问题 → 直接回复，不走检索
agent.run("你好啊")

# 复杂问题 → 分析 → 检索 → 评估 → 重搜（最多3轮）
agent.run("年假政策和加班制度有什么不同？")

# 知识库没有的问题 → 老实说不知道
agent.run("公司允许养宠物吗？")
```

### 5. Agent 自纠正机制（CRAG 模式）

- **analyze**：LLM + 硬规则兜底，闲聊直接回复、知识问题进入检索
- **retrieve**：多路召回（语义 + BM25 关键词），结果合并去重
- **evaluate**：LLM 评估文档主题相关性，不通过则自动换词重搜
- **generate**：检索达标或达到迭代上限后，基于文档生成回答
- **兜底**：3 轮仍未通过，降级使用已有文档生成回答，避免"白板回复"

### 6. 质量评估体系

```python
evaluator = RAGEvaluator()
scores = evaluator.evaluate(TEST_CASES)
# → 检索命中率 + 信息覆盖率 + 幻觉检测（不依赖外部 LLM 评分）
```

---

## 📊 项目亮点

- **模块化解耦**：DocumentProcessor → VectorStoreManager → RAGChain → AgenticRAG 四层独立，每一层可单独替换（如换用 Milvus / Qdrant 向量库）
- **Agent 自纠正**：基于 LangGraph 实现 CRAG 模式，检索结果自评质量，不通过自动换词重搜，三次未达上限仍可降级生成，避免用户白等
- **LLM 可插拔**：LLM 和 Embedding 均通过适配层封装，修改 `src/config.py` 三行配置即可切换千问 / DeepSeek / OpenAI
- **API 双模式**：同服提供基础 RAG（`/qa`）和 Agent RAG（`/agent/qa`），按场景灵活调用
- **自研评估器**：不依赖外部框架，直接量化检索命中率、信息覆盖率与幻觉率，可用于每次 Prompt/参数调优后的 A/B 对比

---

## 📄 License

MIT License

---

## 🙏 致谢

- [LangChain](https://www.langchain.com/) — LLM 应用编排框架
- [LangGraph](https://www.langchain.com/langgraph) — Agent 状态图引擎
- [Chroma](https://www.trychroma.com/) — 开源向量数据库
- [通义千问](https://tongyi.aliyun.com/) — LLM 与 Embedding 能力
- [Streamlit](https://streamlit.io/) — Python Web UI 框架
- [FastAPI](https://fastapi.tiangolo.com/) — 高性能 API 框架
