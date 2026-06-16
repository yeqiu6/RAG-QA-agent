"""
阶段四：基础 RAG 管道测试

运行：uv run python hello_rag_chain.py
"""
from src.document.pipeline import DocumentProcessor
from src.retrieval.vector_store import VectorStoreManager
from src.rag.chain import RAGChain

# 先检查向量库有没有数据，没有就建
import os
if not os.path.exists("./chroma_db") or not os.listdir("./chroma_db"):
    print("📦 正在构建向量库...")
    processor = DocumentProcessor(chunk_size=600, chunk_overlap=120)
    chunks = processor.process("data/sample_docs")
    store = VectorStoreManager()
    store.add_documents(chunks)
    print()

# 创建 RAG 问答链
rag = RAGChain(k=2)

# 测试
questions = [
    "加班费怎么算？加班需要审批吗？",
    "公司密码有什么要求？多久要换一次？",
    "入住二线城市酒店能报销多少钱一晚？",
    "公司允许员工养猫吗？",   # 知识库里没有
]

for q in questions:
    print(f"\n{'='*60}")
    result = rag.ask_with_sources(q)
    print(f"❓ {result['question']}")
    print(f"\n🤖 {result['answer']}")
    print(f"\n📚 参考来源：")
    for s in result["sources"]:
        print(f"   → {s['doc_name']}")