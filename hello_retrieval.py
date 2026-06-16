"""
阶段三：向量检索引擎测试

运行：uv run python hello_retrieval.py
"""
from src.document.pipeline import DocumentProcessor
from src.retrieval.vector_store import VectorStoreManager

# 第一步：处理文档
print("=" * 50)
print("第一步：文档处理")
print("=" * 50)
processor = DocumentProcessor(chunk_size=600, chunk_overlap=120)
chunks = processor.process("data/sample_docs")

# 第二步：存入向量库
print("\n" + "=" * 50)
print("第二步：文档入库（向量化 + 存储）")
print("=" * 50)
store = VectorStoreManager()
store.add_documents(chunks)

# 第三步：测试检索
print("\n" + "=" * 50)
print("第三步：语义检索测试")
print("=" * 50)

test_queries = [
    "加班费怎么算？",
    "密码有什么要求？",
    "住宿报销标准是多少？",
]

for q in test_queries:
    print(f"\n🔍 查询：{q}")
    results = store.search_with_scores(q, k=2)
    for doc, score in results:
        fname = doc.metadata.get("doc_name", "unknown")
        print(f"   [{fname}] 相关度={score:.3f}")
        print(f"   {doc.page_content[:120].replace(chr(10), ' ')}...")

print("\n✅ 阶段三完成！向量库路径：./chroma_db")