"""
阶段五：高级检索策略测试

运行：uv run python hello_advanced_rag.py
"""
from src.rag.chain import RAGChain
from src.rag.advanced import AdvancedRAGChain

questions = [
    "加班给多少钱？要提前申请吗？",
    "公司电脑能用自己装的VPN吗？",
    "密码多久改一次有什么要求？",
    "公司让养宠物吗？",
]

print("=" * 60)
print("测试 A：基础 RAG（阶段四）")
print("=" * 60)
basic = RAGChain(k=3)
for q in questions:
    result = basic.ask_with_sources(q)
    print(f"\n❓ {q}")
    print(f"🤖 {result['answer'][:200]}")

print("\n\n" + "=" * 60)
print("测试 B：高级 RAG（阶段五）")
print("=" * 60)
advanced = AdvancedRAGChain(k=5, rerank_top=2)
for q in questions:
    result = advanced.ask(q)
    print(f"\n{'─'*50}")
    print(f"❓ {result['question']}")
    print(f"🤖 {result['answer'][:300]}")
    print(f"📚 来源:")
    for s in result["sources"]:
        print(f"   → {s['doc_name']}")