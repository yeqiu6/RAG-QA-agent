"""
阶段六：Agentic RAG 测试

运行：uv run python hello_agent.py
"""
from src.rag.chain import RAGChain
from src.agent.graph import AgenticRAG

questions = [
    "年假政策和加班制度有什么不同？",    # 对比类，需要查多份文档
    "密码要求是什么？",                  # 简单问题
    "你好啊",                           # 闲聊，不需要检索
    "公司允许养宠物吗？",                # 知识库没有
]

print("=" * 60)
print("普通 RAG（阶段四）")
print("=" * 60)
basic = RAGChain(k=3)
for q in questions:
    print(f"\n❓ {q}")
    result = basic.ask_with_sources(q)
    print(f"🤖 {result['answer'][:250]}")

print("\n\n" + "=" * 60)
print("Agentic RAG（阶段六）")
print("=" * 60)
agent = AgenticRAG(k=3, max_iterations=3)
for q in questions:
    result = agent.run(q)
    print(f"\n{'─'*50}")
    print(f"❓ {result['question']}")
    print(f"🔄 迭代: {result['iterations']} 轮")
    print(f"🤖 {result['answer'][:300]}")
    print(f"📚 来源:")
    for s in result["sources"]:
        print(f"   → {s['doc_name']}")