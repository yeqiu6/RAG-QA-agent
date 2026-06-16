"""
课时 1.3-1.4：RAG 原理 + Embedding

运行：uv run python hello_rag.py

这个脚本演示了 RAG 最核心的三步：
  1. 把知识库"向量化"（文档 → 数字）
  2. 根据用户问题"检索"最相关的文档（算距离）
  3. 把文档喂给 LLM，"生成"有据可查的回答
"""

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from src.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL

# ============================================================
# 准备工作：创建 LLM 和 Embedding 两个"引擎"
# ============================================================

# 生成引擎 — 负责写回答
llm = ChatOpenAI(
    model=QWEN_MODEL,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    temperature=0,
)

# 检索引擎 — 负责算相似度
# DashScopeEmbeddings 是千问原生的 Embedding 接口
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",        # 千问的向量模型
    dashscope_api_key=QWEN_API_KEY,
)

# ============================================================
# 第一步：准备"企业知识库"（真实项目里这些是 PDF/Word 文档）
# ============================================================
print("=" * 60)
print("第一步：准备知识库")
print("=" * 60)

knowledge_base = [
    {
        "title": "年假制度",
        "content": "员工入职满1年后享有15天带薪年假。年假最多可累积至30天。"
                   "未休完的年假在次年3月31日之前有效，过期清零。",
    },
    {
        "title": "加班政策",
        "content": "工作日加班按1.5倍工资计算，周末加班按2倍工资计算。"
                   "每月加班时长不得超过36小时。加班需提前一天在OA系统提交申请。",
    },
    {
        "title": "差旅报销",
        "content": "差旅费用需在出差结束后7个工作日内通过OA系统提交报销。"
                   "单笔超过5000元的报销需要部门经理审批。住宿标准不超过每晚500元。",
    },
    {
        "title": "远程办公",
        "content": "员工每周最多可以申请2天远程办公。远程办公需要提前1天在OA系统报备。"
                   "远程办公期间需保持企业微信在线。",
    },
    {
        "title": "会议室使用",
        "content": "会议室需要在OA系统提前1天预约。A栋3楼有大中小三个会议室。"
                   "大会议室可容纳30人，中会议室15人，小会议室6人。",
    },
]

print(f"知识库共 {len(knowledge_base)} 份文档")
for doc in knowledge_base:
    print(f"  📄 {doc['title']}")

# ============================================================
# 第二步：把文档转成向量（Embedding）
# ============================================================
print("\n" + "=" * 60)
print("第二步：文档向量化（Embedding）")
print("=" * 60)

doc_texts = [doc["content"] for doc in knowledge_base]
doc_vectors = [embeddings.embed_query(text) for text in doc_texts]

print(f"每份文档 → {len(doc_vectors[0])} 维向量")
print(f"示例（前5个数字）：{doc_vectors[0][:5]}...")
print("💡 含义：文字变成了数学坐标，相似度=坐标距离")

# ============================================================
# 第三步：RAG 问答函数
# ============================================================
print("\n" + "=" * 60)
print("第三步：检索 + 生成 = RAG")
print("=" * 60)


def rag_qa(question: str) -> dict:
    """最简 RAG：一个问题进来 → 检索文档 → 交给 LLM → 回答出去"""

    # 3.1 把问题也转成向量
    q_vec = embeddings.embed_query(question)

    # 3.2 算问题向量和每份文档向量的余弦相似度
    similarities = []
    for i, doc_vec in enumerate(doc_vectors):
        sim = np.dot(q_vec, doc_vec) / (
            np.linalg.norm(q_vec) * np.linalg.norm(doc_vec)
        )
        similarities.append((i, sim))

    # 3.3 按相似度排序，取 Top-2
    similarities.sort(key=lambda x: x[1], reverse=True)
    top2 = similarities[:2]

    # 3.4 构造 Prompt：参考文档 + 用户问题
    context_parts = []
    for idx, score in top2:
        doc = knowledge_base[idx]
        context_parts.append(
            f"【文档：{doc['title']}（相关度：{score:.2f}）】\n{doc['content']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""请根据以下参考文档回答用户问题。
如果文档中没有相关信息，就说"根据现有资料无法回答"。
回答时请引用文档标题。

参考文档：
{context}

用户问题：{question}

回答："""

    # 3.5 LLM 生成
    response = llm.invoke(prompt)

    return {
        "question": question,
        "retrieved": [
            {"title": knowledge_base[idx]["title"], "score": round(score, 3)}
            for idx, score in top2
        ],
        "answer": response.content,
    }


# ============================================================
# 第四步：测试
# ============================================================
test_questions = [
    "年假有几天？能存到明年吗？",     # 应该命中"年假制度"
    "加班费怎么算？",                # 应该命中"加班政策"
    "出差住宿能报多少钱一晚？",       # 应该命中"差旅报销"
    "明天下午有什么电影？",           # 知识库里没有 → 应该承认不知道
]

for q in test_questions:
    print(f"\n{'─' * 50}")
    result = rag_qa(q)
    print(f"❓ 用户：{result['question']}")
    print(f"📄 检索到的文档：")
    for doc in result["retrieved"]:
        print(f"   → {doc['title']}（相关度：{doc['score']}）")
    print(f"🤖 AI 回答：{result['answer']}")

print("\n" + "=" * 60)
print("✅ RAG 三步走通了！")
print("=" * 60)
print("""
这就是 RAG 的核心骨架：
  ① 文档 → 向量 → 存起来
  ② 问题 → 向量 → 算相似度 → 找到最相关文档
  ③ 文档 + 问题 → LLM → 有据可查的回答

后面的所有章节，就是在这三个环节上不断优化升级。
""")
