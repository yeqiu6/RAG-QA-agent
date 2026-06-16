"""
高级 RAG 策略 —— 查询重写 + 重排序 + 多路召回 + 自省检索
"""
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from src.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from src.retrieval.vector_store import VectorStoreManager
from src.rag.chain import _format_docs


class AdvancedRAGChain:
    """集成多种高级检索策略

    用法：
        rag = AdvancedRAGChain(k=5, rerank_top=3)
        result = rag.ask("加班给多少钱？")
    """

    def __init__(self, k: int = 5, rerank_top: int = 3):
        self.k = k          # 第一阶段检索返回多少文档
        self.rerank_top = rerank_top  # 重排序后保留几个
        self._store = VectorStoreManager()
        self._llm = ChatOpenAI(
            model=QWEN_MODEL,
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            temperature=0,
        )

        # ============================================================
        # 5.1 查询重写：口语 → 更好的检索词
        # ============================================================
    def rewrite_query(self, question: str) -> str:
            """用 LLM 把用户口语转成更适合检索的关键词"""
            prompt = f"""你是一个搜索查询优化器。把用户口语问题改写成适合在企业文档中搜索的关键词。

        规则：
        - 提取核心概念和关键实体
        - 补全可能的同义词
        - 去掉语气词和废话
        - 输出纯关键词，用空格分隔，不要超过30个字

        用户问题：{question}
        优化后的检索词："""

            response = self._llm.invoke(prompt)
            rewritten = response.content.strip()
            print(f"   🔄 查询重写: '{question}' → '{rewritten}'")
            return rewritten

        # ============================================================
        # 5.2 重排序：检索完再精筛
        # ============================================================
    def rerank(self, question: str, docs: List[Document]) -> List[Document]:
            """用 LLM 对检索结果重新打分排序，滤掉不相关的"""
            if len(docs) <= self.rerank_top:
                return docs

            # 让 LLM 选出最相关的文档编号
            doc_list = ""
            for i, doc in enumerate(docs):
                preview = doc.page_content[:200].replace("\n", " ")
                doc_list += f"[{i}] {preview}\n\n"

            prompt = f"""根据用户问题，从以下文档片段中选出最相关的 {self.rerank_top} 个。
        只输出数字编号，用逗号分隔，例如: 0, 2, 5

        用户问题：{question}

        文档列表：
        {doc_list}
        最相关的 {self.rerank_top} 个文档编号："""

            response = self._llm.invoke(prompt)

            # 解析 LLM 返回的编号
            try:
                indices = []
                for part in response.content.split(","):
                    part = part.strip().strip("[]()（）")
                    if part.isdigit():
                        indices.append(int(part))
                reranked = [docs[i] for i in indices[:self.rerank_top] if i < len(docs)]
                if len(reranked) > 0:
                    print(f"   📊 重排序: {len(docs)} 篇 → {len(reranked)} 篇（编号: {indices[:self.rerank_top]}）")
                    return reranked
            except Exception:
                pass

            # 如果解析失败，退回用原始排序的前几个
            return docs[:self.rerank_top]

            # ============================================================
            # 5.3 多路召回：语义 + 关键词
            # ============================================================
    def _build_keyword_index(self, docs: List[Document]) -> BM25Retriever:
                """用当前检索结果构建临时 BM25 索引"""
                return BM25Retriever.from_documents(docs)

    def multi_retrieve(self, question: str) -> List[Document]:
                """双路检索：
                路1：语义向量检索（Chroma）
                路2：关键词检索（BM25）
                结果合并去重
                """
                # 先用语义检索拉一批（比最终需要的多拉一些）
                all_docs = self._store.search(question, k=30)

                if len(all_docs) == 0:
                    return []

                # 路1：语义 Top-K
                semantic_results = all_docs[:self.k]

                # 路2：BM25 关键词检索（在全部30篇的基础上建临时索引）
                try:
                    bm25 = BM25Retriever.from_documents(all_docs)
                    keyword_results = bm25.invoke(question)
                    keyword_results = keyword_results[:self.k]
                except Exception:
                    keyword_results = []

                # 合并 + 去重（按 page_content 去重）
                seen = set()
                merged = []
                for doc in semantic_results + keyword_results:
                    key = doc.page_content[:100]  # 用前100字符做指纹
                    if key not in seen:
                        seen.add(key)
                        merged.append(doc)

                print(
                    f"   🔀 多路召回: 语义{len(semantic_results)}篇 + 关键词{len(keyword_results)}篇 → 去重后{len(merged)}篇")
                return merged

                # ============================================================
                # 5.4 自省检索：检索后自我检查
                # ============================================================
    def _check_relevance(self, question: str, docs: List[Document]) -> bool:
                    """让 LLM 判断检索到的文档是否相关"""
                    if not docs:
                        return False

                    previews = []
                    for i, doc in enumerate(docs[:3]):
                        previews.append(f"[{i}] {doc.page_content[:300]}")

                    prompt = f"""判断以下文档能否回答用户问题。只需回答"相关"或"不相关"。

            用户问题：{question}

            文档：
            {chr(10).join(previews)}

            这些文档能回答用户问题吗？（相关/不相关）："""

                    response = self._llm.invoke(prompt)
                    answer = response.content.strip()
                    # 先检查"不相关"，避免 "相关" in "不相关" 的 bug
                    if "不相关" in answer:
                        return False
                    if "相关" in answer:
                        return True

                    # LLM 没按格式答，默认认为相关（宽容策略）
                    print(f"   ⚠️  无法判断: '{answer}'")
                    return True

    def ask(self, question: str) -> dict:
                    """完整的智能检索流程：
                    1. 查询重写
                    2. 多路召回
                    3. 重排序
                    4. 自省检查（不通过就重试）
                    5. LLM 生成
                    """
                    print(f"\n🚀 开始处理: {question}")

                    # Step 1: 查询重写
                    search_query = self.rewrite_query(question)

                    # Step 2+3+4: 检索 + 自省（最多重试 2 次）
                    final_docs = []
                    for attempt in range(3):
                        # 多路召回
                        docs = self.multi_retrieve(search_query)
                        # 重排序
                        docs = self.rerank(question, docs)

                        if self._check_relevance(question, docs):
                            final_docs = docs
                            print(f"   ✅ 第{attempt + 1}次检索通过")
                            break
                        else:
                            print(f"   ⚠️  第{attempt + 1}次检索不相关，重试...")
                            # 用 LLM 生成新的检索词
                            search_query = self.rewrite_query(
                                f"{question}（上次搜索'{search_query}'效果不好，请换个思路）"
                            )
                    else:
                        # 3 次都不行，用最后一次的结果
                        final_docs = docs
                        print(f"   ⚠️  3次检索均不理想，使用最后一次结果")

                    # Step 5: 生成回答
                    context = _format_docs(final_docs[:self.rerank_top])

                    prompt = f"""请根据以下参考文档回答用户问题。

            规则：
            1. 如果参考文档中有相关信息，请准确回答，并注明文档名称
            2. 如果参考文档中没有相关信息，请说"根据现有资料无法回答"
            3. 回答要简洁、专业

            参考文档：
            {context}

            用户问题：{question}

            回答："""

                    response = self._llm.invoke(prompt)

                    sources = []
                    for doc in final_docs[:self.rerank_top]:
                        sources.append({
                            "doc_name": doc.metadata.get("doc_name", "未知"),
                            "source_file": doc.metadata.get("source_file", ""),
                            "preview": doc.page_content[:150],
                        })

                    return {
                        "question": question,
                        "answer": response.content,
                        "sources": sources,
                    }