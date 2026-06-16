"""
Agentic RAG — 基于 LangGraph 的智能检索智能体

核心能力：
  - 分析问题是否真的需要检索文档
  - 检索后自评质量，不行就换检索词重搜
  - 多轮思考，直到满意或达到上限
"""
import json
import re
from typing import TypedDict, List, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from src.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from src.retrieval.vector_store import VectorStoreManager
from src.rag.chain import _format_docs


# ---- 状态定义：图里流转的数据结构 ----
class AgentState(TypedDict):
    question: str
    search_queries: List[str]
    docs: List
    iteration: int
    max_iterations: int
    need_search: bool
    docs_relevant: bool
    answer: str


class AgenticRAG:
    """智能 RAG Agent

    用法：
        agent = AgenticRAG(k=3, max_iterations=3)
        result = agent.run("年假和加班制度有什么不同？")
    """

    def __init__(self, k: int = 3, max_iterations: int = 3):
        self.k = k
        self.max_iterations = max_iterations
        self._store = VectorStoreManager()
        self._llm = ChatOpenAI(
            model=QWEN_MODEL,
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            temperature=0,
        )
        self._graph = self._build_graph()

    def _is_chitchat(self, question: str) -> bool:
        """用确定性规则兜底，避免简单寒暄误进检索链路。"""
        q = question.strip().lower()
        chitchat_patterns = [
            r"^(你好|您好|嗨|哈喽|hello|hi|hey)[啊呀吗！!。.\s]*$",
            r"^(谢谢|感谢|多谢|thank you|thanks)[！!。.\s]*$",
            r"^(再见|拜拜|bye)[！!。.\s]*$",
        ]
        return any(re.match(pattern, q) for pattern in chitchat_patterns)

    def _clean_queries(self, raw_queries) -> List[str]:
        """把 LLM 可能带解释的输出清洗成短检索词。"""
        if isinstance(raw_queries, str):
            candidates = re.split(r"[,，;；\n]+", raw_queries)
        elif isinstance(raw_queries, list):
            candidates = []
            for item in raw_queries:
                candidates.extend(re.split(r"[,，;；\n]+", str(item)))
        else:
            candidates = []

        queries = []
        noise_markers = ["用户", "询问", "问题", "属于", "需要检索", "不需要", "因此", "因为"]
        for query in candidates:
            query = query.strip().strip("-*[]()（）\"'“”")
            query = re.sub(r"^(检索词|关键词)\s*\d*\s*[:：]\s*", "", query).strip()
            if not query or len(query) > 30:
                continue
            if any(marker in query for marker in noise_markers):
                continue
            if query not in queries:
                queries.append(query)

        return queries

    def _parse_analysis(self, decision: str, question: str) -> tuple[bool, List[str], str]:
        """解析查询分析结果，优先 JSON，失败时做保守兜底。"""
        try:
            match = re.search(r"\{.*\}", decision, re.S)
            data = json.loads(match.group(0) if match else decision)
            need_search_value = data.get("need_search")
            if isinstance(need_search_value, str):
                need_search = need_search_value.strip().lower() in {"true", "yes", "1", "需要", "是"}
            else:
                need_search = bool(need_search_value)

            if need_search:
                queries = self._clean_queries(data.get("queries", []))
                return True, queries or [question], ""

            reply = data.get("answer") or data.get("reply") or "您好，有什么可以帮您？"
            return False, [], str(reply).strip()
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        upper_decision = decision.upper()
        if "ANSWER" in upper_decision and "RETRIEVE" not in upper_decision:
            parts = decision.split("|", 1)
            reply = parts[1].strip() if len(parts) > 1 else "您好，有什么可以帮您？"
            return False, [], reply

        if "RETRIEVE" in upper_decision:
            parts = decision.split("|", 1)
            queries_str = parts[1].strip() if len(parts) > 1 else question
            return True, self._clean_queries(queries_str) or [question], ""

        if self._is_chitchat(question):
            return False, [], "您好，有什么可以帮您？"

        return True, [question], ""

    # ============================================================
    # 节点 1：分析问题 — 要不要检索？检索什么？
    # ============================================================
    def _analyze(self, state: AgentState) -> AgentState:
        """判断问题类型并生成检索词"""
        if self._is_chitchat(state["question"]):
            print(f"   💬 无需检索，直接回复")
            return {**state, "need_search": False, "answer": "您好，有什么可以帮您？"}

        prompt = f"""你是查询分析器。判断用户问题是否需要检索企业文档。

需要检索：问公司制度、政策、规定、流程等具体信息
不需要检索：闲聊、打招呼、问常识

只输出 JSON，不要输出解释文字。
格式：
{{"need_search": true, "queries": ["检索词1", "检索词2"], "answer": ""}}
或
{{"need_search": false, "queries": [], "answer": "直接回复的话"}}

用户问题：{state['question']}
你的判断："""

        response = self._llm.invoke(prompt)
        decision = response.content.strip()
        need_search, queries, reply = self._parse_analysis(decision, state["question"])

        if not need_search:
            print(f"   💬 无需检索，直接回复")
            return {**state, "need_search": False, "answer": reply}

        print(f"   🧠 需要检索 → 检索词: {queries}")
        return {**state, "need_search": True, "search_queries": queries}

    # ============================================================
    # 节点 2：检索文档
    # ============================================================
    def _retrieve(self, state: AgentState) -> AgentState:
        """用检索词去 Chroma 搜文档"""
        all_docs = []
        for q in state["search_queries"]:
            docs = self._store.search(q, k=self.k)
            all_docs.extend(docs)

        # 去重
        seen = set()
        unique = []
        for d in all_docs:
            key = d.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(d)

        print(f"   📚 检索到 {len(unique)} 篇去重文档")
        return {**state, "docs": unique}

    # ============================================================
    # 节点 3：评估 — 文档能不能用？
    # ============================================================
    def _evaluate(self, state: AgentState) -> AgentState:
        """检查文档质量，不行就生成新检索词"""
        docs = state["docs"]
        iteration = state["iteration"] + 1

        if not docs:
            return {
                **state,
                "iteration": iteration,
                "search_queries": [],
                "docs_relevant": False,
                "answer": "根据现有资料无法回答。",
            }

        # 让 LLM 判断文档是否相关
        previews = ""
        for i, doc in enumerate(docs[:3]):
            previews += f"[{i}] {doc.page_content[:200].replace(chr(10), ' ')}\n"

        prompt = f"""以下文档能否回答用户问题？只回答一个词："能"或"不能"。

用户问题：{state['question']}

文档：
{previews}

能回答吗？："""

        response = self._llm.invoke(prompt)
        verdict = response.content.strip()
        can_answer = "能" in verdict and "不能" not in verdict

        if can_answer:
            print(f"   ✅ 第{iteration}轮评估通过")
            return {
                **state,
                "iteration": iteration,
                "search_queries": [],
                "docs_relevant": True,
            }  # ← 清空，表示不用再搜


        if iteration >= state["max_iterations"]:
            print(f"   ⚠️ 已达最大轮数({state['max_iterations']})，未找到可用文档")
            return {
                **state,
                "iteration": iteration,
                "docs_relevant": False,
                "answer": "根据现有资料无法回答。",
            }

        # 换检索词重试
        print(f"   ⚠️ 不相关，重试（{iteration}/{state['max_iterations']}）")
        prompt = f"""之前搜出来的文档跟用户问题匹配度低。请换一个角度，生成 1-3 个新的检索词，用逗号分隔。

用户问题：{state['question']}
之前失败的关键词：{', '.join(state['search_queries'])}

新检索词："""

        response = self._llm.invoke(prompt)
        # 生成新检索词那段，过滤掉过长的"检索词"
        new_queries = self._clean_queries(response.content)
        print(f"   🔄 新检索词: {new_queries}")

        return {
            **state,
            "iteration": iteration,
            "search_queries": new_queries,
        }

    # ============================================================
    # 节点 4：生成回答
    # ============================================================
    def _generate(self, state: AgentState) -> AgentState:
        """用文档生成最终回答"""
        if state.get("answer"):
            # analyze 阶段已经生成了答案（闲聊等不需要检索的情况）
            return state

        docs = state["docs"]
        if not docs or not state.get("docs_relevant"):
            return {**state, "answer": "根据现有资料无法回答。"}

        context = _format_docs(docs[:self.k])

        prompt = f"""你是企业知识库助手。根据以下参考文档回答问题。

规则：
1. 如果参考文档中有相关信息，请准确回答，并注明文档名称
2. 如果参考文档中没有相关信息，请说"根据现有资料无法回答"
3. 回答要简洁、专业

参考文档：
{context}

用户问题：{state['question']}

回答："""

        response = self._llm.invoke(prompt)
        print(f"   ✍️ 生成回答完成")
        return {**state, "answer": response.content}

    # ============================================================
    # 路由：决定走哪条路
    # ============================================================
    def _router(self, state: AgentState) -> Literal["retrieve", "generate"]:
        if not state["need_search"]:
            return "generate"
        return "retrieve"


    def _eval_router(self, state: AgentState) -> Literal["retrieve", "generate"]:
        # 如果 evaluate 处理后还有 search_queries，说明需要重新检索
        if state.get("search_queries") and state["iteration"] < state["max_iterations"]:
            return "retrieve"
        return "generate"

    # ============================================================
    # 构建图
    # ============================================================
    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("analyze", self._analyze)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("evaluate", self._evaluate)
        builder.add_node("generate", self._generate)

        builder.set_entry_point("analyze")

        # analyze → retrieve 或 generate
        builder.add_conditional_edges(
            "analyze",
            self._router,
            {"retrieve": "retrieve", "generate": "generate"},
        )

        # retrieve → evaluate
        builder.add_edge("retrieve", "evaluate")

        # evaluate → retrieve（重试）或 generate（通过）
        builder.add_conditional_edges(
            "evaluate",
            self._eval_router,
            {"retrieve": "retrieve", "generate": "generate"},
        )

        builder.add_edge("generate", END)

        return builder.compile()

    # ============================================================
    # 对外接口
    # ============================================================
    def run(self, question: str) -> dict:
        print(f"\n🚀 Agent 处理: {question}")
        print("=" * 50)

        result = self._graph.invoke({
            "question": question,
            "search_queries": [],
            "docs": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "need_search": False,
            "docs_relevant": False,
            "answer": "",
        })

        sources = []
        seen = set()
        docs_for_sources = (
            result.get("docs", [])[:self.k]
            if result.get("docs_relevant")
            else []
        )
        for doc in docs_for_sources:
            doc_name = doc.metadata.get("doc_name", "未知")
            if doc_name in seen:
                continue
            seen.add(doc_name)
            sources.append({
                "doc_name": doc_name,
                "preview": doc.page_content[:150],
            })

        return {
            "question": question,
            "answer": result["answer"],
            "sources": sources,
            "iterations": result["iteration"],
        }
