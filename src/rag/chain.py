"""
RAG Chain —— 检索 + 生成，一步到位
"""
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from src.rag.prompts import RAG_PROMPT, RAG_CHAT_PROMPT
from src.retrieval.vector_store import VectorStoreManager


def _format_docs(docs: List[Document]) -> str:
    """把检索到的文档块拼成一个大字符串，喂给 Prompt"""
    parts = []
    for i, doc in enumerate(docs):
        doc_name = doc.metadata.get("doc_name", "未知文档")
        parts.append(f"【文档{i+1}: {doc_name}】\n{doc.page_content}")
    return "\n\n".join(parts)


class RAGChain:
    """基础 RAG 问答链

    用法：
        rag = RAGChain(k=3)
        answer = rag.ask("年假有几天？")
    """

    def __init__(self, k: int = 3):
        self.k = k
        self._store = VectorStoreManager()
        self._llm = ChatOpenAI(
            model=QWEN_MODEL,
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            temperature=0,
        )
        self._chain = self._build_chain()

    def _build_chain(self):
        """用 LCEL 管道构建 RAG 链

        管道数据流：
            question → retriever → format_docs → prompt → llm → output_parser
                         ↑                  ↑         ↑      ↑
                    从Chroma搜文档     拼接成文本   填入模板  生成回答
        """
        # 包装检索器：question → List[Document]
        retriever = lambda question: self._store.search(question, k=self.k)

        # LCEL 管道
        chain = (
            {
                "context": lambda q: _format_docs(retriever(q)),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )
        return chain

    def ask(self, question: str) -> str:
        """同步问答"""
        return self._chain.invoke(question)

    async def aask(self, question: str) -> str:
        """异步问答（FastAPI 里用）"""
        return await self._chain.ainvoke(question)

    def ask_with_sources(self, question: str) -> dict:
        """问答 + 返回检索到的文档来源"""
        docs = self._store.search(question, k=self.k)
        answer = self._chain.invoke(question)

        sources = []
        for doc in docs:
            sources.append({
                "doc_name": doc.metadata.get("doc_name", "未知"),
                "source_file": doc.metadata.get("source_file", ""),
                "preview": doc.page_content[:150],
            })

        return {"question": question, "answer": answer, "sources": sources}