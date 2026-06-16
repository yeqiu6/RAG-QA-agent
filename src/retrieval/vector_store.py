"""
向量库管理器 —— 基于 Chroma，负责文档块的存储和检索
"""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.retrieval.embeddings import EmbeddingManager


class VectorStoreManager:
    """Chroma 向量库封装

    用法：
        store = VectorStoreManager()
        store.add_documents(chunks)                    # 入库
        results = store.search("年假有几天？", k=3)     # 检索
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.embeddings = EmbeddingManager()
        self._store: Chroma | None = None

    def _get_store(self) -> Chroma:
        """延迟初始化：第一次使用时才创建连接"""
        if self._store is None:
            self._store = Chroma(
                collection_name="enterprise_knowledge",
                embedding_function=self.embeddings._embeddings,
                persist_directory=str(self.persist_dir),
            )
        return self._store

    def add_documents(self, documents: List[Document]) -> None:
        """把文档块加入向量库（自动向量化 + 存储）"""
        store = self._get_store()
        store.add_documents(documents)
        print(f"💾 已入库 {len(documents)} 个文档块 → {self.persist_dir}")

    def search(self, query: str, k: int = 3) -> List[Document]:
        """语义搜索：输入自然语言问题，返回最相关的 k 个文档块"""
        store = self._get_store()
        results = store.similarity_search(query, k=k)
        return results

    def search_with_scores(self, query: str, k: int = 3) -> List[tuple]:
        """带相似度分数的检索"""
        store = self._get_store()
        results = store.similarity_search_with_relevance_scores(query, k=k)
        return results

    def delete_collection(self) -> None:
        """清空向量库（重新索引时用）"""
        store = self._get_store()
        store.delete_collection()
        self._store = None
        print("🗑️  向量库已清空")