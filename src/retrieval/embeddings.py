"""
Embedding 管理器 —— 统一封装文本向量化接口
"""
from langchain_community.embeddings import DashScopeEmbeddings
from src.config import QWEN_API_KEY


class EmbeddingManager:
    """向量化引擎

    用法：
        emb = EmbeddingManager()
        vec = emb.embed_query("年假有几天？")       # 单个文本
        vecs = emb.embed_documents(["文档1", "文档2"]) # 批量文本
    """

    def __init__(self):
        self._embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=QWEN_API_KEY,
        )

    def embed_query(self, text: str) -> list[float]:
        """把一个问题转成向量"""
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把一批文档转成向量（批量处理，比逐个调更快）"""
        return self._embeddings.embed_documents(texts)