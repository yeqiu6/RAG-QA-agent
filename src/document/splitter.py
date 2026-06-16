"""
文档分割器 —— 把长文档切成适合检索的小块
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitter:
    """智能文档分割器

    使用 RecursiveCharacterTextSplitter：
    - 优先按段落分隔符切（\n\n）
    - 再按句子分隔符切（。！？）
    - 最后按字符切
    - 块之间保留重叠，避免关键信息被切断
    """

    def __init__(
        self,
        chunk_size: int = 800,      # 每块最大字符数
        chunk_overlap: int = 150,   # 块之间重叠字符数
        separators: List[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 中文文档的推荐分隔符优先级
        self.separators = separators or [
            "\n\n",    # 段落
            "\n",      # 行
            "。",      # 句号
            "！",      # 感叹号
            "？",      # 问号
            "；",      # 分号
            "，",      # 逗号
            " ",       # 空格
            "",        # 最后手段：逐字符
        ]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """分割文档列表，返回文档块列表"""
        chunks = self._splitter.split_documents(documents)

        # 给每个块补充元数据：块序号、属于原文件的第几段
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        print(f"✂️  将 {len(documents)} 份文档切分为 {len(chunks)} 个块")
        print(f"   块大小: {self.chunk_size} 字符, 重叠: {self.chunk_overlap} 字符")
        return chunks
