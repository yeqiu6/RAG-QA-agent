"""
文档处理管道 —— 加载 + 分块 + 元数据增强，一站式处理
"""
from pathlib import Path
from typing import List
from datetime import datetime
from langchain_core.documents import Document
from src.document.loader import DocumentLoader
from src.document.splitter import DocumentSplitter


class DocumentProcessor:
    """文档处理管道

    用法：
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=150)
        chunks = processor.process("data/sample_docs")
        # 返回统一的、带元数据的文档块列表
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        encoding: str = "utf-8",
    ):
        self.loader = DocumentLoader(encoding=encoding)
        self.splitter = DocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process(self, path: str | Path) -> List[Document]:
        """主流程：加载 → 分块 → 元数据增强"""
        print(f"\n{'='*50}")
        print(f"📂 开始处理文档: {path}")
        print(f"{'='*50}")

        # 第1步：加载
        raw_docs = self.loader.load(path)
        if not raw_docs:
            print("⚠️  未找到支持的文档")
            return []

        # 第2步：分块
        chunks = self.splitter.split(raw_docs)

        # 第3步：元数据增强
        chunks = self._enrich_metadata(chunks)

        # 第4步：打印统计
        self._print_stats(chunks)
        return chunks

    def _enrich_metadata(self, chunks: List[Document]) -> List[Document]:
        """给每个文档块补充有用的元数据"""
        for chunk in chunks:
            # 处理时间戳
            chunk.metadata["processed_at"] = datetime.now().isoformat()

            # 从文件名提取文档分类（如 "员工手册_考勤制度" → "考勤制度"）
            source = chunk.metadata.get("source_file", "")
            fname = Path(source).stem  # 去掉路径和后缀
            chunk.metadata["doc_name"] = fname

            # 提取内容摘要（前100个字符，用于快速预览）
            chunk.metadata["preview"] = chunk.page_content[:100].replace("\n", " ")

        return chunks

    def _print_stats(self, chunks: List[Document]) -> None:
        """打印处理统计"""
        # 统计来源文件数
        sources = set()
        total_chars = 0
        for c in chunks:
            sources.add(c.metadata.get("source_file", ""))
            total_chars += len(c.page_content)

        print(f"\n📊 处理统计：")
        print(f"   源文件数: {len(sources)}")
        print(f"   文档块数: {len(chunks)}")
        print(f"   总字符数: {total_chars}")
        print(f"   平均块大小: {total_chars // max(len(chunks), 1)} 字符")
