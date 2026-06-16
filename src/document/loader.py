"""
文档加载器 —— 把各种格式的文件读进来，统一成 Document 对象
"""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader


class DocumentLoader:
    """多格式文档加载器

    使用方式：
        loader = DocumentLoader()
        docs = loader.load("data/sample_docs")  # 加载整个目录
        docs = loader.load_file("data/sample_docs/政策.md")  # 加载单个文件
    """

    # 文件后缀 → 用什么 Loader 处理
    LOADER_MAP = {
        ".txt": TextLoader,
        ".md": TextLoader,     # Markdown 本质是纯文本，TextLoader 就够用
        ".pdf": PyPDFLoader,
    }

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def load(self, path: str | Path) -> List[Document]:
        """加载目录下所有支持的文档，返回统一的 Document 列表"""
        path = Path(path)
        all_docs = []

        if path.is_file():
            return self.load_file(path)

        if not path.is_dir():
            raise FileNotFoundError(f"路径不存在: {path}")

        # 遍历目录，按后缀匹配对应的 loader
        for ext, loader_cls in self.LOADER_MAP.items():
            for file_path in path.glob(f"**/*{ext}"):
                # 跳过隐藏文件和临时文件
                if file_path.name.startswith("~") or file_path.name.startswith("."):
                    continue
                try:
                    docs = self._load_single(file_path, loader_cls)
                    # 给每个文档打上来源标记
                    for doc in docs:
                        doc.metadata["source_file"] = str(file_path)
                        doc.metadata["file_type"] = ext
                    all_docs.extend(docs)
                    print(f"  ✅ 已加载: {file_path.name}")
                except Exception as e:
                    print(f"  ❌ 加载失败 {file_path.name}: {e}")

        print(f"📄 共加载 {len(all_docs)} 份文档片段")
        return all_docs

    def load_file(self, file_path: str | Path) -> List[Document]:
        """加载单个文件"""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in self.LOADER_MAP:
            raise ValueError(f"不支持的文件格式: {ext}，支持的格式: {list(self.LOADER_MAP.keys())}")

        return self._load_single(file_path, self.LOADER_MAP[ext])

    def _load_single(self, file_path: Path, loader_cls) -> List[Document]:
        """用指定的 loader 加载单个文件"""
        # txt 文件需要指定编码
        if loader_cls == TextLoader:
            loader = loader_cls(str(file_path), encoding=self.encoding)
        else:
            loader = loader_cls(str(file_path))
        return loader.load()
