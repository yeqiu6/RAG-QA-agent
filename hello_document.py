"""
阶段二：文档处理管线 — 完整测试

运行：uv run python hello_document.py
"""
from src.document.pipeline import DocumentProcessor

# 一行代码搞定：加载 → 分块 → 元数据增强
processor = DocumentProcessor(chunk_size=600, chunk_overlap=120)
chunks = processor.process("data/sample_docs")

# 检视每个块的元数据
print(f"\n{'='*50}")
print("文档块详情")
print("=" * 50)

for i, chunk in enumerate(chunks):
    print(f"\n--- 块 #{i} ---")
    print(f"文档名: {chunk.metadata['doc_name']}")
    print(f"文件类型: {chunk.metadata['file_type']}")
    print(f"预览: {chunk.metadata['preview']}...")
    print(f"处理时间: {chunk.metadata['processed_at']}")
