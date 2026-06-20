"""
API 数据模型 —— 输入输出的格式约定
"""
from pydantic import BaseModel, Field
from typing import List, Optional


# ---- 请求：别人发给你什么 ----
class QuestionRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    k: int = Field(default=3, description="检索文档数量", ge=1, le=10)


class DocumentUploadRequest(BaseModel):
    """文档入库请求（预留，可扩展）"""
    path: str = Field(default="data/sample_docs", description="文档目录路径")


# ---- 响应：你返回给别人什么 ----
class SourceItem(BaseModel):
    doc_name: str = Field(..., description="文档名称")
    preview: str = Field(..., description="文档片段预览")


class AnswerResponse(BaseModel):
    question: str = Field(..., description="原始问题")
    answer: str = Field(..., description="AI 回答")
    sources: List[SourceItem] = Field(default_factory=list, description="参考来源")


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    version: str = Field(default="0.1.0")