"""
FastAPI 服务入口

启动：uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router as api_router

app = FastAPI(
    title="企业知识库 RAG 助手",
    description="基于 LangChain + LangGraph 的企业级 RAG 智能检索 API",
    version="0.1.0",
)

# 允许跨域（前端可以调这个 API）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    print("🚀 企业知识库 RAG 助手启动成功！")
    print("📖 API 文档: http://localhost:8000/docs")