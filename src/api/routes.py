"""
API 路由 —— 对外暴露的接口
"""
from fastapi import APIRouter, HTTPException
from src.api.schemas import QuestionRequest, AnswerResponse, HealthResponse, SourceItem
from src.rag.chain import RAGChain
from src.agent.graph import AgenticRAG

router = APIRouter()

# 全局单例（启动时初始化一次，避免每次请求都重建）
_rag_chain: RAGChain | None = None
_agent: AgenticRAG | None = None


def get_rag() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain(k=3)
    return _rag_chain


def get_agent() -> AgenticRAG:
    global _agent
    if _agent is None:
        _agent = AgenticRAG(k=3, max_iterations=3)
    return _agent


# ---- 接口 1：健康检查 ----
@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """服务是否正常"""
    return HealthResponse()


# ---- 接口 2：基础 RAG 问答 ----
@router.post("/qa", response_model=AnswerResponse, tags=["问答"])
async def ask_question(req: QuestionRequest):
    """基础 RAG：检索文档 → 生成回答"""
    try:
        rag = get_rag()
        result = rag.ask_with_sources(req.question)
        return AnswerResponse(
            question=req.question,
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- 接口 3：Agent 智能问答 ----
@router.post("/agent/qa", response_model=AnswerResponse, tags=["问答"])
async def ask_agent(req: QuestionRequest):
    """Agent RAG：智能分析 → 多轮检索 → 生成回答"""
    try:
        agent = get_agent()
        result = agent.run(req.question)
        return AnswerResponse(
            question=req.question,
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))