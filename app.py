"""
企业知识库 RAG 问答 — Web 界面
运行：uv run streamlit run app.py
"""
import streamlit as st
from src.rag.chain import RAGChain


# 页面配置
st.set_page_config(
    page_title="企业知识库助手",
    page_icon="📚",
    layout="wide",
)

st.title("📚 企业知识库 RAG 助手")
st.caption("基于内部文档的智能问答 — 回答均有据可查")

# 初始化 RAG 链（只创建一次，用缓存）
@st.cache_resource
def get_rag():
    return RAGChain(k=3)


rag = get_rag()

# 输入框
question = st.chat_input("输入你的问题，例如：加班费怎么算？")

if question:
    # 显示用户消息
    with st.chat_message("user"):
        st.write(question)

    # 显示 AI 回答
    with st.chat_message("assistant"):
        with st.spinner("正在检索文档..."):
            result = rag.ask_with_sources(question)

        st.write(result["answer"])

        # 展开显示参考来源
        with st.expander("📚 参考来源"):
            for i, s in enumerate(result["sources"]):
                st.markdown(f"**{i+1}. {s['doc_name']}**")
                st.caption(s["preview"] + "...")