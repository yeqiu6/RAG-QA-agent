"""
课时 1.2：第一次调用大模型

目标：用通义千问生成第一句话，理解：
  1. 怎么创建模型连接
  2. 什么是 SystemMessage / HumanMessage
  3. Token 是什么
"""

# ---- 第一步：导入 ----
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL

# ---- 第二步：创建模型连接 ----
# ChatOpenAI 是 LangChain 提供的统一接口
# 千问兼容 OpenAI 格式，所以用同一个类，只改 base_url 就行
llm = ChatOpenAI(
    model=QWEN_MODEL,          # 用哪个模型
    api_key=QWEN_API_KEY,      # API Key（从 .env 读的）
    base_url=QWEN_BASE_URL,    # 千问的地址
    temperature=0.7,           # 0=古板, 1=天马行空, 问答场景建议 0~0.3
)

# ---- 第三步：最简单的调用 ----
print("=" * 50)
print("示例 1：最简单的调用")
print("=" * 50)

response = llm.invoke("用一句话介绍什么是RAG")
print(f"回答：{response.content}")
print()

# ---- 第四步：带角色的对话 ----
# SystemMessage = 设定 AI 的行为和角色
# HumanMessage  = 用户说的话
print("=" * 50)
print("示例 2：带系统角色设定的对话")
print("=" * 50)

messages = [
    SystemMessage(content="你是专业的企业知识库助手，回答要准确简洁。不知道就直接说不知道。"),
    HumanMessage(content="公司的年假政策是什么？"),
]
response = llm.invoke(messages)
print(f"回答：{response.content}")
print()

# ---- 第五步：理解 Token ----
# Token 是模型计费的基本单位
# 中文：1个汉字 ≈ 1-2 个 token
# 英文：1个单词 ≈ 1.3 个 token
print("=" * 50)
print("示例 3：看看 Token 是怎么算的")
print("=" * 50)

import tiktoken
encoder = tiktoken.get_encoding("cl100k_base")

text = "RAG是检索增强生成技术，它结合了信息检索和文本生成。"
tokens = len(encoder.encode(text))
print(f"文本：{text}")
print(f"字符数：{len(text)}")
print(f"Token数：{tokens}")
print(f"💡 千问 Turbo 上下文窗口：1,000,000 tokens ≈ 三本《三体》")
print()

print("✅ hello_llm.py 完成！如果你看到上面有正常的 AI 回答，说明环境全部 OK。")
