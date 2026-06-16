"""
全局配置 —— 所有环境变量和设置都从这里读取

企业项目惯例：
- 敏感信息（Key）放 .env 文件，不写死在代码里
- config.py 负责读取，其他模块 from src.config import config 直接用
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 自动加载项目根目录的 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")


# ---- 千问 API 配置 ----
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL = "tongyi-xiaomi-analysis-flash"

# ---- 快速检查 ----
if not QWEN_API_KEY:
    print("⚠️  未检测到 QWEN_API_KEY！")
    print("   请在项目根目录创建 .env 文件，写入：")
    print("   QWEN_API_KEY=sk-你的key")
    print("   QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1")
