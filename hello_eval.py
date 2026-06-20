"""
阶段七：RAGAS 评估

运行：uv run python hello_eval.py
"""
from src.eval.test_data import TEST_CASES
from src.eval.evaluator import RAGEvaluator

print("=" * 50)
print("阶段七：RAG 系统评估")
print("=" * 50)

evaluator = RAGEvaluator()
scores = evaluator.evaluate(TEST_CASES)
evaluator.print_report(scores)