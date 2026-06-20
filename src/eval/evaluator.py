"""
RAG 系统评估器 —— 不依赖外部评分框架，直接测三个核心指标
"""
import re
from src.rag.chain import RAGChain


class RAGEvaluator:
    """实用 RAG 评估器

    三个核心指标：
      1. 检索命中率 — 文档能不能搜到相关材料
      2. 信息覆盖率 — 回答有没有包含关键信息
      3. 幻觉检测   — 回答里有没有文档里没有的内容
    """

    def __init__(self):
        self._rag = RAGChain(k=3)

    @staticmethod
    def _extract_keywords(text: str, min_len: int = 2) -> set:
        """从文本提取关键词（数字、专有名词、关键概念）"""
        keywords = set()
        # 数字相关：1.5倍、90天、12位、5000元 等
        for m in re.finditer(r'\d+(?:\.\d+)?\s*(?:倍|天|位|元|小时|年|月|日|工作日|个)', text):
            keywords.add(m.group())
        # 短关键概念：OA系统、企业微信、部门经理 等
        for m in re.finditer(r'[A-Z一-鿿]{2,8}(?:系统|部门|审批|政策|制度|规定|流程|标准)', text):
            keywords.add(m.group())
        return keywords if keywords else set(text[:50])

    def evaluate(self, test_cases) -> dict:
        """跑评估，逐条输出结果"""
        results = []

        print(f"正在测试 {len(test_cases)} 个用例...\n")
        for i, tc in enumerate(test_cases):
            result = self._rag.ask_with_sources(tc.question)
            answer = result["answer"]
            sources = result["sources"]

            # ---- 指标 1: 检索命中 ----
            # 检索到的文档名里是否包含期望的关键信息
            source_text = " ".join(s["doc_name"] + s["preview"] for s in sources)
            gt_words = self._extract_keywords(tc.ground_truth)
            hit_count = sum(1 for w in gt_words if w in source_text)
            retrieval_hit = hit_count / max(len(gt_words), 1) if gt_words else 0

            # ---- 指标 2: 信息覆盖 ----
            # 回答里包含了标准答案里的多少关键信息
            answer_hit = sum(1 for w in gt_words if w in answer)
            info_coverage = answer_hit / max(len(gt_words), 1) if gt_words else 0

            # ---- 指标 3: 有无幻觉 ----
            # 回答里有，但在检索到的文档里找不到实质内容 → 可疑
            has_hallucination = False
            # 简单检测：如果回答了具体数字但来源里没有
            answer_numbers = set(re.findall(r'\d+(?:\.\d+)?', answer))
            source_numbers = set(re.findall(r'\d+(?:\.\d+)?', source_text))
            extra_numbers = answer_numbers - source_numbers
            if extra_numbers and len(extra_numbers) > 2:
                has_hallucination = True

            results.append({
                "question": tc.question,
                "answer": answer[:300],
                "hit_docs": [s["doc_name"] for s in sources],
                "retrieval_hit": round(retrieval_hit, 3),
                "info_coverage": round(info_coverage, 3),
                "has_hallucination": has_hallucination,
            })

            print(f"  [{i+1}/{len(test_cases)}] {tc.question}")
            print(f"    检索: {[s['doc_name'] for s in sources]}")
            print(f"    命中率={retrieval_hit:.0%}  覆盖率={info_coverage:.0%}  幻觉={'⚠️' if has_hallucination else '✅'}")
            print()

        # 汇总
        avg_retrieval = sum(r["retrieval_hit"] for r in results) / len(results)
        avg_coverage = sum(r["info_coverage"] for r in results) / len(results)
        hallucination_count = sum(1 for r in results if r["has_hallucination"])

        return {
            "results": results,
            "avg_retrieval_hit": round(avg_retrieval, 3),
            "avg_info_coverage": round(avg_coverage, 3),
            "hallucination_count": hallucination_count,
            "total": len(results),
        }

    def print_report(self, scores: dict) -> None:
        """打印评估报告"""
        print("=" * 55)
        print("📊 RAG 系统评估报告")
        print("=" * 55)

        for i, r in enumerate(scores["results"]):
            h = "⚠️ 疑似幻觉" if r["has_hallucination"] else "✅"
            print(f"\n用例 {i+1}: {r['question']}")
            print(f"  检索命中: {r['retrieval_hit']:.0%} | 信息覆盖: {r['info_coverage']:.0%} | 幻觉: {h}")
            print(f"  回答: {r['answer'][:150]}...")

        print(f"\n{'='*55}")
        print(f"📈 汇总 (共 {scores['total']} 题):")
        print(f"   平均检索命中率: {scores['avg_retrieval_hit']:.0%}")
        print(f"   平均信息覆盖率: {scores['avg_info_coverage']:.0%}")
        print(f"   疑似幻觉数量:   {scores['hallucination_count']}/{scores['total']}")
        print("=" * 55)
        print()
        print("💡 指标说明：")
        print("   检索命中率: 检索阶段是否拿到了相关文档")
        print("   信息覆盖率: 生成阶段是否覆盖了关键信息点")
        print("   幻觉检测:   回答是否包含文档里没有的内容")
