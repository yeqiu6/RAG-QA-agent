"""
RAG 评估测试数据集
"""
from dataclasses import dataclass
from typing import List


@dataclass
class TestCase:
    question: str
    ground_truth: str  # 基于文档的标准答案


# 基于 data/sample_docs 的测试用例
TEST_CASES: List[TestCase] = [
    TestCase(
        question="加班费怎么算？",
        ground_truth="工作日加班按1.5倍工资计算，周末加班按2倍计算，法定节假日按3倍计算。每月加班不超过36小时，需提前一天在OA提交申请。"
    ),
    TestCase(
        question="公司密码有什么要求？",
        ground_truth="密码长度不少于12位，须包含大写字母、小写字母、数字及特殊字符。密码有效期为90天，到期系统提示修改。"
    ),
    TestCase(
        question="出差住宿能报销多少钱？",
        ground_truth="一线城市每晚不超过600元，二线城市不超过450元，其他城市不超过300元，超出标准部分个人承担。"
    ),
    TestCase(
        question="怎么报销差旅费？",
        ground_truth="出差结束后7个工作日内通过OA系统提交报销申请，上传票据照片或电子发票。单笔超过5000元需部门经理审批，超过20000元需总监加签。财务部收到完整材料后5个工作日内完成审核，审核通过后3个工作日内打入工资卡。"
    ),
    TestCase(
        question="公司允许养宠物吗？",
        ground_truth="现有资料中未涉及宠物相关的规定。"
    ),
]