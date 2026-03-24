"""
evaluator.py
============
RAGAS 批量评测：对 FinQA eval 集运行完整 RAG 流程，输出基线指标。

评测指标：
  - faithfulness       : 答案是否忠实于检索内容
  - context_precision  : 检索内容是否与问题相关
  - answer_relevancy   : 答案是否切题

结果保存至 data/eval_results.json

Usage:
    python src/evaluator.py [--n 50] [--output data/eval_results.json]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from datasets import Dataset
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from src.bm25_store import BM25Store
from src.config import config
from src.data_loader import DataLoader
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore

_llm_cfg = config.get("llm", {})

_llm = ChatOpenAI(
    model=_llm_cfg.get("model", "Qwen/Qwen3-8B"),
    base_url=_llm_cfg.get("base_url", "https://api-inference.modelscope.cn/v1"),
    api_key=os.environ.get(_llm_cfg.get("api_key_env", "MODELSCOPE_API_KEY"), ""),
    temperature=_llm_cfg.get("temperature", 0.1),
    max_tokens=_llm_cfg.get("max_tokens", 1024),
)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the question using ONLY the provided context. "
    "Be concise and precise. If the context does not contain enough information, say so."
)


def _generate_answer(question: str, contexts: list[str]) -> str:
    """用 LLM 基于检索内容生成答案。"""

    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    human_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = _llm.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(human_prompt)])
    return response.content


def run_eval(n: int, output_path: Path) -> dict:
    """
    批量评测主函数。

    Args:
        n           : 评测样本数
        output_path : 结果保存路径

    Returns:
        RAGAS 评测分数字典
    """
    # 初始化组件
    retriever = Retriever(VectorStore(), BM25Store(), Reranker())

    # 加载 FinQA eval 记录
    _, eval_records = DataLoader().load()
    samples = eval_records[:n]

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, record in enumerate(tqdm(samples, desc="Evaluating")):
        question     = record.get("question", "")
        ground_truth = record.get("answer", "")
        chunks = retriever.search(question)
        contexts = [c["text"] for c in chunks]
        answer = _generate_answer(question, contexts)
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)


    # 构造 RAGAS Dataset
    dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts_list,
        "ground_truth": ground_truths,
    })

    result = evaluate(dataset, metrics=[faithfulness, context_precision, answer_relevancy])
    scores = result.to_pandas().mean().to_dict() 
    print(f"Score: {scores}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int,   default=50,
                        help="评测样本数（默认 50）")
    parser.add_argument("--output", type=str,   default="data/eval_results.json",
                        help="结果保存路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(n=args.n, output_path=_ROOT / args.output)
