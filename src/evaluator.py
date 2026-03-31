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

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

from src.bm25_store import BM25Store
from src.config import config
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore

_EVAL_PATH = Path(__file__).parent.parent / "data/finqa/eval.jsonl"

_llm_cfg = config.get("llm", {})

_llm = ChatOpenAI(
    model=_llm_cfg.get("model", "Qwen/Qwen3-8B"),
    base_url=_llm_cfg.get("base_url", "https://api-inference.modelscope.cn/v1"),
    api_key=os.environ.get(_llm_cfg.get("api_key_env", "MODELSCOPE_API_KEY"), ""),
    temperature=_llm_cfg.get("temperature", 0.1),
    max_tokens=_llm_cfg.get("max_tokens", 1024),
    extra_body={"enable_thinking": False},
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

    with open(_EVAL_PATH, encoding="utf-8") as f:
        eval_records = [json.loads(line) for line in f if line.strip()]
    samples = eval_records[:n]

    ragas_samples = []

    for record in tqdm(samples, desc="Evaluating"):
        question     = record.get("question", "")
        ground_truth = record.get("answer", "")
        chunks   = retriever.search(question)
        contexts = [c["text"] for c in chunks]
        answer   = _generate_answer(question, contexts)
        ragas_samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))

    dataset = EvaluationDataset(samples=ragas_samples)

    _openai_client = OpenAI(
        base_url=_llm_cfg.get("base_url", "https://api-inference.modelscope.cn/v1"),
        api_key=os.environ.get(_llm_cfg.get("api_key_env", "MODELSCOPE_API_KEY"), ""),
    )
    # Monkey-patch: inject enable_thinking=False for every RAGAS call (required by ModelScope/Qwen3)
    _orig_create = _openai_client.chat.completions.create
    def _patched_create(*args, **kwargs):
        extra = kwargs.get("extra_body") or {}
        extra["enable_thinking"] = False
        kwargs["extra_body"] = extra
        return _orig_create(*args, **kwargs)
    _openai_client.chat.completions.create = _patched_create

    ragas_llm = llm_factory(
        model=_llm_cfg.get("model", "Qwen/Qwen3-8B"),
        client=_openai_client,
    )
    _vs_cfg = config.get("vector_store", {})
    emb_model = _vs_cfg.get("embedding_model_path") or _vs_cfg.get("embedding_model", "BAAI/bge-m3")
    ragas_emb = HuggingFaceEmbeddings(model=emb_model)
    metrics = [
        Faithfulness(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    ]
    result = evaluate(dataset, metrics=metrics)
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
