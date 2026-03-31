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
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

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
    批量评测主函数。三阶段分批执行，避免检索模型与 LLM 同时占用显存。

    Phase 1 — 批量检索：BGE-M3 + Reranker 在 GPU，生成完毕后立即释放。
    Phase 2 — 批量生成：LLM via API，GPU 零占用。
    Phase 3 — RAGAS 评测：LLM via API + BGE-M3 embeddings（仅 AnswerRelevancy 用到）。

    Args:
        n           : 评测样本数
        output_path : 结果保存路径

    Returns:
        RAGAS 评测分数字典
    """
    retriever = Retriever(VectorStore(), BM25Store(), Reranker())

    with open(_EVAL_PATH, encoding="utf-8") as f:
        eval_records = [json.loads(line) for line in f if line.strip()]
    samples = eval_records[:n]

    # ------------------------------------------------------------------
    # Phase 1: 批量检索（BGE-M3 + Reranker 在 GPU）
    # ------------------------------------------------------------------
    retrieved = []
    for record in tqdm(samples, desc="[1/3] Retrieving"):
        question     = record.get("question", "")
        ground_truth = record.get("answer", "")
        chunks   = retriever.search(question)
        contexts = [c["text"] for c in chunks]
        retrieved.append((question, ground_truth, contexts))

    # 检索完成，释放 GPU 显存
    retriever._vs.offload()
    retriever._reranker.offload()

    # ------------------------------------------------------------------
    # Phase 2: 批量生成（ModelScope API，GPU 零占用）
    # ------------------------------------------------------------------
    ragas_samples = []
    for question, ground_truth, contexts in tqdm(retrieved, desc="[2/3] Generating"):
        answer = _generate_answer(question, contexts)
        ragas_samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))

    dataset = EvaluationDataset(samples=ragas_samples)

    # RAGAS LLM: 复用已配置的 _llm（含 enable_thinking=False），用 LangchainLLMWrapper 包装
    ragas_llm = LangchainLLMWrapper(_llm)

    # RAGAS Embeddings: 用 LangchainEmbeddingsWrapper 包装 HuggingFace 模型
    from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
    _vs_cfg = config.get("vector_store", {})
    emb_model = _vs_cfg.get("embedding_model_path") or _vs_cfg.get("embedding_model", "BAAI/bge-m3")
    ragas_emb = LangchainEmbeddingsWrapper(LCHuggingFaceEmbeddings(model_name=emb_model))

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
