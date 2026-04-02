"""
evaluator.py
============
RAGAS 批量评测：对 FinQA eval 集运行完整 RAG 流程，输出基线指标。

评测指标：
  - faithfulness       : 答案是否忠实于检索内容
  - context_precision  : 检索内容是否与问题相关
  - answer_relevancy   : 答案是否切题
  - hit@1/3/5          : 检索命中率
  - mrr@1/3/5          : 平均倒数排名
  - t_retrieval_s      : 检索阶段总耗时（秒）
  - t_generation_s     : 生成阶段总耗时（秒）
  - t_ragas_s          : RAGAS 评测耗时（秒）
  - prompt_tokens      : 生成阶段 prompt token 总量
  - completion_tokens  : 生成阶段 completion token 总量

结果保存至 data/eval_results.json，追加至 data/eval_log.jsonl

Usage:
    python src/evaluator.py [--n 10] [--tag baseline] [--output data/eval_results.json]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

from src.bm25_store import BM25Store
from src.config import config
from src.query_rewriter import QueryRewriter
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore

_EVAL_PATH_QA    = _ROOT / "data/results/qa_pairs.jsonl"
_EVAL_PATH_FINQA = _ROOT / "data/finqa/eval.jsonl"
_KS = [1, 3, 5]


class _BGEEmbeddings:
    """RAGAS-compatible embeddings wrapper，复用 FlagEmbedding BGEM3FlagModel。
    实现 embed_query / embed_documents 接口供 AnswerRelevancy 调用。
    """
    def __init__(self, model_name: str):
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(model_name, use_fp16=True)

    def embed_query(self, text: str) -> list[float]:
        result = self._model.encode_queries([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return result["dense_vecs"][0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self._model.encode_corpus(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return result["dense_vecs"].tolist()


_llm_cfg = config.get("llm", {})

_is_modelscope = "modelscope" in _llm_cfg.get("base_url", "")
_llm = ChatOpenAI(
    model=_llm_cfg.get("model", "Qwen/Qwen3-8B"),
    base_url=_llm_cfg.get("base_url", "https://api-inference.modelscope.cn/v1"),
    api_key=os.environ.get(_llm_cfg.get("api_key_env", "MODELSCOPE_API_KEY"), "") or "local",
    temperature=_llm_cfg.get("temperature", 0.1),
    max_tokens=_llm_cfg.get("max_tokens", 1024),
    extra_body={"enable_thinking": False} if _is_modelscope else {},
)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the question using ONLY the provided context. "
    "Be concise and precise. If the context does not contain enough information, say so."
)


def _generate_answer(question: str, contexts: list[str]) -> tuple[str, int, int]:
    """用 LLM 基于检索内容生成答案。

    Returns:
        (answer, prompt_tokens, completion_tokens)
    """
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    human_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = _llm.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(human_prompt)])
    usage = response.response_metadata.get("token_usage", {})
    return (
        response.content,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )


def _compute_retrieval_metrics(ranks: list[int | None]) -> dict:
    """计算 Hit@K 和 MRR@K。ranks 为每条样本 gold doc 的排名（1-based），未命中为 None。"""
    n = len(ranks)
    metrics = {}
    for k in _KS:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        rr   = sum(1.0 / r for r in ranks if r is not None and r <= k)
        metrics[f"hit@{k}"] = round(hits / n, 4)
        metrics[f"mrr@{k}"] = round(rr / n, 4)
    return metrics


def run_eval(n: int, output_path: Path, tag: str = "default", n_retrieval: int = None, use_qa_pairs: bool = False, meta_filter: bool = False, summary_filter: bool = False, query_rewrite: bool = False) -> dict:
    """
    批量评测主函数。三阶段分批执行，避免检索模型与 LLM 同时占用显存。

    Phase 1 — 批量检索：对 n_retrieval 条样本检索，计算 Hit@K / MRR@K；
               同时保留前 n 条的 contexts 供后续生成使用。完成后释放 GPU 显存。
    Phase 2 — 批量生成：对前 n 条样本调用 LLM API，记录 token 用量和耗时。
    Phase 3 — RAGAS 评测：对前 n 条样本计算 RAGAS 指标。

    Args:
        n           : RAGAS 评测 + 生成的样本数
        n_retrieval : Hit@K / MRR 评测的样本数（默认等于 n）
        output_path : 结果保存路径
        tag         : 实验标签
    """
    if n_retrieval is None:
        n_retrieval = n

    summary_store = None
    if summary_filter:
        from src.summary_store import SummaryStore
        summary_store = SummaryStore()
    rewriter = QueryRewriter() if query_rewrite else None
    retriever = Retriever(VectorStore(), BM25Store(), Reranker(), summary_store=summary_store, query_rewriter=rewriter)

    eval_path = _EVAL_PATH_QA if use_qa_pairs else _EVAL_PATH_FINQA
    with open(eval_path, encoding="utf-8") as f:
        eval_records = [json.loads(line) for line in f if line.strip()]

    # ------------------------------------------------------------------
    # Phase 1: 批量检索（BGE-M3 + Reranker 在 GPU）
    #   - 全量 n_retrieval 条用于计算 Hit@K / MRR@K
    #   - 前 n 条同时保存 contexts 供生成使用
    # ------------------------------------------------------------------
    retrieved = []   # (question, ground_truth, contexts)，仅前 n 条
    ranks = []       # gold doc 排名（1-based），未命中为 None，共 n_retrieval 条

    t0 = time.time()
    for i, record in enumerate(tqdm(eval_records[:n_retrieval], desc="[1/3] Retrieving")):
        question     = record.get("question", "")
        ground_truth = str(record.get("exe_ans", "") or record.get("answer", ""))
        # 兼容旧版 qa_pairs（doc_id 用 path.stem，缺少 .pdf 后缀）
        gold_id = record.get("doc_id", "")
        if gold_id and not gold_id.endswith(".pdf"):
            gold_id = gold_id + ".pdf"

        chunks = retriever.search(question, top_k=max(_KS), rewrite=query_rewrite, meta_filter=meta_filter, summary_filter=summary_filter)

        if i < n:
            contexts = [c["text"] for c in chunks]
            retrieved.append((question, ground_truth, contexts))

        retrieved_ids = [c.get("doc_id", "") for c in chunks]
        if gold_id and gold_id in retrieved_ids:
            ranks.append(retrieved_ids.index(gold_id) + 1)
        else:
            ranks.append(None)

    t_retrieval = round(time.time() - t0, 2)

    # 检索完成，释放 GPU 显存
    retriever._vs.offload()
    retriever._reranker.offload()

    retrieval_metrics = _compute_retrieval_metrics(ranks)

    # ------------------------------------------------------------------
    # Phase 2: 批量生成（ModelScope API，GPU 零占用）
    # ------------------------------------------------------------------
    ragas_samples = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    t0 = time.time()
    for question, ground_truth, contexts in tqdm(retrieved, desc="[2/3] Generating"):
        answer, pt, ct = _generate_answer(question, contexts)
        total_prompt_tokens     += pt
        total_completion_tokens += ct
        ragas_samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))
    t_generation = round(time.time() - t0, 2)

    # ------------------------------------------------------------------
    # Phase 3: RAGAS 评测
    # ------------------------------------------------------------------
    dataset = EvaluationDataset(samples=ragas_samples)

    _api_key = os.environ.get(_llm_cfg.get("api_key_env", "MODELSCOPE_API_KEY"), "") or "local"
    _is_modelscope = "modelscope" in _llm_cfg.get("base_url", "")
    _ragas_chat = ChatOpenAI(
        model=_llm_cfg.get("model", "Qwen/Qwen3-8B"),
        base_url=_llm_cfg.get("base_url", "https://api-inference.modelscope.cn/v1"),
        api_key=_api_key,
        temperature=0,
        max_tokens=8192,   # RAGAS faithfulness 输出较长，需要足够的 token 空间
        extra_body={"enable_thinking": False} if _is_modelscope else {},
    )
    ragas_llm = LangchainLLMWrapper(_ragas_chat)
    _vs_cfg = config.get("vector_store", {})
    emb_model = _vs_cfg.get("embedding_model_path") or _vs_cfg.get("embedding_model", "BAAI/bge-m3")
    ragas_emb = _BGEEmbeddings(emb_model)

    metrics = [
        Faithfulness(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    ]

    t0 = time.time()
    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=RunConfig(timeout=300, max_retries=2),
    )
    t_ragas = round(time.time() - t0, 2)

    ragas_scores = result.to_pandas().select_dtypes(include="number").mean().to_dict()

    # ------------------------------------------------------------------
    # 汇总所有指标
    # ------------------------------------------------------------------
    all_scores = {
        **ragas_scores,
        **retrieval_metrics,
        "t_retrieval_s":    t_retrieval,
        "t_generation_s":   t_generation,
        "t_ragas_s":        t_ragas,
        "t_total_s":        round(t_retrieval + t_generation + t_ragas, 2),
        "prompt_tokens":    total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "avg_prompt_tokens":     round(total_prompt_tokens / n, 1),
        "avg_completion_tokens": round(total_completion_tokens / n, 1),
    }

    print("\n=== Eval Results ===")
    for k, v in all_scores.items():
        print(f"  {k:<26}: {v}")

    # 保存最新结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)

    # 追加到实验日志
    log_entry = {
        "time":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tag":            tag,
        "n":              n,
        "n_retrieval":    n_retrieval,
        "retriever_mode": config.get("retriever", {}).get("mode"),
        "alpha":          config.get("retriever", {}).get("custom", {}).get("alpha"),
        "candidate_k":    config.get("retriever", {}).get("custom", {}).get("candidate_k"),
        "meta_filter":    meta_filter,
        "summary_filter": summary_filter,
        "query_rewrite":  query_rewrite,
        "chunk_strategy": config.get("chunking", {}).get("strategy"),
        "chunk_size":     config.get("chunking", {}).get("chunk_size"),
        "chunk_overlap":  config.get("chunking", {}).get("overlap"),
        **all_scores,
    }
    log_path = output_path.parent / "eval_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    print(f"\nLog appended → {log_path}")

    return all_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=10,
                        help="RAGAS 评测样本数（默认 10）")
    parser.add_argument("--n-retrieval", type=int, default=None,
                        help="Hit@K/MRR 评测样本数（默认与 --n 相同）")
    parser.add_argument("--tag",         type=str, default="default",
                        help="实验标签（如 baseline / alpha0.6 / chunk256）")
    parser.add_argument("--qa-pairs",    action="store_true",
                        help="使用自己生成的 qa_pairs.jsonl（默认使用 FinQA eval.jsonl）")
    parser.add_argument("--meta-filter",    action="store_true",
                        help="从 query 提取 ticker/year 作为 ChromaDB 预过滤")
    parser.add_argument("--summary-filter", action="store_true",
                        help="用 LLM summary 两阶段检索：先找相关文档，再搜 chunk")
    parser.add_argument("--query-rewrite",  action="store_true",
                        help="对每条 query 用 LLM 展开 ticker/模糊公司名再检索")
    parser.add_argument("--output",      type=str, default="data/eval_results.json",
                        help="结果保存路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(n=args.n, output_path=_ROOT / args.output, tag=args.tag,
             n_retrieval=args.n_retrieval, use_qa_pairs=args.qa_pairs,
             meta_filter=args.meta_filter, summary_filter=args.summary_filter,
             query_rewrite=args.query_rewrite)
