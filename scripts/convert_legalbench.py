"""
convert_legalbench.py
=====================
将 LegalBench-RAG 数据转换为项目标准格式。

输入：D:/Python_Projects/legalbenchrag/data/
  corpus/     — 原始法律文本（.txt）
  benchmarks/ — JSON 评测文件

输出：data/legalbench/
  docs/       — 法律文本（.md），每文件一个 doc_id
  eval.jsonl  — 评测集，含 char-level ground truth

doc_id 格式：{subdir}__{filename_without_ext}
  e.g. contractnli__CopAcc_NDA-and-ToP-Mentors_2.0_2017

eval.jsonl 格式：
  {"question": "...", "doc_id": "...", "snippets": [{"char_start": 100, "char_end": 500}]}

Usage:
    python scripts/convert_legalbench.py
    python scripts/convert_legalbench.py --n-per-dataset 194 --seed 42
"""

import argparse
import json
import random
import re
import shutil
import unicodedata
from pathlib import Path

_LEGALBENCH_ROOT = Path("D:/Python_Projects/legalbenchrag/data")
_CORPUS_DIR = _LEGALBENCH_ROOT / "corpus"
_BENCHMARKS_DIR = _LEGALBENCH_ROOT / "benchmarks"

_OUT_ROOT = Path(__file__).parent.parent / "data" / "legalbench"
_DOCS_DIR = _OUT_ROOT / "docs"
_EVAL_PATH = _OUT_ROOT / "eval.jsonl"

_DATASETS = ["contractnli", "cuad", "maud", "privacy_qa"]


def _file_path_to_doc_id(file_path: str) -> str:
    """contractnli/foo bar.txt → contractnli__foo_bar"""
    path = Path(file_path)
    subdir = path.parent.name
    stem = re.sub(r"[^\w\-]", "_", path.stem)
    return f"{subdir}__{stem}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-dataset", type=int, default=194,
                        help="每个数据集采样的查询数（默认 194，共 776）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true",
                        help="不采样，使用全量数据")
    args = parser.parse_args()

    _DOCS_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    all_records: list[dict] = []
    copied_files: set[str] = set()

    for dataset in _DATASETS:
        bench_path = _BENCHMARKS_DIR / f"{dataset}.json"
        with open(bench_path, encoding="utf-8") as f:
            data = json.load(f)
        tests = data["tests"]

        # 过滤掉语料文件不存在的 query（如 maud 中含 || 的 Amendment 附件）
        def _corpus_exists(test: dict) -> bool:
            fp = test["snippets"][0]["file_path"]
            if "|" in fp:
                return False
            src = _CORPUS_DIR / unicodedata.normalize("NFC", fp)
            try:
                return src.exists()
            except Exception:
                return False

        tests = [t for t in tests if _corpus_exists(t)]

        if not args.all:
            tests = random.sample(tests, min(args.n_per_dataset, len(tests)))

        print(f"[{dataset}] {len(tests)} queries")

        for test in tests:
            query = test["query"]
            snippets_raw = test["snippets"]

            # 所有 snippet 来自同一文件（已验证 cross-file=0）
            file_path = snippets_raw[0]["file_path"]
            doc_id = _file_path_to_doc_id(file_path)

            snippets = [
                {"char_start": s["span"][0], "char_end": s["span"][1]}
                for s in snippets_raw
            ]

            all_records.append({
                "question": query,
                "doc_id": doc_id,
                "snippets": snippets,
            })

            # 拷贝语料文件（NFC 规范化处理含组合字符的文件名）
            if file_path not in copied_files:
                src = _CORPUS_DIR / unicodedata.normalize("NFC", file_path)
                dst = _DOCS_DIR / f"{doc_id}.md"
                shutil.copy2(src, dst)
                copied_files.add(file_path)

    # 写 eval.jsonl
    with open(_EVAL_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n完成：{len(all_records)} 条查询，{len(copied_files)} 个文档")
    print(f"docs  → {_DOCS_DIR}")
    print(f"eval  → {_EVAL_PATH}")


if __name__ == "__main__":
    main()
