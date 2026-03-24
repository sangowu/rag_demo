"""Analyze FinQA answer types and program complexity."""
import json, urllib.request
from collections import Counter

url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json"
with urllib.request.urlopen(url, timeout=30) as r:
    data = json.loads(r.read())

# 分类：direct extraction vs numerical calculation
direct, calc = [], []
ops_counter = Counter()

for row in data:
    qa = row["qa"]
    program = qa.get("program", "") or ""
    steps   = qa.get("steps", []) or []

    # program 为空或只有 "none" 表示直接抽取
    if not program or program.strip().lower() in ("", "none", "n/a"):
        direct.append(row)
    else:
        calc.append(row)
        # 统计用到了哪些运算
        for op in ["add", "subtract", "multiply", "divide", "greater", "exp", "table_sum", "table_average", "table_max", "table_min"]:
            if op in program:
                ops_counter[op] += 1

total = len(data)
print(f"Total: {total}")
print(f"Direct extraction : {len(direct):4d}  ({len(direct)/total*100:.1f}%)")
print(f"Numerical calc    : {len(calc):4d}  ({len(calc)/total*100:.1f}%)")

print(f"\nTop operations in calc questions:")
for op, cnt in ops_counter.most_common():
    print(f"  {op:20s}: {cnt}")

# 看几条直接抽取的例子
print("\n--- Direct extraction samples ---")
for row in direct[:3]:
    qa = row["qa"]
    print(f"  Q: {qa['question']}")
    print(f"  A: {qa['answer']}  |  gold_inds: {qa['gold_inds']}")
    print()

# 看几条计算题的例子
print("--- Numerical calc samples ---")
for row in calc[:3]:
    qa = row["qa"]
    print(f"  Q: {qa['question']}")
    print(f"  A: {qa['answer']}  |  program: {qa.get('program','')}")
    print(f"  gold_inds: {qa['gold_inds']}")
    print()
