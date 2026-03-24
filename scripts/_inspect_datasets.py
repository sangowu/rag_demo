"""Inspect FinQA and TAT-QA dataset formats via GitHub raw JSON."""
import json
import urllib.request

# ── FinQA ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("FinQA  (czyssrs/FinQA)")
print("=" * 60)
url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json"
with urllib.request.urlopen(url, timeout=30) as r:
    data = json.loads(r.read())

print(f"Total records: {len(data)}")
row = data[0]
print(f"Top-level keys: {list(row.keys())}")
print()

# pre_text / post_text
print("[pre_text] (first 3 items):", row.get("pre_text", [])[:3])
print("[post_text] (first 3 items):", row.get("post_text", [])[:3])
print()

# table
print("[table] (first 3 rows):", row.get("table", [])[:3])
print()

# qa block
qa = row.get("qa", {})
print("[qa] keys:", list(qa.keys()))
print("  question:", qa.get("question", ""))
print("  answer:", qa.get("answer", ""))
print("  gold_inds:", qa.get("gold_inds", {}))
print("  exe_ans:", qa.get("exe_ans", ""))
print()

# annotation
print("[annotation]:", row.get("annotation", ""))
print("[id]:", row.get("id", ""))


# ── TAT-QA ─────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TAT-QA  (NExTplusplus/TAT-QA)")
print("=" * 60)
url2 = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_train.json"
with urllib.request.urlopen(url2, timeout=30) as r:
    data2 = json.loads(r.read())

print(f"Total paragraphs/docs: {len(data2)}")
doc = data2[0]
print(f"Top-level keys: {list(doc.keys())}")
print()

# table
tbl = doc.get("table", {})
print("[table] keys:", list(tbl.keys()))
print("  uid:", tbl.get("uid", ""))
print("  table (first 3 rows):", tbl.get("table", [])[:3])
print()

# paragraphs
paras = doc.get("paragraphs", [])
print(f"[paragraphs] count: {len(paras)}")
if paras:
    p = paras[0]
    print("  paragraph keys:", list(p.keys()))
    print("  text (first 300 chars):", str(p.get("text", ""))[:300])
print()

# questions
qs = doc.get("questions", [])
print(f"[questions] count: {len(qs)}")
if qs:
    q = qs[0]
    print("  question keys:", list(q.keys()))
    print("  question:", q.get("question", ""))
    print("  answer:", q.get("answer", ""))
    print("  answer_type:", q.get("answer_type", ""))
    print("  answer_from:", q.get("answer_from", ""))
    print("  rel_paragraphs:", q.get("rel_paragraphs", []))
    print("  derivation:", q.get("derivation", ""))
