"""Step 2: find the best chunks for each benchmark question, and score it.

Usage:  python retrieve.py            (top 5 by default)
        python retrieve.py 3          (top 3)
Needs:  chunks.jsonl (from chunk.py) and questions.csv in the same folder.

gold_sections column in questions.csv:
  "A.3.3"              -> this section must be found
  "3.3|4.3|7.3"        -> any ONE of these is enough
  "3.4.1+A.3.1+A.3.5"  -> ALL of these are needed
  ""                   -> answer not in document (retrieval cannot be scored)
"""
import sys, os, csv, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge needs this for questions

# 1) Load chunks and questions
chunks = [json.loads(l) for l in open(os.path.join(HERE, "chunks.jsonl"), encoding="utf-8")]
questions = list(csv.DictReader(open(os.path.join(HERE, "questions.csv"), encoding="utf-8")))

# 2) Turn every chunk into a vector (the title is added so the heading helps the search)
model = SentenceTransformer(MODEL)
print(f"Model max input: {model.max_seq_length} tokens (longer chunks are CUT here)")
texts = [f"{c['section']} {c['title']}\n{c['text']}" for c in chunks]
vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

# 3) Put vectors in a FAISS index (inner product = cosine, because vectors are normalised)
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(np.asarray(vecs, dtype="float32"))

# 4) Search for each question and check against the gold sections
rows, scores = [], []
for q in questions:
    qv = model.encode([QUERY_PREFIX + q["question"]], normalize_embeddings=True)
    sims, ids = index.search(np.asarray(qv, dtype="float32"), K)
    found = [chunks[i]["section"] for i in ids[0]]

    gold = q.get("gold_sections", "").strip()
    if gold:
        groups = [g.split("|") for g in gold.split("+")]          # each group = alternatives
        hit = sum(any(s in found for s in g) for g in groups) / len(groups)
        scores.append(hit)
        result = f"{hit:.0%} of needed sections found"
    else:
        result = "not in document (check the answer step later)"

    print(f"\n{q['id']} (type {q['q_type']}): {q['question'][:80]}...")
    for rank, (i, s) in enumerate(zip(ids[0], sims[0]), 1):
        mark = " <-- gold" if gold and any(chunks[i]["section"] in g.split("|") for g in gold.split("+")) else ""
        print(f"  {rank}. {chunks[i]['section']:<8} {s:.3f}  {chunks[i]['title'][:45]}{mark}")
    print(f"  => {result}")
    rows.append({"id": q["id"], "q_type": q["q_type"], "gold": gold,
                 "retrieved": " ; ".join(found), "recall": result})

print(f"\nAverage recall@{K} over {len(scores)} scorable questions: {np.mean(scores):.0%}")

with open(os.path.join(HERE, f"retrieval_results_k{K}.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"Saved retrieval_results_k{K}.csv")
