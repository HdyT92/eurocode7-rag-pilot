"""Step 3: retrieve chunks, send them to the local LLM (Ollama), save the answers.

Usage:  py answer.py        -> RAG with top 5 chunks
        py answer.py 3      -> RAG with top 3 chunks
        py answer.py 0      -> NO document (baseline, same settings, for a fair comparison)
Needs:  Ollama running, model pulled (ollama pull qwen2.5:7b), chunks.jsonl, questions.csv
Output: answers_k{K}.csv  -> fill the 'label' column yourself (you are the expert reviewer)
"""
import sys, os, csv, json, urllib.request
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LLM = "qwen2.5:7b"
NUM_CTX = 8192          # Ollama default is 4096; long chunks would be cut silently
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_RAG = (
    "You help geotechnical engineers read a Eurocode 7 guidance document. "
    "Answer ONLY from the CONTEXT below. Do not use outside knowledge. "
    "After each fact, give the section number in brackets, e.g. [3.4.1]. "
    "If the CONTEXT does not contain the answer, reply exactly: NOT IN DOCUMENT"
)
SYSTEM_BASE = (
    "You help geotechnical engineers with Eurocode 7. Answer the question. "
    "If you are not sure, reply exactly: NOT SURE"
)

def ask_llm(system, user):
    body = {"model": LLM, "stream": False,
            "options": {"temperature": 0, "num_ctx": NUM_CTX},   # temperature 0 = same answer every run
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out["message"]["content"].strip(), out.get("prompt_eval_count", 0)

questions = list(csv.DictReader(open(os.path.join(HERE, "questions.csv"), encoding="utf-8")))

if K > 0:
    import faiss
    from sentence_transformers import SentenceTransformer
    chunks = [json.loads(l) for l in open(os.path.join(HERE, "chunks.jsonl"), encoding="utf-8")]
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = emb.encode([f"{c['section']} {c['title']}\n{c['text']}" for c in chunks],
                      normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1]); index.add(np.asarray(vecs, dtype="float32"))

rows = []
for q in questions:
    if K > 0:
        qv = emb.encode(["Represent this sentence for searching relevant passages: " + q["question"]],
                        normalize_embeddings=True)
        _, ids = index.search(np.asarray(qv, dtype="float32"), K)
        found = [chunks[i] for i in ids[0]]
        context = "\n\n".join(f"[Section {c['section']} - {c['title']} - pages {', '.join(c['pages'])}]\n{c['text']}"
                              for c in found)
        answer, n_tok = ask_llm(SYSTEM_RAG, f"CONTEXT:\n{context}\n\nQUESTION: {q['question']}")
        sections = " ; ".join(c["section"] for c in found)
    else:
        answer, n_tok = ask_llm(SYSTEM_BASE, q["question"])
        sections = "-"

    warn = "  !! prompt may be CUT (near context limit)" if n_tok >= NUM_CTX - 50 else ""
    print(f"\n=== {q['id']} (type {q['q_type']}) | prompt tokens: {n_tok}/{NUM_CTX}{warn}")
    print(f"GOLD:   {q['gold_answer'][:150]}")
    print(f"ANSWER: {answer}")
    rows.append({"id": q["id"], "q_type": q["q_type"], "question": q["question"],
                 "gold_answer": q["gold_answer"], "answer": answer, "sections_used": sections,
                 "prompt_tokens": n_tok,
                 "label": ""})   # you fill: correct / partial / wrong / abstain / abstain+wrong

out = os.path.join(HERE, f"answers_k{K}.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig so Excel shows symbols correctly
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"\nSaved {out}. Now open it and fill the 'label' column.")
