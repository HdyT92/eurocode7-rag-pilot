# Eurocode 7 RAG: a small pilot

A small, fully local test of **retrieval-augmented generation (RAG)** on a Eurocode 7 guidance document.
The goal is not a finished tool. The goal is to see **where simple RAG fails** on engineering-standard questions, using questions written and checked by a structural engineer.

> Status: **pilot**. 5 questions only. The results show patterns, not proof.

## Why this matters
Engineers using codes and standards often need to read several clauses, follow references, and read tables before reaching an answer. A general LLM may "know" Eurocode in a rough way, but:
- it cannot show **which clause** its answer comes from,
- it has never seen **new or local documents** (new standards, National Annexes, project specifications),
- it may give a **confident wrong answer**.

## Document
*Eurocode 7: Geotechnical Design – Worked examples*, JRC Report EUR 26227 EN (2013), European Commission Joint Research Centre. DOI: [10.2788/3398](https://doi.org/10.2788/3398).
The report states: *"Reproduction is authorised provided the source is acknowledged."* The PDF is **not** included in this repository. Download it from the source and save it as `2013_06_WS_GEO.pdf`.

Note: this is a 2013 guidance report based on the first generation of Eurocode 7, not the standard itself.

## Pipeline (all local)
| Step | File | What it does |
|---|---|---|
| 1 | `chunk.py` | Reads the PDF (PyMuPDF) and cuts it into **one chunk per numbered section** (115 chunks) |
| 2 | `retrieve.py` | Embeds chunks with `BAAI/bge-small-en-v1.5`, searches with FAISS, scores retrieval against gold sections |
| 3 | `answer.py` | Sends the top-k chunks to `qwen2.5:7b` (Ollama, temperature 0, context 8192) and saves answers |

Hardware: laptop, RTX 5050 (8 GB VRAM), 16 GB RAM.

## Benchmark (`questions.csv`)
5 questions, each with a gold answer, page, section and type:

| Type | Meaning | Questions |
|---|---|---|
| 1 | Answer in one section | – |
| 2 | Answer needs several sections | Q04 |
| 3 | Answer is in a table | Q01, Q02, Q03 |
| 4 | Answer is **not** in the document (trap question with a false premise) | Q05 |

All gold answers were checked against the PDF by hand.

**Labelling rule** (by the author, as expert reviewer): `correct` needs all key points; any incorrect claim makes the answer `wrong`; `partial` = correct but incomplete; `abstain` = the model declined to answer.

## Results

### Retrieval (did the right sections come back?)
| | top-3 | top-5 |
|---|---|---|
| Average recall (Q01–Q04) | 92% | 100% |
| Q04 (multi-section) | **67%** | 100% |

With top-3, the chunk holding the key calculation for Q04 (section A.3.5) is lost. Similarity scores were very close (e.g. 0.725 / 0.724 / 0.723), so ranking is fragile.

### Answers
| | No document | RAG, top-5 |
|---|---|---|
| correct | 0 | 1 |
| partial | 0 | 2 |
| wrong | 1 | 1 |
| abstain | 4 | 1 |
| Q05 (not in document) handled correctly | yes | yes |

## Findings

**About the answers**
1. **Multi-step reasoning failed even with full context.** For Q04, all needed sections were retrieved and the prompt fitted in context (5,495 tokens), but the model compared the *allowable* settlement with the *table limit*, instead of the *calculated* settlement (5.4 cm) with the allowable value (5 cm). Finding the evidence was not enough; combining it was the problem.
2. **Citation without the answer.** For Q02, the model returned only `[Table 3.3.3]`, the right source, without the value (1.25).
3. **The prompt changes behaviour strongly.** In an early chat test with no option to abstain, the model gave confident wrong values (e.g. partial factor 1.0 instead of 1.25, and a factor that does not exist in EC7). With an explicit "NOT SURE" option, it abstained on 4 of 5 questions. Safe, but not useful.
4. **Similarity scores cannot detect "not in document".** For Q05, the best chunk scored 0.713, about the same as chunks for answerable questions (~0.72). The top chunk held nearby but wrong values (rotation 1/500, tilt 1/300). The model still answered `NOT IN DOCUMENT`, but one question is not enough to trust this.
5. **Default context would cut the prompt silently.** Q04 needed 5,495 tokens; Ollama's default context is 4,096.

**About the document and PDF extraction**

6. **Broken cross-reference in the source**: "One of the design methods given in *Error! Reference source not found.*" (p. 19).
7. **A lost symbol changes the meaning**: "Ed [≈] Cd = 5 cm" (p. 119). The comparison symbol is lost in extraction, and 5.4 > 5 reads as a contradiction.
8. **Lost Greek letter**: `φ' = 38°` becomes `' = 38°` (p. 112).
9. **Table columns mixed** in Table 3.2.1 (p. 19).
10. **Figure text inserted into a table**: soil labels from the figure above (numbered "Fig.A.3.111" in the source, another numbering error) appear inside Table A.3.2 (p. 112). The table also breaks across two pages.

## Limitations
- 5 questions, one document, one model, one reviewer.
- Mostly Chapter 3 and Annex A.3 (shallow foundations).
- Labels are one engineer's judgement.

## Next steps
- Grow the benchmark to ~20 questions across chapters, with more multi-section and table questions.
- Add a **verifier**: the model must quote the exact supporting sentence; code checks that the quote exists in the retrieved text, otherwise the answer is rejected.
- Compare fixed-size chunking with section chunking.
- Compare with a large hosted model without the document.
- Test page images with a vision model for tables, figures and symbols.

## How to run
```
pip install pymupdf sentence-transformers faiss-cpu
ollama pull qwen2.5:7b
py chunk.py
py retrieve.py 5
py answer.py 5      # RAG
py answer.py 0      # no document (baseline)
```

## Note on AI assistance
Code was written with AI assistance. The research design, benchmark questions, gold answers, labels and interpretation of results are my own.

## Author
Hedayat Maghsood, structural engineer (MSc Structural Engineering; MSc Construction Project Management).
