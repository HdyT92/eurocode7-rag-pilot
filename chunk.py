"""Step 1: read the JRC Eurocode 7 PDF and cut it into one chunk per section.

Usage:  python chunk.py                (uses the PDF in the same folder)
        python chunk.py other_file.pdf
Output: chunks.jsonl (one JSON object per section), saved next to this script
"""
import sys, re, json, statistics, os
from collections import Counter
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "2013_06_WS_GEO.pdf")
doc = pymupdf.open(pdf_path)

# 1) Read every page, keep the printed page number (the number at the bottom of the page)
pages = []
for i, page in enumerate(doc):
    lines = [l.strip() for l in page.get_text().splitlines() if l.strip()]
    # skip table-of-contents pages (many dotted lines)
    if sum(("....." in l or "…" in l) for l in lines) >= 8:
        continue
    # printed page number = the number-only text block lowest on the page
    blocks = [b for b in page.get_text("blocks") if re.fullmatch(r"\d{1,3}", b[4].strip())]
    printed = max(blocks, key=lambda b: b[1])[4].strip() if blocks else None
    if printed and page.rect.height - max(b[1] for b in blocks) > 120:
        printed = None  # a lonely number in the middle of the page is not a page number
    pages.append({"pdf_page": i + 1, "printed": printed, "lines": lines})

# 2) Remove running headers (lines that repeat on many pages, e.g. author names)
counts = Counter(l for p in pages for l in set(p["lines"]))
noise = {l for l, c in counts.items() if c >= 8 and len(l) > 3}

# 3) Walk through the lines and start a new chunk at each numbered heading
#    e.g. "3.4.1. SOME USEFUL DEFINITIONS" or "A.3.3. DESIGN OF A STRIP FOUNDATION"
heading = re.compile(r"^((?:A\.)?\d+(?:\.\d+){1,3})\.?\s+([A-Z].{3,})$")
chunks, cur = [], None
for p in pages:
    for l in p["lines"]:
        if l in noise or re.fullmatch(r"\d{1,3}", l) or "....." in l:
            continue  # skip headers, page numbers, table-of-contents lines
        m = heading.match(l)
        if m:
            if cur: chunks.append(cur)
            cur = {"section": m.group(1), "title": m.group(2), "pages": [], "text": ""}
        if cur is None:
            continue  # front matter before the first section
        cur["text"] += l + "\n"
        if p["printed"] and p["printed"] not in cur["pages"]:
            cur["pages"].append(p["printed"])
if cur: chunks.append(cur)

# 3b) Merge tiny "parent" headings (e.g. "3.2 Design situations...") into the next chunk
merged = []
for c in chunks:
    if merged and len(merged[-1]["text"].split()) < 20:
        prev = merged.pop()
        c["text"] = prev["text"] + c["text"]
        c["pages"] = prev["pages"] + [p for p in c["pages"] if p not in prev["pages"]]
    merged.append(c)
chunks = merged

with open(os.path.join(HERE, "chunks.jsonl"), "w", encoding="utf-8") as f:
    for i, c in enumerate(chunks):
        c["id"] = f"C{i:03d}"
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# 4) Print simple stats so you can SEE if the chunking makes sense
words = [len(c["text"].split()) for c in chunks]
print(f"{len(chunks)} chunks | words per chunk: min {min(words)}, median {statistics.median(words)}, max {max(words)}")
for c, w in sorted(zip(chunks, words), key=lambda x: -x[1])[:5]:
    print(f"  biggest: {c['section']} {c['title'][:50]}  ({w} words, pages {c['pages']})")
