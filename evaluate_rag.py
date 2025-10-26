import os, sys, re, time
import pandas as pd
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# make 'app' importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.rag_chain import build_chain  # uses your current pipeline

# === tiny embedder for semantic quote checks (no GPU needed) ===
from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def cosine(a_str: str, b_str: str) -> float:
    """Cosine similarity between two strings using MiniLM embeddings."""
    ea = _embedder.encode([a_str], normalize_embeddings=True)  # (1, d)
    eb = _embedder.encode([b_str], normalize_embeddings=True)  # (1, d)
    from sklearn.metrics.pairwise import cosine_similarity
    return float(cosine_similarity(ea, eb)[0][0])

QUOTE_RE = re.compile(r'\"([^"]{8,400})\"')  # capture quoted spans (8..400 chars)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_queries.csv')
df = pd.read_csv(DATA_PATH)

def eval_query(persona: str, query: str, sem_thresh=0.65):
    chain = build_chain(persona)
    start = time.time()
    out = chain.invoke({"query": query})
    latency = time.time() - start

    answer = out.get("result", "") or ""
    docs = out.get("source_documents", []) or []

    # collect context text + filenames
    ctx_texts = [d.page_content for d in docs]
    filenames = [os.path.basename((d.metadata or {}).get("source", "unknown")) for d in docs]

    answered = bool(answer.strip())
    # explicit quote present?
    quotes = QUOTE_RE.findall(answer)
    has_explicit_quote = len(quotes) > 0

    # strict grounded: any quoted span is a literal substring of any retrieved chunk
    grounded_strict = False
    for q in quotes:
        if any(q in c for c in ctx_texts):
            grounded_strict = True
            break

    # semantic faithfulness: highest cosine between any quote and any chunk
    best_sim = 0.0
    if quotes and ctx_texts:
        # take the longest quote to stabilize
        q = max(quotes, key=len)
        best_sim = max(cosine(q, c) for c in ctx_texts)

    return {
        "persona": persona,
        "query": query,
        "answered": answered,
        "has_explicit_quote": has_explicit_quote,
        "grounded_strict": grounded_strict,
        "semantic_sim": round(best_sim, 4),
        "latency": round(latency, 2),
        "sources": ", ".join(sorted(set(filenames))),
        "answer": answer.replace("\n", " ")[:800],
    }

def main():
    df = pd.read_csv(DATA_PATH)
    rows = []
    print("🚀 Starting SafeDrop AI Evaluation (strict mode)…")
    for _, r in df.iterrows():
        rows.append(eval_query(r["user_type"], r["query"], sem_thresh=0.55))
    out = pd.DataFrame(rows)

    eff = 100 * out["answered"].mean()
    explicit = 100 * out["has_explicit_quote"].mean()
    grounded = 100 * out["grounded_strict"].mean()
    # semantic >= 0.65 counts as faithful
    faithful = 100 * (out["semantic_sim"] >= 0.65).mean()
    escalate = 100 * out["answer"].str.contains("Escalate:", case=False, na=False).mean()
    median_lat = out["latency"].median()

    print("\n=== 🧾 SafeDrop AI Evaluation Summary (Strict) ===")
    print(f"Effectiveness (answered):          {eff:.1f}%")
    print(f"Semantic Faithfulness (≥0.65):   {faithful:.1f}%")
    print(f"Grounded & Quoted (strict):          {grounded:.1f}%")
    print(f"Answers with explicit quotes:       {explicit:.1f}%")
    print(f"Escalation rate:                      {escalate:.1f}%")
    print(f"Median latency:                     {median_lat:.2f} sec")

    save = os.path.join(os.path.dirname(__file__), '..', 'data', 'eval_results.csv')
    out.to_csv(save, index=False)
    print(f"\n✅ Detailed log saved to {save}")

if __name__ == "__main__":
    main()
