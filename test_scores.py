"""Profile similarity scores for threshold tuning."""
import sys
import statistics
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import CHROMA_DIR, EMBEDDING_MODEL, TOP_K

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# Queries from design
queries = [
    ("¿Qué es BimBam Buy?", "core"),
    ("¿Cuánto cuesta?", "pricing"),
    ("¿Cómo me registro?", "onboarding"),
    ("¿Cuál es el clima hoy?", "out-of-scope"),
    ("¿Quién es el CEO de Google?", "out-of-scope"),
]

in_scope_scores = []
out_scope_scores = []

print("Similarity score profiling (lower = more similar):\n")
for query, category in queries:
    results = vectorstore.similarity_search_with_score(query, k=TOP_K)
    scores = [score for _, score in results]
    avg = statistics.mean(scores)
    print(f"[{category}] {query}")
    print(f"  scores: {[f'{s:.4f}' for s in scores]}")
    print(f"  avg: {avg:.4f}")
    print()
    if category == "out-of-scope":
        out_scope_scores.extend(scores)
    else:
        in_scope_scores.extend(scores)

def percentile(data, p):
    """Compute percentile using linear interpolation."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    k = (p / 100) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])

print("=" * 60)
print("In-scope scores (core, pricing, onboarding):")
print(f"  count: {len(in_scope_scores)}")
if in_scope_scores:
    p25 = percentile(in_scope_scores, 25)
    p50 = percentile(in_scope_scores, 50)
    p75 = percentile(in_scope_scores, 75)
    print(f"  P25: {p25:.4f}")
    print(f"  P50: {p50:.4f}")
    print(f"  P75: {p75:.4f}")
    print(f"\nSuggested SIMILARITY_THRESHOLD (P75 of in-scope): {p75:.4f}")

print("\nOut-of-scope scores:")
print(f"  count: {len(out_scope_scores)}")
if out_scope_scores:
    p25 = percentile(out_scope_scores, 25)
    p50 = percentile(out_scope_scores, 50)
    p75 = percentile(out_scope_scores, 75)
    print(f"  P25: {p25:.4f}")
    print(f"  P50: {p50:.4f}")
    print(f"  P75: {p75:.4f}")

print("\n" + "=" * 60)
if in_scope_scores and out_scope_scores:
    in_p75 = percentile(in_scope_scores, 75)
    out_p25 = percentile(out_scope_scores, 25)
    if out_p25 > in_p75:
        print("✓ Clear separation: out-of-scope P25 > in-scope P75")
    else:
        print("⚠ Overlap detected: consider adjusting threshold")
