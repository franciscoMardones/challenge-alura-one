"""RAG pipeline for BimBam Buy Agent.

Builds the retrieval-augmented generation chain using LCEL,
handles similarity-based retrieval with score thresholding,
and provides source extraction helpers.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_K,
    SIMILARITY_THRESHOLD,
    MMR_LAMBDA,
    FALLBACK_MESSAGE,
)


# ---------------------------------------------------------------------------
# Chain cache (lazy-init)
# ---------------------------------------------------------------------------

_rag_chain = None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = (
    "Eres un asistente de BimBam Buy. Usa la información del contexto para responder.\n\n"
    "CONTEXTO:\n{context}\n\n"
    "PREGUNTA: {question}\n\n"
    "INSTRUCCIONES:\n"
    "1. Responde directamente basándote en el contexto.\n"
    "2. Si el contexto contiene información relevante, úsala sin dudar.\n"
    "3. Sé claro y organiza la información en puntos cuando sea apropiado.\n"
    "4. Solo di 'no tengo información' si el contexto NO tiene nada relacionado con la pregunta.\n"
    "5. No repitas la misma idea con diferentes palabras.\n\n"
    "RESPUESTA:"
)

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_docs(docs: list) -> str:
    """Join documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def extract_sources(docs: list) -> list[str]:
    """Extract unique source filenames from document metadata."""
    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            sources.append(source)
            seen.add(source)
    return sources


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def get_relevant_docs(
    query: str,
    vectorstore: Chroma,
    *,
    use_mmr: bool = True,
) -> tuple[list, list[str]]:
    """Retrieve documents using MMR for diversity, with optional fallback.

    Returns:
        (docs, sources): matched documents and their source filenames.
    """
    if use_mmr:
        # Maximal Marginal Relevance search for diversity
        docs = vectorstore.max_marginal_relevance_search(
            query, k=TOP_K, lambda_mult=MMR_LAMBDA
        )
    else:
        # Fallback: standard similarity search with threshold filtering
        results = vectorstore.similarity_search_with_score(query, k=TOP_K)
        docs = [doc for doc, score in results if score <= SIMILARITY_THRESHOLD]
    sources = extract_sources(docs)
    return docs, sources


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_rag_chain(vectorstore: Chroma):
    """Build and return an LCEL RAG chain.

    The chain: prompt → Groq LLM → string output.
    Accepts {"context": str, "question": str} as input.
    Uses module-level cache to avoid per-request rebuilds.
    """
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    chain = rag_prompt | llm | StrOutputParser()
    _rag_chain = chain
    return chain


def run_rag_stream(question: str, vectorstore: Chroma, history_text: str = ""):
    """Streaming RAG: yields token dicts then a final done event.

    Yields:
        {"token": "<chunk>"}          — one per streamed token
        {"sources": [...], "done": True}  — final event
    """
    docs, sources = get_relevant_docs(question, vectorstore)
    if not docs:
        yield {"sources": [], "done": True}
        return

    context = format_docs(docs)
    if history_text:
        context += history_text

    chain = build_rag_chain(vectorstore)
    full_answer = ""
    for chunk in chain.stream({"context": context, "question": question}):
        full_answer += str(chunk)
        yield {"token": str(chunk)}

    yield {"sources": sources, "done": True}


def run_rag(question: str, vectorstore: Chroma, history_text: str = "") -> dict:
    """End-to-end RAG call: retrieve → format → generate → return answer + sources.

    Returns dict with keys: answer, sources, fallback (bool).
    """
    docs, sources = get_relevant_docs(question, vectorstore)
    if not docs:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": [],
            "fallback": True,
        }
    context = format_docs(docs)
    if history_text:
        context += history_text
    chain = build_rag_chain(vectorstore)
    answer = chain.invoke({"context": context, "question": question})
    return {
        "answer": answer,
        "sources": sources,
        "fallback": False,
    }
