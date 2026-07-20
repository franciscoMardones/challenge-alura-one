"""Quick test for RAG pipeline — multiple questions."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import CHROMA_DIR, EMBEDDING_MODEL
from rag import run_rag

print("Cargando vectorstore...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
print(f"Vectores cargados: {vectorstore._collection.count()}")

questions = [
    "¿Cuál es la política de devoluciones?",
    "¿Cómo funciona el programa de afiliados?",
    "¿Qué métodos de pago aceptan?",
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"Pregunta: {q}")
    result = run_rag(q, vectorstore)
    print(f"Respuesta: {result['answer'][:200]}...")
    print(f"Fuentes: {result['sources']}")
