# Part 3 — Vector store &  retrieval (FAISS + LangChain retriever)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from embed import MODEL_NAME
from ingest import load_and_chunk

_embedding_function = None


def get_embedding_function() -> HuggingFaceEmbeddings:
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return _embedding_function


def build_vectorstore(chunks) -> FAISS:
    """List of LangChain Documents -> a FAISS vectorstore.

    Split out from build_retriever() so callers that need several retrievers
    with different k (e.g. app.py's short/long/one-page summary lengths) can
    reuse one embedded index instead of re-embedding the same chunks per k.
    """
    return FAISS.from_documents(chunks, get_embedding_function())


def build_retriever(chunks, k: int = 3):
    """List of LangChain Documents -> a LangChain retriever backed by FAISS."""
    return build_vectorstore(chunks).as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    chunks = load_and_chunk("data/paper.pdf")
    retriever = build_retriever(chunks)

    query = "What is the total system latency of the pipeline?"
    results = retriever.invoke(query)

    print(f"query: {query!r}\n")
    print(f"retrieved {len(results)} chunks:\n")

    for i, doc in enumerate(results):
        print(f"--- result {i} (page {doc.metadata['page']}) ---")
        print(doc.page_content)
        print()

    hit = any("latency" in doc.page_content.lower() for doc in results)
    print(f"sanity check: at least one retrieved chunk mentions 'latency' -> {hit}")
    assert hit, "expected a latency-related chunk to be retrieved for this query"
    print("sanity check passed")
