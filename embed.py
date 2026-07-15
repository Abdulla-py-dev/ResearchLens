# Part 2 — Embeddings (Sentence-Transformers, all-MiniLM-L6-v2)
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import load_and_chunk

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks) -> np.ndarray:
    """List of LangChain Documents -> (N, 384) array of embeddings."""
    texts = [chunk.page_content for chunk in chunks]
    return get_model().encode(texts, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    chunks = load_and_chunk("data/paper.pdf")
    vectors = embed_chunks(chunks)

    print(f"{vectors.shape[0]} chunks embedded, vector dim = {vectors.shape[1]}\n")

    # Adjacent, overlapping chunks should be more similar than two far-apart ones.
    i, j, k = 10, 11, len(chunks) - 1

    sim_related = cosine_similarity(vectors[i], vectors[j])
    sim_unrelated = cosine_similarity(vectors[i], vectors[k])

    print(f"cosine(chunk {i}, chunk {j}) [adjacent/overlapping]  = {sim_related:.4f}")
    print(f"cosine(chunk {i}, chunk {k}) [far apart]             = {sim_unrelated:.4f}")

    assert sim_related > sim_unrelated, "adjacent chunks should be more similar than distant ones"
    print("\nsanity check passed: related chunks score higher than unrelated ones")
