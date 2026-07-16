# Part 5 —- Verification (DeBERTa NLI entailment check)
import re

from sentence_transformers import CrossEncoder

from generate import build_rag_chain, safe_invoke
from ingest import load_and_chunk
from retrieve import build_retriever

# Small checkpoint by design — deberta-large-mnli (~1.5GB) would recreate the
# RAM-wall problem that already ruled out local Ollama. See PROJECT_STATE.md.
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

_model = None


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def split_sentences(text: str) -> list[str]:
    """Good-enough sentence splitter for decomposing source text into NLI premises."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def verify_sentence(sentence: str, source_text: str) -> dict:
    """NLI check: is `sentence` entailed by any single sentence within `source_text`?

    Passing a whole multi-topic chunk as one premise dilutes the NLI signal
    to a flat "neutral" (confirmed in testing) — the model was trained on
    single-sentence premise/hypothesis pairs, not long paragraphs. Splitting
    the source into individual sentences and keeping the one with the
    highest entailment score is the standard fix for long-premise NLI.
    """
    model = get_model()
    id2label = model.config.id2label
    entailment_idx = next(i for i, label in id2label.items() if label == "entailment")

    premises = split_sentences(source_text) or [source_text]
    pairs = [(premise, sentence) for premise in premises]
    probs = model.predict(pairs, apply_softmax=True)

    best_i = max(range(len(probs)), key=lambda i: probs[i][entailment_idx])
    scores = {id2label[i]: float(probs[best_i][i]) for i in id2label}
    label = max(scores, key=scores.get)
    return {"label": label, "scores": scores, "matched_premise": premises[best_i]}


if __name__ == "__main__":
    chunks = load_and_chunk("data/paper.pdf")
    retriever = build_retriever(chunks)
    chain = build_rag_chain(retriever)

    question = "What is the total system latency of the pipeline, and what dominates it?"
    answer = safe_invoke(chain, question)
    source_docs = retriever.invoke(question)
    source_text = "\n".join(doc.page_content for doc in source_docs)

    print(f"question: {question}\n")
    print(f"generated answer:\n{answer}\n")
    print("--- per-sentence verification of the generated answer ---\n")

    for sentence in split_sentences(answer):
        result = verify_sentence(sentence, source_text)
        rounded = {k: round(v, 3) for k, v in result["scores"].items()}
        print(f"sentence: {sentence}")
        print(f"  -> {result['label']}  {rounded}")
        print(f"  matched premise: {result['matched_premise']!r}\n")

    # Adversarial check: does a deliberately wrong sentence actually get flagged,
    # or does verification just rubber-stamp everything? Must be an actual
    # contradiction, not just a different number — "below 500 ms" was tried
    # first and got (correctly!) marked as entailment, because "below 62 ms"
    # logically implies "below 500 ms". "Exceeding" is a genuine contradiction
    # of "below 62 ms", unlike "below 500 ms".
    false_sentence = "The complete processing pipeline achieves a total system latency exceeding 500 ms."
    false_result = verify_sentence(false_sentence, source_text)
    print("--- adversarial check (deliberately wrong sentence) ---\n")
    print(f"sentence: {false_sentence}")
    print(f"  -> {false_result['label']}  "
          f"{ {k: round(v, 3) for k, v in false_result['scores'].items()} }")

    assert false_result["label"] != "entailment", "expected the false sentence to be flagged"
    print("\nsanity check passed: deliberately wrong sentence was flagged, not entailed")
