# ResearchLens

A RAG (Retrieval-Augmented Generation) research paper summarizer. Upload
a PDF and get a summary (Short / Long / One Page) plus a chat panel for
follow-up questions — every generated sentence is checked against the
source text and shown with a verified/unverified badge.

**Stack:** LangChain (ingestion + retrieval orchestration), Sentence-
Transformers (`all-MiniLM-L6-v2` embeddings), FAISS (vector store), Llama 3
via Groq (generation), DeBERTa NLI (`cross-encoder/nli-deberta-v3-base`,
hallucination-check verification), Streamlit (frontend).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY — free key at console.groq.com/keys
```

## Run

```bash
streamlit run app.py
```

Works without a Groq key too — falls back to showing the retrieved source
passages directly instead of an AI-generated answer.

## Layout

```
ingest.py     ← PDF loading + chunking (LangChain)
embed.py      ← chunk embeddings (Sentence-Transformers)
retrieve.py   ← FAISS index + retriever
generate.py   ← Llama 3 (Groq) RAG chain
verify.py     ← DeBERTa NLI verification
app.py        ← Streamlit UI
```

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On share.streamlit.io: New app → pick this repo/branch → main file
   `app.py`.
3. Before deploying, open **Advanced settings → Secrets** and add:
   ```
   GROQ_API_KEY = "gsk_..."
   ```
   (`app.py` reads this via `st.secrets`, with the same `.env` fallback
   for local runs.)

Notes:
- The Groq free tier's rate limit is shared across every visitor to the
  deployed app, not per-user — handled gracefully (a friendly message
  instead of a crash if it's ever hit).
- `requirements.txt` pins the CPU build of `torch` so the install doesn't
  pull the multi-GB CUDA build on Streamlit Cloud's host.
