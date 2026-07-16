# Part 4 — Generation (Llama 3 via Groq,, LangChain RAG chain)
from dotenv import load_dotenv

load_dotenv()

from groq import AuthenticationError, GroqError, RateLimitError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from ingest import load_and_chunk
from retrieve import build_retriever

MODEL_NAME = "llama-3.1-8b-instant"

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below. If the context
doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, groq_api_key: str | None = None):
    """LangChain retriever -> a runnable chain: question in, grounded answer out.

    groq_api_key is optional so callers can pass one explicitly (e.g. Part 6's
    Streamlit app reading st.secrets) instead of relying on the GROQ_API_KEY
    env var, which only exists for local/CLI use via .env.
    """
    llm_kwargs = {"model": MODEL_NAME, "temperature": 0}
    if groq_api_key:
        llm_kwargs["groq_api_key"] = groq_api_key
    llm = ChatGroq(**llm_kwargs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


RATE_LIMIT_MESSAGE = (
    "Groq's free-tier rate limit was hit (too many requests/tokens in a short "
    "window). Please wait a moment and try again."
)

AUTH_ERROR_MESSAGE = (
    "Groq rejected the API key (invalid, revoked, or expired). Check the key "
    "in your .env / secrets.toml, or generate a new one at "
    "console.groq.com/keys."
)

GROQ_UNAVAILABLE_MESSAGE = (
    "Couldn't reach Groq's API right now. Please try again in a moment."
)


def safe_invoke(chain, chain_input) -> str:
    """Run the chain, turning known Groq failures into a friendly message
    instead of a crash — confirmed live that an unhandled Groq exception
    otherwise surfaces as a raw Python traceback (with local file paths) in
    the Streamlit UI, which is what this guards against.

    chain_input is whatever the chain's first step expects — a bare question
    string for build_rag_chain()'s retriever-driven chains, or a
    {"context": ..., "question": ...} dict for a hand-assembled context (see
    app.py's summary generation, which forces the paper's first chunk into
    context alongside whatever the retriever finds).

    Deployment note: once this app is live, the Groq API key's rate limit
    (6,000 tokens/min on the free tier, confirmed live 15 Jul 2026) is shared
    across every concurrent user, not per-visitor — so hitting this path is
    more likely under real traffic than during solo local testing.
    """
    try:
        return chain.invoke(chain_input)
    except RateLimitError:
        return RATE_LIMIT_MESSAGE
    except AuthenticationError:
        return AUTH_ERROR_MESSAGE
    except GroqError:
        # Base class for every other Groq failure (network issues, 5xx, bad
        # request, etc.) — deliberately after the two specific excepts above
        # so those get their own more useful message first.
        return GROQ_UNAVAILABLE_MESSAGE


ERROR_MESSAGES = {RATE_LIMIT_MESSAGE, AUTH_ERROR_MESSAGE, GROQ_UNAVAILABLE_MESSAGE}


def is_error_message(answer: str) -> bool:
    """True if safe_invoke() returned one of its own fallback strings rather
    than a real generated answer — callers should skip NLI verification and
    render as an error, not a claim to fact-check (confirmed live: without
    this check, an auth-error message got fed through the verifier and shown
    with a nonsensical "N/N sentences verified" badge)."""
    return answer in ERROR_MESSAGES


if __name__ == "__main__":
    chunks = load_and_chunk("data/paper.pdf")
    retriever = build_retriever(chunks)
    chain = build_rag_chain(retriever)

    question = "What is the total system latency of the pipeline, and what dominates it?"
    answer = safe_invoke(chain, question)

    print(f"question: {question}\n")
    print(f"answer:\n{answer}")
