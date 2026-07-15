# Part 6 — Streamlit UI (upload, summary, chat, verification badges)
import os
import tempfile

import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from generate import MODEL_NAME, PROMPT, format_docs, is_error_message, safe_invoke, build_rag_chain
from ingest import load_and_chunk
from retrieve import build_vectorstore
from verify import split_sentences, verify_sentence

# Per the guide's requirement: three summary lengths, each pulling more
# retrieved chunks as the target length grows — a one-page summary needs
# more source material than a 2-3 sentence one. Even so, k alone isn't
# reliable for a "summarize the whole paper" instruction (not a real
# question): its embedding doesn't consistently land near the paper's own
# content, so plain top-k retrieval can miss it entirely (confirmed live —
# Short's k=3 pulled reference-list/author-bio chunks and produced "I don't
# know"). Every academic paper puts its title+abstract in the very first
# chunk, so generate_summary() below always anchors context on chunks[0] in
# addition to whatever the retriever finds, regardless of length/k.
SUMMARY_LENGTHS = {
    "Short": {
        "k": 3,
        "prompt": (
            "Provide a short summary of this paper in 2-3 sentences, covering only "
            "its single most important contribution and result. Be concise."
        ),
    },
    "Long": {
        "k": 6,
        "prompt": (
            "Provide a detailed summary of this paper in several paragraphs, "
            "covering its key contributions, methodology, and results."
        ),
    },
    "One Page": {
        "k": 10,
        "prompt": (
            "Provide a comprehensive, approximately one-page summary of this paper. "
            "Structure it with headers: Overview, Methodology, Results, and "
            "Conclusion, covering the key contributions, technical approach, "
            "experimental setup, and findings in detail."
        ),
    },
}

WELCOME_STEPS = [
    "Upload a research paper (PDF) in the sidebar.",
    "A summary is generated automatically once indexing finishes.",
    "Ask follow-up questions in the chat panel below — every answer is checked "
    "sentence-by-sentence against the source text with a DeBERTa NLI model, "
    "so you can see exactly what's grounded.",
]

st.set_page_config(
    page_title="ResearchLens — Pipeline v2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .rl-card {
        background-color: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .rl-step {
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        margin-bottom: 0.6rem;
    }
    .rl-step-num {
        background: #4f46e5;
        color: white;
        border-radius: 50%;
        width: 1.6rem;
        height: 1.6rem;
        min-width: 1.6rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .rl-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.15rem 0.35rem 0.15rem 0;
        border: 1px solid transparent;
    }
    .rl-badge.verified {
        background: rgba(34, 197, 94, 0.14);
        color: #16a34a;
        border-color: rgba(34, 197, 94, 0.35);
    }
    .rl-badge.flagged {
        background: rgba(245, 158, 11, 0.14);
        color: #b45309;
        border-color: rgba(245, 158, 11, 0.35);
    }
    .rl-badge.info {
        background: rgba(99, 102, 241, 0.14);
        color: #4f46e5;
        border-color: rgba(99, 102, 241, 0.35);
    }
    .rl-sentence {
        padding: 0.5rem 0.75rem;
        border-radius: 10px;
        background-color: rgba(120, 120, 120, 0.08);
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
    }
    .rl-source {
        opacity: 0.65;
        font-size: 0.82rem;
        font-style: italic;
        margin-top: 0.2rem;
    }
    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
        border: 1.5px dashed rgba(128, 128, 128, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key() -> str:
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return key
    except (KeyError, FileNotFoundError):
        pass
    return os.getenv("GROQ_API_KEY", "")


def badge(text: str, kind: str) -> str:
    icon = {"verified": "✅", "flagged": "⚠️", "info": "ℹ️"}[kind]
    return f"<span class='rl-badge {kind}'>{icon} {text}</span>"


class UnusablePDFError(Exception):
    """Raised for any uploaded file build_vectorstore_for_file can't turn
    into chunks — bad PDF structure or no extractable text. Caught at the
    call site and shown as a friendly st.error() instead of the raw
    traceback (with local file paths) that surfaced here during testing."""


@st.cache_resource(show_spinner=False)
def build_vectorstore_for_file(file_bytes: bytes, _filename: str):
    """Cached so re-running the app doesn't re-embed the same PDF. Keyed on
    the file's own bytes (and filename, to avoid same-size collisions).
    Returns the vectorstore itself (not a fixed-k retriever) so callers can
    derive retrievers with different k — e.g. one per summary length."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        chunks = load_and_chunk(tmp_path)
    except Exception as e:
        raise UnusablePDFError(
            "Couldn't read this file as a PDF — it may be corrupted, "
            "password-protected, or not actually a PDF."
        ) from e
    finally:
        os.remove(tmp_path)

    if not chunks:
        raise UnusablePDFError(
            "No extractable text found in this PDF. It might be a "
            "scanned/image-only document — try a text-based PDF instead."
        )
    return build_vectorstore(chunks), chunks[0], len(chunks)


def verify_answer(answer: str, source_docs) -> list[dict]:
    source_text = "\n".join(doc.page_content for doc in source_docs)
    verifications = []
    for sentence in split_sentences(answer):
        result = verify_sentence(sentence, source_text)
        verifications.append(
            {
                "sentence": sentence,
                "verified": result["label"] == "entailment",
                "matched_premise": result["matched_premise"],
            }
        )
    return verifications


def answer_with_verification(chain, retriever, question: str) -> dict:
    answer = safe_invoke(chain, question)
    if is_error_message(answer):
        return {"answer": answer, "verifications": [], "is_error": True}
    source_docs = retriever.invoke(question)
    return {"answer": answer, "verifications": verify_answer(answer, source_docs), "is_error": False}


def anchored_summary_docs(vectorstore, first_chunk, prompt_text: str, k: int) -> list:
    """Retrieved chunks for prompt_text, always including the paper's first
    chunk (title+abstract, for virtually any academic paper) even if plain
    retrieval wouldn't surface it — see the module comment on SUMMARY_LENGTHS
    for why relying on retrieval alone isn't reliable for a summary prompt."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(prompt_text)
    return [first_chunk] + [
        d for d in retrieved_docs if d.page_content != first_chunk.page_content
    ]


def generate_summary(vectorstore, first_chunk, prompt_text: str, k: int, groq_api_key: str) -> dict:
    """Like answer_with_verification, but for whole-document summarization
    using anchored_summary_docs() instead of trusting retrieval alone."""
    source_docs = anchored_summary_docs(vectorstore, first_chunk, prompt_text, k)

    llm = ChatGroq(model=MODEL_NAME, temperature=0, groq_api_key=groq_api_key)
    chain = PROMPT | llm | StrOutputParser()
    answer = safe_invoke(
        chain, {"context": format_docs(source_docs), "question": prompt_text}
    )
    if is_error_message(answer):
        return {"answer": answer, "verifications": [], "is_error": True}
    return {"answer": answer, "verifications": verify_answer(answer, source_docs), "is_error": False}


def render_verified_answer(answer: str, verifications: list[dict], is_error: bool = False):
    if is_error:
        st.error(answer)
        return
    st.markdown(answer)
    if not verifications:
        return

    verified_count = sum(v["verified"] for v in verifications)
    st.markdown(
        badge(f"{verified_count}/{len(verifications)} sentences verified", "info"),
        unsafe_allow_html=True,
    )
    with st.expander("🔍 Verification detail"):
        for v in verifications:
            kind = "verified" if v["verified"] else "flagged"
            label = "Entailed by source" if v["verified"] else "Not clearly supported"
            st.markdown(
                f"<div class='rl-sentence'>{v['sentence']}<br>"
                f"{badge(label, kind)}"
                f"<div class='rl-source'>closest source: “{v['matched_premise']}”</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_offline_answer(source_docs):
    st.info(
        "No `GROQ_API_KEY` configured — showing the most relevant passages "
        "from the paper directly instead of a generated answer."
    )
    for doc in source_docs:
        st.markdown(
            f"<div class='rl-sentence'>{doc.page_content}"
            f"<div class='rl-source'>page {doc.metadata.get('page', '?')}</div></div>",
            unsafe_allow_html=True,
        )


# ---- Sidebar ----------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔬 ResearchLens")
    st.caption("LangChain · FAISS · Llama 3 (Groq) · DeBERTa NLI verification")

    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

    api_key = get_api_key()
    if api_key:
        st.markdown(badge("Groq key configured", "verified"), unsafe_allow_html=True)
    else:
        st.markdown(badge("Offline mode — no Groq key", "flagged"), unsafe_allow_html=True)
        with st.expander("Set up a Groq API key"):
            st.markdown(
                "1. Get a free key at [console.groq.com/keys](https://console.groq.com/keys)\n"
                "2. **Local:** copy `.streamlit/secrets.toml.example` to "
                "`.streamlit/secrets.toml` and paste the key in\n"
                "3. **Streamlit Cloud:** paste it into the app's "
                "**Settings → Secrets** instead — no file needed there"
            )

    if uploaded_file is not None:
        st.divider()
        st.markdown("**Pipeline status**")
        st.markdown(badge("Ingested & chunked", "verified"), unsafe_allow_html=True)
        st.markdown(badge("Embedded (MiniLM)", "verified"), unsafe_allow_html=True)
        st.markdown(badge("Indexed (FAISS)", "verified"), unsafe_allow_html=True)


# ---- Main area ----------------------------------------------------------

if uploaded_file is None:
    # Raw HTML must start at column 0 on each line, or Streamlit's markdown
    # parser can mis-render nested divs as indented text instead of HTML
    # (confirmed via a browser check — the step numbers silently vanished
    # when this block was written with Python-source-matching indentation).
    steps_html = "".join(
        f'<div class="rl-step"><div class="rl-step-num">{i}</div><div>{text}</div></div>'
        for i, text in enumerate(WELCOME_STEPS, start=1)
    )
    st.markdown(
        f'<div class="rl-card"><h3>Welcome to ResearchLens</h3>{steps_html}</div>',
        unsafe_allow_html=True,
    )
    st.stop()

file_id = f"{uploaded_file.name}:{uploaded_file.size}"
if st.session_state.get("file_id") != file_id:
    st.session_state.file_id = file_id
    st.session_state.messages = []
    st.session_state.summaries = {}

with st.spinner("Chunking, embedding, and indexing the paper..."):
    try:
        vectorstore, first_chunk, chunk_count = build_vectorstore_for_file(
            uploaded_file.getvalue(), uploaded_file.name
        )
    except UnusablePDFError as e:
        st.error(f"⚠️ {e}")
        st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # used for chat Q&A
chain = build_rag_chain(retriever, groq_api_key=api_key) if api_key else None

st.markdown(f"### 📄 {uploaded_file.name}")
st.caption(f"{chunk_count} chunks indexed")

st.markdown("#### Summary")
length_choice = st.radio(
    "Summary length", list(SUMMARY_LENGTHS.keys()), horizontal=True, label_visibility="collapsed"
)
length_spec = SUMMARY_LENGTHS[length_choice]

if length_choice not in st.session_state.summaries:
    with st.spinner(f"Generating {length_choice.lower()} summary..."):
        if api_key:
            st.session_state.summaries[length_choice] = generate_summary(
                vectorstore, first_chunk, length_spec["prompt"], length_spec["k"], api_key
            )
        else:
            st.session_state.summaries[length_choice] = None

summary = st.session_state.summaries[length_choice]
if summary:
    render_verified_answer(summary["answer"], summary["verifications"], summary["is_error"])
else:
    # Offline mode still anchors on the abstract chunk, same as the live path.
    docs = anchored_summary_docs(vectorstore, first_chunk, length_spec["prompt"], length_spec["k"])
    render_offline_answer(docs)

st.divider()
st.markdown("#### Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "verifications" in message:
            render_verified_answer(
                message["content"], message["verifications"], message.get("is_error", False)
            )
        else:
            st.markdown(message["content"])

question = st.chat_input("Ask a question about the paper...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if chain:
            with st.spinner("Thinking..."):
                result = answer_with_verification(chain, retriever, question)
            render_verified_answer(result["answer"], result["verifications"], result["is_error"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "verifications": result["verifications"],
                    "is_error": result["is_error"],
                }
            )
        else:
            render_offline_answer(retriever.invoke(question))
            st.session_state.messages.append(
                {"role": "assistant", "content": "(offline mode — see passages above)"}
            )
