# Part 1 — Ingestion & chunking (LangChain PyPDFLoader + RecursiveCharacterTextSplitter)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_and_chunk(pdf_path: str):
    """PDF file path -> list of chunked LangChain Document objects."""
    pages = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


if __name__ == "__main__":
    chunks = load_and_chunk("data/paper.pdf")

    print(f"{len(chunks)} chunks\n")

    for i in (0, len(chunks) // 2, len(chunks) - 1):
        chunk = chunks[i]
        print(f"--- chunk {i} (page {chunk.metadata['page']}) ---")
        print(chunk.page_content)
        print()
