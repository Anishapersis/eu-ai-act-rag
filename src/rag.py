"""EU AI Act RAG assistant, v2.

Same stack as the original notebook (LangChain, Chroma, all-MiniLM-L6-v2,
TinyLlama-1.1B-Chat on CPU) with fixes for the problems the notebook showed:

1. PDF text cleaning: drops the Council header repeated on every page and
   rejoins words split across lines ("high\\n-\\nrisk" -> "high-risk").
2. The vector store is reused: the notebook re-embedded on every run and added
   duplicate chunks to ./chroma_db each time.
3. A relevance threshold on retrieval: if no chunk is similar enough the
   assistant refuses before calling the LLM, instead of relying on the prompt.
4. TinyLlama's chat template, greedy decoding and a repetition penalty, which
   stop the repetition loop and the invented "Human:" turns.
5. Answers cite the PDF pages they came from.

Usage:
    python src/rag.py "Which AI practices are prohibited?"
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "aiact_final_draft.pdf"
DB_DIR = ROOT / "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REFUSAL = "I don't know. The AI Act document does not cover that question."

SYSTEM_PROMPT = (
    "You answer questions about the EU AI Act using only the context provided. "
    "If the context does not contain the answer, reply exactly: \"" + REFUSAL + "\" "
    "Answer in at most five sentences or a short list, and do not repeat yourself."
)

# Header printed at the top of every page of Council document 5662/24.
_HEADER = re.compile(r"^\s*5\s?662/24\s+R\s?B/ek\s+[\d\s]+TREE\.2\.B\s+LIMITE\s+EN\s*", re.S)


# ---------------------------------------------------------------- cleaning --
def clean_page(text: str) -> str:
    """Normalise one page of PDF text extracted by pypdf."""
    text = re.sub(r"\s*\n\s*", "\n", text)                 # strip blank-ish lines
    text = _HEADER.sub("", text)                           # repeated page header
    text = re.sub(r"(\w)\n-\n(\w)", r"\1-\2", text)        # high\n-\nrisk -> high-risk
    text = re.sub(r"(\w)-\n(\w)", r"\1-\2", text)          # high-\nrisk  -> high-risk
    text = re.sub(r"\s+", " ", text)                       # one paragraph of text
    return text.strip()


def load_pages(pdf_path: Path = PDF_PATH) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    if not pdf_path.exists():
        raise FileNotFoundError(f"Put the AI Act PDF at {pdf_path} (see README).")
    pages = PyPDFLoader(str(pdf_path)).load()
    return [Document(page_content=clean_page(p.page_content),
                     metadata={"page": p.metadata.get("page", 0) + 1})
            for p in pages if clean_page(p.page_content)]


def chunk(pages: Sequence[Document], size: int = 800, overlap: int = 100) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                              separators=[". ", "; ", " ", ""])
    return splitter.split_documents(list(pages))


# ------------------------------------------------------------ vector store --
def get_vectorstore(embeddings, docs: Sequence[Document] | None = None, db_dir: Path = DB_DIR):
    """Open the persisted Chroma store, building it once if it is empty."""
    from langchain_chroma import Chroma

    store = Chroma(collection_name="ai_act", embedding_function=embeddings,
                   persist_directory=str(db_dir),
                   collection_metadata={"hnsw:space": "cosine"})
    if store._collection.count() == 0:
        if docs is None:
            docs = chunk(load_pages())
        ids = [f"p{d.metadata['page']}-{i}" for i, d in enumerate(docs)]
        store.add_documents(list(docs), ids=ids)
    return store


# --------------------------------------------------------------- generator --
def tinyllama_generator(model_name: str = LLM_MODEL, max_new_tokens: int = 256) -> Callable[[str, str], str]:
    """Return generate(system, user) -> answer using TinyLlama's chat template."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    def generate(system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.15, no_repeat_ngram_size=4,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return re.split(r"\n\s*(Human|User|<\|user\|>):", text)[0].strip()

    return generate


# --------------------------------------------------------------- assistant --
@dataclass
class Answer:
    text: str
    pages: list[int] = field(default_factory=list)
    best_score: float = 0.0
    refused: bool = False


class AIActAssistant:
    def __init__(self, store, generate: Callable[[str, str], str], k: int = 4, min_score: float = 0.35):
        self.store, self.generate, self.k, self.min_score = store, generate, k, min_score

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_relevance_scores(question, k=self.k)

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)
        best = max((s for _, s in hits), default=0.0)
        if best < self.min_score:
            return Answer(REFUSAL, best_score=best, refused=True)
        context = "\n\n".join(f"[page {d.metadata['page']}] {d.page_content}" for d, _ in hits)
        text = self.generate(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}")
        pages = sorted({d.metadata["page"] for d, _ in hits})
        return Answer(text, pages=pages, best_score=best, refused=text.startswith("I don't know"))


def build_assistant() -> AIActAssistant:
    from langchain_huggingface import HuggingFaceEmbeddings

    store = get_vectorstore(HuggingFaceEmbeddings(model_name=EMBED_MODEL))
    return AIActAssistant(store, tinyllama_generator())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question")
    a = build_assistant().ask(ap.parse_args().question)
    print(a.text)
    print(f"\nSources: pages {a.pages}  (best relevance {a.best_score:.2f})" if a.pages
          else f"\n(refused: best relevance {a.best_score:.2f} below threshold)")
