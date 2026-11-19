"""Offline tests: no model downloads. Uses fake embeddings and a fake generator."""
import json
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
import rag  # noqa: E402


class KeywordEmbeddings(Embeddings):
    """Bag-of-words vectors over a tiny vocabulary, so similarity is predictable."""
    VOCAB = ["prohibited", "practice", "fine", "penalty", "sandbox", "risk", "capital", "usa"]

    def _vec(self, text):
        t = text.lower()
        v = [float(t.count(w)) for w in self.VOCAB] + [0.01]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


DOCS = [Document(page_content="Article 5 Prohibited practice: the following practice shall be prohibited.", metadata={"page": 106}),
        Document(page_content="Article 71 Penalty: a fine of up to 35 000 000 EUR applies.", metadata={"page": 225}),
        Document(page_content="Article 53 Member States shall establish a regulatory sandbox.", metadata={"page": 175})]


@pytest.fixture
def store(tmp_path):
    return rag.get_vectorstore(KeywordEmbeddings(), DOCS, db_dir=tmp_path / "db")


def test_clean_page_removes_header_and_rejoins_hyphens():
    raw = "5662/24 \n \n \nRB/ek \n \n61 \n \nTREE.2.B \nLIMITE \n \nEN \n \nThe high\n-\nrisk system and transparency\n-\nrelated rules"
    assert rag.clean_page(raw) == "The high-risk system and transparency-related rules"


def test_vectorstore_is_not_duplicated_on_reopen(tmp_path):
    rag.get_vectorstore(KeywordEmbeddings(), DOCS, db_dir=tmp_path / "db")
    again = rag.get_vectorstore(KeywordEmbeddings(), DOCS, db_dir=tmp_path / "db")
    assert again._collection.count() == len(DOCS)


def test_out_of_scope_question_is_refused_without_calling_llm(store):
    calls = []
    a = rag.AIActAssistant(store, lambda s, u: calls.append(u) or "answer", k=2).ask("What is the capital of the usa?")
    assert a.refused and a.text == rag.REFUSAL and calls == []


def test_in_scope_answer_cites_pages(store):
    seen = {}
    def fake(system, user):
        seen["user"] = user
        return "Practices listed in Article 5 are prohibited."
    a = rag.AIActAssistant(store, fake, k=2).ask("Which practice is prohibited?")
    assert not a.refused and 106 in a.pages and "[page 106]" in seen["user"]


def test_evaluate_reports_summary(store):
    import evaluate
    qs = {"in_scope": [{"q": "prohibited practice", "pages": [106], "keywords": ["prohibited"]}],
          "out_of_scope": ["capital of usa"]}
    rep = evaluate.evaluate(rag.AIActAssistant(store, lambda s, u: "It is prohibited.", k=1), qs)
    assert rep["summary"]["retrieval_hit_at_k"] == 1.0
    assert rep["summary"]["out_of_scope_refusal_rate"] == 1.0
    assert rep["summary"]["answer_keyword_accuracy"] == 1.0


@pytest.mark.skipif(not rag.PDF_PATH.exists(), reason="PDF not present")
def test_real_pdf_cleaning():
    pages = rag.load_pages()
    text = " ".join(p.page_content for p in pages)
    assert len(pages) == 272 and "TREE.2.B" not in text and text.lower().count("high-risk") > 400


def test_eval_questions_file_is_valid():
    qs = json.loads((ROOT / "eval" / "questions.json").read_text())
    assert len(qs["in_scope"]) >= 5 and len(qs["out_of_scope"]) >= 3
