# EU AI Act RAG Assistant (LangChain · Chroma · TinyLlama)

Ask questions about the EU Artificial Intelligence Act (retrieval hit@4 of 0.88, 100% refusal of out-of-scope questions) and get answers grounded in the text of the Act, generated **entirely on a laptop CPU** with open models: no paid API and no GPU.

| Component | Choice |
|---|---|
| Document loading | `PyPDFLoader` (LangChain), 272 pages |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | Chroma, persisted to disk |
| Generator | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` via Hugging Face Transformers |
| Orchestration | LangChain retrieval chain (v1), plain Python pipeline (v2) |

## v1: baseline notebook

[`notebooks/01_rag_tinyllama_original.ipynb`](notebooks/01_rag_tinyllama_original.ipynb) builds the full pipeline: load the PDF (272 pages), split into 1,507 chunks of 500 characters with 50 overlap, embed with MiniLM, store in Chroma, retrieve the top 3 chunks, and answer with TinyLlama using a prompt that tells the model to answer only from the context.

The outputs are kept exactly as they ran. They revealed two issues that shaped v2:

1. **Repetition loop.** "What are the key risk categories in the AI Act?" produced a generic list that degenerated into repeating *"Risk transfer and governance/management/control risks"* until the token limit. None of the items are the Act's actual tiers: prohibited practices, high-risk systems, transparency obligations and general-purpose AI models.
2. **Grounding ignored.** "What is capital of Usa?" was answered from the model's own knowledge ("Washington, D.C.") and the model then invented a new `Human:` turn, despite the instruction to say it does not know.

## What caused them

| Problem | Cause |
|---|---|
| Repetition loop | Sampling at temperature 0.3 with no repetition penalty, on a 1.1B model |
| Invented `Human:` turns | The prompt went to a plain text-generation pipeline, so TinyLlama's chat template was never applied and the model just continued a transcript |
| Out-of-scope question answered | Refusal depended only on the prompt. Small models follow such instructions unreliably, and the retriever always returns *something*, even for unrelated questions |
| Noisy chunks | pypdf splits hyphenated words across lines (`high\n-\nrisk`: the phrase "high-risk" appears 421 times but did not survive extraction as written) and every page repeats a Council header (`5662/24 RB/ek … LIMITE EN`) that was embedded into the chunks |
| Duplicate vectors | `Chroma.from_documents(..., persist_directory=...)` on every run appends the same chunks again to `./chroma_db` |

## v2: fixes in [`src/rag.py`](src/rag.py)

- **Text cleaning:** removes the page header and rejoins split hyphenated words; 1,043 cleaner chunks of 800 characters, split on sentence boundaries.
- **Persistent store:** builds the Chroma collection once, with stable IDs, and reuses it afterwards.
- **Relevance gate:** if the best cosine relevance is below a threshold, the assistant refuses *before* calling the LLM.
- **Proper chat formatting:** TinyLlama's chat template, greedy decoding, `repetition_penalty=1.15` and `no_repeat_ngram_size=4`, with any stray `Human:` continuation cut off.
- **Citations:** every answer lists the PDF pages it was drawn from.

```bash
pip install -r requirements.txt
# put the PDF at data/aiact_final_draft.pdf
python src/rag.py "Which artificial intelligence practices are prohibited?"
python eval/evaluate.py          # retrieval hit rate, answer keywords, refusal rate
pytest -q                        # offline tests (no model downloads)
```

## Evaluation

[`eval/questions.json`](eval/questions.json) holds 8 in-scope questions, each with the page(s) of the Act that answer it, located in the text (Article 5 on prohibited practices, the Article 71 fine of EUR 35 000 000 or 7 % of turnover, the 10^25 FLOP threshold for systemic-risk models, and so on). It also holds 4 out-of-scope questions. `eval/evaluate.py` reports:

- retrieval hit@k: an expected page is among the retrieved chunks
- answer keyword accuracy
- out-of-scope refusal rate
- in-scope questions wrongly refused

### v2 results

Measured on a Windows laptop CPU with `all-MiniLM-L6-v2` and TinyLlama-1.1B-Chat (k = 4, relevance threshold 0.35):

| Metric | Result |
|---|---|
| Retrieval hit@4 (expected page retrieved) | **0.88** (7 of 8) |
| Out-of-scope questions refused | **1.00** (4 of 4; v1 answered "capital of USA") |
| In-scope questions wrongly refused | **0** of 8 |
| Answer contains the expected key fact | 0.38 (3 of 8) |

**Reading the results:** retrieval and the refusal gate work. The weak link is generation: a 1.1B model often paraphrases loosely or omits the exact figure (for example, the EUR 35 000 000 fine) even when the right page is in its context. Because retrieval already finds the right page 7 times out of 8, the next steps target the generator:

- swap in a larger local model (for example Phi-3-mini or Qwen2.5-3B through Ollama) and re-run the same evaluation
- extract quoted sentences from the retrieved chunks before generating
- add a reranker and test a hybrid BM25 + vector retriever for the one retrieval miss

## Data

The notebook used Council of the EU document 5662/24 (26 January 2024), the final compromise text of the AI Act. That version is marked LIMITE, so it is not redistributed here. The official published text is [Regulation (EU) 2024/1689 on EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/oj). The pipeline works on either PDF, but the evaluation page numbers refer to document 5662/24.

## Project structure

```
├── notebooks/01_rag_tinyllama_original.ipynb   # v1, original run with outputs
├── src/rag.py                                  # v2 pipeline
├── eval/questions.json, eval/evaluate.py       # evaluation set and runner
├── tests/test_rag.py                           # offline tests with fake embeddings
├── data/                                       # put the PDF here
└── requirements.txt
```

## Author

**Anisha Persis Christopher**, Data Scientist · [LinkedIn](https://www.linkedin.com/in/anishapersis) · [GitHub](https://github.com/Anishapersis)
