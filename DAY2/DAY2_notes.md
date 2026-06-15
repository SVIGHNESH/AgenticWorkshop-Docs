# Day 2 — Retrieval Augmented Generation (RAG)

## Theory

### What is RAG?
RAG is an AI architecture that combines **information retrieval** with **LLMs**.  
Instead of relying only on the model's training data, RAG retrieves relevant information from external knowledge sources and uses it to generate accurate, grounded answers.

---

### Five Pillars of RAG

| Pillar | Role | Examples |
|--------|------|---------|
| Documents | Knowledge sources | PDFs, Word files, web pages, databases |
| Embedding Model | Converts text into vectors | all-MiniLM-L6-v2, text-embedding-ada-002 |
| Vector Database | Stores and searches embeddings | FAISS, ChromaDB, Pinecone |
| Retriever | Finds the most relevant chunks | Similarity search (cosine / dot product) |
| LLM | Reasons and generates the final answer | Groq, GPT-4, Gemini |

---

### Embedding Concepts

**Bag of Words (BoW)**  
Represents text using word frequencies. No understanding of meaning or order.
```
"cat sat mat" → {cat: 1, sat: 1, mat: 1}
```

**Word2Vec**  
Dense vectors where similar words have similar representations.  
Famous example: `king − man + woman ≈ queen`

---

### RAG Pipeline — Phase 1: Indexing

```
Documents
    ↓
Chunking  (split into 800-char pieces with 150-char overlap)
    ↓
Embeddings  (each chunk → a 384-dim vector)
    ↓
Vector Database  (FAISS / ChromaDB — stored for fast search)
```

---

### RAG Pipeline — Phase 2: Retrieval & Generation

```
User Query
    ↓
Embed the query  (same model used during indexing)
    ↓
Vector Search  (find top-K most similar chunks)
    ↓
Relevant Chunks  (sent to the LLM as context)
    ↓
LLM  (generates a grounded answer using the context)
    ↓
Final Answer
```

---

### Applications of RAG

- **Enterprise Search** — search internal company documents
- **HR Assistant** — answer policy and procedure questions
- **Legal Assistant** — search contracts, laws, and precedents
- **Healthcare Assistant** — medical knowledge Q&A
- **Customer Support** — product manuals and FAQs
- **Education** — textbook question-answering chatbot

---

### LangChain

Open-source framework for building LLM-powered applications.  
Connects models, prompts, memory, tools, and agents.

**Architecture:**
```
Your App → Chains / Agents → Prompts + Memory + Tools → LLM Wrapper → LLM
```

**Components:**

| Component | Purpose |
|-----------|---------|
| Prompt Templates | Reusable prompt structures with variable placeholders |
| LLM Wrappers | Connect to OpenAI, Gemini, Anthropic, Hugging Face |
| Chains | Link multiple operations sequentially |
| Output Parsers | Convert raw LLM output into structured formats |
| Memory | Store and retrieve conversation history |
| Tools | External functions (search, calculator, database) |
| Agents | AI that decides which tools to use and in what order |

---

### Hugging Face
A platform hosting thousands of open-source AI models for NLP, embeddings, vision, and LLM tasks.

### NLP (Natural Language Processing)
Field of AI enabling computers to understand and generate human language.

---

## Code Examples

### 1. PDF Text Extraction

> Reads all pages of an uploaded PDF and returns plain text.

```python
# Code/DAY2/RAGCHATBOT/rag.py

from pypdf import PdfReader

def extract_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)
```

---

### 2. Chunking Text with Overlap

> Splits a long document into smaller pieces so each chunk fits in the model's context window.  
> Overlap ensures a sentence split across a boundary still appears fully in at least one chunk.

```python
# Code/DAY2/RAGCHATBOT/rag.py

CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 150   # overlap between consecutive chunks

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    text   = " ".join(text.split())   # normalise whitespace
    chunks = []
    start  = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap       # slide forward, keeping overlap
    return chunks
```

---

### 3. Building a FAISS Vector Index

> Embeds all chunks and stores them in a FAISS index for fast similarity search.

```python
# Code/DAY2/RAGCHATBOT/rag.py

from sentence_transformers import SentenceTransformer
import faiss

class PDFIndex:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks   = []
        self.index    = None

    def build(self, text: str) -> int:
        self.chunks    = chunk_text(text)
        embeddings     = self.embedder.encode(
            self.chunks, normalize_embeddings=True
        ).astype("float32")

        # IndexFlatIP = inner product (equivalent to cosine sim on normalised vectors)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        return len(self.chunks)
```

---

### 4. Retrieving Relevant Chunks

> Given a user question, find the top-K most similar chunks from the index.

```python
# Code/DAY2/RAGCHATBOT/rag.py

TOP_K = 4

def retrieve(self, question: str) -> list[str]:
    q = self.embedder.encode(
        [question], normalize_embeddings=True
    ).astype("float32")

    _, idx = self.index.search(q, TOP_K)
    return [self.chunks[i] for i in idx[0] if i != -1]
```

---

### 5. Building the LLM Prompt with Context

> Injects retrieved chunks as context so the LLM answers from the document, not from hallucinated knowledge.

```python
# Code/DAY2/RAGCHATBOT/rag.py

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided context from a PDF document. If the answer is not in the "
    "context, say you don't know. Be concise."
)

def build_messages(question: str, context_chunks: list, history: list) -> list:
    context  = "\n\n---\n\n".join(context_chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)   # previous conversation turns
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })
    return messages
```

---

### 6. Streamlit Chat UI

> Provides a web interface for uploading PDFs and chatting with them.

```python
# Code/DAY2/RAGCHATBOT/app.py  (simplified)

import streamlit as st

st.title("PDF RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    text  = extract_text(uploaded_file)
    index = PDFIndex()
    n     = index.build(text)
    st.success(f"Indexed {n} chunks.")

    question = st.chat_input("Ask a question about the PDF")
    if question:
        chunks   = index.retrieve(question)
        messages = build_messages(question, chunks, st.session_state.history)
        # stream response from Groq LLM
        response = call_groq(messages)
        st.chat_message("assistant").write(response)
```

---

## Full RAG Flow Summary

```
PDF Upload
    ↓  extract_text()
Raw Text
    ↓  chunk_text()
List of Chunks
    ↓  SentenceTransformer.encode()
Embeddings (float32 vectors)
    ↓  faiss.IndexFlatIP.add()
FAISS Index
         ←── User Question ──→ encode question ──→ index.search()
                                                        ↓
                                               Top-K relevant chunks
                                                        ↓
                                               build_messages()
                                                        ↓
                                               Groq LLM (streaming)
                                                        ↓
                                               Final Answer shown in UI
```
