# Day 7 — Multi-Modal AI & Retrieval-Augmented Generation (RAG)

## Theory

### Speech-to-Text & Text-to-Speech Integration
Converting between audio and text enables voice-first interfaces. The Voice Interviewer uses **Faster-Whisper** for speech-to-text (STT) transcription and **pyttsx3** for text-to-speech (TTS) synthesis. This creates a fully conversational, hands-free interaction model without requiring user typing.

### LLM-Based Question Generation
Instead of static interview questions, a language model dynamically generates tailored questions by analyzing the candidate's resume against job description requirements. This matches the candidate's actual experience to the role, making interviews more relevant and personalized.

### Multi-Turn Conversation with History
Maintaining message history (system prompt, user inputs, assistant responses) enables the LLM to track context across exchanges. This allows follow-up questions, acknowledgments, and coherent multi-turn dialogue without losing prior context.

### Retrieval-Augmented Generation (RAG)
RAG solves the "hallucination problem" by grounding LLM responses in retrieved documents. The Knowledge Vault ingests external documents, chunks them, embeds them as vectors, and retrieves the most relevant chunks when answering queries. The LLM then generates answers based solely on retrieved context.

### Vector Embeddings & Semantic Search
Text is converted to numerical vectors that capture semantic meaning. Two pieces of text with similar meaning will have vectors close together in vector space. This enables semantic search: finding documents by meaning rather than keyword matching.

### ChromaDB: Vector Database for Semantic Search
ChromaDB is a lightweight vector database that stores embeddings and supports fast similarity-based retrieval. It enables scaling from a few documents to thousands while maintaining O(log n) or better retrieval performance through approximate nearest neighbor search.

---

## Code Examples

### Parsing Resume & Job Description for Matching

> What this demonstrates: Loading and extracting text from both PDF and text files for document comparison

```python
# From: Code/DAY7/voice_interviewer/document_parser.py
def extract_text(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
```

### Generating Interview Questions with LLM

> What this demonstrates: Using Claude/Ollama to dynamically create interview questions based on resume and JD

```python
# From: Code/DAY7/voice_interviewer/llm_engine.py
def generate_questions(self, resume_text, jd_text):
    prompt = f"""
    You are an expert interviewer from Vault-Tec. 
    Compare the following Resume with the Job Description (JD).
    Identify the most relevant experiences in the Resume that match the JD requirements.
    Generate a list of 5 tailored interview questions based on the candidate's actual experience and how it relates to the JD.
    
    Resume:
    {resume_text}
    
    JD:
    {jd_text}
    
    Return ONLY the list of 5 questions, one per line.
    """
    response = ollama.chat(model=self.model_name, messages=[
        {'role': 'user', 'content': prompt}
    ])
    questions = response['message']['content'].strip().split('\n')
    return [q.strip() for q in questions if q.strip()]
```

### Speech Recognition with Faster-Whisper

> What this demonstrates: Recording audio and transcribing it to text using local Whisper model

```python
# From: Code/DAY7/voice_interviewer/voice_engine.py
def listen(self, duration=5):
    audio_file = self.record_audio(duration)
    segments, info = self.stt_model.transcribe(audio_file, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    os.remove(audio_file)
    print(f"You: {text}")
    return text
```

### Multi-Turn Conversation with Message History

> What this demonstrates: Maintaining conversation state across multiple turns with system context

```python
# From: Code/DAY7/voice_interviewer/llm_engine.py
def get_response(self, user_input, context_prompt):
    if not self.history:
        self.history.append({'role': 'system', 'content': context_prompt})
    
    self.history.append({'role': 'user', 'content': user_input})
    
    response = ollama.chat(model=self.model_name, messages=self.history)
    assistant_message = response['message']['content']
    
    self.history.append({'role': 'assistant', 'content': assistant_message})
    return assistant_message
```

### Document Ingestion: Text Extraction & Chunking

> What this demonstrates: Loading PDFs, splitting into overlapping chunks, and preparing for embedding

```python
# From: Code/DAY7/knowledge_vault/ingest.py
def ingest_pdfs(pdf_folder, db_path):
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(pdf_folder, file))
            doc = Document(page_content=text, metadata={"source": file})
            documents.append(doc)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
    )
    return vectorstore
```

### Retrieval-Augmented Generation (RAG) for Q&A

> What this demonstrates: Retrieving relevant context and using it to ground LLM answers

```python
# From: Code/DAY7/knowledge_vault/chat.py
docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in docs])

prompt = f"""
You are a Vault-Tec AI Assistant. Use the following pieces of retrieved context 
from the Vault's archives to answer the question. If you don't know the answer, 
just say that you don't know.

Context:
{context}

Question: {query}

Vault-Tec Recommended Answer:"""

response = llm.invoke(prompt)
```

### Initializing Vector Database Retriever

> What this demonstrates: Loading embeddings from ChromaDB and creating a semantic search retriever

```python
# From: Code/DAY7/knowledge_vault/chat.py
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory=db_path,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

---

## Project Architecture

### Voice Interviewer
```
voice_interviewer/
├── main.py                 ← orchestrates the interview flow
├── document_parser.py      ← extracts text from PDF/text files
├── voice_engine.py         ← handles speech recognition and synthesis
├── llm_engine.py           ← manages LLM interactions and message history
├── requirements.txt        ← dependencies (ollama, whisper, pyttsx3, etc.)
├── sample_resume.txt       ← example resume for testing
└── sample_jd.txt          ← example job description for testing
```

**Flow**: Load Resume + JD → Parse documents → Generate questions → For each Q: Ask → Listen → Acknowledge → Next Q → End session

### Knowledge Vault
```
knowledge_vault/
├── ingest.py              ← PDF ingestion, chunking, embedding, and ChromaDB storage
├── chat.py                ← semantic search and RAG-based Q&A
└── requirements.txt       ← dependencies (langchain, chroma, ollama, etc.)
```

**Flow**: Scan PDF folder → Extract + Chunk → Create embeddings → Store in ChromaDB → Query → Retrieve + Augment → Generate response

---

## Key Technologies

- **Ollama**: Local LLM inference (llama3 for conversation, nomic-embed-text for embeddings)
- **Faster-Whisper**: Fast, accurate speech-to-text transcription
- **pyttsx3**: Text-to-speech for natural voice output
- **ChromaDB**: Vector database for persistent storage and retrieval of embeddings
- **LangChain**: Framework for building LLM applications with retrieval, memory, and chat
- **PyPDF2 / pymupdf**: PDF parsing and text extraction
- **sounddevice**: Audio recording from microphone
