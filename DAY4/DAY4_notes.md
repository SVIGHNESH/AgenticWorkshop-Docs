# Day 4 — Memory in AI Agents

## Theory

### What is Memory in AI Agents?
Memory is the ability of an AI agent to **remember information from previous interactions** and use it in future responses.

Without memory → every conversation starts from scratch.  
With memory → the agent maintains context, personalises responses, and completes multi-step tasks.

---

### Why Memory is Important

- Maintain conversation context across turns
- Improve user experience (no need to repeat yourself)
- Enable personalised AI responses
- Help AI agents complete long workflows
- Reduce repeated user input

---

### Four Types of Memory

#### 1. Buffer Memory
Stores the **entire** conversation history in sequence. Every user message and AI response is saved.

```
Turn 1: User: "My name is Vighnesh"   AI: "Nice to meet you, Vighnesh!"
Turn 2: User: "What is my name?"      AI: "Your name is Vighnesh."
```

✅ Complete, lossless history  
⚠️ Grows indefinitely — can hit token limits on long conversations

---

#### 2. Summary Memory
Creates a **compressed summary** of the conversation and stores only the important points.

```
Long conversation about a project
    ↓
Summary: "User is building a RAG chatbot using FastAPI and Groq. Needs help with chunking."
```

✅ Stays small regardless of conversation length  
⚠️ Some detail is lost in compression

---

#### 3. Window Memory
Keeps only the **last K interactions**. Older messages are automatically removed.

```
K = 3 → keeps Turn 8, Turn 9, Turn 10 → Turn 7 and earlier are dropped
```

✅ Bounded, predictable token usage  
⚠️ AI forgets things said early in the conversation

---

#### 4. Entity Memory
Stores important **facts** extracted from the conversation — not the whole history, just key entities and what is known about them.

**What it stores:**

| Entity Type | Example |
|-------------|---------|
| Person Name | Vighnesh |
| Company | TechCorp |
| Location | Mumbai |
| Skills | Python, FastAPI |
| Dates | Interview on Monday |
| Projects | RAG Chatbot |

✅ Very efficient — stores exactly the right information  
⚠️ Needs extraction logic to identify and update facts

---

### Memory Comparison Table

| Memory Type | What is Stored | Token Usage | Detail Level |
|-------------|---------------|-------------|--------------|
| Buffer Memory | Full conversation | Grows large | Complete |
| Summary Memory | Compressed summary | Small | Approximate |
| Window Memory | Last K messages | Fixed | Recent only |
| Entity Memory | Key facts (entities) | Minimal | Specific facts |

---

## Code Examples

### 1. Buffer Memory Bot

> Stores every message — AI always has the full conversation history.

```python
# Code/DAY4/BufferMemoryBot/BufferMemoryBot.ipynb

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import OpenAI

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=OpenAI(),
    memory=memory,
    verbose=True
)

conversation.predict(input="My name is Vighnesh")
conversation.predict(input="What is my name?")
# AI: "Your name is Vighnesh."

# Inspect stored history:
print(memory.chat_memory.messages)
# [HumanMessage("My name is Vighnesh"), AIMessage("Nice to meet you, Vighnesh!"), ...]
```

---

### 2. Summary Memory Bot

> Long conversations are compressed into a summary — saves tokens.

```python
# Code/DAY4/BufferMemoryBot/SummaryMemoryBot.ipynb

from langchain.memory import ConversationSummaryMemory
from langchain_openai import OpenAI

llm    = OpenAI()
memory = ConversationSummaryMemory(llm=llm)

conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input="I am building a RAG chatbot.")
conversation.predict(input="I am using FastAPI and Groq LLM.")
conversation.predict(input="I need help with the chunking logic.")

# Instead of storing all 3 raw messages, memory stores a summary:
print(memory.buffer)
# "User is building a RAG chatbot using FastAPI and Groq. Needs help with chunking."
```

---

### 3. Window Memory Bot

> Only keeps the last K turns — older context is automatically discarded.

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_openai import OpenAI

# Keep only the last 3 interactions
memory = ConversationBufferWindowMemory(k=3)

conversation = ConversationChain(llm=OpenAI(), memory=memory)

conversation.predict(input="Turn 1 — I am Vighnesh")
conversation.predict(input="Turn 2 — I work at TechCorp")
conversation.predict(input="Turn 3 — I like Python")
conversation.predict(input="Turn 4 — What did I say first?")
# AI no longer has Turn 1 — it was dropped when Turn 4 came in
```

---

### 4. PDF Memory Bot

> Combines PDF RAG (Day 2) with Buffer Memory — enables follow-up questions on a document.

```python
# Code/DAY4/BufferMemoryBot/PDFMemoryBot.ipynb

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

qa_chain({"question": "What is the main topic of the PDF?"})
qa_chain({"question": "Can you explain the second point?"})
# The second question uses chat_history to understand "second point" in context
```

---

### 5. Entity Memory Bot

> Extracts and stores specific named facts — does not keep the raw conversation.

```python
from langchain.memory import ConversationEntityMemory
from langchain.chains import ConversationChain
from langchain_openai import OpenAI

llm    = OpenAI()
memory = ConversationEntityMemory(llm=llm)

conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input="My name is Vighnesh and I work at TechCorp.")
conversation.predict(input="I am building a project called RAG Chatbot.")

# Memory stores structured facts, not raw text:
print(memory.entity_store.store)
# {
#   "Vighnesh":    "works at TechCorp, building RAG Chatbot",
#   "TechCorp":    "Vighnesh's employer",
#   "RAG Chatbot": "project being built by Vighnesh"
# }
```
