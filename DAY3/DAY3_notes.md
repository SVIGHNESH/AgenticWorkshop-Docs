# Day 3 — LangChain

## Theory

### LangChain
An open-source framework for building applications powered by LLMs.  
It connects models, prompts, memory, tools, and agents into cohesive pipelines.

**Architecture:**
```
Your App
    ↓
Chains / Agents
    ↓
Prompts + Memory + Tools
    ↓
LLM Wrapper
    ↓
LLM Model (OpenAI / Gemini / Anthropic / Hugging Face)
```

**Components:**

| Component | What it does |
|-----------|-------------|
| Prompt Templates | Reusable prompt structures with variable placeholders |
| LLM Wrappers | Connect LangChain to any LLM provider |
| Chains | Link multiple operations sequentially |
| Output Parsers | Convert raw LLM output into structured formats (JSON, lists) |
| Memory | Store and retrieve conversation history / context |
| Tools | External functions — search, APIs, calculators, databases |
| Agents | AI that decides which tools to use and in what order |

---

### Hugging Face
A platform and community hosting thousands of open-source AI models used for NLP, embeddings, vision, and LLM applications.

### NLP (Natural Language Processing)
The field of AI that enables computers to understand and generate human language.

---

## Code Examples

### 1. Prompt Templates

> Reusable prompt structures with variable placeholders.

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

prompt = template.format(topic="Retrieval Augmented Generation")
print(prompt)
# "Explain Retrieval Augmented Generation in simple terms."
```

---

### 2. LLM Wrapper

> Connect LangChain to an LLM provider and call it.

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")

response = llm.invoke([HumanMessage(content="What is LangChain?")])
print(response.content)
```

---

### 3. Chains — Linking Operations Together

> Chain a prompt template and an LLM into one pipeline.

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

llm      = ChatOpenAI(model="gpt-4o-mini")
template = PromptTemplate(
    input_variables=["skill"],
    template="Give me a 2-sentence explanation of {skill}."
)

chain = LLMChain(llm=llm, prompt=template)

result = chain.run(skill="FAISS vector search")
print(result)
```

---

### 4. Output Parsers

> Convert raw LLM text output into structured Python objects.

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

parser = CommaSeparatedListOutputParser()

template = PromptTemplate(
    input_variables=["topic"],
    template="List 5 key concepts in {topic}.\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain  = template | ChatOpenAI(model="gpt-4o-mini") | parser
result = chain.invoke({"topic": "RAG"})
print(result)
# ['Documents', 'Embeddings', 'Vector Database', 'Retriever', 'LLM']
```

---

### 5. Tools

> Give the LLM access to external functions it can call.

```python
from langchain.tools import tool

@tool
def get_word_count(text: str) -> int:
    """Returns the number of words in a given text."""
    return len(text.split())

# The agent can now call get_word_count() when it needs to count words
print(get_word_count.name)        # "get_word_count"
print(get_word_count.description) # "Returns the number of words in a given text."
```

---

### 6. Agents

> An agent decides which tools to use and in what order to achieve a goal.

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    return f"Search results for: {query}"

llm   = ChatOpenAI(model="gpt-4o-mini")
agent = initialize_agent(
    tools=[search_web],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("What is the latest version of LangChain?")
# Agent decides to call search_web("LangChain latest version")
# Then uses the result to answer
```
