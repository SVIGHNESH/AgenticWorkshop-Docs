# Day 5 — Embeddings (Word, Sentence, Document, Contextual, Image, Audio, Multi-Modal)

## Theory

### What are Embeddings?
An embedding is a **numerical vector** that represents data in a mathematical space where **similar things are placed close together**.

```
"cat"  → [0.12, -0.45, 0.89, ..., 0.31]   (384 numbers)
"dog"  → [0.11, -0.43, 0.91, ..., 0.29]   ← close to "cat"
"car"  → [0.98,  0.21, -0.33, ..., 0.05]  ← far from "cat"
```

**Why do we need embeddings?**
- Computers cannot understand words directly
- Vectors let us do **math on meaning**
- We can measure similarity: `cosine(cat, dog) > cosine(cat, car)`

---

### 1. Word Embeddings
Each word or short phrase gets its own fixed vector.  
**Model:** `SentenceTransformer("all-MiniLM-L6-v2")`  
**Output shape:** `(3, 384)` for 3 sentences → each represented by 384 numbers.

---

### 2. Sentence Embeddings
The **entire sentence** is encoded into one single vector.  
Captures overall meaning, not just individual words.  
**Use case:** Find which job description best matches a resume sentence.

---

### 3. Document Embeddings
Entire paragraphs or documents are embedded and compared using **cosine similarity** to find the most relevant document for a query.

**This is the foundation of the RAG pipeline:**
```
PDF → Text Extraction → Chunking (500–1000 chars) → Embeddings → FAISS / ChromaDB → Similarity Search → LLM
```

---

### 4. Contextual Embeddings
Traditional embeddings give the **same vector** to a word regardless of context.  
Contextual embeddings **change the vector** based on surrounding words.

**Example — the word "bank":**
```
"I deposited money in the bank."   → financial context  → vector A
"I sat on the bank of the river."  → geographical context → vector B

cosine_similarity(A, B) = 0.48   ← same word, different meaning, different vector
```

**Model:** `bert-base-uncased`  
**Output:** `(1, T, 768)` — 1 sentence, T tokens, 768 dimensions per token

---

### 5. Image Embeddings
An image is passed through a neural network and converted into a vector that captures its visual features.

```
Image → Neural Network (CNN / ViT / CLIP) → [0.12, -0.45, 0.89, ...]
```

| Model | Output Shape | Notes |
|-------|-------------|-------|
| ViT (`google/vit-base-patch16-224`) | `(1, 768)` | General visual features |
| CLIP (`openai/clip-vit-base-patch32`) | `(1, 512)` | Images AND text in the same space |

**CLIP is special:** it embeds both images and text in the same vector space, enabling **text-to-image search**.

---

### 6. Audio Embeddings
Audio signals are converted into vectors that capture:
- Speech content (what was said)
- Speaker characteristics (who said it)
- Emotion or tone
- Music style and instruments
- Environmental sounds (rain, traffic, birds, etc.)

Just like text embeddings convert sentences to vectors, audio embeddings convert **sound waves** to vectors.

**Model:** `facebook/wav2vec2-base-960h` → `(1, 768)` vector

---

### 7. Multi-Modal Embeddings
A multi-modal embedding model converts **different types of data** into vectors that **all live in the same vector space**.

```
Text  → Vector ┐
Image → Vector ┼── all in the same vector space
Audio → Vector ┘
Video → Vector ┘
```

This enables **cross-modal search:**
- "Find images similar to this text description"
- "Find audio clips similar to this image"

---

## Code Examples

### 1. Word Embeddings

> Generate vector representations for sentences using SentenceTransformer.

```python
# From: Code/DAY5/Embeddings/

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Artificial Intelligence is amazing.",
    "Machine Learning is a subset of AI.",
    "I am learning Transformers.",
]

embeddings = model.encode(sentences)
print("Embedding Shape:", embeddings.shape)
# (3, 384)

print("First Embedding:", embeddings[0])
# [-0.016, -0.085, 0.070, ...]
```

---

### 2. Sentence Embeddings — Job Finding Use Case

> Find which job best matches a candidate's background.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Artificial Intelligence is transforming the world.",
    "Machine Learning is a branch of AI.",
    "I am building a RAG chatbot using Hugging Face.",
]

embeddings = model.encode(sentences)
print("Embedding shape:", embeddings.shape)
# (3, 384)

for i, sentence in enumerate(sentences):
    print(f"\nSentence: {sentence}")
    print(f"Embedding length: {len(embeddings[i])}")
    print(f"First 10 values: {embeddings[i][:10]}")
```

---

### 3. Document Embeddings + Best Match

> Embed entire documents and find the one most relevant to a query using cosine similarity.

```python
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Java is a high-level, object-oriented programming language. It is widely used for enterprise applications.",
    "Spring Boot simplifies Java application development by providing auto-configuration and embedded servers.",
    "Transformers are deep learning models that have revolutionized natural language processing.",
]

document_embeddings = model.encode(documents, convert_to_tensor=True)
print("Shape:", document_embeddings.shape)
# torch.Size([3, 384])

query           = "How can I build backend APIs with Java?"
query_embedding = model.encode(query, convert_to_tensor=True)

scores     = util.cos_sim(query_embedding, document_embeddings)
best_match = torch.argmax(scores)

print("Query:", query)
print("Most Relevant Document:", documents[best_match])
# → "Java is a high-level, object-oriented programming language..."
```

---

### 4. Contextual Embeddings with BERT

> Every token gets its own context-aware 768-dim vector.

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = AutoModel.from_pretrained("bert-base-uncased")

sentence = "I am learning contextual embeddings using BERT."
inputs   = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

contextual_embeddings = outputs.last_hidden_state
print("Shape:", contextual_embeddings.shape)
# torch.Size([1, 14, 768])
# 1 sentence, 14 tokens, 768 dimensions each

# Each token has its own vector:
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
for token, embedding in zip(tokens, contextual_embeddings[0]):
    print(token, embedding.shape)
# [CLS] torch.Size([768])
# i     torch.Size([768])
# am    torch.Size([768])
# ...
```

---

### 5. Proving "bank" Has Different Vectors in Different Contexts

> Same word, different embedding — this is what makes BERT powerful.

```python
sentence1 = "I deposited money in the bank."
sentence2 = "I sat on the bank of the river."

inputs1 = tokenizer(sentence1, return_tensors="pt")
inputs2 = tokenizer(sentence2, return_tensors="pt")

with torch.no_grad():
    emb1 = model(**inputs1).last_hidden_state
    emb2 = model(**inputs2).last_hidden_state

tokens1 = tokenizer.convert_ids_to_tokens(inputs1["input_ids"][0])
tokens2 = tokenizer.convert_ids_to_tokens(inputs2["input_ids"][0])

bank_vec1 = emb1[0][tokens1.index("bank")]
bank_vec2 = emb2[0][tokens2.index("bank")]

similarity = torch.nn.functional.cosine_similarity(
    bank_vec1.unsqueeze(0),
    bank_vec2.unsqueeze(0)
)
print("Similarity:", similarity.item())
# 0.48 → same word, different context → different vectors
```

---

### 6. Image Embeddings with ViT

> Convert an image into a 768-dim feature vector using Vision Transformer.

```python
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model     = AutoModel.from_pretrained("google/vit-base-patch16-224")

image  = Image.open("image.jpeg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# [CLS] token (index 0) summarises the entire image
embedding = outputs.last_hidden_state[:, 0, :]
print(embedding.shape)
# torch.Size([1, 768])
```

---

### 7. Image Embeddings with CLIP (OpenAI)

> CLIP puts images AND text in the same 512-dim space — enables cross-modal comparison.

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image  = Image.open("image.jpeg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    image_features = model.get_image_features(**inputs)

embedding = image_features.pooler_output
print(embedding.shape)
# torch.Size([1, 512])
```

---

### 8. Audio Embeddings with Wav2Vec2

> Convert a speech audio file into a 768-dim embedding vector.

```python
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

waveform, sample_rate = torchaudio.load("audio.wav")

# Wav2Vec2 expects 16 kHz audio
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform  = resampler(waveform)

inputs = processor(
    waveform.squeeze().numpy(),
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

# Average over time steps → one vector per audio clip
embedding = outputs.last_hidden_state.mean(dim=1)
print(embedding.shape)
# torch.Size([2, 768])
```

---

### 9. Multi-Modal Embeddings — Concept

> Different data types, same vector space.

```python
# Text, Image, and Audio all become vectors in the same space
# CLIP (by OpenAI) is the most widely used multi-modal model

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("image.jpeg")

# Encode IMAGE
img_inputs     = processor(images=image, return_tensors="pt")
image_vector   = model.get_image_features(**img_inputs)   # shape: (1, 512)

# Encode TEXT
text_inputs    = processor(text=["a dog running in a park"], return_tensors="pt")
text_vector    = model.get_text_features(**text_inputs)   # shape: (1, 512)

# Compare across modalities — same vector space!
similarity = torch.nn.functional.cosine_similarity(image_vector, text_vector)
print("Image-Text Similarity:", similarity.item())
# High score → image matches the text description
```

---

## All Embedding Types — Summary Table

| # | Type | Model | Output Shape | Use Case |
|---|------|-------|-------------|---------|
| 1 | Word / Sentence | `all-MiniLM-L6-v2` | `(N, 384)` | Text search, RAG, job matching |
| 2 | Document | `all-MiniLM-L6-v2` | `(N, 384)` | RAG retrieval, document ranking |
| 3 | Contextual | `bert-base-uncased` | `(1, T, 768)` | NLP, context-aware understanding |
| 4 | Image (ViT) | `vit-base-patch16-224` | `(1, 768)` | Image search, classification |
| 5 | Image (CLIP) | `clip-vit-base-patch32` | `(1, 512)` | Text-to-image cross-modal search |
| 6 | Audio | `wav2vec2-base-960h` | `(1, 768)` | Speech recognition, audio search |
| 7 | Multi-Modal | CLIP (text + image jointly) | `(1, 512)` | Cross-modal retrieval |
