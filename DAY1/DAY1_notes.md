# Day 1 — AI Fundamentals & Job Agent

## Theory

### Artificial Intelligence (AI)
AI enables machines to perform tasks that normally require human intelligence — learning, reasoning, and decision-making.

**Types of AI:**

| Type | Description | Status |
|------|-------------|--------|
| Narrow AI | Designed for a specific task (voice assistants, recommendation systems) | Exists today |
| General AI | Can perform any intellectual task a human can do | Under research |
| Super AI | Theoretical; would surpass human intelligence | Does not exist yet |

---

### Machine Learning (ML)
Systems learn patterns from data and improve automatically. Widely used for predictions and classification.

### Deep Learning (DL)
Subset of ML that uses neural networks. Highly effective for image, audio, and video processing.

### Generative AI
Creates new content — text, images, audio, code — based on user input.

### Agentic AI
Can perform tasks **autonomously** with minimal human intervention. Makes decisions and takes actions to achieve goals.

---

### Key Python Libraries for ML/AI

| Library | Purpose |
|---------|---------|
| NumPy | Numerical and mathematical computations |
| Pandas | Data analysis and manipulation |
| Scikit-learn | ML algorithms and model training |
| TensorFlow | Deep learning framework by Google |
| PyTorch | Flexible deep learning, popular in research |

---

### HTTP Methods
- **GET** — Retrieve data from a server
- **POST** — Send data to a server

Both are common HTTP methods used in APIs (like FastAPI).

---

### Large Language Model (LLM)
Trained on large amounts of text data to understand and generate human-like language.  
Examples: ChatGPT, Gemini, Grok.

---

### Prompt Engineering
The practice of designing effective prompts to improve the quality of AI-generated outputs.

#### Zero-shot Prompting
No examples provided — the AI relies entirely on its pre-trained knowledge.
```
Translate this sentence to French: "Good morning."
```

#### One-shot Prompting
One example is given before the actual task to guide the model.
```
English: Hello → French: Bonjour
English: Thank you → French: ?
```

#### Chain-of-Thought Prompting
The AI is encouraged to reason step by step before answering. Useful for complex problems.
```
Q: If I have 3 apples and buy 2 more, then give away 1, how many do I have?
A: I start with 3. I buy 2 more → 5. I give away 1 → 4. The answer is 4.
```

---

## Code Examples

### 1. Skill Extraction from Resume / JD Text

> Scans text for known skills using a keyword list.

```python
# Code/DAY1/jobagents/backend/services/analyzer.py

POSSIBLE_SKILLS = [
    'python', 'java', 'react', 'fastapi', 'sql', 'postgresql',
    'machine learning', 'docker', 'aws', 'git', 'kubernetes',
    'javascript', 'typescript', 'data analysis', 'agile', 'scrum'
]

def extract_skills(text: str) -> list:
    text_lower = text.lower()
    found = [s for s in POSSIBLE_SKILLS if s in text_lower]
    return sorted(set(found))

# Example:
# text = "Experienced Python developer with Docker and AWS skills"
# extract_skills(text) → ['aws', 'docker', 'python']
```

---

### 2. Skill Gap Matching

> Compares resume skills against job description skills and computes a match score.

```python
# Code/DAY1/jobagents/backend/services/matcher.py

def calculate_match(resume_skills: list, jd_skills: list) -> tuple:
    """Returns (match_percentage, missing_skills)"""
    if not jd_skills:
        return 0.0, []

    matched = set(resume_skills).intersection(set(jd_skills))
    missing = set(jd_skills) - set(resume_skills)
    score   = (len(matched) / len(jd_skills)) * 100

    return round(score, 2), list(missing)

# Example:
# resume_skills = ['python', 'docker', 'git']
# jd_skills     = ['python', 'docker', 'kubernetes', 'aws']
# score, gaps   = calculate_match(resume_skills, jd_skills)
# → score = 50.0, gaps = ['kubernetes', 'aws']
```

---

### 3. Course Recommendations for Skill Gaps

> Maps missing skills to learning resources.

```python
# Code/DAY1/jobagents/backend/services/recommender.py

COURSE_MAP = {
    "python":     "Python for Automation (Coursera)",
    "docker":     "Docker: Containerization Basics (Docker Docs)",
    "aws":        "AWS: Cloud Infrastructure (A Cloud Guru)",
    "kubernetes": "Kubernetes: Orchestration (KodeKloud)",
    "react":      "React.js: Building User Interfaces (Frontend Masters)",
    "sql":        "SQL: Data Retrieval (Mode Analytics)",
}

def generate_recommendations(missing_skills: list) -> list:
    return [
        COURSE_MAP[skill.lower()]
        for skill in missing_skills
        if skill.lower() in COURSE_MAP
    ]

# Example:
# generate_recommendations(['kubernetes', 'aws'])
# → ['Kubernetes: Orchestration (KodeKloud)', 'AWS: Cloud Infrastructure (A Cloud Guru)']
```

---

### 4. FastAPI Endpoint for Resume Analysis

> Ties everything together — upload a resume and JD, get back a match score and recommendations.

```python
# Code/DAY1/jobagents/backend/api/routes.py

from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/analyze")
async def analyze(resume: UploadFile, jd: UploadFile):
    resume_text = parse_file(resume)        # extract text from PDF/DOCX
    jd_text     = parse_file(jd)

    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(jd_text)

    score, missing = calculate_match(resume_skills, jd_skills)
    courses        = generate_recommendations(missing)

    return {
        "match_score":    score,
        "missing_skills": missing,
        "courses":        courses
    }
```

---

### 5. Parsing PDF and DOCX Files

> Extracts plain text from uploaded resume files.

```python
# Code/DAY1/jobagents/backend/services/parser.py

import PyPDF2
from docx import Document

def parse_file(file) -> str:
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file.file)
        return " ".join(page.extract_text() for page in reader.pages)

    elif filename.endswith(".docx"):
        doc = Document(file.file)
        return " ".join(p.text for p in doc.paragraphs)

    return ""
```

---

## Project Architecture

```
jobagents/
├── backend/
│   ├── main.py              ← FastAPI app + CORS setup
│   ├── api/routes.py        ← REST endpoints
│   ├── services/
│   │   ├── parser.py        ← PDF / DOCX text extraction
│   │   ├── analyzer.py      ← Skill extraction
│   │   ├── matcher.py       ← Match score + gap analysis
│   │   ├── recommender.py   ← Course suggestions
│   │   └── gemini_client.py ← Google Gemini LLM integration
│   └── models/schemas.py    ← Pydantic validation schemas
└── frontend/
    └── streamlit_app.py     ← Interactive web UI
```
