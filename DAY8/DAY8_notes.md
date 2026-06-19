# Day 8 — Agentic Systems & Recommendation Engines

## Theory

### Content-Based Recommendations
Content-based recommendation systems find similar items by comparing their feature vectors using mathematical similarity metrics. Each item (destination, product, etc.) is converted into a numeric feature vector, then cosine similarity measures how similar two vectors are. Unlike collaborative filtering, content-based methods don't require user history—only item features and a similarity function.

### Destination Profiling
A destination profile is a numerical summary of travel patterns for a given location. For each destination, the system aggregates trip data to compute average costs (accommodation, transportation, total), duration, traveler demographics (age, gender), and preferences (accommodation type, transportation method). This creates a multi-dimensional vector that captures the "signature" of a destination.

### Cosine Similarity for Recommendations
Cosine similarity measures the angle between two feature vectors in high-dimensional space. Similar destinations (e.g., beach resorts) cluster together, and the system ranks recommendations by similarity score. This approach handles diverse feature types (numerical costs/duration, categorical preferences as ratios) in a single metric without needing labeled data.

### Data Cleaning & Normalization
Raw travel data contains inconsistencies: currency symbols, varied date formats, alias destinations (e.g., "NYC" vs "New York"), and missing values. The DataCleaner systematically normalizes this by standardizing currency, resolving aliases, mapping transportation types to canonical names, and dropping incomplete records. Clean data is essential for accurate feature engineering.

### Natural Language Query Routing
An agentic system interprets natural language queries to determine user intent and route them to appropriate handlers. The system uses keyword matching to classify intent: "stats" routes to dataset summaries, "list" returns all destinations, "about [X]" fetches destination profiles, and "recommend/similar" triggers similarity-based suggestions. This rule-based approach avoids overhead while remaining practical for common queries.

### Feature Scaling & Heterogeneous Features
The feature matrix combines scaled numerical features (cost, duration, age—which have different ranges) with ratio features (accommodation/transport preferences—already normalized to 0–1). StandardScaler normalizes numericals to mean 0, variance 1, while ratios are stacked as-is. This balanced weighting prevents high-range features from dominating similarity calculations.

### Voice Integration in Agents
Voice-enabled agents bridge speech recognition and text processing pipelines. Raw audio is converted to text using speech recognition services (e.g., Google Speech Recognition), then processed through the same NLP pipeline as text input. Voice output requires separate text-to-speech synthesis. This creates a fully conversational interface.

### Local LLM Inference
Running language models locally via Ollama provides offline AI capabilities without external API calls. Models like Qwen3 process requests on-device, enabling privacy-preserving applications. The trade-off is latency—local inference is slower than cloud APIs but maintains full data privacy and reduces costs.

---

## Code Examples

### Data Cleaning: Currency Normalization

> What this demonstrates: removing formatting noise from currency fields

```python
# From: Code/DAY8/TravelInfo/recommender.py
@staticmethod
def _clean_currency(value):
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r'[\$,USD\s"\']', '', str(value))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan
```

### Destination Profiling: Aggregating Trip Data

> What this demonstrates: grouping trips by destination and computing aggregate metrics

```python
# From: Code/DAY8/TravelInfo/recommender.py
def build_profiles(self):
    df = self.df.copy()
    df['Total cost'] = df['Accommodation cost'].fillna(0) + df['Transportation cost'].fillna(0)
    
    profiles = []
    for dest, group in df.groupby('Destination'):
        n = len(group)
        profile = {
            'destination': dest,
            'total_trips': n,
            'avg_total_cost': group['Total cost'].mean(),
            'avg_accommodation_cost': group['Accommodation cost'].mean(),
            'avg_transportation_cost': group['Transportation cost'].mean(),
            'avg_duration': group['Duration (days)'].mean(),
            'avg_age': group['Traveler age'].mean(),
            'gender_male_ratio': (group['Traveler gender'] == 'Male').mean(),
        }
        profiles.append(profile)
```

### Feature Extraction: Mixed Data Types

> What this demonstrates: combining scaled numerical features with ratio features

```python
# From: Code/DAY8/TravelInfo/recommender.py
def get_feature_matrix(self):
    df = self.destination_profiles.copy()
    numerical_cols = [
        'avg_total_cost', 'avg_accommodation_cost', 'avg_transportation_cost',
        'avg_duration', 'avg_age', 'total_trips',
    ]
    ratio_cols = [c for c in df.columns if c not in numerical_cols + ['destination']]
    
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numerical_cols].values)
    X_ratios = df[ratio_cols].values
    X = np.hstack([X_num, X_ratios])
    
    self.feature_matrix = X
    return X, scaler
```

### Recommendation Engine: Similarity Scoring

> What this demonstrates: computing cosine similarity and ranking results

```python
# From: Code/DAY8/TravelInfo/recommender.py
def recommend(self, destination, top_n=5):
    dest = self._resolve_destination(destination)
    if dest is None:
        sample = sorted(self.destinations)[:10]
        return None, f"Destination '{destination}' not found. Try: {', '.join(sample)}..."
    
    idx = self.destinations.index(dest)
    scores = list(enumerate(self.similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [(i, s) for i, s in scores if i != idx][:top_n]
```

### Query Intent Classification: Keyword Matching

> What this demonstrates: routing user queries to appropriate handlers based on keywords

```python
# From: Code/DAY8/TravelInfo/agent.py
def _process_query(self, text):
    low = text.lower()
    
    stats_keywords = ['stats', 'overview', 'summary', 'statistics', 'how many',
                      'male', 'female', 'gender', 'breakdown', 'overall',
                      'average budget', 'avg budget', 'avg cost', 'total trips',
                      'demographic', 'count', 'report']
    if any(kw in low for kw in stats_keywords) and not self._has_dest(low):
        stats = self.rec.overall_stats()
        return self._format_stats_response(stats, text)
```

### Voice Input Capture: Speech-to-Text

> What this demonstrates: capturing and converting speech audio to text using Google's API

```python
# From: Code/DAY8/TravelInfo/agent.py
def _get_voice_input(self):
    """Capture voice input and convert to text."""
    if not self.voice_enabled:
        return None
    try:
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print(f"{c('🎤', BLUE)} listening...", flush=True)
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
        text = self.recognizer.recognize_google(audio)
        print(f"{c('✓', GREEN)} heard: \"{text}\"")
        return text
```

### Interactive Shell Loop with Voice Toggle

> What this demonstrates: maintaining state, toggling features, and handling conversation flow

```python
# From: Code/DAY8/TravelInfo/agent.py
def run(self):
    print(c("Welcome to TravelBot ✈️", CYAN))
    
    while True:
        try:
            # Attempt voice input if enabled, fallback to text
            if self.voice_enabled:
                user_input = self._get_voice_input()
                if user_input is None:
                    print(f"{c('▶', GREEN)} {BOLD}You{RESET}: ", end='', flush=True)
                    user_input = input().strip()
            else:
                user_input = input(f"{c('▶', GREEN)} {BOLD}You{RESET}: ").strip()
            
            if not user_input:
                continue
            
            # Route to handler and display response
            response = self._handle(user_input)
            print(response)
        except KeyboardInterrupt:
            print(self._say("safe travels! ✈️"))
            break
```

### Local LLM Integration for Math

> What this demonstrates: calling a local Ollama model for mathematical reasoning

```python
# From: Code/DAY8/VoiceCalculator/server.py
@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    user_text = data.get('text', '').strip()
    
    # Prompt designed to extract math expression and compute result
    prompt = f"""You are a calculator. The user says: "{user_text}"
Extract the mathematical expression and compute the result.
Reply with ONLY the final numeric answer, nothing else."""
    
    # Make synchronous call to Ollama (local LLM)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            'model': "qwen3:4b",  # Local model
            'prompt': prompt,
            'stream': False  # Wait for complete response
        },
        timeout=30
    )
    
    result_text = response.json().get('response', '').strip()
    return jsonify({'result': result_text, 'original_text': user_text})
```

### Ollama Service Health Check

> What this demonstrates: checking if local LLM backend is running before processing requests

```python
# From: Code/DAY8/VoiceCalculator/server.py
@app.route('/status', methods=['GET'])
def status():
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            return jsonify({
                'status': 'connected',
                'models': model_names,
                'active_model': 'qwen3:4b'
            })
    except:
        pass
    
    return jsonify({
        'status': 'disconnected',
        'error': 'Cannot connect to Ollama'
    }), 503
```

---

## Project Architecture

### TravelInfo — Content-Based Travel Recommender

```
DAY8/TravelInfo/
├── recommender.py      ← Core recommendation engine
│   ├── DataCleaner     Loads & normalizes raw CSV data
│   ├── FeatureEngineer Builds destination feature vectors
│   └── Recommender     Computes similarity & generates recommendations
├── agent.py            ← Interactive CLI chatbot
│   ├── TravelAgent     Main conversational interface
│   ├── _process_query() Intent routing & query handling
│   └── Voice I/O       Speech recognition integration
├── cli.py              ← Command-line interface (list, stats, info, recommend)
└── requirements.txt    speech-recognition, pydub
```

**Data Flow:**
1. Load raw CSV → **DataCleaner** normalizes and validates
2. Cleaned data → **FeatureEngineer** builds destination profiles & feature matrix
3. Feature matrix → **Recommender** computes cosine similarity
4. User query → **TravelAgent** classifies intent → calls Recommender → formats rich output
5. Optional: speech input via microphone → Google Speech Recognition API → text query

---

### VoiceCalculator — Offline Voice-Powered LLM Calculator

```
DAY8/VoiceCalculator/
├── server.py           ← Flask backend
│   ├── /calculate      POST endpoint: voice→text→LLM→answer
│   └── /status         GET endpoint: check Ollama connection
├── index.html          ← Web frontend (browser-based UI)
├── requirements.txt    flask, flask-cors, requests
└── START_HERE.txt      Setup & usage instructions
```

**Execution flow:**
User speaks → Browser captures audio → Sends text to Flask → Flask calls Ollama (local LLM) → LLM computes math → Returns numeric result → Browser speaks answer

**Key design:**
- **Local inference:** No internet, full privacy. Ollama manages model loading & caching.
- **Stateless:** Each request is independent; no session state needed.
- **CORS enabled:** Browser frontend can call Flask backend across origins.
- **Model flexibility:** Change `MODEL_NAME` in server.py to swap models (mistral, llama2, etc.)

---

## Key Takeaways

1. **Content-based recommendations** don't require user history—they only need item features and a similarity metric.
2. **Feature scaling matters:** StandardScaler on numeric features prevents outliers from dominating; category ratios naturally normalize to [0, 1].
3. **Natural language routing** enables conversational UIs without rigid command syntax.
4. **Voice integration** multiplies usability: the same NLP pipeline works for speech and text.
5. **Local LLMs** trade latency for privacy and cost—Ollama makes offline AI practical for consumer devices.
