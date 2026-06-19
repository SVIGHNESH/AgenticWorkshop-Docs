# Day 9 — AI Agent with Tool Use (Medical Receptionist)

## Theory

### Agentic Tool Use
An AI agent is given a set of callable tools and autonomously decides which to invoke based on the user's query. Instead of generating a static answer, the model reasons about what data it needs, calls the appropriate tool, and incorporates the result into its final response. This turns the LLM from a knowledge store into an orchestrator that can query live data.

### Function Calling vs Prompt-Based Tool Parsing
Two approaches are shown side-by-side:
- **Native function calling** (Gemini API): tools are registered as Python callables; the model emits structured `FunctionCall` objects, and the SDK handles execution automatically with `enable_automatic_function_calling=True`.
- **Prompt-engineered tool calling** (Ollama): tools are described in the system prompt; the model emits `<tool_use>tool_name(param=value)</tool_use>` tags that are parsed with regex and dispatched manually. This works with any model that follows instructions.

### Conversation History Management
Both agents maintain a rolling `conversation_history` list so the model has context across turns. The Gemini backend stores the SDK's native `chat.history` object; the Ollama backend stores plain `{"role", "content"}` dicts and keeps only the last 10 turns to stay within the context window.

### Data Layer Abstraction
All hospital data lives in CSV files loaded at startup by `MedicalDataLoader`. The agent never queries files directly — it calls methods on the loader, which return formatted strings the LLM can read. This separation means the agent logic is data-agnostic and the loader can be swapped for a real database without touching the agent.

### Multi-Backend Architecture
The project ships two complete implementations (Gemini cloud, Ollama local) behind the same Flask API surface. The frontend `index.html` talks to `/api/chat` and `/api/reset` regardless of backend, demonstrating how to decouple the UI from the inference provider.

---

## Code Examples

### Registering tools as Python functions (Gemini)

> What this demonstrates: passing native Python callables as tools so the SDK handles function-call dispatch automatically

```python
# From: Code/DAY9/AIMedicalReceptionist/receptionist_agent.py
def setup_tools(self):
    def search_patients(query: str) -> str:
        """Search for patient information by patient ID or partial name"""
        return self.data_loader.search_patients(query)

    def check_bed_availability(ward_id: int = None) -> str:
        """Check available beds in hospital or specific ward"""
        return self.data_loader.check_bed_availability(ward_id)

    # ... more tools ...

    self.tools = [search_patients, check_bed_availability, ...]
```

### Enabling automatic function calling in Gemini

> What this demonstrates: letting the SDK automatically call tools and loop until the model produces a final text response

```python
# From: Code/DAY9/AIMedicalReceptionist/receptionist_agent.py
model = genai.GenerativeModel(
    model_name=self.model,
    tools=self.tools,
    system_instruction=system_prompt
)
chat = model.start_chat(history=self.conversation_history, enable_automatic_function_calling=True)
response = chat.send_message(user_message)
self.conversation_history = chat.history
```

### Prompt-based tool description for Ollama

> What this demonstrates: instructing a model without native tool support to emit parseable tool calls via custom XML tags

```python
# From: Code/DAY9/AIMedicalReceptionist/receptionist_agent_ollama.py
tools_desc = "Available tools:\n"
for tool_name, desc in self.tools.items():
    tools_desc += f"- {tool_name}: {desc}\n"

system_prompt = f"""...
{tools_desc}
When you need data, use: <tool_use>tool_name(param=value)</tool_use>
..."""
```

### Regex-based tool call parsing (Ollama)

> What this demonstrates: extracting tool names and parameters from free-text LLM output and dispatching them manually

```python
# From: Code/DAY9/AIMedicalReceptionist/receptionist_agent_ollama.py
while "<tool_use>" in response and iterations < max_iterations:
    tool_calls = re.findall(r'<tool_use>(\w+)\(([^)]*)\)</tool_use>', response)

    tool_results = []
    for tool_name, params_str in tool_calls:
        params = {}
        if params_str:
            param_pairs = re.findall(r'(\w+)=(["\']?)([^,"\')]+)\2', params_str)
            for param_name, _, param_value in param_pairs:
                params[param_name] = param_value

        result = self.execute_tool(tool_name, **params)
        tool_results.append(f"Tool {tool_name}: {result[:300]}")
```

### CSV data loader with joined queries

> What this demonstrates: loading multiple CSVs at startup and merging them to answer compound queries

```python
# From: Code/DAY9/AIMedicalReceptionist/data_loader.py
def load_all_data(self):
    for file in Path(self.data_dir).glob("*.csv"):
        df_name = file.stem
        self.dataframes[df_name] = pd.read_csv(file)

def get_patient_admissions(self, patient_id: int) -> str:
    result = self.dataframes['admission'][
        self.dataframes['admission']['patient_id'] == patient_id
    ]
    result = result.merge(
        self.dataframes['disease'][['disease_id', 'disease_name']], on='disease_id', how='left'
    )
    result = result.merge(
        self.dataframes['department'][['department_id', 'department_name']], on='department_id', how='left'
    )
    return result[['admission_id', 'admission_date', 'discharge_date', 'disease_name', 'department_name']].to_string()
```

### Real-time bed availability check

> What this demonstrates: deriving computed state (available beds) by cross-referencing two DataFrames

```python
# From: Code/DAY9/AIMedicalReceptionist/data_loader.py
def check_bed_availability(self, ward_id: int = None) -> str:
    beds = self.dataframes['bed']
    admissions = self.dataframes['admission']

    occupied_beds = set(
        admissions[admissions['admission_status'].isin(['Admitted', 'Pending'])]['bed_id'].unique()
    )

    if ward_id:
        available = beds[(beds['ward_id'] == ward_id) & (~beds['bed_id'].isin(occupied_beds))]
    else:
        available = beds[~beds['bed_id'].isin(occupied_beds)]

    return f"Available beds: {len(available)}\n{available.to_string()}"
```

### Flask API exposing the agent

> What this demonstrates: wrapping an agent in a REST API so any frontend can consume it

```python
# From: Code/DAY9/AIMedicalReceptionist/app.py
receptionist = ReceptionistAgent()

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    response = receptionist.chat(user_message)
    return jsonify({'response': response})

@app.route('/api/reset', methods=['POST'])
def reset():
    receptionist.reset_conversation()
    return jsonify({'status': 'reset'})
```

---

## Project Architecture

```
DAY9/
└── AIMedicalReceptionist/
    ├── app.py                        ← Flask app using Gemini backend
    ├── app_ollama.py                 ← Flask app using Ollama backend
    ├── receptionist_agent.py         ← Agent with Gemini native function calling
    ├── receptionist_agent_ollama.py  ← Agent with prompt-based tool parsing for Ollama
    ├── data_loader.py                ← Loads CSVs and exposes query methods
    ├── requirements.txt              ← flask, google-generativeai, pandas, python-dotenv
    ├── QUICKSTART.txt                ← Setup guide for all 3 run modes
    ├── data/                         ← CSV files: patient, doctor, admission, billing, etc.
    └── templates/
        └── index.html                ← Chat UI (served by both Flask apps)
```
