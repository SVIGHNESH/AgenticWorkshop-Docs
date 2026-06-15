# Day 6 — OpenAI API Integration & Multi-Modal AI

## Theory

### OpenAI Chat API
The OpenAI Chat API enables real-time conversation with large language models like GPT-4. It uses a message-based architecture where you maintain a conversation history (list of messages with roles: system, user, assistant) and send it with each request. The model responds based on the entire context, maintaining conversational coherence. This allows you to build stateful chatbots that remember previous interactions within a single session.

### Message-Based Conversation Architecture
Conversations are managed as a list of message dictionaries, each with a `role` (system, user, or assistant) and `content` (the text). The system role sets behavioral instructions, user messages represent input, and assistant messages are model responses. By appending all messages to this list, you preserve context for subsequent requests, enabling multi-turn dialogue that feels natural and contextually aware.

### OpenAI Whisper API for Audio Transcription
Whisper is OpenAI's speech-to-text model that converts audio files into written text. It supports multiple audio formats (MP3, WAV, M4A, WebM, etc.) and can process audio files up to 25MB. Whisper is robust to background noise and accents, making it reliable for real-world audio. The transcription happens server-side via API, requiring only the audio file and an API key.

### Multi-Modal AI Applications
Multi-modal AI combines multiple types of input (text, audio, images) in a single application. Day 6 demonstrates this by pairing a chat API (text-based) with audio transcription. A real-world multi-modal bot could accept voice input, transcribe it to text, send it to a chat model, and speak the response back—creating a seamless voice-based assistant.

### Error Handling and API Integration
When calling external APIs, robust error handling is critical. This includes file validation (checking existence and format), API error catching, and graceful failure messages. Using environment variables for sensitive data (API keys) keeps credentials out of source code, following security best practices.

---

## Code Examples

### Initialize OpenAI Client and Start Chatbot Loop

> What this demonstrates: setting up the OpenAI client and implementing a basic conversation loop

```python
# From: Code/DAY6/app.py
from openai import OpenAI

client = OpenAI(api_key="your-api-key-here")

def start_chatbot():
    messages = [
        {"role": "system", "content": "You are a helpful, friendly, and knowledgeable AI assistant."}
    ]
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower().strip() in ['quit', 'exit', 'bye']:
            print("AI: Goodbye! Have a great day!")
            break
        
        messages.append({"role": "user", "content": user_input})
```

### Send Message to OpenAI Chat API and Collect Response

> What this demonstrates: making API calls with conversation history and extracting model responses

```python
# From: Code/DAY6/app.py
response = client.chat.completions.create(
    model="gpt-4o-mini",  
    messages=messages,
    temperature=0.7       
)

assistant_reply = response.choices[0].message.content
print(f"\nAI: {assistant_reply}\n")

messages.append({"role": "assistant", "content": assistant_reply})
```

### Maintain Conversation State Across Turns

> What this demonstrates: appending both user and assistant messages to preserve context

```python
# From: Code/DAY6/app.py
messages = [
    {"role": "system", "content": "You are a helpful, friendly, and knowledgeable AI assistant."}
]

messages.append({"role": "user", "content": user_input})
# ... call API ...
messages.append({"role": "assistant", "content": assistant_reply})
```

### Transcribe Audio Files Using Whisper API

> What this demonstrates: converting audio files to text using OpenAI Whisper

```python
# From: Code/DAY6/voice_to_text_bot.py
def transcribe_audio(audio_file_path: str, api_key: str = None) -> str:
    client = OpenAI(api_key=api_key)
    audio_path = Path(audio_file_path)
    
    with open(audio_path, 'rb') as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        )
    
    return transcript.text
```

### Validate Audio File Format Before Transcription

> What this demonstrates: checking file existence and format constraints before API calls

```python
# From: Code/DAY6/voice_to_text_bot.py
audio_path = Path(audio_file_path)
if not audio_path.exists():
    raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

supported_formats = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
if audio_path.suffix.lower() not in supported_formats:
    raise ValueError(f"Unsupported format. Supported: {supported_formats}")
```

### CLI Interface for Voice-to-Text Tool

> What this demonstrates: building a command-line tool with argument parsing for flexibility

```python
# From: Code/DAY6/voice_to_text_bot.py
parser = argparse.ArgumentParser(
    description="Convert audio files to text using OpenAI Whisper API"
)
parser.add_argument("audio_file", help="Path to audio file (mp3, wav, m4a, etc.)")
parser.add_argument("--api-key", help="OpenAI API key (uses OPENAI_API_KEY env var if not provided)")
parser.add_argument("-o", "--output", help="Output file path (prints to stdout if not provided)")

args = parser.parse_args()
text = transcribe_audio(args.audio_file, args.api_key)
```

### Handle Errors Gracefully and Exit with Status Codes

> What this demonstrates: catching specific exceptions and providing clear error messages

```python
# From: Code/DAY6/voice_to_text_bot.py
try:
    text = transcribe_audio(args.audio_file, args.api_key)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(text)
except FileNotFoundError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
```

---

## Project Architecture

```
DAY6/
├── app.py                    ← Interactive chatbot using OpenAI GPT-4o-mini
├── voice_to_text_bot.py      ← Audio transcription tool using Whisper API
├── selfembeddings.py         ← (Empty/placeholder for embeddings exploration)
├── requirements.txt          ← Project dependencies (openai SDK)
└── venv/                     ← Python virtual environment
```

### Key Technologies
- **OpenAI Python SDK** (`openai>=1.3.0`): Official client library for OpenAI APIs
- **OpenAI Chat Completions**: GPT-4o-mini model for conversational AI
- **OpenAI Whisper**: Audio-to-text transcription API
- **argparse**: Python standard library for CLI argument parsing
- **pathlib**: Python standard library for file path operations

### Workflow
1. **Chat Bot** (`app.py`): User enters text → sent to GPT-4o-mini → response displayed → context maintained
2. **Voice Bot** (`voice_to_text_bot.py`): User provides audio file → validated → transcribed via Whisper → text output or saved to file
