# RAG Document Q&A System

A production-grade Retrieval-Augmented Generation (RAG) system built with LangGraph, featuring:

- **DeepSeek OCR** via vLLM for document text extraction
- **LangGraph** for agentic workflow orchestration with validation loops
- **ChromaDB** for persistent vector storage
- **OpenAI GPT** for embeddings and generation
- **Contextual Chunking** based on Anthropic's research (49% retrieval improvement)
- **Streamlit** web interface for document upload and chat

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Upload Page    │  │   Chat Page     │  │  Manage Page    │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼──────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌───────────────────────┐  ┌─────────────────────────────────────┐
│   Ingestion Pipeline  │  │          LangGraph Workflow          │
│  ┌─────────────────┐  │  │  ┌─────────┐  ┌─────────┐  ┌──────┐ │
│  │  PDF Processor  │  │  │  │Retriever│→ │Generator│→ │Valid.│ │
│  └────────┬────────┘  │  │  └─────────┘  └────┬────┘  └──┬───┘ │
│           ▼           │  │                    │          │      │
│  ┌─────────────────┐  │  │                    ◄──────────┘      │
│  │  DeepSeek OCR   │  │  │                 (retry if invalid)   │
│  └────────┬────────┘  │  │                    │                 │
│           ▼           │  │                    ▼                 │
│  ┌─────────────────┐  │  │              ┌─────────┐             │
│  │ Text Cleaner    │  │  │              │Response │             │
│  └────────┬────────┘  │  │              └─────────┘             │
│           ▼           │  └─────────────────────────────────────┘
│  ┌─────────────────┐  │                     │
│  │Contextual Chunk │  │                     │
│  └────────┬────────┘  │                     │
└───────────┼───────────┘                     │
            │                                 │
            ▼                                 ▼
      ┌─────────────────────────────────────────────┐
      │              ChromaDB Vector Store           │
      └─────────────────────────────────────────────┘
```

## Project Structure

```
assignment-langraph/
├── pyproject.toml              # UV workspace root
├── .env.example                # Environment template
├── data/                       # Sample PDFs
├── src/
│   ├── common/                 # Shared utilities
│   │   └── src/common/
│   │       ├── config.py       # Pydantic settings
│   │       └── logging.py      # structlog config
│   │
│   ├── ingestion/              # Document ingestion
│   │   └── src/ingestion/
│   │       ├── ocr/            # DeepSeek OCR client
│   │       ├── processor/      # PDF, cleaner, chunker
│   │       ├── vectorstore/    # ChromaDB
│   │       ├── pipeline.py     # Main orchestrator
│   │       └── cli.py          # CLI interface
│   │
│   ├── agents/                 # LangGraph RAG
│   │   └── src/agents/
│   │       ├── nodes/          # Retriever, Generator, Validator, Response
│   │       ├── state.py        # Shared state schema
│   │       ├── graph.py        # LangGraph workflow
│   │       └── chat.py         # Chat session
│   │
│   └── app/                    # Streamlit UI
│       └── src/app/
│           ├── main.py         # Entry point
│           └── pages/          # Upload, Chat pages
│
└── tests/
```

## Prerequisites

1. **Python 3.11+**
2. **UV** package manager: https://docs.astral.sh/uv/
3. **Poppler** (for PDF processing):
   - Windows: Download from https://github.com/osber/poppler-windows/releases
   - Mac: `brew install poppler`
   - Linux: `apt install poppler-utils`
4. **DeepSeek OCR vLLM server** (for OCR):
   ```bash
   vllm serve deepseek-ai/DeepSeek-OCR \
     --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
     --no-enable-prefix-caching \
     --mm-processor-cache-gb 0
   ```

## Setup

1. **Clone and navigate:**
   ```bash
   git clone <repository-url>
   cd assignment-langraph
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Install dependencies:**
   ```bash
   uv sync --all-packages
   ```

## Usage

### Command Line Interface

**Ingest a document:**
```bash
uv run ingestion ingest data/document.pdf

# With page range:
uv run ingestion ingest data/document.pdf --start-page 1 --end-page 10

# Without contextual chunking (faster):
uv run ingestion ingest data/document.pdf --no-context
```

**List ingested documents:**
```bash
uv run ingestion list
```

**Search documents:**
```bash
uv run ingestion search "What is the main topic?"
```

**Delete a document:**
```bash
uv run ingestion delete <document_id>
```

### Streamlit Web Interface

```bash
uv run streamlit run src/app/src/app/main.py
```

Then open http://localhost:8501 in your browser.

### Python API

```python
from ingestion import IngestionPipeline
from agents import ChatSession

# Ingest a document
pipeline = IngestionPipeline()
result = pipeline.ingest("document.pdf", start_page=1, end_page=10)
print(f"Ingested {result['chunks_created']} chunks")

# Chat with documents
session = ChatSession()
response = session.chat("What is the main topic of the document?")
print(response.content)
```

## LangGraph Workflow

The RAG workflow follows this pattern with a retry loop for validation:

```
START → Retriever → Generator → Validator
                                    │
                            [is_valid?]
                           /           \
                      Yes /             \ No (retry < 3)
                         ↓               ↓
                    Response ←── Generator (with feedback)
                         ↓
                        END
```

**Agents:**
1. **Retriever**: Fetches top-k relevant chunks from ChromaDB
2. **Generator**: Generates answer using GPT-4o-mini with context
3. **Validator**: Checks for hallucinations using structured output
4. **Response**: Formats final response with sources

## Observability

### Langfuse (Recommended)

Self-host or use cloud for tracing:

```bash
# .env
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### structlog

JSON logging for production:

```bash
# .env
LOG_FORMAT=json
LOG_LEVEL=INFO
```

## Configuration

All settings via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `DEEPSEEK_OCR_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store path |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap tokens |
| `ENABLE_CONTEXTUAL_CHUNKING` | `true` | Use Anthropic-style chunking |
| `RETRIEVAL_TOP_K` | `5` | Chunks to retrieve |
| `MAX_RETRY_COUNT` | `3` | Max validation retries |

## Sample Interaction

```
User: What were the major events in Indian history around 122 AD?
