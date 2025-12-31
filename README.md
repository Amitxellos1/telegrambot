# 🤖 Telegram RAG Bot

A lightweight GenAI Telegram bot that uses **Retrieval Augmented Generation (RAG)** to answer questions from a knowledge base and can describe images using vision models.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 📚 **RAG-based Q&A**: Answers questions using retrieved context from document knowledge base
- 🖼️ **Image Description**: Describes uploaded images using vision models
- 💬 **Conversation History**: Maintains last 3 interactions per user for context-aware responses
- 🔍 **Source Attribution**: Shows which documents were used to generate answers
- 📝 **Conversation Summary**: Summarize recent chat history
- ⚡ **Query Caching**: Caches embeddings for repeated queries

## 🏗️ System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Telegram      │────▶│   Bot Server    │────▶│   RAG Pipeline  │
│   User          │◀────│   (bot.py)      │◀────│                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │                        │
                                │                        ▼
                                │               ┌─────────────────┐
                                │               │   Embedder      │
                                │               │ (sentence-      │
                                │               │  transformers)  │
                                │               └────────┬────────┘
                                │                        │
                                │                        ▼
                                │               ┌─────────────────┐
                                │               │   ChromaDB      │
                                │               │  Vector Store   │
                                │               └────────┬────────┘
                                │                        │
                                ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   LLM Client    │◀────│   Retrieved     │
                        │ (OpenAI/Ollama) │     │   Context       │
                        └─────────────────┘     └─────────────────┘
```

## 📁 Project Structure

```
Avivo_task/
├── bot.py                 # Main Telegram bot application
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── env.example.txt        # Environment variables template
├── README.md              # This file
├── rag/
│   ├── __init__.py        # RAG module exports
│   ├── embedder.py        # Text embedding & chunking
│   ├── retriever.py       # Vector search with ChromaDB
│   └── llm.py             # LLM clients (OpenAI/Ollama)
├── data/                  # Knowledge base documents
│   ├── company_policies.md
│   ├── tech_faq.md
│   ├── product_info.md
│   ├── onboarding_guide.md
│   └── security_guidelines.md
└── db/                    # ChromaDB persistence directory
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Telegram account
- OpenAI API key (or Ollama installed locally)

### 1. Clone and Setup

```bash
# Navigate to project directory
cd Avivo_task

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the bot token you receive

### 3. Configure Environment

1. Copy `env.example.txt` to `.env`:
   ```bash
   # Windows
   copy env.example.txt .env
   # Linux/Mac
   cp env.example.txt .env
   ```

2. Edit `.env` with your credentials:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   OPENAI_API_KEY=your_openai_key_here
   ```

### 4. Run the Bot

```bash
python bot.py
```

The bot will:
1. Initialize the embedding model
2. Index documents from `data/` directory (first run only)
3. Start listening for Telegram messages

## 📱 Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and introduction |
| `/ask <question>` | Ask a question from the knowledge base |
| `/image` | Describe an uploaded image |
| `/sources` | Show documents used in last answer |
| `/summarize` | Summarize recent conversation |
| `/help` | Show help message |

### Example Usage

```
User: /ask What is the remote work policy?
Bot: 📝 Answer:
     Employees are eligible for remote work after completing their 
     3-month probation period. Remote work requests must be submitted 
     to HR at least one week in advance...
     
     📚 Sources: company_policies.md
```

## ⚙️ Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token | Required |
| `LLM_PROVIDER` | LLM provider (`openai` or `ollama`) | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `OLLAMA_MODEL` | Ollama model for text | `mistral` |
| `OLLAMA_VISION_MODEL` | Ollama model for images | `llava` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `TOP_K_RESULTS` | Number of documents to retrieve | `3` |
| `MAX_HISTORY_PER_USER` | Conversation history length | `3` |

## 🔧 Using Ollama (Local LLM)

For fully local operation without API costs:

1. Install Ollama from [ollama.ai](https://ollama.ai)

2. Pull required models:
   ```bash
   ollama pull mistral
   ollama pull llava  # For image description
   ```

3. Update `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_HOST=http://localhost:11434
   ```

4. Run the bot:
   ```bash
   python bot.py
   ```

## 🐳 Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]
```

Build and run:

```bash
docker build -t telegram-rag-bot .
docker run -d --env-file .env telegram-rag-bot
```

## 📚 Adding Custom Documents

1. Add your `.md` files to the `data/` directory
2. Delete the `db/` folder to force re-indexing
3. Restart the bot

The bot will automatically:
- Load all `.md` files
- Split them into chunks
- Generate embeddings
- Store in ChromaDB

## 🎯 Tech Stack

| Component | Technology |
|-----------|------------|
| **Bot Framework** | python-telegram-bot 21.x |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Store** | ChromaDB |
| **LLM** | OpenAI GPT-4o-mini / Ollama |
| **Image Description** | GPT-4o-mini Vision / LLaVA |

## 📊 Model Selection Rationale

- **all-MiniLM-L6-v2**: Fast, lightweight (80MB), good quality embeddings. Ideal for local deployment.
- **GPT-4o-mini**: Cost-effective, fast responses, supports vision. Best balance of quality and price.
- **Ollama + Mistral**: Free, runs locally, good quality. Best for privacy-focused deployments.

## 🔒 Security Considerations

- Never commit `.env` file to version control
- Use environment variables for all secrets
- Consider rate limiting for production deployments
- Validate and sanitize user inputs

## 📝 License

MIT License - feel free to use and modify.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


