# poc-chatbot

Minimal proof-of-concept chatbot using Qdrant for vector storage and Gemini for LLM/embeddings.

Prerequisites

- Python 3.13
- Docker & docker-compose
- Copy `.env.example` → `.env` and fill GEMINI_API_KEY, QDRANT_API_KEY, etc.

Quick setup

1. Copy env:
   - cp .env.example .env
2. Start Qdrant:
   - docker compose -f docker-compose.yaml up -d
3. Python venv & deps:
   - python3.13 -m venv .venv
   - source .venv/bin/activate
   - pip install --upgrade pip
   - pip install -e .
4. Ingest example docs:
   - python -c "from poc_chatbot.backend.main import add_documents; add_documents()"
5. Run interactive query:
   - python -m poc_chatbot.backend.main

Notes

- Edit collection/settings in app_config.py or poc_chatbot/backend/main.py.
- Check Qdrant logs: docker logs qdrant
