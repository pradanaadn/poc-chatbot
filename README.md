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

   ```bash
    uv sync
   ```

Notes

- Edit collection/settings in app_config.py or poc_chatbot/backend/main.py.
- Check Qdrant logs: docker logs qdrant
