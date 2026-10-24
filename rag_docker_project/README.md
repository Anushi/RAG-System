# Document RAG Assistant (Docker)

A Retrieval-Augmented Generation app: documents -> chunks -> embeddings -> cosine-similarity retrieval -> OpenAI answer -> Streamlit UI.

## Project structure
```
app/rag.py        chunking, embeddings, retrieval, LLM generation
app/main.py       Streamlit interface
app/evaluate.py   Precision@k / Recall@k on data/eval_questions.json
data/             4 sample documents (fictional, educational) + eval questions
Dockerfile, docker-compose.yml, .env.example, requirements.txt
```

## Run with Docker
1. Install and start Docker Desktop.
2. Create the env file:
   - PowerShell: `Copy-Item .env.example .env`
   - macOS/Linux: `cp .env.example .env`
3. Edit `.env` and set `OPENAI_API_KEY`. Without a key the app still retrieves passages but cannot generate answers.
4. `docker compose up --build`
5. Open http://localhost:8501

Stop with Ctrl+C or `docker compose down`.

## Use your own documents
Put `.txt` or `.md` files in `data/` (replace the samples), then restart: `docker compose restart`.
Update `data/eval_questions.json` with your own questions and expected source files.

## Evaluate retrieval
```
docker compose exec rag python app/evaluate.py
```
Report the numbers it prints. Do not quote results you have not measured.

## Run without Docker
```
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app/main.py
```

## Troubleshooting
- **Port 8501 in use:** change the left side in `docker-compose.yml`, e.g. `"8502:8501"`.
- **First start is slow:** the embedding model (~90 MB) downloads once and is cached in a Docker volume.
- **"No OPENAI_API_KEY set":** check `.env`, then `docker compose up -d --force-recreate`.
- **Docker engine not running:** open Docker Desktop and wait until it says Running.
- **Build is slow/fails on pip:** check your internet connection and retry.

## Host online
Local Docker is only reachable at localhost. For a public URL, deploy the same image to a container host
(Render, Railway, Fly.io, Azure Container Apps, AWS App Runner, Google Cloud Run, or a Hugging Face Docker Space).
- Set `OPENAI_API_KEY` as a secret/environment variable in the host dashboard. Never commit `.env`.
- Expose port 8501 (Hugging Face Spaces expects 7860, so change `--server.port` in the Dockerfile CMD and `EXPOSE`).
- Add authentication or a spending limit before sharing publicly, since visitors would use your API credits.
