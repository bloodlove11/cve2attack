FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHAT_REQUIRE_API_KEY=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY evals ./evals
COPY traces/.gitkeep ./traces/.gitkeep
COPY .streamlit ./.streamlit

RUN pip install --upgrade pip && \
    pip install ".[ui]" && \
    mkdir -p traces

EXPOSE 8000 8501

# Default: API server. UI: streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
# Override for CLI: docker run ... python -m src.main "query"
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
