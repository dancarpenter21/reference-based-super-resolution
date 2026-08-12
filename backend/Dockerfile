FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY backend ./backend
COPY ml_engine ./ml_engine
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e .

CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
