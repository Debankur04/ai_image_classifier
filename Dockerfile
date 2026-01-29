FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY Requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r Requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
