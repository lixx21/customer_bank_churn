FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data_source/ ./data_source/
COPY .env .
COPY src/ ./src/
COPY run_pipeline.sh .

RUN chmod +x run_pipeline.sh

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]