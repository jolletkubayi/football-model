FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
# writable data folder (hosts like Hugging Face run the app as a non-root user)
RUN mkdir -p /data && chmod 777 /data
ENV DB_PATH=/data/football.db
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
