FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# main.py, не app.py!
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]