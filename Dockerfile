FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Kopieer en installeer dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Kopieer de app
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.headless=true"]
