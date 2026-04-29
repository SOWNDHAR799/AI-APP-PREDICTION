FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if any (e.g. for some scikit-learn or other libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8080

CMD ["streamlit", "run", "ai_prediction_app.py"]
