FROM python:3.9-slim

WORKDIR /app

# Install Poppler for pdf2image and other required libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the required files
COPY requirements.txt .
COPY .env .
COPY main.py .
COPY htmlTemplates.py .
COPY model.py .

# Install Python dependencies
RUN pip install -r requirements.txt

EXPOSE 8501

# Healthcheck for the Streamlit service
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start the Streamlit app
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
