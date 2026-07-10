# Use a Python base image
FROM python:3.8-slim

# Set environment variables to improve Python behavior in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install FFmpeg and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    gcc \
    g++ \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the package files
COPY setup.py requirements.txt ./

# Install base requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install the package itself
COPY . .
RUN pip install -e .

# Create directories for input and output if they don't exist
RUN mkdir -p /app/input /app/output

# Set volume mount points
VOLUME ["/app/input", "/app/output", "/app/models"]

# Expose port for Streamlit
EXPOSE 8501

# Run the web app by default
CMD ["chorus-detection-web"]