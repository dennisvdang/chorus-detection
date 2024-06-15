# Use the slim version of the Python 3.9 base image
FROM python:3.9-slim

# Install FFmpeg dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script and any other necessary files to the working directory
COPY src/ .

# Copy the model file to the working directory
COPY models/CRNN/best_model_V3.h5 /app/models/CRNN/

# Run the script when the container starts
CMD ["python", "chorus_finder.py"]