# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create directories for logs and docs if they don't exist
RUN mkdir -p logs docs

# Install curl for health checks or other utilities if needed
# Already installed above

# Define the default command to run the application
# These default arguments can be overridden when running the container
CMD ["python", "main.py", "--query", "V-Probing", "--docs-path", "/app/docs/PHYSICSOFLLM.pdf"]
