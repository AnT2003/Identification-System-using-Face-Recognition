# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory to the root directory
WORKDIR /

# Install system dependencies for dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-thread-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
# Expose the port Flask will run on
EXPOSE 5000

# Command to run your Flask app
CMD ["python", "app.py"]
