# Use TensorFlow GPU base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the application code
COPY requirements.txt .
COPY app.py .
COPY embeddings.pkl .
COPY filenames.pkl .

# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
