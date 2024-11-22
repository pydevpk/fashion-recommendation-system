# Use TensorFlow GPU base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Accept AWS credentials as build arguments
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

# Set AWS credentials as environment variables inside the container
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# Install required packages for S3 access and Python dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws


RUN mkdir -p /app/templates && chmod 777 /app/uploads

# Copy the application code
COPY requirements.txt .
COPY app.py .
COPY ./templates /app/templates
# COPY .env .

# Create the uploads directory and set permissions
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Download required files from S3
# Replace `<S3_BUCKET>` and `<S3_KEY>` with the actual S3 bucket and object keys
RUN aws s3 cp s3://ashi-similar-styles-ai-engine/embeddings/embeddings.pkl /app/embeddings.pkl && \
    aws s3 cp s3://ashi-similar-styles-ai-engine/embeddings/products.pkl /app/products.pkl && \
    aws s3 cp s3://ashi-similar-styles-ai-engine/embeddings/column_encoders.pkl /app/column_encoders.pkl && \
    aws s3 cp s3://ashi-similar-styles-ai-engine/embeddings/combined_features.pkl /app/combined_features.pkl && \
    aws s3 cp s3://ashi-similar-styles-ai-engine/embeddings/categorical_recommendation_model.pkl /app/categorical_recommendation_model.pkl && \
    aws s3 cp s3://ashi-similar-styles-ai-engine/jewelry-data/ASHI_FINAL_DATA.csv /app/ASHI_FINAL_DATA.csv

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
