
# Image similarity search

This project is to dedicated to find similarity between images using cosine similarity. By default this project will return product IDs. Steps to run and deploy the application:

- FastAPI Container Setup

    - Build the container:

    ```bash
    docker build -t fastapi-app .
    ```

    or with AWS credentials

    ```bash
    docker build -t fastapi-app . --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

    ```
    - Run the container:

    ```bash
    docker run --name fastapi-app -d -p 8000:8000 fastapi-app
    ```

    Ensure the FastAPI app is running by visiting http://localhost:8000.

- Set Up Nginx as a Reverse Proxy
    - Create a Dockerfile for Nginx:

    ```
    nginx/Dockerfile
    ```

    ```bash
    FROM nginx:latest

    # Copy the custom Nginx configuration
    COPY nginx.conf /etc/nginx/nginx.conf
    ```

- Nginx Configuration

    - Create the custom Nginx configuration file:

    ```
    nginx/nginx.conf
    ```

    ```bash
    user  nginx;
    worker_processes  auto;

    error_log  /var/log/nginx/error.log;
    pid        /var/run/nginx.pid;

    events {
        worker_connections 1024;
    }

    http {
        include       /etc/nginx/mime.types;
        default_type  application/octet-stream;

        sendfile        on;
        keepalive_timeout  65;

        upstream fastapi_app {
            server fastapi-app:8000; # Reference the FastAPI container by its name
        }

        server {
            listen 80;

            location / {
                proxy_pass http://fastapi_app;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
            }
        }
    }
    ```

- Set Up docker-compose

    To run both the FastAPI and Nginx containers, use docker-compose.

    ```
    docker-compose.yml
    ```

    ```bash
    version: '3.8'

    services:
        fastapi-app:
            build:
            context: .
            container_name: fastapi-app
            expose:
            - 8000
            restart: always

        nginx:
            build:
            context: ./nginx
            container_name: nginx
            ports:
            - "80:80"
            depends_on:
            - fastapi-app
            restart: always

    ```

- Build and Deploy with Docker Compose
    
    - Build and start the containers:

    ```bash
    docker-compose up --build -d
    ```
    or with AWS credentials

    ```bash
    AWS_ACCESS_KEY_ID=AKIAYOURACCESSKEY AWS_SECRET_ACCESS_KEY=yoursecretkey AWS_DEFAULT_REGION=us-west-2 \
    docker-compose up --build -d
    ```

    - Verify the deployment:

        - Visit http://localhost in your browser.
        - Nginx should serve your FastAPI app.

- Logs and Debugging

    - View logs for the services:

    ```bash
    docker-compose logs fastapi-app
    docker-compose logs nginx
    ```

    - Restart containers if necessary:

    ```bash
    docker-compose restart
    ```

Happy coding :)

