#!/bin/bash

# Stop script on error
set -e

echo "Stopping and removing Docker containers..."

# Define container names to stop and remove
CONTAINERS=("nginx" "fastapi-app" "ashi_db")

# Stop and remove containers
for CONTAINER in "${CONTAINERS[@]}"; do
    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER$"; then
        echo "Stopping container: $CONTAINER"
        docker stop $CONTAINER || true
        echo "Removing container: $CONTAINER"
        docker rm -f $CONTAINER || true
    else
        echo "Container $CONTAINER does not exist, skipping..."
    fi
done

# Remove all images dynamically
echo "Removing all Docker images..."
IMAGES=$(docker images -q)

if [ -n "$IMAGES" ]; then
    docker rmi -f $IMAGES
    echo "All images have been removed."
else
    echo "No Docker images to remove."
fi

# Final cleanup and status
echo "Cleanup complete!"
# docker system prune -f
docker images
docker ps -a