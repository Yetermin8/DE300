#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Give docker permission (only if needed, you might skip this line if not required)
sudo chmod 666 /var/run/docker.sock

# Create a Docker network for container communication
echo "Creating Docker network: hw2-network"
docker network create hw2-network

# Build Jupyter Docker image
echo "Building Jupyter Docker image from dockerfile-jupyter"
docker build -f dockerfiles/dockerfile-jupyter -t jupyter-hw2-image .

# Check if the Jupyter container is already running and stop it
CONTAINER_NAME="jupyter-hw2"
if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
    echo "Stopping existing Jupyter container..."
    docker stop $CONTAINER_NAME
fi

# Remove if the container exists
if [ $(docker ps -aq -f status=exited -f name=$CONTAINER_NAME) ]; then
    docker rm $CONTAINER_NAME
fi

# Run Jupyter container with network setup
echo "Starting Jupyter container"
docker run -it --rm --network hw2-network \
       --name $CONTAINER_NAME \
       -v "$(pwd)/src:/home/jovyan/work" \
       -p 8888:8888 \
       jupyter-hw2-image \
       start-notebook.sh --NotebookApp.token=''

echo "Jupyter is running at http://localhost:8888 with no token"
