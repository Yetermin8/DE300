#!/bin/bash

# Set variables
IMAGE_NAME="jupyter-image"
CONTAINER_NAME="jupyter-hw2"
NOTEBOOK_DIR="/home/jovyan"
HOST_DIR="$(pwd)"
LOCAL_PORT=8888

# Check if the Docker container is already running and stop it
echo "Checking if the container $CONTAINER_NAME is already running..."
if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME
fi

# Remove if the container exists
if [ $(docker ps -aq -f status=exited -f name=$CONTAINER_NAME) ]; then
    docker rm $CONTAINER_NAME
fi

# Free the port if in use
if lsof -i:$LOCAL_PORT; then
    echo "Port $LOCAL_PORT is busy, attempting to free it..."
    sudo kill -9 $(lsof -ti:$LOCAL_PORT)
    sleep 2 # Wait for the port to be freed
fi

# Run Jupyter notebook with appropriate permissions
echo "Starting Jupyter notebook..."
docker run -it --rm \
    -p $LOCAL_PORT:8888 \
    -v $HOST_DIR:$NOTEBOOK_DIR \
    -u $(id -u):$(id -g) \
    --name $CONTAINER_NAME $IMAGE_NAME \
    jupyter notebook --notebook-dir=$NOTEBOOK_DIR --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''

echo "Container started. Visit http://localhost:$LOCAL_PORT to access Jupyter Notebook."
