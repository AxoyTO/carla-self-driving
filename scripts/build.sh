#!/bin/bash

# Build script for CARLA RL Training Application
set -e

echo "Building CARLA RL Training Docker image..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
docker build -t carla-rl:latest .

echo "Build completed successfully!"
echo "Image: carla-rl:latest"

# Show image size
echo "Image size:"
docker images carla-rl:latest 