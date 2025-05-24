# Use Ubuntu 22.04 as base image for compatibility with CARLA
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CARLA_VERSION=0.9.15
ENV CARLA_ROOT=/opt/carla

# Install system dependencies including virtual display for headless operation
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    unzip \
    libc++1 \
    libc++abi1 \
    libpng16-16 \
    libjpeg8 \
    libtiff5 \
    fontconfig \
    libfreetype6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    xvfb \
    xauth \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Download and install CARLA
RUN mkdir -p ${CARLA_ROOT} && \
    cd /tmp && \
    wget -q https://github.com/carla-simulator/carla/releases/download/${CARLA_VERSION}/CARLA_${CARLA_VERSION}.tar.gz && \
    tar -xzf CARLA_${CARLA_VERSION}.tar.gz -C ${CARLA_ROOT} --strip-components=1 && \
    rm CARLA_${CARLA_VERSION}.tar.gz

# Install CARLA Python API
RUN cd ${CARLA_ROOT}/PythonAPI/carla/dist && \
    python3 -m pip install carla-*.whl

# Copy application code
COPY app/ ./app/
COPY utils/ ./utils/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create directories for data persistence
RUN mkdir -p /app/data/sensor_capture /app/data/logs /app/models

# Set Python path to include current directory
ENV PYTHONPATH=/app:${PYTHONPATH}

# Expose ports for CARLA
EXPOSE 2000 2001 2002

# Create entrypoint script with enhanced CARLA server management
RUN echo '#!/bin/bash\n\
    set -e\n\
    \n\
    # Function to start virtual display for headless operation\n\
    start_virtual_display() {\n\
    echo "Starting virtual display for headless operation..."\n\
    export DISPLAY=:99\n\
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &\n\
    XVFB_PID=$!\n\
    echo "Virtual display started with PID: $XVFB_PID"\n\
    sleep 2\n\
    }\n\
    \n\
    # Function to start CARLA server\n\
    start_carla_server() {\n\
    echo "Starting CARLA server in headless mode..."\n\
    cd ${CARLA_ROOT}\n\
    \n\
    # CARLA server arguments for headless operation\n\
    CARLA_ARGS="-carla-server"\n\
    CARLA_ARGS="$CARLA_ARGS -RenderOffScreen"\n\
    CARLA_ARGS="$CARLA_ARGS -quality-level=Low"\n\
    CARLA_ARGS="$CARLA_ARGS -world-port=2000"\n\
    CARLA_ARGS="$CARLA_ARGS -resx=800"\n\
    CARLA_ARGS="$CARLA_ARGS -resy=600"\n\
    CARLA_ARGS="$CARLA_ARGS -opengl"\n\
    \n\
    # Additional arguments for containerized deployment\n\
    if [ "$CARLA_HEADLESS" = "true" ]; then\n\
    CARLA_ARGS="$CARLA_ARGS -nullrhi"\n\
    fi\n\
    \n\
    # Custom port if specified\n\
    if [ ! -z "$CARLA_PORT" ] && [ "$CARLA_PORT" != "2000" ]; then\n\
    CARLA_ARGS=$(echo "$CARLA_ARGS" | sed "s/-world-port=2000/-world-port=$CARLA_PORT/")\n\
    fi\n\
    \n\
    echo "Starting CARLA with arguments: $CARLA_ARGS"\n\
    ./CarlaUE4.sh $CARLA_ARGS &\n\
    CARLA_PID=$!\n\
    echo "CARLA server started with PID: $CARLA_PID"\n\
    \n\
    # Store PID for cleanup\n\
    echo $CARLA_PID > /app/carla_server.pid\n\
    \n\
    # Wait for CARLA to be ready\n\
    echo "Waiting for CARLA server to be ready..."\n\
    CARLA_HOST=${CARLA_HOST:-localhost}\n\
    CARLA_CHECK_PORT=${CARLA_PORT:-2000}\n\
    \n\
    python3 -c "import carla; import time; import sys; \
    client = carla.Client('\''$CARLA_HOST'\'', $CARLA_CHECK_PORT); \
    client.set_timeout(60.0); \
    max_attempts = 30; \
    attempts = 0; \
    while attempts < max_attempts: \
    try: \
    version = client.get_server_version(); \
    print(f'\''CARLA server is ready! Version: {version}'\''); \
    sys.exit(0); \
    except Exception as e: \
    attempts += 1; \
    print(f'\''Waiting for CARLA... (attempt {attempts}/{max_attempts})'\''); \
    time.sleep(2); \
    print('\''Failed to connect to CARLA server after maximum attempts'\''); \
    sys.exit(1)"\n\
    \n\
    if [ $? -ne 0 ]; then\n\
    echo "Failed to start CARLA server properly"\n\
    exit 1\n\
    fi\n\
    }\n\
    \n\
    # Function to cleanup on exit\n\
    cleanup() {\n\
    echo "Cleaning up..."\n\
    if [ -f /app/carla_server.pid ]; then\n\
    CARLA_PID=$(cat /app/carla_server.pid)\n\
    if kill -0 $CARLA_PID 2>/dev/null; then\n\
    echo "Stopping CARLA server (PID: $CARLA_PID)"\n\
    kill $CARLA_PID\n\
    wait $CARLA_PID 2>/dev/null || true\n\
    fi\n\
    rm -f /app/carla_server.pid\n\
    fi\n\
    if [ ! -z "$XVFB_PID" ] && kill -0 $XVFB_PID 2>/dev/null; then\n\
    echo "Stopping virtual display (PID: $XVFB_PID)"\n\
    kill $XVFB_PID\n\
    fi\n\
    }\n\
    \n\
    # Set up signal handlers\n\
    trap cleanup EXIT INT TERM\n\
    \n\
    # Start CARLA server if requested\n\
    if [ "$START_CARLA_SERVER" = "true" ]; then\n\
    # Start virtual display if not already set\n\
    if [ -z "$DISPLAY" ] || [ "$DISPLAY" = ":99" ]; then\n\
    start_virtual_display\n\
    fi\n\
    \n\
    start_carla_server\n\
    \n\
    # If we are only running the CARLA server, wait forever\n\
    if [ "$1" = "carla-server-only" ]; then\n\
    echo "CARLA server is running. Waiting..."\n\
    wait\n\
    exit 0\n\
    fi\n\
    fi\n\
    \n\
    # Execute the main command\n\
    exec "$@"\n\
    ' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python3", "app/main.py"] 