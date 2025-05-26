#!/bin/bash

# Start Celery worker with eventlet pool for better async performance on macOS
echo "Starting Celery worker with eventlet pool for macOS..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
fi

# Start Celery worker with eventlet pool
celery -A fluentpro worker \
    --pool=eventlet \
    --concurrency=100 \
    --loglevel=info \
    --prefetch-multiplier=1

echo "Celery worker stopped" 