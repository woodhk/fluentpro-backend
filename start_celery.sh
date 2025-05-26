#!/bin/bash

# Start Celery worker with thread pool for macOS compatibility
echo "Starting Celery worker with thread pool for macOS..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
fi

# Start Celery worker with thread pool
celery -A fluentpro worker \
    --pool=threads \
    --concurrency=4 \
    --loglevel=info \
    --prefetch-multiplier=1

echo "Celery worker stopped" 