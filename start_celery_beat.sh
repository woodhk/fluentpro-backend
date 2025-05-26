#!/bin/bash

# Start Celery Beat scheduler
echo "Starting Celery Beat scheduler..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
fi

# Start Celery Beat
celery -A fluentpro beat \
    --loglevel=info \
    --scheduler django_celery_beat.schedulers:DatabaseScheduler

echo "Celery Beat stopped" 