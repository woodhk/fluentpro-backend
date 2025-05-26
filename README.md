# fluentpro-backend

## Setup

1. Copy `credentials.json.example` to `credentials.json`
2. Replace with your actual Google OAuth credentials from Google Cloud Console
3. Never commit `credentials.json` or `token.json` to the repository

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_GEMINI_API_KEY=your_gemini_key
SECRET_KEY=your_django_secret_key
```

4. Run Django migrations:
```bash
python manage.py migrate
```

## Running the Application

### Start Redis (required for Celery)
```bash
redis-server
```

### Start Django Development Server
```bash
python manage.py runserver
```

### Start Celery Worker (macOS-compatible)

**Option 1: Thread Pool (Recommended for stability)**
```bash
./start_celery.sh
```

**Option 2: Eventlet Pool (Better for high concurrency)**
```bash
./start_celery_eventlet.sh
```

Or manually:
```bash
# Thread pool
celery -A fluentpro worker --pool=threads --concurrency=4 --loglevel=info

# Eventlet pool (requires: pip install eventlet)
celery -A fluentpro worker --pool=eventlet --concurrency=100 --loglevel=info
```

### Start Celery Beat (for scheduled tasks)
```bash
./start_celery_beat.sh
```

Or manually:
```bash
celery -A fluentpro beat --loglevel=info --scheduler django_celery_beat.schedulers:DatabaseScheduler
```

## macOS-Specific Notes

This application has been configured to work properly on macOS by using thread-based workers instead of fork-based workers, which resolves the common `objc[pid]: +[__SwiftNativeNSStringBase initialize]` error that occurs with Celery on macOS.

The configuration automatically:
- Uses thread pool instead of fork for workers
- Sets appropriate concurrency levels
- Maintains parallel processing capabilities for speed and quality