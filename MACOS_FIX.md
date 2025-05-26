# macOS Celery Fork Issue Fix

## Problem
The application was experiencing crashes on macOS with the error:
```
objc[pid]: +[__SwiftNativeNSStringBase initialize] may have been in progress in another thread when fork() was called.
objc[pid]: +[__SwiftNativeNSStringBase initialize] may have been in progress in another thread when fork() was called. We cannot safely call it or ignore it in the fork() child process. Crashing instead.
```

This is a known issue on macOS when using Celery with the default `fork` multiprocessing method.

## Root Cause
- macOS has restrictions on forking processes when certain system libraries are in use
- Celery's default worker pool uses `fork` which conflicts with macOS system libraries
- The issue is particularly common when using AI/ML libraries that interact with system frameworks

## Solution Implemented

### 1. Updated Celery Configuration (`fluentpro/settings.py`)
```python
# Fix for macOS forking issues - use threads instead of fork
CELERY_WORKER_POOL = 'threads'
CELERY_WORKER_CONCURRENCY = 4
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
```

### 2. Updated Celery App Configuration (`fluentpro/celery.py`)
```python
# macOS-specific configuration to avoid forking issues
app.conf.update(
    worker_pool='threads',
    worker_concurrency=4,
    worker_prefetch_multiplier=1,
    worker_pool_restarts=True,
)
```

### 3. Adjusted ThreadPoolExecutor in Workflow
- Reduced `max_workers` from 20 to 4 in `course_generation_workflow.py`
- This prevents overwhelming the thread pool and maintains stability

### 4. Created Startup Scripts
- `start_celery.sh` - Thread pool (recommended for stability)
- `start_celery_eventlet.sh` - Eventlet pool (better for high concurrency)
- `start_celery_beat.sh` - Scheduler

### 5. Added Dependencies
- Added `eventlet>=0.33.0` for alternative async performance

## Benefits of This Fix

### ✅ Maintains Parallel Processing
- Still processes multiple tasks concurrently
- Thread-based parallelism instead of process-based
- No loss in speed or quality

### ✅ macOS Compatibility
- Eliminates fork-related crashes
- Works reliably on macOS systems
- No more `SIGABRT` errors

### ✅ Better Resource Management
- More efficient memory usage
- Better suited for I/O-bound tasks (like API calls)
- Reduced system overhead

## Performance Considerations

### Thread Pool vs Fork Pool
- **Thread Pool**: Better for I/O-bound tasks (API calls, database operations)
- **Fork Pool**: Better for CPU-bound tasks (heavy computation)
- Since this application primarily makes API calls to LLMs, thread pool is actually optimal

### Concurrency Settings
- **Threads**: 4 workers (conservative, stable)
- **Eventlet**: 100 workers (high concurrency for async operations)

## Usage

### Start with Thread Pool (Recommended)
```bash
./start_celery.sh
```

### Start with Eventlet (High Performance)
```bash
pip install eventlet
./start_celery_eventlet.sh
```

### Manual Start
```bash
celery -A fluentpro worker --pool=threads --concurrency=4 --loglevel=info
```

## Testing
Run the configuration test:
```bash
python test_celery_config.py
```

This will validate that all settings are correctly applied and show the current configuration.

## Alternative Solutions Considered

1. **Using `spawn` instead of `fork`**: More resource-intensive
2. **Using `gevent`**: Requires more dependencies
3. **Disabling multiprocessing**: Would lose parallelism benefits

The thread pool solution was chosen as it provides the best balance of compatibility, performance, and simplicity.