import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fluentpro.settings')

app = Celery('fluentpro')
app.config_from_object('django.conf:settings', namespace='CELERY')

# macOS-specific configuration to avoid forking issues
app.conf.update(
    worker_pool='threads',
    worker_concurrency=4,
    worker_prefetch_multiplier=1,
    # Ensure we don't use fork on macOS
    worker_pool_restarts=True,
)

app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')