from django.db import models
from django.utils import timezone

# Create your models here.
class GoogleDocument(models.Model):
    doc_id = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=500)
    content = models.TextField()
    last_modified = models.DateTimeField()
    processed_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-processed_at']
    
    def __str__(self):
        return self.title

class ProcessingStatus(models.Model):
    STATUS_CHOICES = [
        ('idle', 'Idle'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('error', 'Error'),
    ]
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='idle')
    message = models.TextField(blank=True)
    last_check = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name_plural = "Processing Status"