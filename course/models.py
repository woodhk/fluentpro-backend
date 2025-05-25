from django.db import models
from django.utils import timezone
from django.db.models import JSONField
import json

class GoogleDocument(models.Model):
    doc_id = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=500)
    content = models.TextField()
    last_modified = models.DateTimeField()
    processed_at = models.DateTimeField(auto_now_add=True)
    
    # New fields for RAG processing
    embeddings = models.TextField(blank=True, null=True)  # Store as JSON
    structured_content = models.TextField(blank=True, null=True)  # Store as JSON
    processing_completed = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-processed_at']
    
    def __str__(self):
        return self.title
    
    def get_embeddings(self):
        """Get embeddings as list"""
        if self.embeddings:
            return json.loads(self.embeddings)
        return None
    
    def set_embeddings(self, embeddings_list):
        """Set embeddings from list"""
        self.embeddings = json.dumps(embeddings_list)
    
    def get_structured_content(self):
        """Get structured content as dict"""
        if self.structured_content:
            return json.loads(self.structured_content)
        return None
    
    def set_structured_content(self, content_dict):
        """Set structured content from dict"""
        self.structured_content = json.dumps(content_dict)

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
    current_stage = models.CharField(max_length=100, blank=True)
    
    class Meta:
        verbose_name_plural = "Processing Status"