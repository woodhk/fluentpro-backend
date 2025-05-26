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

class GeneratedCourse(models.Model):
    """Model for storing generated courses"""
    document = models.ForeignKey(GoogleDocument, on_delete=models.CASCADE, related_name='courses')
    course_name = models.CharField(max_length=500)
    role = models.CharField(max_length=200)
    industry = models.CharField(max_length=200)
    topic_description_pair = models.TextField()  # Store as JSON
    created_at = models.DateTimeField(auto_now_add=True)
    processing_status = models.CharField(max_length=50, default='pending')
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.course_name} - {self.role}"
    
    def get_topic_description(self):
        """Get topic description as dict"""
        if self.topic_description_pair:
            return json.loads(self.topic_description_pair)
        return None
    
    def set_topic_description(self, data):
        """Set topic description from dict"""
        self.topic_description_pair = json.dumps(data)

class GeneratedLesson(models.Model):
    """Model for storing generated lessons"""
    course = models.ForeignKey(GeneratedCourse, on_delete=models.CASCADE, related_name='lessons')
    lesson_number = models.IntegerField()
    lesson_title = models.CharField(max_length=500)
    lesson_introduction = models.TextField()
    skill_aims = models.TextField()  # Store as JSON
    language_learning_aims = models.TextField()  # Store as JSON
    lesson_summary = models.TextField()  # Store as JSON
    is_bonus = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['course', 'lesson_number']
    
    def __str__(self):
        return f"Lesson {self.lesson_number}: {self.lesson_title}"
    
    def get_skill_aims(self):
        """Get skill aims as list"""
        if self.skill_aims:
            return json.loads(self.skill_aims)
        return []
    
    def set_skill_aims(self, aims_list):
        """Set skill aims from list"""
        self.skill_aims = json.dumps(aims_list)
    
    def get_language_learning_aims(self):
        """Get language learning aims as dict"""
        if self.language_learning_aims:
            return json.loads(self.language_learning_aims)
        return {}
    
    def set_language_learning_aims(self, aims_dict):
        """Set language learning aims from dict"""
        self.language_learning_aims = json.dumps(aims_dict)
    
    def get_lesson_summary(self):
        """Get lesson summary as list"""
        if self.lesson_summary:
            return json.loads(self.lesson_summary)
        return []
    
    def set_lesson_summary(self, summary_list):
        """Set lesson summary from list"""
        self.lesson_summary = json.dumps(summary_list)

class CourseGenerationStatus(models.Model):
    """Track the status of course generation workflow"""
    document = models.ForeignKey(GoogleDocument, on_delete=models.CASCADE)
    status = models.CharField(max_length=50, default='idle')
    current_step = models.CharField(max_length=100, blank=True)
    orchestrator_output = models.TextField(blank=True)  # Store as JSON
    worker_outputs = models.TextField(blank=True)  # Store as JSON
    evaluator_feedback = models.TextField(blank=True)  # Store as JSON
    final_output = models.TextField(blank=True)  # Store as JSON
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"Generation for {self.document.title} - {self.status}"