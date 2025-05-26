from django.contrib import admin
from .models import (
    GoogleDocument, ProcessingStatus, GeneratedCourse, 
    GeneratedLesson, CourseGenerationStatus
)
# Register your models here.
@admin.register(GoogleDocument)
class GoogleDocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'doc_id', 'last_modified', 'processed_at']
    list_filter = ['processed_at', 'last_modified']
    search_fields = ['title', 'content']
    readonly_fields = ['doc_id', 'last_modified', 'processed_at']

@admin.register(ProcessingStatus)
class ProcessingStatusAdmin(admin.ModelAdmin):
    list_display = ['status', 'message', 'last_check']
    readonly_fields = ['status', 'message', 'last_check']

@admin.register(GeneratedCourse)
class GeneratedCourseAdmin(admin.ModelAdmin):
    list_display = ['course_name', 'role', 'industry', 'document', 'created_at']
    list_filter = ['role', 'industry', 'created_at']
    search_fields = ['course_name', 'role', 'industry']
    readonly_fields = ['created_at']

@admin.register(GeneratedLesson)
class GeneratedLessonAdmin(admin.ModelAdmin):
    list_display = ['lesson_title', 'lesson_number', 'course', 'is_bonus']
    list_filter = ['is_bonus', 'course__role', 'course__industry']
    search_fields = ['lesson_title', 'lesson_introduction']
    ordering = ['course', 'lesson_number']

@admin.register(CourseGenerationStatus)
class CourseGenerationStatusAdmin(admin.ModelAdmin):
    list_display = ['document', 'status', 'current_step', 'started_at', 'completed_at']
    list_filter = ['status', 'started_at']
    readonly_fields = ['started_at', 'completed_at']