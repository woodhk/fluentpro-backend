from django.contrib import admin
from .models import GoogleDocument, ProcessingStatus
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