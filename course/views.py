from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from .models import GoogleDocument, ProcessingStatus
from .tasks import check_for_new_docs
import json

def home(request):
    # Get documents with pagination
    documents = GoogleDocument.objects.all()
    paginator = Paginator(documents, 10)  # Show 10 docs per page
    
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get current status
    try:
        status = ProcessingStatus.objects.get(pk=1)
    except ProcessingStatus.DoesNotExist:
        status = ProcessingStatus.objects.create(pk=1)
    
    context = {
        'documents': page_obj,
        'status': status,
    }
    return render(request, 'course/home.html', context)

@require_http_methods(["POST"])
def check_new_docs(request):
    """API endpoint to trigger checking for new docs"""
    try:
        # Trigger the Celery task
        check_for_new_docs.delay()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Started checking for new documents'
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@require_http_methods(["GET"])
def get_status(request):
    """API endpoint to get current processing status"""
    try:
        status = ProcessingStatus.objects.get(pk=1)
        return JsonResponse({
            'status': status.status,
            'message': status.message,
            'last_check': status.last_check.isoformat()
        })
    except ProcessingStatus.DoesNotExist:
        return JsonResponse({
            'status': 'idle',
            'message': 'No processing started yet'
        })

@require_http_methods(["GET"])
def get_documents(request):
    """API endpoint to get documents"""
    documents = GoogleDocument.objects.all()[:10]  # Get latest 10
    
    docs_data = [{
        'id': doc.id,
        'title': doc.title,
        'content': doc.content[:200] + '...' if len(doc.content) > 200 else doc.content,
        'processed_at': doc.processed_at.isoformat()
    } for doc in documents]
    
    return JsonResponse({'documents': docs_data})