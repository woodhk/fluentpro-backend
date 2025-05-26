from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from .models import GoogleDocument, ProcessingStatus, GeneratedCourse, GeneratedLesson, CourseGenerationStatus
from .tasks import check_for_new_docs, process_all_documents_with_rag, generate_courses_for_document, generate_courses_for_all_documents, export_courses_to_json
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

def document_detail(request, doc_id):
    """View to show detailed document with structured content"""
    document = get_object_or_404(GoogleDocument, id=doc_id)
    
    # Get structured content if available
    structured_content = document.get_structured_content()
    
    context = {
        'document': document,
        'structured_content': structured_content,
        'has_embeddings': bool(document.embeddings),
    }
    return render(request, 'course/document_detail.html', context)

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

@require_http_methods(["POST"])
def process_documents_rag(request):
    """API endpoint to trigger RAG processing for all documents"""
    try:
        # Trigger the Celery task
        process_all_documents_with_rag.delay()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Started RAG processing for all documents'
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
            'current_stage': status.current_stage,
            'last_check': status.last_check.isoformat()
        })
    except ProcessingStatus.DoesNotExist:
        return JsonResponse({
            'status': 'idle',
            'message': 'No processing started yet',
            'current_stage': ''
        })

@require_http_methods(["GET"])
def get_documents(request):
    """API endpoint to get documents"""
    documents = GoogleDocument.objects.all()[:10]  # Get latest 10
    
    docs_data = [{
        'id': doc.id,
        'title': doc.title,
        'content': doc.content[:200] + '...' if len(doc.content) > 200 else doc.content,
        'processed_at': doc.processed_at.isoformat(),
        'processing_completed': doc.processing_completed,
        'has_structured_content': bool(doc.structured_content)
    } for doc in documents]
    
    return JsonResponse({'documents': docs_data})


def courses_list(request):
    """View to list all generated courses"""
    courses = GeneratedCourse.objects.all()
    paginator = Paginator(courses, 10)
    
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get current status
    try:
        status = ProcessingStatus.objects.get(pk=1)
    except ProcessingStatus.DoesNotExist:
        status = ProcessingStatus.objects.create(pk=1)
    
    context = {
        'courses': page_obj,
        'status': status,
    }
    return render(request, 'course/course_list.html', context)

def course_detail(request, course_id):
    """View to show course details with all lessons"""
    course = get_object_or_404(GeneratedCourse, id=course_id)
    
    context = {
        'course': course,
        'lessons': course.lessons.all().order_by('lesson_number'),
        'topic_description': course.get_topic_description(),
    }
    return render(request, 'course/course_detail.html', context)

def generation_status(request):
    """View to show course generation status"""
    generation_statuses = CourseGenerationStatus.objects.all()[:20]
    
    context = {
        'generation_statuses': generation_statuses,
    }
    return render(request, 'course/generation_status.html', context)

@require_http_methods(["POST"])
def generate_courses(request):
    """API endpoint to trigger course generation"""
    document_id = request.POST.get('document_id')
    
    try:
        if document_id:
            # Generate for specific document
            generate_courses_for_document.delay(int(document_id))
            message = 'Started course generation for document'
        else:
            # Generate for all documents
            generate_courses_for_all_documents.delay()
            message = 'Started course generation for all documents'
        
        return JsonResponse({
            'status': 'success',
            'message': message
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@require_http_methods(["POST"])
def export_courses(request):
    """API endpoint to export courses"""
    document_id = request.POST.get('document_id')
    
    try:
        if document_id:
            export_courses_to_json.delay(int(document_id))
        else:
            export_courses_to_json.delay()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Started course export'
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@require_http_methods(["GET"])
def get_generation_progress(request, document_id):
    """API endpoint to get generation progress for a document"""
    try:
        gen_status = CourseGenerationStatus.objects.filter(
            document_id=document_id
        ).latest('started_at')
        
        return JsonResponse({
            'status': gen_status.status,
            'current_step': gen_status.current_step,
            'error_message': gen_status.error_message,
            'started_at': gen_status.started_at.isoformat(),
            'completed_at': gen_status.completed_at.isoformat() if gen_status.completed_at else None
        })
    except CourseGenerationStatus.DoesNotExist:
        return JsonResponse({
            'status': 'not_started',
            'current_step': '',
            'error_message': ''
        })

@require_http_methods(["POST"])
def clear_docs(request):
    """API endpoint to clear all documents from memory"""
    try:
        # Count documents before deletion
        doc_count = GoogleDocument.objects.count()
        
        # Delete all documents (this will cascade to related objects)
        GoogleDocument.objects.all().delete()
        
        # Reset processing status
        try:
            status = ProcessingStatus.objects.get(pk=1)
            status.status = 'idle'
            status.message = f'Cleared {doc_count} documents from memory'
            status.current_stage = ''
            status.save()
        except ProcessingStatus.DoesNotExist:
            ProcessingStatus.objects.create(
                pk=1,
                status='idle',
                message=f'Cleared {doc_count} documents from memory'
            )
        
        return JsonResponse({
            'status': 'success',
            'message': f'Successfully cleared {doc_count} documents from memory'
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)