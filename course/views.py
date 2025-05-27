from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from .models import GoogleDocument, ProcessingStatus, GeneratedCourse, GeneratedLesson, CourseGenerationStatus
from .tasks import check_for_new_docs, process_all_documents_with_rag, generate_courses_for_document, generate_courses_for_all_documents, export_courses_to_json
import json
from .supabase_service import SupabaseService
import logging

logger = logging.getLogger(__name__)

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
    

def supabase_courses_list(request):
    """View to list all courses from Supabase"""
    try:
        supabase = SupabaseService()
        courses = supabase.get_all_courses()
        
        # Add lesson count for each course
        for course in courses:
            # Get lessons count (you might need to adjust based on your schema)
            lessons_response = supabase.client.table('lessons').select('id').eq('course_id', course['id']).execute()
            course['lessons'] = len(lessons_response.data) if lessons_response.data else 0
        
        context = {
            'courses': courses
        }
        return render(request, 'course/supabase_courses.html', context)
    except Exception as e:
        logger.error(f"Error fetching Supabase courses: {str(e)}")
        context = {
            'courses': [],
            'error': str(e)
        }
        return render(request, 'course/supabase_courses.html', context)

def supabase_course_edit(request, course_id):
    """View to edit a course in Supabase"""
    try:
        supabase = SupabaseService()
        course = supabase.get_course_with_lessons(course_id)
        
        if not course:
            return JsonResponse({'error': 'Course not found'}, status=404)
        
        # Parse JSON fields for display
        if course.get('topic_description') and isinstance(course['topic_description'], str):
            import json as json_lib
            try:
                course['topic_description'] = json_lib.loads(course['topic_description'])
            except:
                pass
        
        # Parse lesson fields
        for lesson in course.get('lessons', []):
            # Parse skill_aims
            if lesson.get('skill_aims') and isinstance(lesson['skill_aims'], str):
                try:
                    lesson['skill_aims'] = json_lib.loads(lesson['skill_aims'])
                except:
                    lesson['skill_aims'] = []
            
            # Parse language_learning_aims
            if lesson.get('language_learning_aims') and isinstance(lesson['language_learning_aims'], str):
                try:
                    lesson['language_learning_aims'] = json_lib.loads(lesson['language_learning_aims'])
                except:
                    lesson['language_learning_aims'] = {}
            
            # Parse lesson_summary
            if lesson.get('lesson_summary') and isinstance(lesson['lesson_summary'], str):
                try:
                    lesson['lesson_summary'] = json_lib.loads(lesson['lesson_summary'])
                except:
                    lesson['lesson_summary'] = []
        
        context = {
            'course': course
        }
        return render(request, 'course/supabase_courses_edit.html', context)
    except Exception as e:
        logger.error(f"Error loading course {course_id} for edit: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["POST"])
def update_course_field(request):
    """API endpoint to update a course field in Supabase"""
    try:
        course_id = request.POST.get('course_id')
        field = request.POST.get('field')
        value = request.POST.get('value')
        
        if not all([course_id, field, value]):
            return JsonResponse({'error': 'Missing required fields'}, status=400)
        
        supabase = SupabaseService()
        success = supabase.update_course(
            int(course_id),
            {field: value},
            user=request.user.username if request.user.is_authenticated else 'anonymous'
        )
        
        if success:
            return JsonResponse({'status': 'success'})
        else:
            return JsonResponse({'error': 'Failed to update course'}, status=500)
            
    except Exception as e:
        logger.error(f"Error updating course field: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["POST"])
def update_lesson_field(request):
    """API endpoint to update a lesson field in Supabase"""
    try:
        lesson_id = request.POST.get('lesson_id')
        field = request.POST.get('field')
        value = request.POST.get('value')
        
        if not all([lesson_id, field, value]):
            return JsonResponse({'error': 'Missing required fields'}, status=400)
        
        # Parse array fields
        if field in ['skill_aims', 'lesson_summary']:
            # Convert newline-separated text to array
            value = [line.strip() for line in value.split('\n') if line.strip()]
        elif field == 'language_learning_aims':
            # This would need more complex parsing - for now keep as is
            pass
        
        supabase = SupabaseService()
        success = supabase.update_lesson(
            int(lesson_id),
            {field: value},
            user=request.user.username if request.user.is_authenticated else 'anonymous'
        )
        
        if success:
            return JsonResponse({'status': 'success'})
        else:
            return JsonResponse({'error': 'Failed to update lesson'}, status=500)
            
    except Exception as e:
        logger.error(f"Error updating lesson field: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_edit_history(request, course_id):
    """API endpoint to get edit history for a course"""
    try:
        supabase = SupabaseService()
        history = supabase.get_edit_history(course_id)
        
        return JsonResponse({'history': history})
    except Exception as e:
        logger.error(f"Error fetching edit history: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)