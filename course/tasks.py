from celery import shared_task
from django.utils import timezone
from datetime import datetime
from .models import GoogleDocument, ProcessingStatus
from .google_service import GoogleDocsService
from .embedding_service import EmbeddingService
from .langgraph_rag import RAGProcessor
import json
from .course_generation_workflow import CourseGenerationWorkflow
from .models import GeneratedCourse, GeneratedLesson, CourseGenerationStatus

@shared_task
def check_for_new_docs():
    """Periodically check for new Google Docs"""
    status, _ = ProcessingStatus.objects.get_or_create(pk=1)
    status.status = 'processing'
    status.message = 'Checking for new documents...'
    status.current_stage = 'Fetching documents'
    status.save()
    
    try:
        service = GoogleDocsService()
        recent_docs = service.get_recent_docs()
        
        new_docs_count = 0
        
        for doc in recent_docs:
            doc_id = doc['id']
            
            # Check if document already exists
            if not GoogleDocument.objects.filter(doc_id=doc_id).exists():
                # Extract document content
                doc_content = service.get_document_content(doc_id)
                
                if doc_content:
                    # Parse modified time
                    modified_time = datetime.fromisoformat(
                        doc['modifiedTime'].replace('Z', '+00:00')
                    )
                    
                    # Save to database
                    google_doc = GoogleDocument.objects.create(
                        doc_id=doc_id,
                        title=doc_content['title'],
                        content=doc_content['content'],
                        last_modified=modified_time
                    )
                    new_docs_count += 1
                    
                    # Trigger RAG processing
                    process_document_with_rag.delay(google_doc.id)
        
        status.status = 'completed'
        status.message = f'Found {new_docs_count} new documents'
        status.current_stage = ''
        status.save()
        
        return f"Processed {new_docs_count} new documents"
        
    except Exception as e:
        status.status = 'error'
        status.message = str(e)
        status.current_stage = ''
        status.save()
        raise

@shared_task
def process_document_with_rag(document_id):
    """Process a document through the RAG pipeline"""
    try:
        # Get the document
        doc = GoogleDocument.objects.get(id=document_id)
        
        # Update status
        status, _ = ProcessingStatus.objects.get_or_create(pk=1)
        status.status = 'processing'
        status.message = f'Processing document: {doc.title}'
        status.current_stage = 'Generating embeddings'
        status.save()
        
        # Generate embeddings
        embedding_service = EmbeddingService()
        embeddings = embedding_service.generate_embeddings(doc.content)
        doc.set_embeddings(embeddings)
        
        # Update status
        status.current_stage = 'Extracting structured content'
        status.save()
        
        # Process with LangGraph RAG
        rag_processor = RAGProcessor()
        result = rag_processor.process_document(doc.content, embeddings)
        
        if result['error']:
            raise Exception(result['error'])
        
        # Save structured content
        doc.set_structured_content(result['structured_output'])
        doc.processing_completed = True
        doc.save()
        
        # Update status
        status.status = 'completed'
        status.message = f'Successfully processed: {doc.title}'
        status.current_stage = ''
        status.save()
        
        return f"Successfully processed document {document_id}"
        
    except Exception as e:
        if 'status' in locals():
            status.status = 'error'
            status.message = f'Error processing document: {str(e)}'
            status.current_stage = ''
            status.save()
        raise

@shared_task
def process_all_documents_with_rag():
    """Process all documents that haven't been processed yet"""
    unprocessed_docs = GoogleDocument.objects.filter(processing_completed=False)
    
    for doc in unprocessed_docs:
        process_document_with_rag.delay(doc.id)
    
    return f"Started processing {unprocessed_docs.count()} documents"


@shared_task
def generate_courses_for_document(document_id):
    """Generate courses from a processed document"""
    try:
        # Get the document
        doc = GoogleDocument.objects.get(id=document_id)
        
        # Check if document has structured content
        structured_content = doc.get_structured_content()
        if not structured_content:
            raise Exception("Document has no structured content")
        
        # Create generation status
        gen_status = CourseGenerationStatus.objects.create(
            document=doc,
            status='processing',
            current_step='Initializing workflow'
        )
        
        # Update main status
        status, _ = ProcessingStatus.objects.get_or_create(pk=1)
        status.status = 'processing'
        status.message = f'Generating courses for: {doc.title}'
        status.current_stage = 'Course Generation Workflow'
        status.save()
        
        # Initialize and run workflow
        workflow = CourseGenerationWorkflow()
        
        # Process document
        result = workflow.process_document(document_id, structured_content)
        
        # Save results
        if result['error']:
            gen_status.status = 'error'
            gen_status.error_message = result['error']
            gen_status.save()
            raise Exception(result['error'])
        
        # Save courses and lessons
        for course_data in result['final_courses']:
            course = GeneratedCourse.objects.create(
                document=doc,
                course_name=course_data['course_name'],
                role=result['role'],
                industry=result['industry']
            )
            course.set_topic_description(course_data['topic_pair'])
            course.processing_status = 'completed'
            course.save()
            
            # Save lessons
            for lesson_data in course_data['lessons']:
                lesson = GeneratedLesson.objects.create(
                    course=course,
                    lesson_number=lesson_data['lesson_number'],
                    lesson_title=lesson_data['lesson_title'],
                    lesson_introduction=lesson_data['lesson_introduction'],
                    is_bonus=lesson_data.get('is_bonus', False)
                )
                
                # Set structured fields
                lesson.set_skill_aims(lesson_data.get('skill_aims', []))
                
                # Convert language learning aims
                lang_aims = {}
                for aim in lesson_data.get('language_learning_aims', []):
                    lang_aims[aim.get('aim_category', '')] = aim.get('examples', [])
                lesson.set_language_learning_aims(lang_aims)
                
                lesson.set_lesson_summary(lesson_data.get('lesson_summary', []))
                lesson.save()
        
        # Update generation status
        gen_status.status = 'completed'
        gen_status.final_output = json.dumps(result['final_courses'])
        gen_status.completed_at = timezone.now()
        gen_status.save()
        
        # Update main status
        status.status = 'completed'
        status.message = f'Successfully generated {len(result["final_courses"])} courses'
        status.current_stage = ''
        status.save()
        
        return f"Successfully generated {len(result['final_courses'])} courses for document {document_id}"
        
    except Exception as e:
        if 'gen_status' in locals():
            gen_status.status = 'error'
            gen_status.error_message = str(e)
            gen_status.save()
        
        if 'status' in locals():
            status.status = 'error'
            status.message = f'Error generating courses: {str(e)}'
            status.current_stage = ''
            status.save()
        raise

@shared_task
def generate_courses_for_all_documents():
    """Generate courses for all documents with structured content"""
    docs_with_content = GoogleDocument.objects.filter(
        processing_completed=True,
        structured_content__isnull=False
    ).exclude(
        courses__isnull=False  # Exclude docs that already have courses
    )
    
    for doc in docs_with_content:
        generate_courses_for_document.delay(doc.id)
    
    return f"Started course generation for {docs_with_content.count()} documents"

@shared_task
def export_courses_to_json(document_id=None):
    """Export courses to JSON format for Supabase"""
    if document_id:
        courses = GeneratedCourse.objects.filter(document_id=document_id)
    else:
        courses = GeneratedCourse.objects.all()
    
    export_data = {
        "courses": [],
        "export_date": timezone.now().isoformat(),
        "total_courses": courses.count()
    }
    
    for course in courses:
        course_data = {
            "course_name": course.course_name,
            "role": course.role,
            "industry": course.industry,
            "document_title": course.document.title,
            "topic_description": course.get_topic_description(),
            "lessons": []
        }
        
        for lesson in course.lessons.all():
            lesson_data = {
                "lesson_number": lesson.lesson_number,
                "lesson_title": lesson.lesson_title,
                "lesson_introduction": lesson.lesson_introduction,
                "skill_aims": lesson.get_skill_aims(),
                "language_learning_aims": lesson.get_language_learning_aims(),
                "lesson_summary": lesson.get_lesson_summary(),
                "is_bonus": lesson.is_bonus
            }
            course_data["lessons"].append(lesson_data)
        
        export_data["courses"].append(course_data)
    
    # Save to file
    export_path = settings.BASE_DIR / 'exports' / f'courses_export_{timezone.now().strftime("%Y%m%d_%H%M%S")}.json'
    export_path.parent.mkdir(exist_ok=True)
    
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return f"Exported {courses.count()} courses to {export_path}"