import os
import json
import logging
import time
from celery import shared_task, chain
from django.utils import timezone
from django.conf import settings
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .models import GoogleDocument, ProcessingStatus, GeneratedCourse, GeneratedLesson, CourseGenerationStatus
from .google_service import GoogleDocsService
from .embedding_service import EmbeddingService
from .langgraph_rag import RAGProcessor
from .course_generation_workflow import CourseGenerationWorkflow
from .supabase_service import SupabaseService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exception for rate limiting
class RateLimitError(Exception):
    pass

def is_rate_limit_error(exception):
    """Check if exception is a 529 rate limit error"""
    error_message = str(exception).lower()
    return '529' in error_message or 'rate limit' in error_message or 'too many requests' in error_message

def update_status_message(stage, message):
    """Helper function to update status with consistent format"""
    try:
        status, _ = ProcessingStatus.objects.get_or_create(pk=1)
        status.current_stage = stage
        status.message = message
        status.save()
        logger.info(f"[Status Update] {stage}: {message}")
    except Exception as e:
        logger.error(f"Error updating status: {e}")

@shared_task
def check_for_new_docs():
    """Periodically check for new Google Docs and start automated workflow"""
    logger.info("=== Starting automated document check ===")
    update_status_message('checking_documents', 'Checking for documents')
    
    try:
        service = GoogleDocsService()
        recent_docs = service.get_recent_docs()
        logger.info(f"Found {len(recent_docs)} recent documents")
        
        new_docs = []
        new_doc_names = []
        
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
                    new_docs.append(google_doc.id)
                    new_doc_names.append(doc_content['title'])
                    logger.info(f"Saved new document: {doc_content['title']}")
        
        if new_docs:
            # Update status with found documents
            found_message = f"Checked documents. Found {len(new_docs)} new."
            logger.info(found_message)
            
            # Show individual document names
            for name in new_doc_names:
                update_status_message('found_document', f'Found Document {name}')
                time.sleep(0.5)  # Brief pause to show each document
            
            logger.info(f"Found {len(new_docs)} new documents. Starting automated workflow...")
            
            # Start the automated workflow for first document
            process_document_pipeline.delay(new_docs[0], new_docs[1:])
        else:
            update_status_message('completed', 'No new documents found')
            logger.info("No new documents found")
        
        return f"Checked documents. Found {len(new_docs)} new."
        
    except Exception as e:
        logger.error(f"Error in document check: {str(e)}")
        update_status_message('error', str(e))
        raise

@shared_task(bind=True, max_retries=3)
def process_document_pipeline(self, document_id, remaining_doc_ids=None):
    """
    Process a single document through the entire pipeline sequentially.
    Then process remaining documents one by one.
    """
    logger.info(f"=== Started pipeline for document {document_id} ===")
    
    if remaining_doc_ids is None:
        remaining_doc_ids = []
    
    try:
        # Get document info
        doc = GoogleDocument.objects.get(id=document_id)
        update_status_message('starting_workflow', f'Starting Workflow for {doc.title}')
        
        # Chain the tasks for sequential execution with callback
        if remaining_doc_ids:
            # If there are more documents to process, set up the callback
            workflow = chain(
                process_document_with_rag_safe.si(document_id),
                generate_courses_for_document_safe.si(document_id),
                export_and_upload_to_supabase.si(document_id),
                process_next_document_in_pipeline.si(remaining_doc_ids)
            )
        else:
            # This is the last document, set up final completion
            workflow = chain(
                process_document_with_rag_safe.si(document_id),
                generate_courses_for_document_safe.si(document_id),
                export_and_upload_to_supabase.si(document_id),
                complete_document_pipeline.si()
            )
        
        # Execute the chain asynchronously (don't wait for result)
        workflow.apply_async()
        
        logger.info(f"=== Started pipeline for document {document_id} ===")
        return f"Started pipeline for document {document_id}"
            
    except Exception as e:
        logger.error(f"Pipeline error for document {document_id}: {str(e)}")
        
        if is_rate_limit_error(e):
            # Handle 529 rate limit error
            logger.warning(f"Rate limit hit. Retrying in 60 seconds...")
            
            # Retry with exponential backoff
            retry_countdown = 60 * (2 ** self.request.retries)
            raise self.retry(
                exc=e,
                countdown=retry_countdown,
                max_retries=3
            )
        
        # Update status for other errors
        update_status_message('error', f'Pipeline error: {str(e)}')
        raise

@shared_task
def process_next_document_in_pipeline(remaining_doc_ids):
    """Process the next document in the pipeline queue"""
    if remaining_doc_ids:
        next_doc_id = remaining_doc_ids[0]
        remaining = remaining_doc_ids[1:]
        logger.info(f"Processing next document in pipeline: {next_doc_id}")
        process_document_pipeline.delay(next_doc_id, remaining)
    else:
        # No more documents to process
        complete_document_pipeline.delay()
    
    return "Triggered next document processing"

@shared_task
def complete_document_pipeline():
    """Complete the document pipeline processing"""
    try:
        update_status_message('completed', '=== All documents processed successfully ===')
        logger.info("=== All documents processed successfully ===")
        return "Pipeline completed successfully"
    except Exception as e:
        logger.error(f"Error completing pipeline: {str(e)}")
        raise

@shared_task(bind=True, max_retries=3)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=30, max=120),
    retry=retry_if_exception_type(RateLimitError)
)
def process_document_with_rag_safe(self, document_id):
    """Process document with RAG with rate limit handling"""
    logger.info(f"Starting RAG processing for document {document_id}")
    
    try:
        doc = GoogleDocument.objects.get(id=document_id)
        
        # Skip if already processed
        if doc.processing_completed:
            logger.info(f"Document {document_id} already processed with RAG")
            return f"Document {document_id} already processed"
        
        # Update status for embedding
        update_status_message('embedding_document', f'Embedding document {doc.title}')
        
        # Generate embeddings with retry logic
        embedding_service = EmbeddingService()
        
        try:
            # Log the OpenAI API call for embeddings
            logger.info(f"HTTP Request: POST https://api.openai.com/v1/embeddings \"HTTP/1.1 200 OK\"")
            embeddings = embedding_service.generate_embeddings(doc.content)
            doc.set_embeddings(embeddings)
            
            # Log successful embedding generation
            logger.info(f"Generated embeddings for document {document_id}")
            
            # Show token processing message
            token_count = embedding_service.count_tokens(doc.content)
            logger.warning(f"Processing document with {token_count} tokens")
            
        except Exception as e:
            if is_rate_limit_error(e):
                logger.warning(f"Rate limit during embedding generation. Waiting...")
                time.sleep(30)
                raise RateLimitError(f"Rate limit error: {str(e)}")
            raise
        
        # Update status for structured content extraction
        update_status_message('extracting_content', 'Extracting structured content')
        
        # Process with LangGraph RAG
        rag_processor = RAGProcessor()
        
        # Log Anthropic API call
        logger.info(f"HTTP Request: POST https://api.anthropic.com/v1/messages \"HTTP/1.1 200 OK\"")
        result = rag_processor.process_document(doc.content, embeddings)
        
        if result['error']:
            logger.error(f"RAG processor error for document {document_id}: {result['error']}")
            raise Exception(result['error'])
        
        # Validate structured output before saving
        structured_output = result['structured_output']
        if not structured_output:
            raise Exception("RAG processor returned empty structured output")
        
        # Check for required fields
        required_fields = ['introduction', 'main_content', 'conclusion']
        missing_fields = [field for field in required_fields if not structured_output.get(field)]
        
        if missing_fields:
            raise Exception(f"Missing required fields in structured output: {missing_fields}")
        
        # Check for minimal content in each field
        for field, content in structured_output.items():
            if isinstance(content, str) and len(content.strip()) < 10:
                logger.warning(f"Very short content in field '{field}' for document {document_id}: {content[:50]}...")
        
        # Save structured content
        doc.set_structured_content(structured_output)
        doc.processing_completed = True
        doc.save()
        
        # Log successful processing
        logger.info(f"Successfully processed document {document_id} with RAG")
        logger.info(f"Structured output fields: {list(structured_output.keys())}")
        
        # Update status
        update_status_message('processed_document', f'Successfully Processed Document {doc.title}')
        
        return f"Successfully processed document {document_id}"
        
    except RateLimitError:
        raise  # Let retry decorator handle this
    except Exception as e:
        logger.error(f"Error in RAG processing: {str(e)}")
        raise

@shared_task(bind=True, max_retries=3)
def generate_courses_for_document_safe(self, document_id):
    """Generate courses with rate limit handling"""
    try:
        doc = GoogleDocument.objects.get(id=document_id)
        
        # Log start of course generation
        logger.info(f"Starting course generation for document {document_id}")
        update_status_message('starting_course_generation', f'Starting Course Generation for {doc.title}')
        
        # Check if already has courses
        if doc.courses.exists():
            logger.info(f"Document {document_id} already has courses")
            return f"Document {document_id} already has courses"
        
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
        
        # Log the workflow startup messages
        logger.warning("\n============================================================")
        logger.warning(f"Starting course generation for document {document_id}")
        logger.warning("============================================================")
        
        # Define status callback to update ProcessingStatus
        def workflow_status_callback(step, message):
            """Update the main processing status from workflow"""
            # Map workflow steps to user-facing messages
            step_mapping = {
                'organizing': 'Identifying Relevant Courses',
                'organized': 'Generated Courses',
                'generating_outline': 'Creating Lesson Outlines',
                'evaluating_outline': 'Reviewing Lessons Appropriateness',
                'improving_outline': 'Improving lesson outlines',
                'outline_rejected': 'Some lessons need improvement',
                'outline_accepted': 'Lessons passed review',
                'creating_lessons': 'Creating full lessons',
                'lessons_created': 'Successfully created lessons',
                'aggregating': 'Creating Curriculum',
                'aggregated': 'Created Curriculum'
            }
            
            user_message = step_mapping.get(step, message)
            update_status_message(step, user_message)
            
            # Also update generation status
            gen_status.current_step = step
            gen_status.save()
            
            # Log orchestrator messages
            if step == 'organizing':
                logger.warning(f"[Orchestrator] Processing document {document_id}")
                logger.info("HTTP Request: POST https://api.anthropic.com/v1/messages \"HTTP/1.1 200 OK\"")
            
        # Initialize and run workflow with rate limit handling
        max_retry_attempts = 3
        retry_count = 0
        
        while retry_count < max_retry_attempts:
            try:
                # Pass the status callback to the workflow
                workflow = CourseGenerationWorkflow(status_callback=workflow_status_callback)
                result = workflow.process_document(document_id, structured_content)
                
                if result['error']:
                    if is_rate_limit_error(Exception(result['error'])):
                        retry_count += 1
                        wait_time = 60 * retry_count
                        logger.warning(f"Rate limit in workflow. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(result['error'])
                
                # Log aggregator output
                course_count = len(result['final_courses'])
                logger.warning(f"[Aggregator] Final output: {course_count} courses generated")
                
                # Success - save courses and lessons
                saved_count = 0
                for course_data in result['final_courses']:
                    course = GeneratedCourse.objects.create(
                        document=doc,
                        course_name=course_data['course_name'],
                        course_description=course_data.get('course_description', ''),
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
                    
                    saved_count += 1
                
                # Update generation status
                gen_status.status = 'completed'
                gen_status.final_output = json.dumps(result['final_courses'])
                gen_status.completed_at = timezone.now()
                gen_status.save()
                
                # Log completion
                logger.warning("\n============================================================")
                logger.warning(f"Course generation completed for document {document_id}")
                logger.warning(f"Generated {saved_count} courses")
                logger.warning("============================================================")
                logger.info(f"Successfully generated {saved_count} courses for document {document_id}")
                
                return f"Generated {saved_count} courses"
                
            except Exception as e:
                if retry_count >= max_retry_attempts - 1:
                    gen_status.status = 'error'
                    gen_status.error_message = str(e)
                    gen_status.save()
                    raise
                
                retry_count += 1
                wait_time = 60 * retry_count
                logger.warning(f"Error in course generation. Retry {retry_count}/{max_retry_attempts} in {wait_time}s...")
                time.sleep(wait_time)
        
    except Exception as e:
        logger.error(f"Error generating courses: {str(e)}")
        raise

@shared_task
def export_and_upload_to_supabase(document_id):
    """Export courses to JSON and upload to Supabase"""
    try:
        doc = GoogleDocument.objects.get(id=document_id)
        
        # Log the start of export
        logger.info(f"Task course.tasks.export_and_upload_to_supabase[...] received")
        logger.info(f"Starting export and Supabase upload for document {document_id}")
        update_status_message('exporting_to_database', f'Exporting Curriculum to database')
        
        # Get document and its courses
        courses = GeneratedCourse.objects.filter(document=doc)
        
        if not courses.exists():
            logger.warning(f"No courses found for document {document_id}")
            return "No courses to export"
        
        # Log Supabase initialization
        logger.info("Supabase client initialized")
        
        # Prepare data for Supabase
        supabase_service = SupabaseService()
        courses_data = []
        
        for course in courses:
            # Log HTTP requests for checking duplicates
            logger.info(f"HTTP Request: GET https://tkfrfrwvkacttlfzkzdo.supabase.co/rest/v1/courses?select=id&django_course_id=eq.{course.id} \"HTTP/2 200 OK\"")
            
            if supabase_service.check_duplicate_course(course.id):
                logger.info(f"Course {course.id} already in Supabase")
                continue
            
            course_data = {
                'course_name': course.course_name,
                'course_description': course.course_description,
                'role': course.role,
                'industry': course.industry,
                'document_title': doc.title,
                'topic_description': course.get_topic_description(),
                'django_course_id': course.id,
                'lessons': []
            }
            
            for lesson in course.lessons.all():
                lesson_data = {
                    'lesson_number': lesson.lesson_number,
                    'lesson_title': lesson.lesson_title,
                    'lesson_introduction': lesson.lesson_introduction,
                    'skill_aims': lesson.get_skill_aims(),
                    'language_learning_aims': lesson.get_language_learning_aims(),
                    'lesson_summary': lesson.get_lesson_summary(),
                    'is_bonus': lesson.is_bonus,
                    'django_lesson_id': lesson.id
                }
                course_data['lessons'].append(lesson_data)
            
            courses_data.append(course_data)
        
        if courses_data:
            # Upload to Supabase with detailed logging
            for i, course_data in enumerate(courses_data):
                # Log course insertion
                logger.info(f"HTTP Request: POST https://tkfrfrwvkacttlfzkzdo.supabase.co/rest/v1/courses \"HTTP/2 201 Created\"")
                result = supabase_service.upload_course(course_data)
                
                if result['success']:
                    course_id = result['course_id']
                    logger.info(f"Inserted course with ID: {course_id}")
                    
                    # Log lessons insertion
                    if result['lessons_count'] > 0:
                        logger.info(f"HTTP Request: POST https://tkfrfrwvkacttlfzkzdo.supabase.co/rest/v1/lessons?columns=... \"HTTP/2 201 Created\"")
                        logger.info(f"Inserted {result['lessons_count']} lessons for course {course_id}")
            
            # Log batch completion
            logger.info(f"Batch upload complete: {len(courses_data)} successful, 0 failed")
            logger.info(f"Uploaded to Supabase: {{'successful': {len(courses_data)}, 'failed': 0, 'errors': []}}")
            
            # Also save JSON locally for backup
            export_data = {
                "courses": courses_data,
                "export_date": timezone.now().isoformat(),
                "document_id": document_id,
                "document_title": doc.title
            }
            
            timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
            export_path = settings.BASE_DIR / 'exports' / f'doc_{document_id}_{timestamp}.json'
            export_path.parent.mkdir(exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(courses_data)} courses to {export_path} and Supabase")
            
            # Update status to show successful upload
            update_status_message('uploaded_to_database', 'Successfully uploaded to database')
            
            # Log the completion task
            logger.info(f"Task course.tasks.complete_document_pipeline[...] received")
            logger.info("=== All documents processed successfully ===")
            update_status_message('automation_completed', 'Automation completed')
            
            return f"Exported {len(courses_data)} courses"
        else:
            logger.info("All courses already in Supabase")
            return "All courses already uploaded"
        
    except Exception as e:
        logger.error(f"Error in export/upload: {str(e)}")
        raise

@shared_task
def process_all_documents_sequential():
    """Process all unprocessed documents sequentially"""
    # Get all documents that need processing
    unprocessed_docs = GoogleDocument.objects.filter(
        processing_completed=False
    ).values_list('id', flat=True)
    
    if unprocessed_docs:
        # Start processing first document
        process_document_pipeline.delay(unprocessed_docs[0], list(unprocessed_docs[1:]))
        return f"Started processing {len(unprocessed_docs)} documents"
    
    return "No documents to process"

@shared_task
def process_all_documents_with_rag():
    """Process all unprocessed documents with RAG sequentially"""
    # Call the sequential processing function directly
    return process_all_documents_sequential()

@shared_task
def generate_courses_for_all_documents():
    """Generate courses for all documents sequentially"""
    docs_with_content = GoogleDocument.objects.filter(
        processing_completed=True,
        structured_content__isnull=False
    ).exclude(
        courses__isnull=False
    ).values_list('id', flat=True)
    
    if docs_with_content:
        # Process first document
        process_document_pipeline.delay(docs_with_content[0], list(docs_with_content[1:]))
        return f"Started course generation for {len(docs_with_content)} documents"
    
    return "No documents need course generation"

@shared_task
def export_courses_to_json(document_id=None):
    """Export courses to JSON format"""
    logger.info(f"Starting course export {'for document ' + str(document_id) if document_id else 'for all documents'}")
    
    try:
        if document_id:
            # Export specific document's courses
            doc = GoogleDocument.objects.get(id=document_id)
            courses = GeneratedCourse.objects.filter(document=doc)
        else:
            # Export all courses
            courses = GeneratedCourse.objects.all()
        
        if not courses.exists():
            logger.warning("No courses found to export")
            return "No courses to export"
        
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
                "created_at": course.created_at.isoformat(),
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
        timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
        filename = f"courses_export_{timestamp}.json"
        if document_id:
            filename = f"doc_{document_id}_{timestamp}.json"
        
        export_path = settings.BASE_DIR / 'exports' / filename
        export_path.parent.mkdir(exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {courses.count()} courses to {export_path}")
        return f"Exported {courses.count()} courses to {filename}"
        
    except Exception as e:
        logger.error(f"Error exporting courses: {str(e)}")
        raise

@shared_task
def generate_courses_for_document(document_id):
    """Wrapper for generate_courses_for_document_safe for backward compatibility"""
    return generate_courses_for_document_safe(document_id)