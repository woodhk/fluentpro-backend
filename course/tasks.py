from celery import shared_task
from django.utils import timezone
from datetime import datetime
from .models import GoogleDocument, ProcessingStatus
from .google_service import GoogleDocsService
from .embedding_service import EmbeddingService
from .langgraph_rag import RAGProcessor
import json

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