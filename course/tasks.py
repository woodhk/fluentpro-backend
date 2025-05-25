from celery import shared_task
from django.utils import timezone
from datetime import datetime
from .models import GoogleDocument, ProcessingStatus
from .google_service import GoogleDocsService

@shared_task
def check_for_new_docs():
    """Periodically check for new Google Docs"""
    status, _ = ProcessingStatus.objects.get_or_create(pk=1)
    status.status = 'processing'
    status.message = 'Checking for new documents...'
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
                    GoogleDocument.objects.create(
                        doc_id=doc_id,
                        title=doc_content['title'],
                        content=doc_content['content'],
                        last_modified=modified_time
                    )
                    new_docs_count += 1
        
        status.status = 'completed'
        status.message = f'Found {new_docs_count} new documents'
        status.save()
        
        return f"Processed {new_docs_count} new documents"
        
    except Exception as e:
        status.status = 'error'
        status.message = str(e)
        status.save()
        raise