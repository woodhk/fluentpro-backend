import os
import pickle
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from django.conf import settings

class GoogleDocsService:
    def __init__(self):
        self.creds = None
        self.docs_service = None
        self.drive_service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate and create Google services"""
        if os.path.exists(settings.GOOGLE_DOCS_TOKEN_FILE):
            with open(settings.GOOGLE_DOCS_TOKEN_FILE, 'rb') as token:
                self.creds = pickle.load(token)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not os.path.exists(settings.GOOGLE_DOCS_CREDENTIALS_FILE):
                    raise Exception("credentials.json file not found. Please download it from Google Cloud Console.")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.GOOGLE_DOCS_CREDENTIALS_FILE, 
                    settings.GOOGLE_DOCS_SCOPES
                )
                self.creds = flow.run_local_server(port=0)
            
            with open(settings.GOOGLE_DOCS_TOKEN_FILE, 'wb') as token:
                pickle.dump(self.creds, token)
        
        self.docs_service = build('docs', 'v1', credentials=self.creds)
        self.drive_service = build('drive', 'v3', credentials=self.creds)
    
    def get_recent_docs(self, max_results=10):
        """Get list of recent Google Docs"""
        try:
            results = self.drive_service.files().list(
                q="mimeType='application/vnd.google-apps.document'",
                pageSize=max_results,
                fields="files(id, name, modifiedTime)",
                orderBy="modifiedTime desc"
            ).execute()
            
            return results.get('files', [])
        except Exception as e:
            print(f"Error fetching docs: {e}")
            return []
    
    def get_document_content(self, doc_id):
        """Extract content from a Google Doc"""
        try:
            document = self.docs_service.documents().get(documentId=doc_id).execute()
            
            # Extract text content
            content = ""
            for element in document.get('body', {}).get('content', []):
                if 'paragraph' in element:
                    for text_run in element['paragraph'].get('elements', []):
                        if 'textRun' in text_run:
                            content += text_run['textRun'].get('content', '')
            
            return {
                'title': document.get('title', 'Untitled'),
                'content': content.strip()
            }
        except Exception as e:
            print(f"Error fetching document content: {e}")
            return None