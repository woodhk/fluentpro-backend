import os
from typing import List, Dict, Any
import openai
from django.conf import settings
import tiktoken
import numpy as np

class EmbeddingService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "text-embedding-3-large"
        self.encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
        
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, max_tokens: int = 8000) -> List[str]:
        """
        Split text into chunks that fit within token limits.
        Each chunk will be under max_tokens.
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            
            if paragraph_tokens > max_tokens:
                # If a single paragraph is too long, split it by sentences
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence + '. ')
                    if current_tokens + sentence_tokens > max_tokens:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
                        current_tokens = sentence_tokens
                    else:
                        current_chunk += sentence + '. '
                        current_tokens += sentence_tokens
            elif current_tokens + paragraph_tokens > max_tokens:
                # Start a new chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_tokens = paragraph_tokens
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for a text using OpenAI's embedding model.
        Handles long texts by chunking if necessary.
        """
        try:
            # Check token count
            token_count = self.count_tokens(text)
            
            if token_count > 8000:
                # Text is too long, need to chunk and average embeddings
                chunks = self.chunk_text(text)
                all_embeddings = []
                
                for chunk in chunks:
                    response = self.client.embeddings.create(
                        input=chunk,
                        model=self.model
                    )
                    all_embeddings.append(response.data[0].embedding)
                
                # Average the embeddings
                averaged_embedding = np.mean(all_embeddings, axis=0).tolist()
                return averaged_embedding
            else:
                # Text fits within limits
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def generate_chunk_embeddings(self, text: str) -> List[Dict[str, Any]]:
        """
        Generate embeddings for each chunk of text separately.
        Returns a list of dictionaries with chunk text and embeddings.
        """
        chunks = self.chunk_text(text)
        chunk_embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                response = self.client.embeddings.create(
                    input=chunk,
                    model=self.model
                )
                chunk_embeddings.append({
                    "chunk_index": i,
                    "text": chunk,
                    "embedding": response.data[0].embedding,
                    "token_count": self.count_tokens(chunk)
                })
            except Exception as e:
                print(f"Error embedding chunk {i}: {e}")
                raise
        
        return chunk_embeddings