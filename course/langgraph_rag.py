import os
from typing import TypedDict, Dict, Any, List, Annotated, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from django.conf import settings
import json
import tiktoken

# Pydantic models for structured output
class StructuredDocument(BaseModel):
    """Structured representation of a document"""
    introduction: str = Field(description="The first paragraph(s) that introduce the document")
    main_content: str = Field(description="The main body of the document (everything between intro and conclusion)")
    conclusion: str = Field(description="The final paragraph(s) that conclude the document")

# Define the state
class GraphState(TypedDict):
    """State that is passed between nodes in the graph"""
    content: str
    embeddings: List[float]
    structured_output: Dict[str, Any]
    error: str
    
class RAGProcessor:
    def __init__(self):
        # Initialize Claude with Anthropic API
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=12000,  # Increased from 8192 to handle larger documents
            temperature=0
        )
        
        # Initialize tokenizer for counting tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Create the graph
        self.workflow = StateGraph(GraphState)
        
        # Add nodes
        self.workflow.add_node("process_content", self.process_content)
        
        # Set entry point
        self.workflow.set_entry_point("process_content")
        
        # Set finish point
        self.workflow.add_edge("process_content", END)
        
        # Compile the graph
        self.app = self.workflow.compile()
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.encoding.encode(text))
    
    def chunk_content(self, content: str, max_input_tokens: int = 15000) -> List[str]:
        """
        Split content into manageable chunks for processing.
        Leaves room for prompt overhead and response tokens.
        """
        token_count = self.count_tokens(content)
        
        if token_count <= max_input_tokens:
            return [content]
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            
            if paragraph_tokens > max_input_tokens:
                # If a single paragraph is too long, split it by sentences
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence + '. ')
                    if current_tokens + sentence_tokens > max_input_tokens:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
                        current_tokens = sentence_tokens
                    else:
                        current_chunk += sentence + '. '
                        current_tokens += sentence_tokens
            elif current_tokens + paragraph_tokens > max_input_tokens:
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

    def create_extraction_prompt(self, content: str, is_chunk: bool = False, chunk_index: int = 0, total_chunks: int = 1) -> str:
        """Create the prompt for Claude to extract and structure content"""
        if is_chunk and total_chunks > 1:
            return f"""You are a document analysis expert. Your task is to extract and structure the content of this document chunk.

This is chunk {chunk_index + 1} of {total_chunks} from a larger document.

CRITICAL INSTRUCTIONS:
1. Extract the content VERBATIM - do not paraphrase, summarize, or modify any text
2. For this chunk, provide what you can identify as:
   - Introduction: Any opening content that introduces topics (even if partial)
   - Main Content: The substantive body content in this chunk
   - Conclusion: Any concluding content (even if partial)
3. If a section is not present or incomplete in this chunk, provide what you can and note "[PARTIAL]" or "[CONTINUATION]" as appropriate
4. Ignore any citations or references (text in brackets like [1] or (Author, Year))
5. Preserve all original formatting, punctuation, and wording exactly as it appears

DOCUMENT CHUNK CONTENT:
{content}

Remember: Copy the text EXACTLY as it appears. Do not modify, summarize, or paraphrase any content."""
        else:
            return f"""You are a document analysis expert. Your task is to extract and structure the content of the following document.

CRITICAL INSTRUCTIONS:
1. Extract the content VERBATIM - do not paraphrase, summarize, or modify any text
2. Divide the content into three sections:
   - Introduction: The opening paragraph(s) that introduce the document
   - Main Content: The body of the document between the introduction and conclusion
   - Conclusion: The final paragraph(s) that conclude the document
3. Ignore any citations or references (text in brackets like [1] or (Author, Year))
4. Preserve all original formatting, punctuation, and wording exactly as it appears

If the document doesn't have a clear introduction or conclusion, use your best judgment to identify logical sections based on the content structure.

DOCUMENT CONTENT:
{content}

Remember: Copy the text EXACTLY as it appears. Do not modify, summarize, or paraphrase any content."""

    def merge_chunk_results(self, chunk_results: List[Dict[str, str]]) -> Dict[str, str]:
        """Merge results from multiple chunks into a single structured document"""
        merged = {
            "introduction": "",
            "main_content": "",
            "conclusion": ""
        }
        
        # Collect all parts
        all_intros = []
        all_main = []
        all_conclusions = []
        
        for result in chunk_results:
            if result.get("introduction") and not result["introduction"].startswith("[PARTIAL]"):
                all_intros.append(result["introduction"])
            if result.get("main_content"):
                all_main.append(result["main_content"])
            if result.get("conclusion") and not result["conclusion"].startswith("[PARTIAL]"):
                all_conclusions.append(result["conclusion"])
        
        # Merge sections
        merged["introduction"] = "\n\n".join(all_intros) if all_intros else "Document introduction not clearly identified."
        merged["main_content"] = "\n\n".join(all_main) if all_main else "Main content processing incomplete."
        merged["conclusion"] = "\n\n".join(all_conclusions) if all_conclusions else "Document conclusion not clearly identified."
        
        return merged

    def process_content(self, state: GraphState) -> GraphState:
        """Process content with Claude to extract structured data"""
        try:
            content = state["content"]
            token_count = self.count_tokens(content)
            
            print(f"Processing document with {token_count} tokens")
            
            # Check if content needs chunking
            chunks = self.chunk_content(content)
            
            if len(chunks) == 1:
                # Process single chunk
                result = self._process_single_chunk(content, False, 0, 1)
                if result:
                    state["structured_output"] = result
                    state["error"] = ""
                else:
                    state["error"] = "Failed to extract structured content from document"
                    state["structured_output"] = {}
            else:
                # Process multiple chunks and merge
                print(f"Document split into {len(chunks)} chunks for processing")
                chunk_results = []
                
                for i, chunk in enumerate(chunks):
                    result = self._process_single_chunk(chunk, True, i, len(chunks))
                    if result:
                        chunk_results.append(result)
                    else:
                        print(f"Warning: Failed to process chunk {i+1}")
                
                if chunk_results:
                    merged_result = self.merge_chunk_results(chunk_results)
                    state["structured_output"] = merged_result
                    state["error"] = ""
                else:
                    state["error"] = "Failed to process any document chunks"
                    state["structured_output"] = {}
            
        except Exception as e:
            state["error"] = f"Error processing content: {str(e)}"
            state["structured_output"] = {}
            
        return state
    
    def _process_single_chunk(self, content: str, is_chunk: bool, chunk_index: int, total_chunks: int) -> Dict[str, str]:
        """Process a single chunk of content"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Create structured output tool
                structured_llm = self.llm.with_structured_output(
                    StructuredDocument,
                    method="function_calling"
                )
                
                # Create the system message
                system_msg = SystemMessage(content="""You are a precise document analyst. 
Your role is to extract and structure document content exactly as it appears, without any modifications.
Always preserve the original text verbatim. 

IMPORTANT: You must provide content for all three required fields (introduction, main_content, conclusion).
If a section is not clearly present, provide a brief note explaining what you found instead.
Never leave any field empty or null.""")
                
                # Create the human message with our prompt
                human_msg = HumanMessage(content=self.create_extraction_prompt(content, is_chunk, chunk_index, total_chunks))
                
                # Get structured response
                response = structured_llm.invoke([system_msg, human_msg])
                
                # Validate response has all required fields
                if not hasattr(response, 'introduction') or not hasattr(response, 'main_content') or not hasattr(response, 'conclusion'):
                    print(f"Warning: Incomplete response from LLM for chunk {chunk_index + 1}, attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        # Last attempt failed, create fallback response
                        return self._create_fallback_response(content, chunk_index)
                
                # Check for empty or very short responses (indicating truncation)
                intro = response.introduction or ""
                main = response.main_content or ""
                conclusion = response.conclusion or ""
                
                if len(intro.strip()) < 10 and len(main.strip()) < 10 and len(conclusion.strip()) < 10:
                    print(f"Warning: Very short response from LLM for chunk {chunk_index + 1}, attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return self._create_fallback_response(content, chunk_index)
                
                # Convert Pydantic model to dict
                structured_data = {
                    "introduction": intro or "No introduction identified in this section.",
                    "main_content": main or "No main content identified in this section.",
                    "conclusion": conclusion or "No conclusion identified in this section."
                }
                
                return structured_data
                
            except Exception as e:
                print(f"Error processing chunk {chunk_index + 1}, attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                else:
                    # Last attempt failed, create fallback response
                    return self._create_fallback_response(content, chunk_index)
        
        return None
    
    def _create_fallback_response(self, content: str, chunk_index: int) -> Dict[str, str]:
        """Create a fallback response when AI processing fails"""
        print(f"Creating fallback response for chunk {chunk_index + 1}")
        
        # Simple heuristic-based content splitting
        paragraphs = content.split('\n\n')
        total_paragraphs = len(paragraphs)
        
        if total_paragraphs <= 3:
            # Short content, treat as main content
            return {
                "introduction": f"[Fallback] Chunk {chunk_index + 1} content (AI processing failed)",
                "main_content": content,
                "conclusion": "[Fallback] End of chunk content"
            }
        else:
            # Longer content, split heuristically
            intro_end = max(1, total_paragraphs // 4)
            conclusion_start = min(total_paragraphs - 1, total_paragraphs * 3 // 4)
            
            intro = '\n\n'.join(paragraphs[:intro_end])
            main = '\n\n'.join(paragraphs[intro_end:conclusion_start])
            conclusion = '\n\n'.join(paragraphs[conclusion_start:])
            
            return {
                "introduction": intro or f"[Fallback] Introduction from chunk {chunk_index + 1}",
                "main_content": main or f"[Fallback] Main content from chunk {chunk_index + 1}",
                "conclusion": conclusion or f"[Fallback] Conclusion from chunk {chunk_index + 1}"
            }

    def process_document(self, content: str, embeddings: List[float]) -> Dict[str, Any]:
        """Main method to process a document through the RAG pipeline"""
        # Prepare initial state
        initial_state = {
            "content": content,
            "embeddings": embeddings,
            "structured_output": {},
            "error": ""
        }
        
        # Run the graph
        result = self.app.invoke(initial_state)
        
        return {
            "structured_output": result["structured_output"],
            "error": result["error"]
        }