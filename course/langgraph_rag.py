import os
from typing import TypedDict, Dict, Any, List, Annotated, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from django.conf import settings
import json

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
            model="claude-sonnet-4-20250514",  # Using available Sonnet model
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=8192,
            temperature=0
        )
        
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
    
    def create_extraction_prompt(self, content: str) -> str:
        """Create the prompt for Claude to extract and structure content"""
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

    def process_content(self, state: GraphState) -> GraphState:
        """Process content with Claude to extract structured data"""
        try:
            content = state["content"]
            
            # Create structured output tool
            structured_llm = self.llm.with_structured_output(
                StructuredDocument,
                method="function_calling"
            )
            
            # Create the system message
            system_msg = SystemMessage(content="""You are a precise document analyst. 
Your role is to extract and structure document content exactly as it appears, without any modifications.
Always preserve the original text verbatim.""")
            
            # Create the human message with our prompt
            human_msg = HumanMessage(content=self.create_extraction_prompt(content))
            
            # Get structured response
            response = structured_llm.invoke([system_msg, human_msg])
            
            # Convert Pydantic model to dict
            structured_data = {
                "introduction": response.introduction,
                "main_content": response.main_content,
                "conclusion": response.conclusion
            }
            
            state["structured_output"] = structured_data
            state["error"] = ""
            
        except Exception as e:
            state["error"] = f"Error processing content: {str(e)}"
            state["structured_output"] = {}
            
        return state
    
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