import os
import asyncio
from typing import TypedDict, Dict, Any, List, Optional, Annotated, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel, Field
from django.conf import settings

# Pydantic models for structured outputs
class RoleIndustry(BaseModel):
    """Role and industry extraction"""
    role: str = Field(description="The professional role/job title")
    industry: str = Field(description="The industry or sector")

class TopicDescriptionPair(BaseModel):
    """A topic with its description"""
    topic: str = Field(description="The main topic")
    description: str = Field(description="Detailed description of the topic")

class OrchestratorOutput(BaseModel):
    """Output from the orchestrator"""
    role_industry: RoleIndustry
    topic_pairs: List[TopicDescriptionPair]

class LessonIntro(BaseModel):
    """Basic lesson with introduction"""
    lesson_number: int = Field(description="Sequential lesson number")
    lesson_title: str = Field(description="Title of the lesson")
    lesson_introduction: str = Field(description="Introduction paragraph for the lesson")
    is_bonus: bool = Field(default=False, description="Whether this is a bonus lesson")

class CourseWithLessons(BaseModel):
    """Course with its lessons"""
    course_name: str = Field(description="Name of the course")
    lessons: List[LessonIntro] = Field(description="List of lessons in the course")

class EvaluationResult(BaseModel):
    """Evaluation result from evaluator"""
    passed: bool = Field(description="Whether the output passed evaluation")
    feedback: Optional[str] = Field(description="Feedback if evaluation failed")

class LanguageLearningAim(BaseModel):
    """Language learning aim with examples"""
    aim_category: str = Field(description="Category of language aim (e.g., 'Announcing Urgency')")
    examples: List[str] = Field(description="Example phrases for this aim")

class FullLesson(BaseModel):
    """Complete lesson with all details"""
    lesson_number: int
    lesson_title: str
    lesson_introduction: str
    skill_aims: List[str] = Field(description="List of skill aims")
    language_learning_aims: List[LanguageLearningAim]
    lesson_summary: List[str] = Field(description="Summary points")
    is_bonus: bool = False

class CourseWithFullLessons(BaseModel):
    """Course with fully detailed lessons"""
    course_name: str
    lessons: List[FullLesson]

# Define the workflow state
class WorkflowState(TypedDict):
    """State that flows through the workflow"""
    # Input
    document_id: int
    introduction: str
    main_content: str
    conclusion: str
    
    # Orchestrator outputs
    role: str
    industry: str
    topic_pairs: List[Dict[str, str]]
    
    # Worker outputs
    worker_outputs: List[Dict[str, Any]]
    
    # Evaluator feedback
    evaluation_results: List[Dict[str, Any]]
    retry_count: int
    
    # Final outputs
    final_courses: List[Dict[str, Any]]
    
    # Status
    current_step: str
    error: str

class CourseGenerationWorkflow:
    def __init__(self):
        # Initialize LLMs
        self.orchestrator_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",  # Using available model
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=8192,
            temperature=0.3
        )
        
        self.worker1_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-preview-05-06",  # Using available model
            google_api_key=settings.GOOGLE_GEMINI_API_KEY,
            temperature=0.5,
            max_output_tokens=8192
        )
        
        self.evaluator_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=4096,
            temperature=0
        )
        
        self.worker2_llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash-preview-05-20",  # Using available model
            google_api_key=settings.GOOGLE_GEMINI_API_KEY,
            temperature=0.7,
            max_output_tokens=8192
        )
        
        # Build the workflow
        self.app = self._build_workflow()
    
    def _build_workflow(self) -> CompiledGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("orchestrator", self.orchestrator_node)
        workflow.add_node("parallel_workers", self.parallel_workers_node)
        workflow.add_node("evaluator", self.evaluator_node)
        workflow.add_node("worker2", self.worker2_node)
        workflow.add_node("aggregator", self.aggregator_node)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Add edges
        workflow.add_edge("orchestrator", "parallel_workers")
        workflow.add_edge("parallel_workers", "evaluator")
        
        # Conditional edge from evaluator
        workflow.add_conditional_edges(
            "evaluator",
            self.should_retry,
            {
                "retry": "parallel_workers",
                "continue": "worker2"
            }
        )
        
        workflow.add_edge("worker2", "aggregator")
        workflow.add_edge("aggregator", END)
        
        return workflow.compile()
    
    def orchestrator_node(self, state: WorkflowState) -> WorkflowState:
        """Orchestrator that analyzes content and delegates to workers"""
        state["current_step"] = "Orchestrator analyzing content"
        
        try:
            # Extract role and industry from introduction
            intro_prompt = f"""Analyze this introduction and extract the professional role and industry it's intended for:

Introduction:
{state['introduction']}

Extract the specific job role/profession and the industry/sector."""

            role_industry_llm = self.orchestrator_llm.with_structured_output(RoleIndustry)
            role_industry = role_industry_llm.invoke([
                SystemMessage(content="You are an expert at analyzing professional documents."),
                HumanMessage(content=intro_prompt)
            ])
            
            # Extract topic-description pairs from main content
            pairs_prompt = f"""Analyze this content and identify all distinct topic-description pairs. Each topic should be a specific speaking scenario or situation that a {role_industry.role} in {role_industry.industry} would encounter.

Content:
{state['main_content']}

Extract each distinct topic with its corresponding description. These should be practical, real-world speaking scenarios."""

            orchestrator_output_llm = self.orchestrator_llm.with_structured_output(OrchestratorOutput)
            orchestrator_output = orchestrator_output_llm.invoke([
                SystemMessage(content="You are an expert at analyzing professional communication scenarios."),
                HumanMessage(content=pairs_prompt)
            ])
            
            # Update state
            state["role"] = role_industry.role
            state["industry"] = role_industry.industry
            state["topic_pairs"] = [
                {"topic": pair.topic, "description": pair.description}
                for pair in orchestrator_output.topic_pairs
            ]
            
        except Exception as e:
            state["error"] = f"Orchestrator error: {str(e)}"
        
        return state
    
    def parallel_workers_node(self, state: WorkflowState) -> WorkflowState:
        """Run Worker 1 in parallel for each topic pair"""
        state["current_step"] = "Running parallel workers"
        
        worker_outputs = []
        
        # Process each topic pair in parallel
        # Reduced max_workers for better thread pool compatibility
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            
            for idx, pair in enumerate(state["topic_pairs"]):
                future = executor.submit(
                    self._process_single_topic,
                    pair,
                    state["role"],
                    state["industry"],
                    idx
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    worker_outputs.append(result)
                except Exception as e:
                    worker_outputs.append({
                        "error": str(e),
                        "course_name": "Error",
                        "lessons": []
                    })
        
        state["worker_outputs"] = worker_outputs
        return state
    
    def _process_single_topic(self, topic_pair: Dict[str, str], role: str, industry: str, index: int) -> Dict[str, Any]:
        """Process a single topic-description pair"""
        prompt = f"""You are a subject matter expert for {role} professionals in the {industry} industry.

Topic: {topic_pair['topic']}
Description: {topic_pair['description']}

Your task is to create a course that breaks down this speaking scenario into sequential lessons. Each lesson should represent a different part of the conversation, starting from the beginning and progressing to the end.

Requirements:
1. Create lessons that follow the natural flow of the conversation
2. Each lesson should focus on a specific part of the speaking interaction
3. Add 1-2 bonus lessons for important skills that don't fit the sequential flow
4. All lessons must be speaking/verbal communication focused
5. Generate an appropriate course name

The lessons should help the learner navigate through the entire conversation step by step."""

        try:
            structured_llm = self.worker1_llm.with_structured_output(CourseWithLessons)
            result = structured_llm.invoke([
                SystemMessage(content=f"You are an expert in professional communication training for {industry}."),
                HumanMessage(content=prompt)
            ])
            
            return {
                "index": index,
                "topic_pair": topic_pair,
                "course_name": result.course_name,
                "lessons": [lesson.dict() for lesson in result.lessons]
            }
        except Exception as e:
            raise Exception(f"Worker 1 error for topic {topic_pair['topic']}: {str(e)}")
    
    def evaluator_node(self, state: WorkflowState) -> WorkflowState:
        """Evaluate worker outputs"""
        state["current_step"] = "Evaluating worker outputs"
        
        evaluation_results = []
        all_passed = True
        
        for output in state["worker_outputs"]:
            if "error" in output:
                evaluation_results.append({
                    "index": output.get("index", -1),
                    "passed": False,
                    "feedback": output["error"]
                })
                all_passed = False
                continue
            
            # Evaluate each output
            eval_prompt = f"""Evaluate this course structure:

Course: {output['course_name']}
Lessons: {json.dumps(output['lessons'], indent=2)}

Check if:
1. Lessons are in sequential order (except bonus lessons)
2. All lessons are relevant to the topic: {output['topic_pair']['topic']}
3. All lessons are speaking/verbal communication related
4. Output is properly structured

Provide feedback if any criteria are not met."""

            try:
                eval_llm = self.evaluator_llm.with_structured_output(EvaluationResult)
                eval_result = eval_llm.invoke([
                    SystemMessage(content="You are a quality assurance expert for educational content."),
                    HumanMessage(content=eval_prompt)
                ])
                
                evaluation_results.append({
                    "index": output["index"],
                    "passed": eval_result.passed,
                    "feedback": eval_result.feedback
                })
                
                if not eval_result.passed:
                    all_passed = False
                    
            except Exception as e:
                evaluation_results.append({
                    "index": output["index"],
                    "passed": False,
                    "feedback": f"Evaluation error: {str(e)}"
                })
                all_passed = False
        
        state["evaluation_results"] = evaluation_results
        
        # Increment retry count if not all passed
        if not all_passed:
            state["retry_count"] = state.get("retry_count", 0) + 1
        
        return state
    
    def should_retry(self, state: WorkflowState) -> Literal["retry", "continue"]:
        """Decide whether to retry or continue"""
        # Check if all evaluations passed
        all_passed = all(result["passed"] for result in state["evaluation_results"])
        
        # Maximum 3 retries
        if not all_passed and state.get("retry_count", 0) < 3:
            # Provide feedback to workers for retry
            for i, output in enumerate(state["worker_outputs"]):
                for eval_result in state["evaluation_results"]:
                    if eval_result["index"] == output.get("index", i):
                        if not eval_result["passed"]:
                            output["feedback"] = eval_result["feedback"]
            return "retry"
        
        return "continue"
    
    def worker2_node(self, state: WorkflowState) -> WorkflowState:
        """Generate full lesson content"""
        state["current_step"] = "Generating full lesson content"
        
        final_courses = []
        
        for output in state["worker_outputs"]:
            if "error" in output:
                continue
            
            # Check if this output passed evaluation
            passed = True
            for eval_result in state["evaluation_results"]:
                if eval_result["index"] == output.get("index", -1) and not eval_result["passed"]:
                    passed = False
                    break
            
            if not passed and state.get("retry_count", 0) >= 3:
                # Skip failed outputs after max retries
                continue
            
            try:
                full_lessons = self._generate_full_lessons(
                    output["course_name"],
                    output["lessons"],
                    state["role"],
                    state["industry"]
                )
                
                final_courses.append({
                    "course_name": output["course_name"],
                    "topic_pair": output["topic_pair"],
                    "lessons": full_lessons
                })
                
            except Exception as e:
                state["error"] = f"Worker 2 error: {str(e)}"
        
        state["final_courses"] = final_courses
        return state
    
    def _generate_full_lessons(self, course_name: str, lessons: List[Dict], role: str, industry: str) -> List[Dict]:
        """Generate full content for each lesson"""
        full_lessons = []
        
        for lesson in lessons:
            prompt = f"""You are creating a complete lesson for {role} professionals in {industry}.

Course: {course_name}
Lesson {lesson['lesson_number']}: {lesson['lesson_title']}
Introduction: {lesson['lesson_introduction']}

Generate the complete lesson content including:
1. Skill Aims (4-5 specific communication skills)
2. Language Learning Aims (3-4 categories with 3 example phrases each)
3. Lesson Summary (4 key takeaways)

Focus on practical verbal communication skills and real phrases professionals would use."""

            try:
                structured_llm = self.worker2_llm.with_structured_output(FullLesson)
                full_lesson = structured_llm.invoke([
                    SystemMessage(content=f"You are an expert communication trainer for {industry} professionals."),
                    HumanMessage(content=prompt)
                ])
                
                full_lessons.append(full_lesson.dict())
                
            except Exception as e:
                # Use original lesson with empty additional fields
                full_lessons.append({
                    **lesson,
                    "skill_aims": [],
                    "language_learning_aims": [],
                    "lesson_summary": []
                })
        
        return full_lessons
    
    def aggregator_node(self, state: WorkflowState) -> WorkflowState:
        """Aggregate all courses and lessons"""
        state["current_step"] = "Aggregating final output"
        
        # The final courses are already structured in state["final_courses"]
        # This node could perform additional formatting or validation if needed
        
        return state
    
    def process_document(self, document_id: int, structured_content: Dict[str, str]) -> Dict[str, Any]:
        """Main entry point to process a document"""
        initial_state = {
            "document_id": document_id,
            "introduction": structured_content.get("introduction", ""),
            "main_content": structured_content.get("main_content", ""),
            "conclusion": structured_content.get("conclusion", ""),
            "role": "",
            "industry": "",
            "topic_pairs": [],
            "worker_outputs": [],
            "evaluation_results": [],
            "retry_count": 0,
            "final_courses": [],
            "current_step": "Starting",
            "error": ""
        }
        
        # Run the workflow
        result = self.app.invoke(initial_state)
        
        return {
            "final_courses": result["final_courses"],
            "role": result["role"],
            "industry": result["industry"],
            "error": result["error"],
            "current_step": result["current_step"]
        }