import os
import asyncio
from typing import TypedDict, Dict, Any, List, Optional, Annotated, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import time

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
    lesson_number: int = Field(description="Lesson number")
    lesson_title: str = Field(description="Title of the lesson")
    lesson_introduction: str = Field(description="Introduction paragraph for the lesson")

class CourseWithLessons(BaseModel):
    """Course with its lessons"""
    course_name: str = Field(description="Name of the course")
    course_description: str = Field(description="Description of the course")
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
        # Initialize LLMs with correct model names
        self.orchestrator_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=8192,
            temperature=0.3
        )
        
        # Fixed model names for Google Gemini
        self.worker1_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-preview-05-06",  # Fixed model name
            google_api_key=settings.GOOGLE_GEMINI_API_KEY,
            temperature=0.5,
            max_output_tokens=8192,
            max_retries=3,
            request_timeout=60
        )
        
        self.evaluator_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=4096,
            temperature=0
        )
        
        self.worker2_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Fixed model name
            google_api_key=settings.GOOGLE_GEMINI_API_KEY,
            temperature=0.7,
            max_output_tokens=8192,
            max_retries=3,
            request_timeout=60
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
        print(f"[Orchestrator] Processing document {state['document_id']}")
        
        try:
            # Extract role and industry from introduction
            intro_prompt = f"""
            Analyze this introduction and extract the professional role and industry it's intended for:

            Introduction:
            {state['introduction']}

            Extract the specific job role/profession and the industry/sector.
            """

            role_industry_llm = self.orchestrator_llm.with_structured_output(RoleIndustry)
            role_industry = role_industry_llm.invoke([
                SystemMessage(content="You are an expert at analyzing professional documents."),
                HumanMessage(content=intro_prompt)
            ])
            
            # Extract topic-description pairs from main content
            pairs_prompt = f"""
            Identify all the distinct topic-description pairs in the content below. Each topic-description in the content below is a specific speaking scenario or situation that a {role_industry.role} in {role_industry.industry} would encounter.

            Content:
            {state['main_content']}

            Extract each distinct topic with its corresponding description VERBATIM.
            """

            orchestrator_output_llm = self.orchestrator_llm.with_structured_output(OrchestratorOutput)
            orchestrator_output = orchestrator_output_llm.invoke([
                SystemMessage(content=f"You are a subject matter expert for {role_industry.role} professionals in the {role_industry.industry} industry, analysing a report for every task that involves speaking/workplace conversations."),
                HumanMessage(content=pairs_prompt)
            ])
            
            # Update state
            state["role"] = role_industry.role
            state["industry"] = role_industry.industry
            state["topic_pairs"] = [
                {"topic": pair.topic, "description": pair.description}
                for pair in orchestrator_output.topic_pairs
            ]
            
            print(f"[Orchestrator] Found {len(state['topic_pairs'])} topic pairs for {state['role']} in {state['industry']}")
            
        except Exception as e:
            state["error"] = f"Orchestrator error: {str(e)}"
            print(f"[Orchestrator] Error: {str(e)}")
        
        return state
    
    def parallel_workers_node(self, state: WorkflowState) -> WorkflowState:
        """Run Worker 1 in parallel for each topic pair"""
        state["current_step"] = "Running parallel workers"
        print(f"[Parallel Workers] Processing {len(state['topic_pairs'])} topics in parallel")
        
        worker_outputs = []
        
        # Process each topic pair in parallel with proper concurrency
        # Reduced workers to prevent API rate limits
        max_workers = min(5, len(state["topic_pairs"]))  # Max 5 concurrent workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for idx, pair in enumerate(state["topic_pairs"]):
                future = executor.submit(
                    self._process_single_topic_with_retry,
                    pair,
                    state["role"],
                    state["industry"],
                    idx
                )
                future_to_index[future] = idx
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    worker_outputs.append(result)
                    print(f"[Worker 1] Completed topic {idx + 1}/{len(state['topic_pairs'])}: {result['course_name']}")
                except Exception as e:
                    print(f"[Worker 1] Failed topic {idx + 1}: {str(e)}")
                    worker_outputs.append({
                        "index": idx,
                        "error": str(e),
                        "course_name": "Error",
                        "course_description": "Error occurred during processing",
                        "lessons": []
                    })
        
        # Sort by index to maintain order
        worker_outputs.sort(key=lambda x: x["index"])
        state["worker_outputs"] = worker_outputs
        return state
    
    def _process_single_topic_with_retry(self, topic_pair: Dict[str, str], role: str, industry: str, index: int) -> Dict[str, Any]:
        """Process a single topic with retry logic"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Add delay between attempts to avoid rate limits
                if attempt > 0:
                    time.sleep(retry_delay * attempt)
                
                return self._process_single_topic(topic_pair, role, industry, index)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[Worker 1] Retry {attempt + 1} for topic {index} after error: {str(e)}")
                continue
    
    def _process_single_topic(self, topic_pair: Dict[str, str], role: str, industry: str, index: int) -> Dict[str, Any]:
        """Process a single topic-description pair"""
        prompt = f"""
        Your task is to analyse the topic-description pair and create a course that breaks down the topic-description pair into lessons. Each lesson zooms in on a specific moment within that bigger theme — a particular type of conversation, challenge, or task a {role} in the {industry} industry might face on the job. Lessons break the course down into realistic, manageable practice moments, so you can build your skills one situation at a time.

        Topic-description pair:
        Topic: {topic_pair['topic']}
        Description: {topic_pair['description']}

        Requirements:
        1. Analayse and leverage the information in the topic-description pair description to create lessons
        2. Ensure all lessons must be speaking/verbal communication focused
        3. Make the course name the topic of the topic-description pair.
        4. Make the course description the description of the topic-description pair.
        5. Make the lesson names concise and to the point.

        Analogy example (for understanding purposes ONLY):
        Imagine a course as a training program on "confident client communication." Each lesson is like attending a different training session within that program — one might focus on asking questions, another on explaining technical terms, another on handling misunderstandings. All are connected, but each teaches a different skill you need to succeed in that bigger goal.


        """

        structured_llm = self.worker1_llm.with_structured_output(CourseWithLessons)
        result = structured_llm.invoke([
            SystemMessage(content=f"You are a subject matter expert for building Business English speaking curriculums for {role} professionals in the {industry} industry."),
            HumanMessage(content=prompt)
        ])
        
        return {
            "index": index,
            "topic_pair": topic_pair,
            "course_name": result.course_name,
            "course_description": result.course_description,
            "lessons": [lesson.dict() for lesson in result.lessons]
        }
    
    def evaluator_node(self, state: WorkflowState) -> WorkflowState:
        """Evaluate worker outputs"""
        state["current_step"] = "Evaluating worker outputs"
        print(f"[Evaluator] Evaluating {len(state['worker_outputs'])} course outputs")
        
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
            eval_prompt = f"""
            Evaluate this course structure:

            Course: {output['course_name']}
            Course Description: {output['course_description']}
            Topic: {output['topic_pair']['topic']}
            Topic Description: {output['topic_pair']['description']}
            Lessons: {json.dumps(output['lessons'], indent=2)}

            Check if:
            1. All lessons are speaking/verbal communication related
            2. Output is properly structured
            3. The course name is the same as the topic of the topic-description pair.
            4. The course description is the same as the description of the topic-description pair.
            """

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
                    print(f"[Evaluator] Course '{output['course_name']}' failed: {eval_result.feedback}")
                    
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
        
        print(f"[Evaluator] {sum(1 for r in evaluation_results if r['passed'])} passed, {sum(1 for r in evaluation_results if not r['passed'])} failed")
        
        return state
    
    def should_retry(self, state: WorkflowState) -> Literal["retry", "continue"]:
        """Decide whether to retry or continue"""
        # Check if all evaluations passed
        all_passed = all(result["passed"] for result in state["evaluation_results"])
        
        # Maximum 3 retries
        if not all_passed and state.get("retry_count", 0) < 3:
            print(f"[Workflow] Retrying failed courses (attempt {state['retry_count'] + 1}/3)")
            # Provide feedback to workers for retry
            for i, output in enumerate(state["worker_outputs"]):
                for eval_result in state["evaluation_results"]:
                    if eval_result["index"] == output.get("index", i):
                        if not eval_result["passed"]:
                            output["feedback"] = eval_result["feedback"]
            return "retry"
        
        return "continue"
    
    def worker2_node(self, state: WorkflowState) -> WorkflowState:
        """Generate full lesson content in parallel"""
        state["current_step"] = "Generating full lesson content"
        passed_courses = [
            output for output in state["worker_outputs"]
            if not any(
                eval_result["index"] == output.get("index", -1) and not eval_result["passed"]
                for eval_result in state["evaluation_results"]
            ) and "error" not in output
        ]
        
        print(f"[Worker 2] Expanding {len(passed_courses)} courses with full lesson details")
        
        final_courses = []
        
        # Process courses in parallel
        max_workers = min(3, len(passed_courses))  # Max 3 concurrent workers for Worker 2
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_course = {}
            
            for output in passed_courses:
                future = executor.submit(
                    self._generate_full_lessons_with_retry,
                    output["course_name"],
                    output["lessons"],
                    state["role"],
                    state["industry"]
                )
                future_to_course[future] = output
            
            for future in as_completed(future_to_course):
                output = future_to_course[future]
                try:
                    full_lessons = future.result()
                    final_courses.append({
                        "course_name": output["course_name"],
                        "course_description": output["course_description"],
                        "topic_pair": output["topic_pair"],
                        "lessons": full_lessons
                    })
                    print(f"[Worker 2] Completed course: {output['course_name']}")
                except Exception as e:
                    print(f"[Worker 2] Failed to expand course '{output['course_name']}': {str(e)}")
                    state["error"] = f"Worker 2 error: {str(e)}"
        
        state["final_courses"] = final_courses
        return state
    
    def _generate_full_lessons_with_retry(self, course_name: str, lessons: List[Dict], role: str, industry: str) -> List[Dict]:
        """Generate full lessons with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(retry_delay * attempt)
                return self._generate_full_lessons(course_name, lessons, role, industry)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[Worker 2] Retry {attempt + 1} for course '{course_name}' after error: {str(e)}")
                continue
    
    def _generate_full_lessons(self, course_name: str, lessons: List[Dict], role: str, industry: str) -> List[Dict]:
        """Generate full content for each lesson"""
        full_lessons = []
        
        for lesson in lessons:
            prompt = f"""
            You are creating a complete lesson for {role} professionals in {industry}.

            Course: {course_name}
            Lesson {lesson['lesson_number']}: {lesson['lesson_title']}
            Introduction: {lesson['lesson_introduction']}

            Generate the complete lesson content including:
            1. Skill Aims (4-5 specific communication skills)
            2. Language Learning Aims (3-4 categories with 3 example phrases each)
            3. Lesson Summary (4 key takeaways)

            Focus on practical verbal communication skills and real phrases professionals would use.
            """

            try:
                structured_llm = self.worker2_llm.with_structured_output(FullLesson)
                full_lesson = structured_llm.invoke([
                    SystemMessage(content=f"You are a subject matter expert for enhancing English speaking lessons for {role} professionals in the {industry} industry."),
                    HumanMessage(content=prompt)
                ])
                
                full_lessons.append(full_lesson.dict())
                
            except Exception as e:
                print(f"[Worker 2] Error generating full lesson: {str(e)}")
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
        print(f"[Aggregator] Final output: {len(state['final_courses'])} courses generated")
        
        return state
    
    def process_document(self, document_id: int, structured_content: Dict[str, str]) -> Dict[str, Any]:
        """Main entry point to process a document"""
        print(f"\n{'='*60}")
        print(f"Starting course generation for document {document_id}")
        print(f"{'='*60}\n")
        
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
        
        print(f"\n{'='*60}")
        print(f"Course generation completed for document {document_id}")
        print(f"Generated {len(result['final_courses'])} courses")
        print(f"{'='*60}\n")
        
        return {
            "final_courses": result["final_courses"],
            "role": result["role"],
            "industry": result["industry"],
            "error": result["error"],
            "current_step": result["current_step"]
        }