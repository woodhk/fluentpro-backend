import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from supabase import create_client, Client
from django.conf import settings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class SupabaseService:
    def __init__(self):
        """Initialize Supabase client with credentials from settings"""
        self.url = settings.SUPABASE_URL
        self.key = settings.SUPABASE_SERVICE_KEY
        self.client: Client = create_client(self.url, self.key)
        logger.info("Supabase client initialized")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def upload_course(self, course_data: Dict[str, Any]) -> Dict[str, Any]:
        """Upload a single course with its lessons to Supabase"""
        try:
            logger.debug(f"Uploading course: {course_data.get('course_name', 'Unknown')}")
            
            # Prepare course data
            course_record = {
                'course_name': course_data['course_name'],
                'course_description': course_data.get('course_description', ''),  # Add this
                'role': course_data['role'],
                'industry': course_data['industry'],
                'document_title': course_data.get('document_title', ''),
                'topic_description': course_data.get('topic_description', {}),
                'django_course_id': course_data.get('django_course_id')
            }
            
            # Insert course
            course_result = self.client.table('courses').insert(course_record).execute()
            
            if not course_result.data:
                raise Exception("Failed to insert course")
            
            course_id = course_result.data[0]['id']
            logger.info(f"Inserted course with ID: {course_id}")
            
            # Insert lessons
            lessons_to_insert = []
            for lesson in course_data.get('lessons', []):
                lesson_record = {
                    'course_id': course_id,
                    'lesson_number': lesson['lesson_number'],
                    'lesson_title': lesson['lesson_title'],
                    'lesson_introduction': lesson.get('lesson_introduction', ''),
                    'skill_aims': lesson.get('skill_aims', []),
                    'language_learning_aims': lesson.get('language_learning_aims', {}),
                    'lesson_summary': lesson.get('lesson_summary', []),
                    'is_bonus': lesson.get('is_bonus', False),
                    'django_lesson_id': lesson.get('django_lesson_id')
                }
                lessons_to_insert.append(lesson_record)
            
            if lessons_to_insert:
                lessons_result = self.client.table('lessons').insert(lessons_to_insert).execute()
                logger.info(f"Inserted {len(lessons_to_insert)} lessons for course {course_id}")
            
            return {
                'success': True,
                'course_id': course_id,
                'lessons_count': len(lessons_to_insert)
            }
            
        except Exception as e:
            logger.error(f"Error uploading course: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def upload_courses_batch(self, courses_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upload multiple courses to Supabase"""
        results = {
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        for course in courses_data:
            result = self.upload_course(course)
            if result['success']:
                results['successful'] += 1
            else:
                results['failed'] += 1
                results['errors'].append({
                    'course_name': course.get('course_name', 'Unknown'),
                    'error': result.get('error', 'Unknown error')
                })
        
        logger.info(f"Batch upload complete: {results['successful']} successful, {results['failed']} failed")
        return results
    
    def get_all_courses(self) -> List[Dict[str, Any]]:
        """Retrieve all courses from Supabase"""
        try:
            response = self.client.table('courses').select('*').order('created_at', desc=True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching courses: {str(e)}")
            return []
    
    def get_course_with_lessons(self, course_id: int) -> Optional[Dict[str, Any]]:
        """Get a single course with all its lessons"""
        try:
            # Get course
            course_response = self.client.table('courses').select('*').eq('id', course_id).single().execute()
            
            if not course_response.data:
                return None
            
            course = course_response.data
            
            # Get lessons
            lessons_response = self.client.table('lessons').select('*').eq('course_id', course_id).order('lesson_number').execute()
            
            course['lessons'] = lessons_response.data
            return course
            
        except Exception as e:
            logger.error(f"Error fetching course {course_id}: {str(e)}")
            return None
    
    def update_course(self, course_id: int, updates: Dict[str, Any], user: str = 'user') -> bool:
        """Update course fields and log the change"""
        try:
            # Get current values for audit log
            current = self.client.table('courses').select('*').eq('id', course_id).single().execute()
            
            if not current.data:
                return False
            
            # Update course
            self.client.table('courses').update(updates).eq('id', course_id).execute()
            
            # Log changes
            for field, new_value in updates.items():
                old_value = current.data.get(field)
                if old_value != new_value:
                    self._log_edit(
                        course_id=course_id,
                        field_name=f'course.{field}',
                        old_value=old_value,
                        new_value=new_value,
                        edited_by=user
                    )
            
            logger.info(f"Updated course {course_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating course {course_id}: {str(e)}")
            return False
    
    def update_lesson(self, lesson_id: int, updates: Dict[str, Any], user: str = 'user') -> bool:
        """Update lesson fields and log the change"""
        try:
            # Get current values for audit log
            current = self.client.table('lessons').select('*, course_id').eq('id', lesson_id).single().execute()
            
            if not current.data:
                return False
            
            # Update lesson
            self.client.table('lessons').update(updates).eq('id', lesson_id).execute()
            
            # Log changes
            for field, new_value in updates.items():
                old_value = current.data.get(field)
                if old_value != new_value:
                    self._log_edit(
                        course_id=current.data['course_id'],
                        lesson_id=lesson_id,
                        field_name=f'lesson.{field}',
                        old_value=old_value,
                        new_value=new_value,
                        edited_by=user
                    )
            
            logger.info(f"Updated lesson {lesson_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating lesson {lesson_id}: {str(e)}")
            return False
    
    def _log_edit(self, course_id: int, field_name: str, old_value: Any, 
                  new_value: Any, edited_by: str, lesson_id: Optional[int] = None):
        """Log an edit to the audit table"""
        try:
            log_entry = {
                'course_id': course_id,
                'field_name': field_name,
                'old_value': old_value if isinstance(old_value, (dict, list)) else str(old_value),
                'new_value': new_value if isinstance(new_value, (dict, list)) else str(new_value),
                'edited_by': edited_by
            }
            
            if lesson_id:
                log_entry['lesson_id'] = lesson_id
            
            self.client.table('course_edits').insert(log_entry).execute()
            
        except Exception as e:
            logger.error(f"Error logging edit: {str(e)}")
    
    def get_edit_history(self, course_id: int) -> List[Dict[str, Any]]:
        """Get edit history for a course"""
        try:
            response = self.client.table('course_edits').select('*').eq('course_id', course_id).order('edited_at', desc=True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching edit history: {str(e)}")
            return []
    
    def check_duplicate_course(self, django_course_id: int) -> bool:
        """Check if a course with this Django ID already exists"""
        try:
            response = self.client.table('courses').select('id').eq('django_course_id', django_course_id).execute()
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Error checking duplicate: {str(e)}")
            return False