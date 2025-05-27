#!/usr/bin/env python
"""
Test and debug Supabase integration
Run this from your Django project root: python test_supabase_integration.py
"""

import os
import sys
import django
import logging
from datetime import datetime

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fluentpro.settings')
django.setup()

from django.conf import settings
from course.supabase_service import SupabaseService
from course.models import GeneratedCourse, GeneratedLesson

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_connection():
    """Test basic Supabase connection"""
    logger.info("Testing Supabase connection...")
    try:
        service = SupabaseService()
        logger.info(f"✅ Connected to Supabase at: {settings.SUPABASE_URL}")
        return service
    except Exception as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        logger.error("Check your SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
        return None

def test_table_access(service):
    """Test access to tables"""
    logger.info("\nTesting table access...")
    
    tables = ['courses', 'lessons', 'course_edits']
    for table in tables:
        try:
            result = service.client.table(table).select('id').limit(1).execute()
            logger.info(f"✅ Can access '{table}' table")
        except Exception as e:
            logger.error(f"❌ Cannot access '{table}' table: {str(e)}")
            return False
    return True

def test_course_upload(service):
    """Test uploading a sample course"""
    logger.info("\nTesting course upload...")
    
    test_course = {
        'course_name': 'Test Course - ' + datetime.now().strftime('%Y%m%d_%H%M%S'),
        'role': 'Test Engineer',
        'industry': 'Software Testing',
        'document_title': 'Test Document',
        'topic_description': {'topic': 'Testing', 'description': 'Test description'},
        'django_course_id': 999999,  # Use a high number unlikely to conflict
        'lessons': [
            {
                'lesson_number': 1,
                'lesson_title': 'Introduction to Testing',
                'lesson_introduction': 'This is a test lesson',
                'skill_aims': ['Test skill 1', 'Test skill 2'],
                'language_learning_aims': {
                    'Greetings': ['Hello', 'Hi', 'Hey'],
                    'Farewells': ['Goodbye', 'See you', 'Bye']
                },
                'lesson_summary': ['Summary point 1', 'Summary point 2'],
                'is_bonus': False,
                'django_lesson_id': 999999
            }
        ]
    }
    
    try:
        result = service.upload_course(test_course)
        if result['success']:
            logger.info(f"✅ Successfully uploaded test course with ID: {result['course_id']}")
            return result['course_id']
        else:
            logger.error(f"❌ Failed to upload course: {result['error']}")
            return None
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}")
        return None

def test_course_retrieval(service, course_id):
    """Test retrieving a course"""
    logger.info(f"\nTesting course retrieval for ID: {course_id}...")
    
    try:
        course = service.get_course_with_lessons(course_id)
        if course:
            logger.info(f"✅ Retrieved course: {course['course_name']}")
            logger.info(f"   - Lessons: {len(course.get('lessons', []))}")
            return True
        else:
            logger.error("❌ Course not found")
            return False
    except Exception as e:
        logger.error(f"❌ Retrieval error: {str(e)}")
        return False

def test_course_update(service, course_id):
    """Test updating a course"""
    logger.info(f"\nTesting course update for ID: {course_id}...")
    
    try:
        success = service.update_course(
            course_id,
            {'course_name': 'Updated Test Course'},
            user='test_script'
        )
        if success:
            logger.info("✅ Successfully updated course")
            return True
        else:
            logger.error("❌ Failed to update course")
            return False
    except Exception as e:
        logger.error(f"❌ Update error: {str(e)}")
        return False

def test_django_to_supabase_sync(service):
    """Test syncing Django courses to Supabase"""
    logger.info("\nTesting Django to Supabase sync...")
    
    try:
        # Get first unsynced course from Django
        django_courses = GeneratedCourse.objects.all()[:1]
        
        if not django_courses:
            logger.warning("⚠️  No courses found in Django database")
            return
        
        course = django_courses[0]
        logger.info(f"Found Django course: {course.course_name}")
        
        # Check if already in Supabase
        if service.check_duplicate_course(course.id):
            logger.info("ℹ️  Course already exists in Supabase")
            return
        
        # Prepare course data
        course_data = {
            'course_name': course.course_name,
            'role': course.role,
            'industry': course.industry,
            'document_title': course.document.title,
            'topic_description': course.get_topic_description(),
            'django_course_id': course.id,
            'lessons': []
        }
        
        # Add lessons
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
        
        # Upload to Supabase
        result = service.upload_course(course_data)
        if result['success']:
            logger.info(f"✅ Successfully synced Django course to Supabase")
        else:
            logger.error(f"❌ Failed to sync: {result['error']}")
            
    except Exception as e:
        logger.error(f"❌ Sync error: {str(e)}")

def cleanup_test_data(service):
    """Clean up test data"""
    logger.info("\nCleaning up test data...")
    
    try:
        # Delete test courses
        result = service.client.table('courses').delete().eq('django_course_id', 999999).execute()
        logger.info("✅ Cleaned up test data")
    except Exception as e:
        logger.warning(f"⚠️  Cleanup warning: {str(e)}")

def main():
    """Run all tests"""
    logger.info("🚀 Starting Supabase Integration Tests")
    logger.info("=" * 50)
    
    # Test 1: Connection
    service = test_connection()
    if not service:
        logger.error("\n❌ Connection test failed. Exiting.")
        return
    
    # Test 2: Table access
    if not test_table_access(service):
        logger.error("\n❌ Table access test failed. Check your database schema.")
        return
    
    # Test 3: Course upload
    course_id = test_course_upload(service)
    if course_id:
        # Test 4: Course retrieval
        test_course_retrieval(service, course_id)
        
        # Test 5: Course update
        test_course_update(service, course_id)
    
    # Test 6: Django sync
    test_django_to_supabase_sync(service)
    
    # Cleanup
    cleanup_test_data(service)
    
    logger.info("\n" + "=" * 50)
    logger.info("✅ All tests completed!")
    logger.info("\nNext steps:")
    logger.info("1. Check your Supabase dashboard to see the uploaded data")
    logger.info("2. Visit /supabase/ in your Django app to see the courses")
    logger.info("3. Try editing a course at /supabase/edit/<course_id>/")

if __name__ == "__main__":
    main()