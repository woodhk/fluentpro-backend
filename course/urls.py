from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('document/<int:doc_id>/', views.document_detail, name='document_detail'),
    path('courses/', views.courses_list, name='courses_list'),
    path('course/<int:course_id>/', views.course_detail, name='course_detail'),
    path('generation-status/', views.generation_status, name='generation_status'),
    
    # API endpoints
    path('api/check-new-docs/', views.check_new_docs, name='check_new_docs'),
    path('api/process-rag/', views.process_documents_rag, name='process_documents_rag'),
    path('api/generate-courses/', views.generate_courses, name='generate_courses'),
    path('api/export-courses/', views.export_courses, name='export_courses'),
    path('api/clear-docs/', views.clear_docs, name='clear_docs'),
    path('api/status/', views.get_status, name='get_status'),
    path('api/documents/', views.get_documents, name='get_documents'),
    path('api/generation-progress/<int:document_id>/', views.get_generation_progress, name='get_generation_progress'),
]