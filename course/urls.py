from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/check-new-docs/', views.check_new_docs, name='check_new_docs'),
    path('api/status/', views.get_status, name='get_status'),
    path('api/documents/', views.get_documents, name='get_documents'),
]
