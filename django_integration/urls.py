"""
Django URL配置

添加到: tutor_backend/tutor_backend/tutor_backend/urls.py
"""

from django.urls import path
from .views import rag  # 假设rag.py在views目录下

# RAG相关路由
urlpatterns = [
    # ... 现有路由
    
    # RAG智能助教
    path('api/rag/ask/', rag.ask_rag, name='rag_ask'),
    path('api/rag/session/<str:session_id>/', rag.get_session_history, name='rag_session'),
    path('api/rag/session/<str:session_id>/clear/', rag.clear_session, name='rag_clear'),
    path('api/rag/status/', rag.rag_status, name='rag_status'),
]
