"""
Django视图：RAG智能助教接口

放置位置: tutor_backend/tutor_backend/tutor_backend/views/rag.py

功能:
1. 调用RAG服务
2. 处理用户认证
3. 记录使用日志
"""

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import requests
import json
import logging

logger = logging.getLogger(__name__)

# RAG服务地址（根据实际部署修改）
RAG_SERVICE_URL = "http://localhost:8000"


@csrf_exempt
@require_http_methods(["POST"])
def ask_rag(request):
    """
    智能助教问答接口
    
    请求体:
        {
            "question": "这个算法的时间复杂度是多少?",
            "logical_content_id": 2,  # logicalContent ID
            "session_id": "可选，前端生成的会话ID"
        }
    
    返回:
        {
            "success": true,
            "data": {
                "answer": "...",
                "references": [...],
                "session_id": "..."
            }
        }
    """
    try:
        # 解析请求
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        logical_content_id = data.get('logical_content_id')
        session_id = data.get('session_id')
        
        # 验证参数
        if not question:
            return JsonResponse({
                'success': False,
                'error': '问题不能为空'
            }, status=400)
        
        if not logical_content_id:
            return JsonResponse({
                'success': False,
                'error': '缺少logical_content_id'
            }, status=400)
        
        # 生成会话ID（如果前端没提供）
        if not session_id:
            user_id = request.user.id if request.user.is_authenticated else 'anonymous'
            import time
            session_id = f"user_{user_id}_{int(time.time())}"
        
        # 构造RAG服务请求
        rag_request = {
            'question': question,
            'algorithm_id': logical_content_id,  # 直接使用logicalContent ID
            'session_id': session_id,
            'user_id': request.user.id if request.user.is_authenticated else None,
            'top_k': 5,
            'retrieval_mode': 'hybrid',
            'enable_tools': True,
            'enable_memory': True
        }
        
        # 记录日志
        logger.info(f"RAG请求: user={rag_request['user_id']}, "
                   f"algorithm_id={logical_content_id}, "
                   f"question={question[:50]}")
        
        # 调用RAG服务
        response = requests.post(
            f"{RAG_SERVICE_URL}/api/rag/context_ask",
            json=rag_request,
            timeout=30
        )
        
        # 处理响应
        if response.status_code == 200:
            result = response.json()
            
            # 记录成功日志
            logger.info(f"RAG响应成功: session={session_id}")
            
            return JsonResponse({
                'success': True,
                'data': result['data']
            })
        else:
            # 记录错误
            logger.error(f"RAG服务错误: status={response.status_code}, "
                        f"response={response.text[:200]}")
            
            return JsonResponse({
                'success': False,
                'error': 'RAG服务暂时不可用',
                'details': response.text if response.status_code != 500 else None
            }, status=500)
            
    except requests.exceptions.Timeout:
        logger.error("RAG服务超时")
        return JsonResponse({
            'success': False,
            'error': '请求超时，请稍后重试'
        }, status=504)
        
    except requests.exceptions.ConnectionError:
        logger.error("无法连接到RAG服务")
        return JsonResponse({
            'success': False,
            'error': 'RAG服务连接失败'
        }, status=503)
        
    except Exception as e:
        logger.exception("RAG请求异常")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_session_history(request, session_id):
    """
    获取会话历史
    
    URL: /api/rag/session/<session_id>/
    """
    try:
        limit = int(request.GET.get('limit', 10))
        
        response = requests.get(
            f"{RAG_SERVICE_URL}/api/rag/session/{session_id}",
            params={'limit': limit},
            timeout=10
        )
        
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to get session history'
            }, status=response.status_code)
            
    except Exception as e:
        logger.exception("获取会话历史失败")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["DELETE"])
def clear_session(request, session_id):
    """
    清空会话（新对话）
    
    URL: /api/rag/session/<session_id>/
    """
    try:
        response = requests.delete(
            f"{RAG_SERVICE_URL}/api/rag/session/{session_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            return JsonResponse({
                'success': True,
                'message': '会话已清空'
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to clear session'
            }, status=response.status_code)
            
    except Exception as e:
        logger.exception("清空会话失败")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def rag_status(request):
    """
    检查RAG服务状态
    
    URL: /api/rag/status/
    """
    try:
        response = requests.get(
            f"{RAG_SERVICE_URL}/health",
            timeout=5
        )
        
        if response.status_code == 200:
            return JsonResponse({
                'success': True,
                'status': 'online',
                'service_url': RAG_SERVICE_URL
            })
        else:
            return JsonResponse({
                'success': False,
                'status': 'offline'
            })
            
    except:
        return JsonResponse({
            'success': False,
            'status': 'offline',
            'error': 'Cannot connect to RAG service'
        })
