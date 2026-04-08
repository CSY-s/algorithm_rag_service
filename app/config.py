"""项目配置模块。

统一从 .env 读取数据库、模型、MCP 等配置，其他模块只需要导入 settings。"""


import os
from dataclasses import dataclass

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))


@dataclass
class Settings:
    """系统运行时使用的全部配置项。"""
    deepseek_api_key: str = os.getenv('DEEPSEEK_API_KEY', '')
    deepseek_model: str = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    deepseek_base_url: str = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1/chat/completions')
    planner_use_llm: bool = os.getenv('PLANNER_USE_LLM', '1') == '1'
    planner_model: str = os.getenv('PLANNER_MODEL', os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'))
    mcp_enabled: bool = os.getenv('MCP_ENABLED', '0') == '1'
    mcp_base_url: str = os.getenv('MCP_BASE_URL', '')
    mcp_timeout: int = int(os.getenv('MCP_TIMEOUT', '30'))
    mysql_host: str = os.getenv('MYSQL_HOST', '127.0.0.1')
    mysql_port: int = int(os.getenv('MYSQL_PORT', '3306'))
    mysql_user: str = os.getenv('MYSQL_USER', 'root')
    mysql_password: str = os.getenv('MYSQL_PASSWORD', '')
    mysql_db: str = os.getenv('MYSQL_DB', 'tutor')
    store_path: str = os.getenv('STORE_PATH', os.path.join(BASE_DIR, 'data', 'rag_store.json'))
    rag_top_k: int = int(os.getenv('RAG_TOP_K', '5'))


settings = Settings()
