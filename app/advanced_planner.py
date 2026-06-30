"""高级规划器模块

支持多步规划、动态调整、规划验证"""

from typing import Any, Dict, List, Optional
import json
from .deepseek import deepseek_chat
from .config import settings


def multi_step_plan(question: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """多步规划：分解复杂问题为多个子问题
    
    参数:
        question: 用户问题
        context: 上下文信息（可选）
    
    返回:
        {
            'main_goal': '主目标',
            'sub_questions': ['子问题1', '子问题2', ...],
            'execution_plan': [
                {'step': 1, 'action': 'tool', 'target': 'knowledge_graph', 'reason': '...'},
                {'step': 2, 'action': 'rag', 'query': '...', 'reason': '...'},
                ...
            ],
            'dependencies': {'step2': ['step1'], 'step3': ['step1', 'step2']},
            'estimated_complexity': 'low/medium/high'
        }
    """
    if not settings.planner_use_llm or not settings.deepseek_api_key:
        return _simple_decomposition(question)
    
    prompt = f"""
你是高级规划器，负责分解复杂问题并制定执行计划。

用户问题：{question}

任务：
1. 识别主目标
2. 分解为2-4个子问题（如果是简单问题，可以只有1个）
3. 为每个子问题制定执行步骤
4. 标注步骤间的依赖关系

可用动作类型：
- tool: 调用工具（knowledge_graph, complexity_extract, external_search）
- rag: 检索知识库
- reasoning: 推理分析
- synthesis: 综合多个结果

返回JSON格式（仅JSON对象，不要其他文字）：
{{
  "main_goal": "主目标描述",
  "sub_questions": ["子问题1", "子问题2"],
  "execution_plan": [
    {{"step": 1, "action": "tool", "target": "knowledge_graph", "reason": "需要了解算法关系"}},
    {{"step": 2, "action": "rag", "query": "子问题1", "reason": "检索相关资料"}}
  ],
  "dependencies": {{"2": [1]}},
  "estimated_complexity": "medium"
}}
"""
    
    try:
        raw = deepseek_chat(
            [
                {'role': 'system', 'content': '你是高级规划器，只输出JSON对象。'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        # 提取JSON
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            obj = json.loads(raw[start:end+1])
            obj['planner_type'] = 'multi_step_llm'
            return obj
    except Exception as e:
        print(f"多步规划失败: {e}")
    
    return _simple_decomposition(question)


def _simple_decomposition(question: str) -> Dict[str, Any]:
    """简单的问题分解（启发式）"""
    
    # 检测是否是对比问题
    if any(kw in question for kw in ['对比', '区别', '比较', '异同', 'vs', '和']):
        return {
            'main_goal': '对比分析',
            'sub_questions': [
                '第一个算法的特点',
                '第二个算法的特点',
                '两者的异同点'
            ],
            'execution_plan': [
                {'step': 1, 'action': 'rag', 'query': '算法A', 'reason': '检索第一个算法'},
                {'step': 2, 'action': 'rag', 'query': '算法B', 'reason': '检索第二个算法'},
                {'step': 3, 'action': 'reasoning', 'reason': '综合对比'}
            ],
            'dependencies': {'3': [1, 2]},
            'estimated_complexity': 'medium',
            'planner_type': 'heuristic'
        }
    
    # 检测是否是实现问题
    elif any(kw in question for kw in ['实现', '代码', '编写', '怎么写']):
        return {
            'main_goal': '代码实现',
            'sub_questions': [
                '算法原理',
                '实现步骤',
                '代码示例'
            ],
            'execution_plan': [
                {'step': 1, 'action': 'rag', 'query': '算法步骤', 'reason': '了解原理'},
                {'step': 2, 'action': 'tool', 'target': 'code_search', 'reason': '查找代码'},
                {'step': 3, 'action': 'synthesis', 'reason': '生成完整答案'}
            ],
            'dependencies': {'3': [1, 2]},
            'estimated_complexity': 'high',
            'planner_type': 'heuristic'
        }
    
    # 简单问题
    else:
        return {
            'main_goal': '回答问题',
            'sub_questions': [question],
            'execution_plan': [
                {'step': 1, 'action': 'rag', 'query': question, 'reason': '直接检索'}
            ],
            'dependencies': {},
            'estimated_complexity': 'low',
            'planner_type': 'heuristic'
        }


def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """验证规划的合理性
    
    返回:
        {
            'is_valid': True/False,
            'issues': ['问题1', '问题2', ...],
            'suggestions': ['建议1', '建议2', ...]
        }
    """
    issues = []
    suggestions = []
    
    # 检查必需字段
    required_fields = ['main_goal', 'execution_plan']
    for field in required_fields:
        if field not in plan:
            issues.append(f"缺少必需字段: {field}")
    
    # 检查执行步骤
    if 'execution_plan' in plan:
        steps = plan['execution_plan']
        
        if len(steps) == 0:
            issues.append("执行计划为空")
        
        if len(steps) > 10:
            suggestions.append("执行步骤过多，可能导致效率低下")
        
        # 检查步骤编号连续性
        step_nums = [s.get('step', 0) for s in steps]
        if step_nums != list(range(1, len(steps) + 1)):
            issues.append("步骤编号不连续")
    
    # 检查依赖关系
    if 'dependencies' in plan:
        deps = plan['dependencies']
        step_nums = {s.get('step') for s in plan.get('execution_plan', [])}
        
        for step, depends_on in deps.items():
            step_num = int(step) if isinstance(step, str) else step
            
            if step_num not in step_nums:
                issues.append(f"依赖关系引用了不存在的步骤: {step_num}")
            
            for dep in depends_on:
                if dep not in step_nums:
                    issues.append(f"步骤{step_num}依赖不存在的步骤: {dep}")
                if dep >= step_num:
                    issues.append(f"步骤{step_num}依赖后续步骤{dep}（循环依赖）")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'suggestions': suggestions
    }


def adjust_plan_dynamically(
    plan: Dict[str, Any],
    execution_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """根据执行结果动态调整计划
    
    参数:
        plan: 原始计划
        execution_results: 已执行步骤的结果
    
    返回:
        调整后的计划
    """
    adjusted_plan = plan.copy()
    
    # 检查已执行步骤的成功率
    success_count = sum(1 for r in execution_results if r.get('success'))
    total_count = len(execution_results)
    
    if total_count > 0:
        success_rate = success_count / total_count
        
        # 如果成功率低，添加重试或备用步骤
        if success_rate < 0.5:
            adjusted_plan['adjustments'] = [
                {
                    'reason': f'前{total_count}步成功率仅{success_rate:.0%}，添加备用策略',
                    'action': 'add_fallback',
                    'fallback_plan': [
                        {'step': len(plan['execution_plan']) + 1, 'action': 'rag', 'query': '备用检索'}
                    ]
                }
            ]
    
    return adjusted_plan
