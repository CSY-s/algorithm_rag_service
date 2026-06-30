"""思维链可视化模块

将推理过程结构化输出，便于前端展示"""

from typing import Any, Dict, List
import re


def parse_reasoning_chain(answer: str) -> Dict[str, Any]:
    """从答案中解析思维链步骤
    
    参数:
        answer: LLM生成的答案（包含推理过程）
    
    返回:
        {
            'steps': [
                {'step_num': 1, 'type': 'analysis', 'content': '...'},
                {'step_num': 2, 'type': 'reasoning', 'content': '...'},
                {'step_num': 3, 'type': 'conclusion', 'content': '...'}
            ],
            'summary': '最终答案摘要',
            'has_chain': True/False
        }
    """
    if not answer:
        return {'steps': [], 'summary': '', 'has_chain': False}
    
    steps = []
    
    # 模式1: 数字序号 (1. 2. 3.)
    pattern1 = r'(\d+)\.\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)'
    matches1 = re.findall(pattern1, answer)
    
    # 模式2: 关键词标记 (首先, 其次, 然后, 最后)
    keywords = ['首先', '其次', '然后', '接着', '最后', '综上']
    
    # 模式3: 步骤标记 (步骤1, 步骤2)
    pattern3 = r'步骤\s*(\d+)[：:]\s*([^\n]+(?:\n(?!步骤)[^\n]+)*)'
    matches3 = re.findall(pattern3, answer)
    
    if matches1 and len(matches1) >= 2:
        # 使用数字序号
        for i, (num, content) in enumerate(matches1, 1):
            step_type = _classify_step_type(content, i, len(matches1))
            steps.append({
                'step_num': i,
                'type': step_type,
                'content': content.strip(),
                'marker': f'{num}.'
            })
    
    elif matches3:
        # 使用步骤标记
        for i, (num, content) in enumerate(matches3, 1):
            step_type = _classify_step_type(content, i, len(matches3))
            steps.append({
                'step_num': i,
                'type': step_type,
                'content': content.strip(),
                'marker': f'步骤{num}'
            })
    
    else:
        # 使用关键词分段
        segments = []
        current_segment = []
        current_keyword = None
        
        for line in answer.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # 检查是否包含关键词
            found_keyword = None
            for kw in keywords:
                if line.startswith(kw):
                    found_keyword = kw
                    break
            
            if found_keyword:
                if current_segment:
                    segments.append((current_keyword, '\n'.join(current_segment)))
                current_segment = [line]
                current_keyword = found_keyword
            else:
                current_segment.append(line)
        
        if current_segment:
            segments.append((current_keyword, '\n'.join(current_segment)))
        
        for i, (keyword, content) in enumerate(segments, 1):
            if keyword:
                step_type = _classify_step_type(content, i, len(segments))
                steps.append({
                    'step_num': i,
                    'type': step_type,
                    'content': content.strip(),
                    'marker': keyword
                })
    
    # 提取摘要（最后一段或包含"综上"的段落）
    summary = ''
    if steps:
        last_step = steps[-1]['content']
        if len(last_step) < 300:
            summary = last_step
        else:
            # 查找包含"综上"、"总结"、"因此"的段落
            for para in answer.split('\n\n'):
                if any(kw in para for kw in ['综上', '总结', '因此', '所以']):
                    summary = para.strip()
                    break
    
    if not summary and len(answer) < 300:
        summary = answer
    elif not summary:
        summary = answer[:200] + '...'
    
    return {
        'steps': steps,
        'summary': summary,
        'has_chain': len(steps) >= 2
    }


def _classify_step_type(content: str, step_num: int, total_steps: int) -> str:
    """分类步骤类型
    
    返回:
        'analysis' - 分析依据
        'reasoning' - 推理过程
        'conclusion' - 结论
        'example' - 举例说明
    """
    content_lower = content.lower()
    
    # 关键词匹配
    if any(kw in content_lower for kw in ['根据', '依据', '基于', '从']):
        return 'analysis'
    elif any(kw in content_lower for kw in ['例如', '比如', '举例']):
        return 'example'
    elif step_num == total_steps or any(kw in content_lower for kw in ['因此', '所以', '综上', '总结']):
        return 'conclusion'
    else:
        return 'reasoning'


def create_chain_graph(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """创建思维链图结构（供前端可视化）
    
    返回:
        {
            'nodes': [
                {'id': 'step1', 'label': '分析', 'content': '...', 'type': 'analysis'},
                ...
            ],
            'edges': [
                {'from': 'step1', 'to': 'step2'},
                ...
            ]
        }
    """
    nodes = []
    edges = []
    
    for i, step in enumerate(steps):
        node_id = f"step{i+1}"
        nodes.append({
            'id': node_id,
            'label': step.get('marker', f"步骤{i+1}"),
            'content': step['content'],
            'type': step['type'],
            'step_num': step['step_num']
        })
        
        # 添加边（连接到下一步）
        if i < len(steps) - 1:
            edges.append({
                'from': node_id,
                'to': f"step{i+2}"
            })
    
    return {
        'nodes': nodes,
        'edges': edges
    }


def extract_references_from_chain(answer: str) -> List[Dict[str, Any]]:
    """从答案中提取引用标记
    
    提取形如 [chunk_id=123] 的引用
    
    返回:
        [
            {'chunk_id': 123, 'position': 50, 'context': '...'},
            ...
        ]
    """
    pattern = r'\[chunk_id=(\d+)\]'
    matches = re.finditer(pattern, answer)
    
    references = []
    for match in matches:
        chunk_id = int(match.group(1))
        position = match.start()
        
        # 提取周围上下文（前后50个字符）
        start = max(0, position - 50)
        end = min(len(answer), position + 100)
        context = answer[start:end].replace('\n', ' ').strip()
        
        references.append({
            'chunk_id': chunk_id,
            'position': position,
            'context': context
        })
    
    return references
