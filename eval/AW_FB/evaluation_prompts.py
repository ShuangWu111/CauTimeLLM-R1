"""
评估模型推理过程的prompt模板
"""


def get_readability_prompt(reasoning_text):
    """可读性评估prompt"""
    prompt = f"""You are an expert evaluator assessing the readability of AI-generated reasoning text.

Please evaluate the following reasoning text on a scale of 0-3 based on these criteria:
- Text clarity and coherence
- Grammar and language quality
- Logical flow and organization
- Ease of understanding

Scoring guidelines:
0: Completely unreadable, incoherent
1: Poor readability, frequent errors, hard to follow
2: Acceptable readability, some errors, generally understandable
3: Excellent readability, clear, well-organized, and easy to understand

Reasoning text to evaluate:
{reasoning_text}

Please provide:
1. A score from 0 to 3
2. A brief explanation (1-2 sentences)

Format your response exactly as:
Score: 0-3
Explanation: your explanation
"""
    return prompt


def get_logical_consistency_prompt(reasoning_text, question, label):
    """逻辑一致性评估prompt"""
    prompt = f"""You are an expert evaluator assessing the logical consistency of AI-generated reasoning text.

Question: {question}
Label: {label}

Please evaluate the following reasoning text on a scale of 0-3 based on these criteria:
- Presence of logical errors or contradictions
- Coherence and rigor of the reasoning chain
- Soundness and validity of conclusions drawn

Scoring guidelines:
0: Completely illogical, contains major contradictions, reasoning chain is invalid
1: Poor logical consistency, several errors or gaps in reasoning
2: Acceptable logical reasoning, minor inconsistencies, generally sound
3: Excellent logical consistency, no logical errors, reasoning chain is rigorous and well-structured

Reasoning text to evaluate:
{reasoning_text}

Please provide:
1. A score from 0 to 3
2. A brief explanation (1-2 sentences)

Format your response exactly as:
Score: 0-3
Explanation: your explanation"""
    
    return prompt


def get_comprehensiveness_prompt(reasoning_text, question):
    """推理全面性评估prompt"""
    prompt = f"""You are an expert evaluator assessing the comprehensiveness of AI-generated reasoning.

Question: {question}

Please evaluate the following reasoning text on a scale of 0-3 based on these criteria:
- Coverage of each piece of provided user information and data
- Depth and completeness of analysis
- Consideration of multiple relevant aspects
- Thoroughness in addressing the question

Scoring guidelines:
0: No relevant analysis, ignores most or all user information and data
1: Very limited analysis, considers few factors or data points
2: Acceptable analysis, covers some key factors but misses minor aspects
3: Excellent analysis, considers all provided information and data, with thorough and well-structured reasoning

Reasoning text to evaluate:
{reasoning_text}

Please provide:
1. A score from 0 to 3
2. A brief explanation (1-2 sentences)

Format your response as:
Score: 0-3
Explanation: your explanation"""
    
    return prompt


def extract_score(response_text):
    """从评估响应中提取分数"""
    import re
    
    # 尝试匹配 "Score: X" 格式
    match = re.search(r'Score:\s*([0-3])', response_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None

