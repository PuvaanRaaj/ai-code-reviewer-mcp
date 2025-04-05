# utils/prompt_builder.py
from typing import List, Dict, Any, Optional

def extract_code_blocks(content: str) -> List[str]:
    """
    Extract code blocks marked with triple backticks from content.
    """
    code_blocks = []
    in_code_block = False
    current_block = []
    
    for line in content.split('\n'):
        if line.strip().startswith('```'):
            if in_code_block:
                # End of a code block
                code_blocks.append('\n'.join(current_block))
                current_block = []
                in_code_block = False
            else:
                # Start of a code block
                in_code_block = True
        elif in_code_block:
            current_block.append(line)
            
    return code_blocks

def build_code_review_prompt(
    code: str, 
    language: Optional[str] = None,
    focus_areas: Optional[List[str]] = None
) -> str:
    """
    Build a prompt for code review with specific focus areas.
    
    Args:
        code: The code to review
        language: Programming language of the code
        focus_areas: Specific areas to focus on in the review
        
    Returns:
        Formatted prompt for the LLM
    """
    # Default focus areas if none specified
    if not focus_areas:
        focus_areas = [
            "security vulnerabilities", 
            "code quality",
            "performance issues",
            "maintainability",
            "best practices"
        ]
    
    # Format the focus areas as a comma-separated string
    focus_str = ", ".join(focus_areas)
    
    # Language-specific instruction
    lang_str = f"The code is written in {language}. " if language else ""
    
    # Build the prompt
    prompt = f"""You are an expert code reviewer. {lang_str}Review the following code focusing on {focus_str}.

Provide specific, actionable feedback including:
1. A summary of the code's purpose and structure
2. Potential bugs or issues that need to be fixed
3. Security concerns if any exist
4. Performance improvements that could be made
5. Code organization and readability enhancements
6. Best practices and design patterns that should be applied

Format your response as a structured code review with clear sections.
IMPORTANT: DO NOT include the original code in your response, only your analysis.

Code to review:

```
{code}
```

Code Review:"""
    
    return prompt