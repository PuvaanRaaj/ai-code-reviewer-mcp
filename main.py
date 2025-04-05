# main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from anthropic import Anthropic
import json
import logging
import secrets
import time
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

# Configure logging - ensure no code is logged
class SensitiveDataFilter(logging.Filter):
    def filter(self, record):
        # Convert any potential sensitive data in logs to [REDACTED]
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if "code" in record.msg.lower() or "```" in record.msg:
                record.msg = "[REDACTED CODE]"
        return True

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
logger.addFilter(SensitiveDataFilter())

# Initialize FastAPI app
app = FastAPI(title="Privacy-Focused Code Reviewer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.environ.get("API_KEY"))

# Security utilities
def verify_api_key(api_key: str = None):
    # In production, implement proper API key verification
    if not api_key:
        return False
    # Demo only - replace with actual validation
    return True

class CodeReviewRequest(BaseModel):
    messages: List[Dict[str, Any]]
    api_key: Optional[str] = None

def build_code_review_prompt(code_content: str, review_focus: Optional[str] = None):
    """Build a prompt for code review without storing the code."""
    
    # Define what to focus on in the review
    focus_areas = review_focus or "security vulnerabilities, code quality, performance issues, best practices"
    
    prompt = f"""You are an expert code reviewer. Review the following code for {focus_areas}.
Provide specific, actionable feedback including:
1. Potential bugs or issues
2. Security concerns
3. Performance improvements
4. Code organization and readability
5. Best practices and design patterns

Format your response as a structured code review with clear sections. 
IMPORTANT: DO NOT include the original code in your response, only your analysis.

Code to review:

```
{code_content}
```

Code Review:"""
    
    return prompt

@app.post("/mcp/code-review")
async def handle_code_review(request: Request):
    """Handle MCP requests for code review without storing code."""
    request_start_time = time.time()
    
    try:
        # Parse the request body
        body = await request.json()
        logger.info(f"Received code review request")
        
        # Extract code from the request
        messages = body.get("messages", [])
        if not messages:
            return Response(
                content=json.dumps({"error": "No messages provided"}),
                media_type="application/json",
                status_code=400
            )
        
        # Find the latest user message with code
        code_content = None
        review_focus = None
        
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Check if this message contains code blocks
                if "```" in content:
                    code_content = content
                    break
                elif not review_focus:
                    # Use as review focus if no code found yet
                    review_focus = content
        
        if not code_content:
            return Response(
                content=json.dumps({"error": "No code found in messages"}),
                media_type="application/json",
                status_code=400
            )
        
        # Build prompt for code review
        prompt = build_code_review_prompt(code_content, review_focus)
        
        # Call the model
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract the review
        review = response.content[0].text
        
        # Construct the MCP response
        mcp_response = {
            "messages": [
                {
                    "role": "assistant",
                    "content": review
                }
            ]
        }
        
        # Explicitly clear variables containing code
        code_content = None
        prompt = None
        
        # Log completion time (without code details)
        request_time = time.time() - request_start_time
        logger.info(f"Code review completed in {request_time:.2f}s")
        
        return Response(
            content=json.dumps(mcp_response),
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return Response(
            content=json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500
        )
    finally:
        # Ensure code is removed from memory even in case of errors
        if 'code_content' in locals():
            del code_content
        if 'prompt' in locals():
            del prompt

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Privacy-Focused Code Reviewer MCP Server")
    uvicorn.run(app, host="0.0.0.0", port=8000)