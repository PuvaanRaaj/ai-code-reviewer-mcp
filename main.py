# main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import time
from typing import List, Dict, Any, Optional
import importlib

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
app = FastAPI(title="Multi-LLM Privacy-Focused Code Reviewer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM Provider Factory
class LLMFactory:
    @staticmethod
    def get_llm_provider(provider_name: str):
        """
        Factory method to get the appropriate LLM provider client.
        """
        provider_name = provider_name.lower()
        
        if provider_name == "anthropic":
            return AnthropicProvider()
        elif provider_name == "openai":
            return OpenAIProvider()
        elif provider_name == "ollama":
            return OllamaProvider()
        elif provider_name == "deepseek":
            return DeepseekProvider()
        elif provider_name == "huggingface":
            return HuggingFaceProvider()
        elif provider_name == "mistral":
            return MistralProvider()
        else:
            logger.error(f"Unknown LLM provider: {provider_name}")
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

# LLM Provider Interface
class LLMProvider:
    def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in the response
            
        Returns:
            The LLM's response as a string
        """
        raise NotImplementedError("Each LLM provider must implement this method")

# Anthropic (Claude) Provider
class AnthropicProvider(LLMProvider):
    def __init__(self):
        try:
            from anthropic import Anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            self.client = Anthropic(api_key=api_key)
            self.model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
            logger.info(f"Initialized Anthropic provider with model: {self.model}")
        except ImportError:
            logger.error("Anthropic package not installed. Run: pip install anthropic")
            raise
            
    def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

# OpenAI Provider
class OpenAIProvider(LLMProvider):
    def __init__(self):
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.client = OpenAI(api_key=api_key)
            self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")
            logger.info(f"Initialized OpenAI provider with model: {self.model}")
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
            
    def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Ollama Provider (Local)
class OllamaProvider(LLMProvider):
    def __init__(self):
        try:
            import ollama
            self.client = ollama
            self.model = os.environ.get("OLLAMA_MODEL", "codellama:latest")
            logger.info(f"Initialized Ollama provider with model: {self.model}")
        except ImportError:
            logger.error("Ollama package not installed. Run: pip install ollama")
            raise
            
    def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0.2, "num_predict": max_tokens}
        )
        return response['response']

# DeepSeek Provider
class DeepseekProvider(LLMProvider):
    def __init__(self):
        try:
            from openai import OpenAI  # DeepSeek uses OpenAI-compatible API
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
            base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-coder")
            logger.info(f"Initialized DeepSeek provider with model: {self.model}")
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
            
    def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Mistral Provider
class MistralProvider(LLMProvider):
    def __init__(self):
        try:
            from mistralai.client import MistralClient
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY environment variable is not set")
            self.client = MistralClient(api_key=api_key)
            self.model = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
            logger.info(f"Initialized Mistral provider with model: {self.model}")
        except ImportError:
            logger.error("Mistral package not installed. Run: pip install mistralai")
            raise
            
    def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

# HuggingFace Provider
class HuggingFaceProvider(LLMProvider):
    def __init__(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Load model name from environment or use default
            self.model_name = os.environ.get("HUGGINGFACE_MODEL", "bigcode/starcoder2-15b")
            logger.info(f"Initializing HuggingFace provider with model: {self.model_name}")
            
            # This will download the model if not already present
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"HuggingFace model loaded on {self.device}")
        except ImportError:
            logger.error("Transformers package not installed. Run: pip install transformers torch")
            raise
            
    def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
            )
            
        # Decode and extract only the new content (not the prompt)
        full_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        prompt_len = len(prompt)
        response = full_output[prompt_len:]
        
        return response

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
        
        # Get the configured LLM provider
        llm_provider_name = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        logger.info(f"Using LLM provider: {llm_provider_name}")
        
        try:
            llm_provider = LLMFactory.get_llm_provider(llm_provider_name)
            review = llm_provider.generate_response(prompt)
        except Exception as e:
            logger.error(f"Error with LLM provider {llm_provider_name}: {str(e)}")
            return Response(
                content=json.dumps({"error": f"LLM provider error: {str(e)}"}),
                media_type="application/json",
                status_code=500
            )
        
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
    provider_name = os.environ.get("LLM_PROVIDER", "anthropic")
    return {
        "status": "healthy", 
        "provider": provider_name,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Multi-LLM Code Reviewer with provider: {os.environ.get('LLM_PROVIDER', 'anthropic')}")
    uvicorn.run(app, host="0.0.0.0", port=8000)