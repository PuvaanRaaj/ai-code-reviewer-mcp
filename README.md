[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/puvaanraaj-ai-code-reviewer-mcp-badge.png)](https://mseep.ai/app/puvaanraaj-ai-code-reviewer-mcp)

# Privacy-Focused AI Code Reviewer

An MCP (Model Context Protocol) server that provides AI-powered code reviews without storing or logging any source code.

## Features

- Privacy-first design: No code is stored or logged
- Memory-only processing: All code remains in memory and is explicitly deleted after processing
- Comprehensive code reviews focusing on:
  - Security vulnerabilities
  - Code quality
  - Performance issues
  - Best practices
  - Design patterns

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API key:
   ```
   API_KEY=your-api-key-here
   LOG_LEVEL=INFO
   ```
5. Run the server:
   ```bash
   python main.py
   ```

## Usage

The server exposes an endpoint at `/mcp/code-review` that accepts POST requests with MCP-formatted messages.

Example request:
```bash
curl -X POST http://localhost:8000/mcp/code-review \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "```python\ndef insecure_function(user_input):\n    query = \"SELECT * FROM users WHERE id = \" + user_input\n    return db.execute(query)\n```"
      }
    ]
  }'
```

## Privacy Guarantees

- No code is written to disk
- No code is included in logs
- All variables containing code are explicitly deleted after use
- Response contains only analysis, not the original code

## Deployment Recommendations

For maximum security and compliance:
- Deploy on your own infrastructure
- Use HTTPS encryption
- Implement proper authentication
- Consider network isolation for sensitive code bases