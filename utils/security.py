# utils/security.py
import secrets
import hashlib
import base64
from typing import Dict, Any, Optional

def generate_request_id() -> str:
    """Generate a cryptographically secure random request ID."""
    return secrets.token_hex(16)

def hash_content(content: str) -> str:
    """Create a hash of content for logging without storing the content itself."""
    return hashlib.sha256(content.encode()).hexdigest()

def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive data from log information."""
    sanitized = {}
    
    # Copy non-sensitive fields
    for key, value in data.items():
        if key.lower() in ['code', 'source', 'content', 'prompt']:
            if isinstance(value, str):
                # Replace with length and hash
                sanitized[key] = f"[REDACTED - {len(value)} chars]"
            else:
                sanitized[key] = "[REDACTED]"
        else:
            sanitized[key] = value
    
    return sanitized

class MemoryOnlyStorage:
    """
    Class to handle sensitive data that should never be written to disk.
    Implements context manager pattern to ensure data is deleted after use.
    """
    
    def __init__(self, data: Any):
        self.data = data
    
    def __enter__(self):
        return self.data
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Explicitly delete the data
        del self.data