# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List, Dict, Any, Optional
import requests
import tiktoken
from .base import BaseLLM
from .rate_limiter import RateLimiter

class OpenRouterLLM(BaseLLM):
    """OpenRouter API wrapper for multiple LLM providers."""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        rate_limits: Optional[Dict[str, Any]] = None
    ):
        """Initialize OpenRouter LLM.
        
        Args:
            api_key: OpenRouter API key
            model: Model identifier (e.g., "anthropic/claude-3-opus")
            rate_limits: Optional dictionary with rate limit settings
        """
        self.api_key = api_key
        self.model = model
        self.api_base = "https://openrouter.ai/api/v1"
        
        try:
            # Initialize tokenizer (using cl100k_base as default)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
            
        # Default rate limits for OpenRouter
        default_limits = {
            "requests_per_minute": 50,
            "input_tokens_per_minute": 100000,
            "output_tokens_per_minute": 50000,
            "input_token_price_per_million": 4.0,
            "output_token_price_per_million": 16.0
        }
        
        # Use provided rate limits or defaults
        limits = rate_limits or default_limits
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            provider="OpenRouter",
            requests_per_minute=limits.get("requests_per_minute", default_limits["requests_per_minute"]),
            input_tokens_per_minute=limits.get("input_tokens_per_minute", default_limits["input_tokens_per_minute"]),
            output_tokens_per_minute=limits.get("output_tokens_per_minute", default_limits["output_tokens_per_minute"]),
            input_token_price_per_million=limits.get("input_token_price_per_million", default_limits["input_token_price_per_million"]),
            output_token_price_per_million=limits.get("output_token_price_per_million", default_limits["output_token_price_per_million"])
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a string using the model's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        if not text:
            return 0
            
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            # Fallback: rough estimate if tokenizer fails
            return len(text.split()) * 1.3
    
    def _count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in all messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Total token count
        """
        if not messages:
            return 0
            
        total_tokens = 0
        
        for message in messages:
            if "content" in message and message["content"]:
                total_tokens += self._count_tokens(message["content"])
            
        # Add overhead for message formatting
        total_tokens += 4 * len(messages)
        total_tokens += 3  # System overhead
        
        return total_tokens
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> str:
        """Generate a response using OpenRouter API with rate limiting.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
            
        Raises:
            requests.exceptions.RequestException: If the API call fails
        """
        # Count input tokens
        input_tokens = self._count_messages_tokens(messages)
        
        # Wait if approaching rate limits
        self.rate_limiter.wait_if_needed(input_tokens, max_tokens)
        
        # Prepare the API call
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/facebookresearch/DocAgent",  # Required by OpenRouter
            "X-Title": "DocAgent"  # Required by OpenRouter
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens else None
        }
        
        # Make the API call
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        
        result_text = result["choices"][0]["message"]["content"]
        
        # Record request with token counts
        output_tokens = self._count_tokens(result_text)
        self.rate_limiter.record_request(input_tokens, output_tokens)
        
        return result_text
    
    def format_message(self, role: str, content: str) -> Dict[str, str]:
        """Format a message for the OpenRouter API.
        
        Args:
            role: Message role (system, user, assistant)
            content: Message content
            
        Returns:
            Formatted message dictionary
        """
        return {
            "role": role,
            "content": content
        }
