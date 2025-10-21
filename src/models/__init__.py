"""
Model clients for solving NYT Connections puzzles.

This module provides a clean interface to various LLM providers:
- OpenAI (GPT-4, etc.) via official SDK
- Anthropic (Claude) via official SDK  
- Google (Gemini) via official SDK
- OSS models (Llama, Qwen, Mistral) via Together AI
- Local models via vLLM/Ollama

Usage:
    from src.models import solve_with_model
    
    result = solve_with_model(
        words=["apple", "banana", ...],
        model="gpt4o"  # or "sonnet", "gemini", "llama-70b", etc.
    )
    
    print(result["groups"])
"""

from .base_client import SolveResult, BaseClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .oss_client import OSSClient, LocalClient
from .client_factory import (
    solve_with_model,
    solve_with_llm,  # backward compatibility
    get_client_for_model,
    MODEL_PRESETS,
)

__all__ = [
    # Main functions
    "solve_with_model",
    "solve_with_llm",
    "get_client_for_model",
    
    # Client classes (for advanced usage)
    "OpenAIClient",
    "AnthropicClient", 
    "GeminiClient",
    "OSSClient",
    "LocalClient",
    
    # Base types
    "SolveResult",
    "BaseClient",
    
    # Constants
    "MODEL_PRESETS",
]