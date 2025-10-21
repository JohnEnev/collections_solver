from __future__ import annotations
from typing import List, Dict, Any

from .base_client import SolveResult
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .oss_client import OSSClient, LocalClient


# Model presets for convenience
MODEL_PRESETS: Dict[str, str] = {
    # OpenAI models
    "gpt4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "o1": "o1-preview",
    "o1-mini": "o1-mini",
    # Note: GPT-5 not publicly available yet
    
    # Anthropic models (use dated versions for reliability)
    "sonnet": "claude-3-5-sonnet-20241022",
    "sonnet-new": "claude-3-5-sonnet-20250101",  # if available on your account
    "opus": "claude-3-opus-20240229",
    "haiku": "claude-3-5-haiku-20241022",
    
    # Google models (using actual model names from genai.list_models())
    "gemini": "models/gemini-2.5-flash",  # Latest stable Flash (fast, good quality)
    "gemini-pro": "models/gemini-2.5-pro",  # Latest stable Pro (best quality)
    "gemini-flash": "models/gemini-2.5-flash",  # Alias for stable Flash
    "gemini-flash-lite": "models/gemini-2.5-flash-lite",  # Lighter/faster
    "gemini-exp": "models/gemini-2.0-flash-exp",  # Experimental
    
    # OSS models via Together AI
    "llama-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "llama-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral-small": "mistralai/Mistral-Small-Instruct-2503",
}


def get_client_for_model(model: str):
    """
    Factory function to get the appropriate client for a model.
    
    Args:
        model: Model identifier (can be preset name or full model string)
        
    Returns:
        Appropriate client instance (OpenAI, Anthropic, Gemini, or OSS)
        
    Raises:
        ValueError: If model type cannot be determined
    """
    # Resolve preset aliases
    resolved_model = MODEL_PRESETS.get(model, model)
    
    # Route to appropriate client based on model name
    if any(x in resolved_model for x in ["gpt", "o1", "o3"]):
        return OpenAIClient(), resolved_model
    
    elif "claude" in resolved_model.lower():
        return AnthropicClient(), resolved_model
    
    elif "gemini" in resolved_model.lower():
        return GeminiClient(), resolved_model
    
    elif any(x in resolved_model.lower() for x in ["llama", "qwen", "mistral", "mixtral"]):
        return OSSClient(), resolved_model
    
    elif resolved_model.startswith("local/"):
        # For local models, strip the "local/" prefix
        actual_model = resolved_model.replace("local/", "")
        return LocalClient(), actual_model
    
    else:
        raise ValueError(
            f"Unknown model type: {model}\n"
            f"Supported: OpenAI (gpt-*), Anthropic (claude-*), "
            f"Google (gemini-*), OSS (llama, qwen, mistral), or local/* for local models"
        )


def solve_with_model(
    words: List[str], 
    model: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to solve a puzzle with any model.
    Automatically routes to the correct client.
    
    Args:
        words: 16 words to group
        model: Model identifier (can be preset or full name)
        **kwargs: Additional parameters passed to the client
        
    Returns:
        Dict with groups, rationales, and metadata
        
    Example:
        >>> result = solve_with_model(
        ...     words=["apple", "banana", ...],
        ...     model="gpt4o"
        ... )
        >>> print(result["groups"])
    """
    client, resolved_model = get_client_for_model(model)
    result = client.solve(words, model=resolved_model, **kwargs)
    
    # Convert SolveResult to dict for backward compatibility
    return {
        "groups": result.groups,
        "rationales": result.rationales,
        "raw": result.raw_response,
        "model": result.model
    }


# Backward compatibility alias
def solve_with_llm(words: List[str], model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Legacy function name for backward compatibility.
    Use solve_with_model() in new code.
    """
    return solve_with_model(words, model)