# Model Client Usage Guide

Clean, provider-specific clients for evaluating LLMs on NYT Connections.

## Installation

```bash
pip install openai anthropic google-generativeai litellm tenacity rich typer orjson
```

## Setup API Keys

Create a `.env` file in your project root:

```bash
# Commercial models (choose what you need)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# For OSS models via Together AI
TOGETHER_API_KEY=...
```

## Quick Start

### Running Baselines

```bash
# OpenAI GPT-4o
python run_baseline_llm.py --model gpt4o --limit 100

# Anthropic Claude Sonnet 4
python run_baseline_llm.py --model sonnet4 --limit 100

# Google Gemini
python run_baseline_llm.py --model gemini --limit 100

# OSS: Llama 3.1 70B via Together AI
python run_baseline_llm.py --model llama-70b --limit 100

# OSS: Qwen 2.5 72B
python run_baseline_llm.py --model qwen-72b --limit 100
```

### Using in Code

```python
from src.models import solve_with_model

# Simple usage with preset
result = solve_with_model(
    words=["apple", "banana", "cherry", ...],  # 16 words
    model="gpt4o"  # or "sonnet", "gemini", "llama-70b", etc.
)

print(result["groups"])      # [[w1, w2, w3, w4], ...]
print(result["rationales"])  # ["theme 1", "theme 2", ...]
```

### Advanced Usage

```python
from src.models import OpenAIClient, AnthropicClient, GeminiClient

# Direct client usage for more control
openai_client = OpenAIClient()
result = openai_client.solve(
    words=my_words,
    model="gpt-4o",
    temperature=0.1,  # custom temperature
    max_tokens=1024   # custom token limit
)

# Access structured result
print(result.groups)
print(result.rationales)
print(result.raw_response)  # for debugging
print(result.model)         # actual model used
```

## Available Model Presets

### Commercial Models

**OpenAI:**
- `gpt4o` → gpt-4o
- `gpt4o-mini` → gpt-4o-mini
- `gpt5` → gpt-5 (requires access)
- `o1` → o1-preview

**Anthropic:**
- `sonnet` → claude-3-5-sonnet-20241022
- `sonnet4` → claude-sonnet-4-5-20250929
- `opus` → claude-3-opus-20240229
- `haiku` → claude-3-haiku-20240307

**Google:**
- `gemini` → gemini-2.0-flash-exp
- `gemini-pro` → gemini-1.5-pro-latest
- `gemini-flash` → gemini-1.5-flash-latest

### OSS Models (via Together AI)

- `llama-70b` → Meta-Llama-3.1-70B-Instruct-Turbo
- `llama-8b` → Meta-Llama-3.1-8B-Instruct-Turbo
- `qwen-72b` → Qwen2.5-72B-Instruct-Turbo
- `mistral-7b` → Mistral-7B-Instruct-v0.3
- `mixtral-8x7b` → Mixtral-8x7B-Instruct-v0.1

### Local Fine-tuned Models

For your fine-tuned models served via vLLM:

```bash
# Start vLLM server
vllm serve ./my-finetuned-model --port 8000

# Use in code
result = solve_with_model(
    words=my_words,
    model="local/my-finetuned-model"
)
```

Or via Ollama:

```bash
ollama create connections-model -f Modelfile
ollama serve

# Then use LocalClient with Ollama's base URL
```

## Project Structure

```
src/models/
├── __init__.py              # Exports main functions
├── base_client.py           # Base protocol + shared prompts
├── openai_client.py         # OpenAI official SDK
├── anthropic_client.py      # Anthropic official SDK
├── gemini_client.py         # Google official SDK
├── oss_client.py            # Together AI + local models
└── client_factory.py        # Router + convenience functions
```

## Why This Architecture?

1. **Clean separation** - Each provider in its own file
2. **Easy debugging** - No abstraction layers to fight
3. **Official SDKs** - Better reliability and features
4. **Future-proof** - Easy to add fine-tuned model support
5. **Portfolio-ready** - Professional code structure

## Next Steps

1. **Baseline Evaluation**: Run all models you want to compare
2. **Analysis**: Compare metrics (exact_rate, avg_mistakes, etc.)
3. **Fine-tuning**: Pick best OSS model as base (likely Llama 3.1 8B)
4. **Redeployment**: Serve fine-tuned model via vLLM or Together AI
5. **Final Evaluation**: Compare fine-tuned vs baselines

## Troubleshooting

**"API key not found":**
- Check your `.env` file has the right key
- Run with `--debug` to see which keys are detected

**"Model not supported":**
- Use `--debug` to see available presets
- Check if you need to update the preset name

**Rate limits:**
- Add retry logic (already included via `tenacity`)
- Lower `--limit` or add sleep between calls
- Use cheaper models for large-scale testing (gpt4o-mini, llama-8b)