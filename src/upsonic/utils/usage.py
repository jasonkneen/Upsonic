"""
Usage and cost calculation utilities for the Upsonic framework.

This module provides functions for calculating and estimating costs based on
token usage and model pricing data. These are pure calculation utilities
without any console/terminal printing dependencies.
"""
from typing import TYPE_CHECKING, Any, Dict, Union

if TYPE_CHECKING:
    from upsonic.model_base import Model
    from upsonic.usage import RequestUsage, RunUsage


# =============================================================================
# Model Pricing Data
# =============================================================================

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI GPT-4o family
    'gpt-4o': {'input_cost_per_1m': 2.50, 'output_cost_per_1m': 10.00},
    'gpt-4o-2024-05-13': {'input_cost_per_1m': 2.50, 'output_cost_per_1m': 10.00},
    'gpt-4o-2024-08-06': {'input_cost_per_1m': 2.50, 'output_cost_per_1m': 10.00},
    'gpt-4o-2024-11-20': {'input_cost_per_1m': 2.50, 'output_cost_per_1m': 10.00},
    'gpt-4o-mini': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.60},
    'gpt-4o-mini-2024-07-18': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.60},
    
    # OpenAI GPT-4 family
    'gpt-4-turbo': {'input_cost_per_1m': 10.00, 'output_cost_per_1m': 30.00},
    'gpt-4-turbo-2024-04-09': {'input_cost_per_1m': 10.00, 'output_cost_per_1m': 30.00},
    'gpt-4': {'input_cost_per_1m': 30.00, 'output_cost_per_1m': 60.00},
    'gpt-4-0613': {'input_cost_per_1m': 30.00, 'output_cost_per_1m': 60.00},
    'gpt-4-32k': {'input_cost_per_1m': 60.00, 'output_cost_per_1m': 120.00},
    'gpt-4-32k-0613': {'input_cost_per_1m': 60.00, 'output_cost_per_1m': 120.00},
    
    # OpenAI GPT-3.5 family
    'gpt-3.5-turbo': {'input_cost_per_1m': 0.50, 'output_cost_per_1m': 1.50},
    'gpt-3.5-turbo-1106': {'input_cost_per_1m': 0.50, 'output_cost_per_1m': 1.50},
    'gpt-3.5-turbo-16k': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 4.00},
    'gpt-3.5-turbo-16k-0613': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 4.00},
    
    # OpenAI GPT-5 family (future)
    'gpt-5': {'input_cost_per_1m': 5.00, 'output_cost_per_1m': 15.00},
    'gpt-5-2025-08-07': {'input_cost_per_1m': 5.00, 'output_cost_per_1m': 15.00},
    'gpt-5-mini': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 1.20},
    'gpt-5-mini-2025-08-07': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 1.20},
    'gpt-5-nano': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.40},
    'gpt-5-nano-2025-08-07': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.40},
    
    # OpenAI GPT-4.1 family
    'gpt-4.1': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 12.00},
    'gpt-4.1-2025-04-14': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 12.00},
    'gpt-4.1-mini': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.80},
    'gpt-4.1-mini-2025-04-14': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.80},
    'gpt-4.1-nano': {'input_cost_per_1m': 0.08, 'output_cost_per_1m': 0.32},
    'gpt-4.1-nano-2025-04-14': {'input_cost_per_1m': 0.08, 'output_cost_per_1m': 0.32},
    
    # OpenAI O-series reasoning models
    'o1': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 60.00},
    'o1-2024-12-17': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 60.00},
    'o1-mini': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 12.00},
    'o1-mini-2024-09-12': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 12.00},
    'o1-preview': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 60.00},
    'o1-preview-2024-09-12': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 60.00},
    'o1-pro': {'input_cost_per_1m': 60.00, 'output_cost_per_1m': 180.00},
    'o1-pro-2025-03-19': {'input_cost_per_1m': 60.00, 'output_cost_per_1m': 180.00},
    'o3': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 80.00},
    'o3-2025-04-16': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 80.00},
    'o3-mini': {'input_cost_per_1m': 4.00, 'output_cost_per_1m': 16.00},
    'o3-mini-2025-01-31': {'input_cost_per_1m': 4.00, 'output_cost_per_1m': 16.00},
    'o3-pro': {'input_cost_per_1m': 80.00, 'output_cost_per_1m': 240.00},
    'o3-pro-2025-06-10': {'input_cost_per_1m': 80.00, 'output_cost_per_1m': 240.00},
    'o3-deep-research': {'input_cost_per_1m': 100.00, 'output_cost_per_1m': 300.00},
    'o3-deep-research-2025-06-26': {'input_cost_per_1m': 100.00, 'output_cost_per_1m': 300.00},
    
    # Anthropic Claude family
    'claude-3-5-sonnet-20241022': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
    'claude-3-5-sonnet-latest': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
    'claude-3-5-sonnet-20240620': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
    'claude-3-5-haiku-20241022': {'input_cost_per_1m': 0.80, 'output_cost_per_1m': 4.00},
    'claude-3-5-haiku-latest': {'input_cost_per_1m': 0.80, 'output_cost_per_1m': 4.00},
    'claude-3-7-sonnet-20250219': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
    'claude-3-7-sonnet-latest': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
    'claude-3-opus-20240229': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 75.00},
    'claude-3-opus-latest': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 75.00},
    'claude-3-haiku-20240307': {'input_cost_per_1m': 0.25, 'output_cost_per_1m': 1.25},
    'claude-4-opus-20250514': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 100.00},
    'claude-4-sonnet-20250514': {'input_cost_per_1m': 4.00, 'output_cost_per_1m': 20.00},
    'claude-opus-4-0': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 100.00},
    'claude-opus-4-1-20250805': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 100.00},
    'claude-opus-4-20250514': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 100.00},
    'claude-sonnet-4-0': {'input_cost_per_1m': 4.00, 'output_cost_per_1m': 20.00},
    'claude-sonnet-4-20250514': {'input_cost_per_1m': 4.00, 'output_cost_per_1m': 20.00},
    
    # Google Gemini family
    'gemini-2.0-flash': {'input_cost_per_1m': 0.075, 'output_cost_per_1m': 0.30},
    'gemini-2.0-flash-lite': {'input_cost_per_1m': 0.0375, 'output_cost_per_1m': 0.15},
    'gemini-2.5-flash': {'input_cost_per_1m': 0.075, 'output_cost_per_1m': 0.30},
    'gemini-2.5-flash-lite': {'input_cost_per_1m': 0.0375, 'output_cost_per_1m': 0.15},
    'gemini-2.5-pro': {'input_cost_per_1m': 1.25, 'output_cost_per_1m': 5.00},
    'gemini-1.5-pro': {'input_cost_per_1m': 1.25, 'output_cost_per_1m': 5.00},
    'gemini-1.5-flash': {'input_cost_per_1m': 0.075, 'output_cost_per_1m': 0.30},
    'gemini-1.0-pro': {'input_cost_per_1m': 0.50, 'output_cost_per_1m': 1.50},
    
    # Groq / Meta Llama
    'llama-3.3-70b-versatile': {'input_cost_per_1m': 0.59, 'output_cost_per_1m': 0.79},
    'llama-3.1-8b-instant': {'input_cost_per_1m': 0.05, 'output_cost_per_1m': 0.05},
    'llama3-70b-8192': {'input_cost_per_1m': 0.59, 'output_cost_per_1m': 0.79},
    'llama3-8b-8192': {'input_cost_per_1m': 0.05, 'output_cost_per_1m': 0.05},
    'mixtral-8x7b-32768': {'input_cost_per_1m': 0.24, 'output_cost_per_1m': 0.24},
    'gemma2-9b-it': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.10},
    
    # Mistral
    'mistral-large-latest': {'input_cost_per_1m': 2.00, 'output_cost_per_1m': 6.00},
    'mistral-small-latest': {'input_cost_per_1m': 1.00, 'output_cost_per_1m': 3.00},
    'codestral-latest': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.20},
    
    # Cohere
    'command': {'input_cost_per_1m': 1.00, 'output_cost_per_1m': 2.00},
    'command-light': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 0.30},
    'command-r': {'input_cost_per_1m': 0.50, 'output_cost_per_1m': 1.50},
    'command-r-plus': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
    
    # DeepSeek
    'deepseek-chat': {'input_cost_per_1m': 0.14, 'output_cost_per_1m': 0.28},
    'deepseek-reasoner': {'input_cost_per_1m': 0.55, 'output_cost_per_1m': 2.19},
    
    # xAI Grok
    'grok-4': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
    'grok-4-0709': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
    'grok-3': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
    'grok-3-mini': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
    'grok-3-fast': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
    'grok-3-mini-fast': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
    
    # Moonshot / Kimi
    'moonshot-v1-8k': {'input_cost_per_1m': 0.012, 'output_cost_per_1m': 0.012},
    'moonshot-v1-32k': {'input_cost_per_1m': 0.024, 'output_cost_per_1m': 0.024},
    'moonshot-v1-128k': {'input_cost_per_1m': 0.06, 'output_cost_per_1m': 0.06},
    'kimi-latest': {'input_cost_per_1m': 0.012, 'output_cost_per_1m': 0.012},
    'kimi-thinking-preview': {'input_cost_per_1m': 0.012, 'output_cost_per_1m': 0.012},
    
    # Open source / Cerebras / Hugging Face
    'gpt-oss-120b': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.10},
    'llama3.1-8b': {'input_cost_per_1m': 0.05, 'output_cost_per_1m': 0.05},
    'llama-3.3-70b': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.20},
    'llama-4-scout-17b-16e-instruct': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.15},
    'llama-4-maverick-17b-128e-instruct': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.15},
    'qwen-3-235b-a22b-instruct-2507': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 0.30},
    'qwen-3-32b': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.10},
    'qwen-3-coder-480b': {'input_cost_per_1m': 0.50, 'output_cost_per_1m': 0.50},
    'qwen-3-235b-a22b-thinking-2507': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 0.30},
    
    # HuggingFace / Together AI paths
    'Qwen/QwQ-32B': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.10},
    'Qwen/Qwen2.5-72B-Instruct': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.20},
    'Qwen/Qwen3-235B-A22B': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 0.30},
    'Qwen/Qwen3-32B': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.10},
    'deepseek-ai/DeepSeek-R1': {'input_cost_per_1m': 0.55, 'output_cost_per_1m': 2.19},
    'meta-llama/Llama-3.3-70B-Instruct': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.20},
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.15},
    'meta-llama/Llama-4-Scout-17B-16E-Instruct': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.15},
    
    # Test model
    'test': {'input_cost_per_1m': 0.00, 'output_cost_per_1m': 0.00},
}

# Default pricing for unknown models
DEFAULT_PRICING: Dict[str, float] = {
    'input_cost_per_1m': 0.50,
    'output_cost_per_1m': 1.50
}

# Provider prefixes to strip from model names
PROVIDER_PREFIXES = [
    'anthropic:',
    'google-gla:',
    'google-vertex:',
    'groq:',
    'mistral:',
    'cohere:',
    'deepseek:',
    'grok:',
    'moonshotai:',
    'cerebras:',
    'huggingface:',
    'heroku:',
    'bedrock:',
    'openai:',
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_name(model: Union["Model", str]) -> str:
    """
    Extract the model name from a Model instance or string.
    
    Args:
        model: Model instance or model name string (e.g., "openai/gpt-4o-mini")
    
    Returns:
        The extracted model name without provider prefix.
    """
    if isinstance(model, str):
        # Handle provider/model format like "openai/gpt-4o-mini"
        if '/' in model:
            return model.split('/', 1)[1]
        return model
    elif hasattr(model, 'model_name'):
        model_name = model.model_name
        # Handle case where model_name might be a coroutine (in tests)
        if hasattr(model_name, '__await__'):
            return "test-model"
        return model_name
    else:
        return str(model)


def get_model_pricing(model_name: str) -> Dict[str, float]:
    """
    Get pricing data for a model.
    
    Args:
        model_name: The model name to look up pricing for.
    
    Returns:
        Dictionary with 'input_cost_per_1m' and 'output_cost_per_1m' keys.
        Returns default pricing if model is not found.
    """
    # Handle case where model_name might be a coroutine (in tests)
    if hasattr(model_name, '__await__'):
        model_name = "test-model"
    
    # Ensure model_name is a string
    model_name = str(model_name)
    
    # Strip provider prefixes
    for prefix in PROVIDER_PREFIXES:
        if model_name.startswith(prefix):
            model_name = model_name[len(prefix):]
            break
    
    return MODEL_PRICING.get(model_name, DEFAULT_PRICING)


# =============================================================================
# Cost Calculation Functions
# =============================================================================

def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: Union["Model", str],
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> float:
    """
    Calculate the cost in dollars based on token usage and model.
    
    This is the primary cost calculation function that returns a float value.
    It first attempts to use genai_prices library for accurate pricing,
    then falls back to built-in pricing data.
    
    Args:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        model: Model instance or model name string.
        cache_write_tokens: Number of cache write tokens (if applicable).
        cache_read_tokens: Number of cache read tokens (if applicable).
        reasoning_tokens: Number of reasoning tokens (for o1/o3 models).
    
    Returns:
        The calculated cost as a float (in dollars).
    """
    if input_tokens is None or output_tokens is None:
        return 0.0
    
    try:
        input_tokens = max(0, int(input_tokens))
        output_tokens = max(0, int(output_tokens))
        cache_write_tokens = max(0, int(cache_write_tokens or 0))
        cache_read_tokens = max(0, int(cache_read_tokens or 0))
        reasoning_tokens = max(0, int(reasoning_tokens or 0))
    except (ValueError, TypeError):
        return 0.0
    
    # Try genai_prices first for accurate pricing
    try:
        from genai_prices import calculate_cost as genai_calculate_cost
        from upsonic.usage import RequestUsage
        
        usage = RequestUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
        )
        
        model_name = get_model_name(model)
        cost = genai_calculate_cost(usage, model_name)
        
        # Add reasoning token cost if applicable (o1/o3 models)
        if reasoning_tokens > 0:
            pricing = get_model_pricing(model_name)
            # Reasoning tokens are typically priced at output rate
            reasoning_cost = (reasoning_tokens / 1_000_000) * pricing['output_cost_per_1m']
            cost += reasoning_cost
        
        return float(cost)
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fall back to built-in pricing
    model_name = get_model_name(model)
    pricing = get_model_pricing(model_name)
    
    input_cost = (input_tokens / 1_000_000) * pricing['input_cost_per_1m']
    output_cost = (output_tokens / 1_000_000) * pricing['output_cost_per_1m']
    
    # Cache tokens typically have different pricing (usually discounted)
    # For simplicity, we use 50% of input rate for cache reads, 100% for writes
    cache_read_cost = (cache_read_tokens / 1_000_000) * (pricing['input_cost_per_1m'] * 0.5)
    cache_write_cost = (cache_write_tokens / 1_000_000) * pricing['input_cost_per_1m']
    
    # Reasoning tokens are typically priced at output rate
    reasoning_cost = (reasoning_tokens / 1_000_000) * pricing['output_cost_per_1m']
    
    total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost + reasoning_cost
    return total_cost


def calculate_cost_from_usage(
    usage: Union[Dict[str, int], "RequestUsage", "RunUsage"],
    model: Union["Model", str]
) -> float:
    """
    Calculate cost from a usage object (RequestUsage, RunUsage, or dict).
    
    Args:
        usage: Usage object or dictionary with token counts.
        model: Model instance or model name string.
    
    Returns:
        The calculated cost as a float (in dollars).
    """
    if isinstance(usage, dict):
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        cache_write_tokens = usage.get('cache_write_tokens', 0)
        cache_read_tokens = usage.get('cache_read_tokens', 0)
        reasoning_tokens = usage.get('reasoning_tokens', 0)
    else:
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)
        cache_write_tokens = getattr(usage, 'cache_write_tokens', 0)
        cache_read_tokens = getattr(usage, 'cache_read_tokens', 0)
        reasoning_tokens = getattr(usage, 'reasoning_tokens', 0)
    
    return calculate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def calculate_cost_from_run_output(
    run_output: Any,
    model: Union["Model", str]
) -> float:
    """
    Calculate cost from an AgentRunOutput object.
    
    If the run_output has a usage attribute with cost already set, return it.
    Otherwise, calculate from the messages in the run output.
    
    Args:
        run_output: AgentRunOutput object.
        model: Model instance or model name string.
    
    Returns:
        The calculated cost as a float (in dollars).
    """
    # First check if usage already has cost
    if hasattr(run_output, 'usage') and run_output.usage:
        if hasattr(run_output.usage, 'cost') and run_output.usage.cost is not None:
            return run_output.usage.cost
        # Calculate from usage tokens
        return calculate_cost_from_usage(run_output.usage, model)
    
    # Fall back to calculating from messages
    total_input_tokens = 0
    total_output_tokens = 0
    
    if hasattr(run_output, 'all_messages'):
        messages = run_output.all_messages()
        for message in messages:
            if hasattr(message, 'usage') and message.usage:
                if hasattr(message, 'kind') and message.kind == 'response':
                    usage = message.usage
                    total_input_tokens += getattr(usage, 'input_tokens', 0)
                    total_output_tokens += getattr(usage, 'output_tokens', 0)
    
    return calculate_cost(total_input_tokens, total_output_tokens, model)


def calculate_cost_from_agent(agent: Any) -> float:
    """
    Calculate cost from an Agent's current run or session.
    
    Args:
        agent: Agent instance.
    
    Returns:
        The calculated cost as a float (in dollars).
    """
    # Try to get session usage first
    if hasattr(agent, 'get_session_usage'):
        session_usage = agent.get_session_usage()
        if session_usage and hasattr(session_usage, 'cost') and session_usage.cost is not None:
            return session_usage.cost
    
    # Fall back to run output
    if hasattr(agent, 'get_run_output'):
        run_output = agent.get_run_output()
        if run_output:
            model = getattr(agent, 'model', 'gpt-4o-mini')
            return calculate_cost_from_run_output(run_output, model)
    
    return 0.0


# =============================================================================
# Formatted String Functions (for display)
# =============================================================================

def format_cost(cost: float, approximate: bool = True) -> str:
    """
    Format a cost value as a string.
    
    Args:
        cost: The cost value in dollars.
        approximate: Whether to prefix with "~" for approximate values.
    
    Returns:
        Formatted cost string (e.g., "~$0.0123" or "$0.0123").
    """
    prefix = "~" if approximate else ""
    
    if cost < 0.0001:
        return f"{prefix}${cost:.6f}"
    elif cost < 0.01:
        return f"{prefix}${cost:.5f}"
    else:
        return f"{prefix}${cost:.4f}"


def get_estimated_cost(
    input_tokens: int,
    output_tokens: int,
    model: Union["Model", str]
) -> str:
    """
    Calculate and format estimated cost as a string.
    
    This function is provided for backward compatibility and display purposes.
    For programmatic use, prefer calculate_cost() which returns a float.
    
    Args:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        model: Model instance or model name string.
    
    Returns:
        Formatted cost string (e.g., "~$0.0123").
    """
    cost = calculate_cost(input_tokens, output_tokens, model)
    return format_cost(cost, approximate=True)


def get_estimated_cost_from_usage(
    usage: Union[Dict[str, int], Any],
    model: Union["Model", str]
) -> str:
    """
    Calculate and format estimated cost from usage data as a string.
    
    Args:
        usage: Usage object or dictionary with token counts.
        model: Model instance or model name string.
    
    Returns:
        Formatted cost string (e.g., "~$0.0123").
    """
    cost = calculate_cost_from_usage(usage, model)
    return format_cost(cost, approximate=True)


def get_estimated_cost_from_run_output(
    run_output: Any,
    model: Union["Model", str]
) -> str:
    """
    Calculate and format estimated cost from AgentRunOutput as a string.
    
    Args:
        run_output: AgentRunOutput object.
        model: Model instance or model name string.
    
    Returns:
        Formatted cost string (e.g., "~$0.0123").
    """
    cost = calculate_cost_from_run_output(run_output, model)
    return format_cost(cost, approximate=True)


def get_estimated_cost_from_agent(agent: Any) -> str:
    """
    Calculate and format estimated cost from Agent as a string.
    
    Args:
        agent: Agent instance.
    
    Returns:
        Formatted cost string (e.g., "~$0.0123").
    """
    cost = calculate_cost_from_agent(agent)
    return format_cost(cost, approximate=True)

