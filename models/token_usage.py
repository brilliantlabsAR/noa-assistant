from __future__ import annotations
from typing import Dict

from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0

    def add(self, token_usage: TokenUsage):
        self.input += token_usage.input
        self.output += token_usage.output
        self.total += token_usage.total

def accumulate_token_usage(token_usage_by_model: Dict[str, TokenUsage], **kwargs):
    """
    Tracks token usage by model in a dictionary.

    Parameters
    ----------
    token_usage_by_model : Dict[str, TokenUsage]
        Dictionary that will be mutated. Key is model name (e.g, "gpt-4o").
    usage : openai.types.completion_usage.CompletionUsage | TokenUsage
        If this keyword argument is present, it is either the usage in OpenAI format to add to the
        dictionary or in our own TokenUsage format. Requires specifying the `model` argument.
    model : str
        If `usage` keyword argument is present, this must be specified. Indicates the model that
        `usage` corresponds to.
    other : Dict[str, TokenUsage]
        If this keyword argument is present, it is a token usage dictionary whose contents will be
        added to `token_usage_by_model`.
    """
    if "usage" in kwargs:
        assert "model" in kwargs, "'usage' argument requires 'model' to be specified"
        model = kwargs["model"]
        usage = kwargs["usage"]
        assert isinstance(usage, CompletionUsage) or isinstance(usage, TokenUsage)
        token_usage = usage if isinstance(usage, TokenUsage) else TokenUsage(input=usage.prompt_tokens, output=usage.completion_tokens, total=usage.total_tokens)
        if model not in token_usage_by_model:
            token_usage_by_model[model] = token_usage
        else:
            token_usage_by_model[model].add(token_usage=token_usage)
    
    if "other" in kwargs:
        other = kwargs["other"]
        assert isinstance(other, dict)
        for model, token_usage in other.items():
            if model not in token_usage_by_model:
                token_usage_by_model[model] = token_usage
            else:
                token_usage_by_model[model].add(token_usage=token_usage)