from __future__ import annotations
from typing import Dict

from pydantic import BaseModel


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0

    def add(self, token_usage: TokenUsage):
        self.input += token_usage.input
        self.output += token_usage.output
        self.total += token_usage.total

def accumulate_token_usage(token_usage_by_model: Dict[str, TokenUsage], model: str, input_tokens: int, output_tokens: int, total_tokens: int):
    token_usage = TokenUsage(input=input_tokens, output=output_tokens, total=total_tokens)
    if model not in token_usage_by_model:
        token_usage_by_model[model] = token_usage
    else:
        token_usage_by_model[model].add(token_usage=token_usage)