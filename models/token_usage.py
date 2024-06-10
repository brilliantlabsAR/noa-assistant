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