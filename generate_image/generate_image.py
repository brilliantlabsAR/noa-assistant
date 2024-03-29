# 
#  Generate image from text or image.
# 

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from pydantic import BaseModel



class GenerateImage(ABC):
    @abstractmethod
    def generate_image(
        query: str,
        use_image: bool,
        image_bytes: bytes | None,
    ) -> str:
        pass