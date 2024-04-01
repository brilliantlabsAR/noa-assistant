#
# web_search.py
#
# Web search tool base class and result structure.
#

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WebSearchResult:
    """
    Web search result, used for all concrete implementations of WebSearch.
    """

    #
    # Summarized result, to be used as the tool response string.
    #
    summary: str

    #
    # Implementation-specific metadata for debugging. Can contain e.g. search result links, etc. If
    # we want to break out search result links for e.g., the mobile companion app, we should create
    # a new field and avoid using this one.
    #
    search_provider_metadata: str

class WebSearch(ABC):
    @abstractmethod
    async def search_web(self, query: str, use_photo: bool = False, image_bytes: bytes | None = None, location: str | None = None) -> WebSearchResult:
        pass


