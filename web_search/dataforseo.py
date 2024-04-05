#
# dataforseo.py
#
# Web search tool implementation using DataForSEO (dataforseo.com). Does not support images.
#

from base64 import b64encode
import json
import os
from typing import Any, List, Optional, Tuple

import aiohttp
from pydantic import BaseModel
import geopy.geocoders

from .web_search import WebSearch, WebSearchResult


DATAFORSEO_USERNAME =  os.environ.get("DATAFORSEO_USERNAME", None)
DATAFORSEO_PASSWORD =  os.environ.get("DATAFORSEO_PASSWORD", None)

class Price(BaseModel):
    current: float = None
    display_price: str = None
    currency: str = None

class Rating(BaseModel):
    rating_type: str = None
    value: float = None
    votes_count: int = None
    rating_max: int = None

class SubItem(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    price : Optional[Price] = None
    rating: Optional[Rating] = None

class Item(BaseModel):
    type: str
    title: Optional[str] = None
    description: Optional[str] = None
    items:Optional[List[SubItem]|List[str]] = None

class Result(BaseModel):
    keyword: str
    type: str
    check_url: str
    items: List[Item]

class Task(BaseModel):
    id: str
    status_code: int
    status_message: str
    cost: float
    result: List[Result]|None

# /v3/serp/google/organic/live/advanced response object
class V3SerpGoogleOrganicLiveAdvancedResponse(BaseModel):
    status_code: int
    status_message: str
    cost: float
    tasks_count: int
    tasks_error: int
    tasks: List[Task]

    def summarise(self, max_search_results: int = 5) -> List[str]:
        item_types = [ "stock_box", "organic", "knowledge_graph", "local_pack", "popular_products", "top_stories" ]
        summaries = []
        for task in self.tasks:
            if not task.result:
                continue
            for result in task.result:
                for item in result.items:
                    if  item.type in item_types and max_search_results > 0 and (item.description or item.items):
                        # print(summaries)
                        if item.items:
                            for subitem in item.items:
                                #print(subitem)

                                if isinstance(subitem, SubItem) and subitem.title and max_search_results > 0 and subitem.description:
                                    summary = (f"{subitem.title}: " if subitem.title else "") + subitem.description
                                    if subitem.price:
                                        if subitem.price.display_price:
                                            summary += f"\nprice: {subitem.price.currency} {subitem.price.display_price}"
                                        elif subitem.price.currency:
                                            summary += f"\nprice: {subitem.price.currency} {subitem.price.current}"
                                    if subitem.rating:
                                        if subitem.rating.value:
                                            summary += f"\nrating: {subitem.rating.value} of {subitem.rating.rating_max} ({subitem.rating.votes_count} votes)"
                                    summaries.append(summary)
                                    max_search_results = max_search_results -1
                        if  item.description:
                            summary = (f"{item.title}: " if item.title else "") + item.description
                            summaries.append(summary)
                            max_search_results = max_search_results -1
        content = "\n".join(summaries) if len(summaries) > 0 else "No result found"
        return content

class DataForSEOClient:
    def __init__(self):
        self._session = aiohttp.ClientSession()
        self._base_url = "https://api.dataforseo.com"

        base64_bytes = b64encode(
                ("%s:%s" % (DATAFORSEO_USERNAME, DATAFORSEO_PASSWORD)).encode("ascii")
                ).decode("ascii")
        self._headers = {'Authorization' : 'Basic %s' %  base64_bytes, 'Content-Encoding' : 'gzip'}

    def __del__(self):
        self._session.detach()

    async def _request(self, path, method, data=None) -> Any | None:
        url = self._base_url + path
        async with self._session.request(method=method, url=url, headers=self._headers, data=data) as response:
            if response.status != 200:
                print(f"DataForSEO search failed: {await response.text()}")
                return None
            return await response.json()

    async def _get(self, path):
        return await self._request(path=path, method='GET')

    async def _post(self, path, data):
        if isinstance(data, str):
            data_str = data
        else:
            data_str = json.dumps(data)
        return await self._request(path=path, method='POST', data=data_str)
    
    async def perform_search(self, query: str, location_coordinate: Tuple[float, float] | None = None, save_to_file: str | None = None) -> V3SerpGoogleOrganicLiveAdvancedResponse | None:
        print("Searching web:")
        print(f"  query: {query}")

        post_data = dict()
        post_data[len(post_data)] = dict(
            language_code = "en",
            location_coordinate = f"{location_coordinate[0]},{location_coordinate[1]}" if location_coordinate else None,
            keyword = query
        )
        response_obj = await self._post("/v3/serp/google/organic/live/advanced", post_data)
        if response_obj is None:
            return None
        if save_to_file:
            with open(save_to_file, mode="w") as fp:
                fp.write(json.dumps(response_obj, indent=2))
        return V3SerpGoogleOrganicLiveAdvancedResponse.model_validate(response_obj)

class DataForSEOWebSearch(WebSearch):
    def __init__(self, save_to_file: str | None = None, max_search_results: int = 5):
        super().__init__()
        self._save_to_file = save_to_file
        self._max_search_results = max_search_results
        self._client = None
    
    async def _lazy_init(self):
        if self._client is None:
            # This instantiation must happen inside of an async event loop because
            # aiohttp.ClientSession()'s initializer requires that
            self._client = DataForSEOClient()

    # DataForSEO does not have reverse image search, so photos are always ignored
    async def search_web(self, query: str, use_photo: bool = False, image_bytes: bytes | None = None, location: str | None = None) -> WebSearchResult:
        await self._lazy_init()
        if location:
            # DataForSEO expects lat,long+
            location_coords = geopy.geocoders.Nominatim(user_agent="GetLoc").geocode(location)
            coordinates = (location_coords.latitude, location_coords.longitude)
        response = await self._client.perform_search(query=query, location_coordinate=coordinates, save_to_file=self._save_to_file)
        summary = response.summarise(max_search_results=self._max_search_results) if response is not None else "No results found"
        return WebSearchResult(summary=summary, search_provider_metadata="")

WebSearch.register(DataForSEOWebSearch)