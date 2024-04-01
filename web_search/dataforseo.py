#
# dataforseo.py
#
# Web search tool implementation using DataForSEO (dataforseo.com). Does not support images.
#

from base64 import b64encode
import json
import os
from typing import List, Optional, Tuple

from http.client import HTTPSConnection
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
        self._username = DATAFORSEO_USERNAME
        self._password = DATAFORSEO_PASSWORD

    def _request(self, path, method, data=None):
        connection = HTTPSConnection("api.dataforseo.com")
        try:
            base64_bytes = b64encode(
                ("%s:%s" % (self._username, self._password)).encode("ascii")
                ).decode("ascii")
            headers = {'Authorization' : 'Basic %s' %  base64_bytes, 'Content-Encoding' : 'gzip'}
            connection.request(method, path, headers=headers, body=data)
            response = connection.getresponse()
            return json.loads(response.read().decode())
        finally:
            connection.close()

    def _get(self, path):
        return self._request(path, 'GET')

    def _post(self, path, data):
        if isinstance(data, str):
            data_str = data
        else:
            data_str = json.dumps(data)
        return self._request(path, 'POST', data_str)
    
    def perform_search(self, query: str, location_coordinate: Tuple[float, float] | None = None, save_to_file: str | None = None) -> V3SerpGoogleOrganicLiveAdvancedResponse:
        print("Searching web:")
        print(f"  query: {query}")

        post_data = dict()
        post_data[len(post_data)] = dict(
            language_code = "en",
            location_coordinate = f"{location_coordinate[0]},{location_coordinate[1]}" if location_coordinate else None,
            keyword = query
        )
        response_obj = self._post("/v3/serp/google/organic/live/advanced", post_data)
        if save_to_file:
            with open(save_to_file, mode="w") as fp:
                fp.write(json.dumps(response_obj, indent=2))
        return V3SerpGoogleOrganicLiveAdvancedResponse.model_validate(response_obj)

class DataForSEOWebSearch(WebSearch):
    def __init__(self, save_to_file: str | None = None, max_search_results: int = 5):
        super().__init__()
        self._save_to_file = save_to_file
        self._max_search_results = max_search_results

    # DataForSEO does not have reverse image search, so photos are always ignored
    async def search_web(self, query: str, use_photo: bool = False, image_bytes: bytes | None = None, location: str | None = None) -> WebSearchResult:
        if location:
            # DataForSEO expects lat,long+
            location_coords = geopy.geocoders.Nominatim(user_agent="GetLoc").geocode(location)
            coordinates = (location_coords.latitude, location_coords.longitude)
        response = DataForSEOClient().perform_search(query=query, location_coordinate=coordinates, save_to_file=self._save_to_file)
        return WebSearchResult(summary=response.summarise(max_search_results=self._max_search_results), search_provider_metadata="")

WebSearch.register(DataForSEOWebSearch)