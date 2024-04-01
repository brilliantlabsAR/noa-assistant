#
# serp.py
#
# Web search tool implementation using SerpApi (serpapi.com). Can perform web searches with images
# using multiple methods (reverse image search, Google Lens).
#

import json
from io import BytesIO
import requests
from typing import List, Tuple, Any, Dict, Optional

from pydantic import BaseModel
import serpapi
import uule_grabber
import os

from .web_search import WebSearch, WebSearchResult


TEST_CDN = os.environ.get("IMAGE_CDN", None)
SERP_API_KEY =  os.environ.get("SERPAPI_API_KEY", None)

class BottomRichSnippetExtensions(BaseModel):
    extensions: List[str] = None
    detected_extensions: Optional[dict] = None

class BottomRichSnippet(BaseModel):
    bottom: BottomRichSnippetExtensions = None
    top: BottomRichSnippetExtensions = None
    left: BottomRichSnippetExtensions = None
    right: BottomRichSnippetExtensions = None

class OrganicResultItem(BaseModel):
    title: str = None
    link: str = None
    redirect_link: str = None
    displayed_link: str = None
    thumbnail: str = None
    favicon: str = None
    snippet: str = None
    snippet_highlighted_words: List[str] = None
    rich_snippet: Optional[BottomRichSnippet] = None
    about_page_link: str = None
    about_page_serpapi_link: str = None
    cached_page_link: str = None
    related_pages_link: str = None
    source: str = None

class RelatedQuestion(BaseModel):
    question: str = None
    snippet: str = None

class PriceMovement(BaseModel):
    price: float = None
    percentage: float = None
    movement: str = None
    date: str = None

class TableRows(BaseModel):
    name: str = None
    value: Any = None

class AnswerBox(BaseModel):
    type: str = None
    title: str = None
    exchange: str = None
    currency: str = None
    price: float = None
    date: str = None
    snippet: str = None
    previous_close: float = None
    price_movement: PriceMovement = None
    table: List[TableRows] = None
    stock: str = None

class ImageResult(BaseModel):
    title: str = None
    snippet: str = None
    link: str = None
    source: str = None

class Price(BaseModel):
    value: str
    extracted_value: float
    currency: str

class VisualMatch(BaseModel):
    position: int = None 
    title: str = None
    link: str = None
    source: str = None
    source_icon: str = None
    rating: float = None
    reviews: int = None
    price: Price = None
    thumbnail: str = None

class AvailableOn(BaseModel):
    link: str = None
    name: str = None
    price : str = None
    avaliable : str = None

class ShoppingResult(BaseModel):
    title : str = None
    price: str = None
    extracted_price : float = None
    link : str = None
    source : str = None
    rating : float = None
    reviews : int = None

class ShoppingResult(BaseModel):
    title : str = None
    price: str = None
    extracted_price : float = None
    link : str = None
    source : str = None
    rating : float = None
    reviews : int = None
    shipping : str = None
    thumbnail: str = None
    extensions : List[str] = []

class InlineProduct(BaseModel):
      title: str = None
      tag: str = None
      source: str = None
      price: float = None
      currency: str = None
      original_price: float = None
      rating: float = None,
      reviews: float = None,
      thumbnail: str = None,
      specifications :Any = None

class Filter(BaseModel):
    title: Optional[str] = None
    link: Optional[str] = None
    serpapi_link: Optional[str] = None

class Job(BaseModel):
    title: Optional[str] = None
    link: Optional[str] = None
    company_name: Optional[str] = None
    location: Optional[str] = None
    via: Optional[str] = None
    extensions: Optional[List[str]] = None
    detected_extensions: Optional[dict] = None

class JobsResults(BaseModel):
    location: Optional[str] = None
    link: Optional[str] = None
    link_text: Optional[str] = None
    serpapi_link: Optional[str] = None
    filters: Optional[List[Filter]] = None
    jobs: Optional[List[Job]] = None

class NewsResult(BaseModel):
    link: str
    title: str
    source: str
    date: str
    snippet: str
    thumbnail: str

class Perspective(BaseModel):
    author: str = None
    source: str = None
    thumbnails: List[str] = None
    title: str = None
    snippet: str = None
    link: str = None
    date: str = None

class Perday(BaseModel):
    time: str = None
    info: str = None
    busyness_score: int = None

class Days(BaseModel):
    monday: List[Perday] = None
    tuesday: List[Perday] = None
    wednesday: List[Perday] = None
    thursday: List[Perday] = None
    friday: List[Perday] = None
    saturday: List[Perday] = None
    sunday: List[Perday] = None

class PopularLive(BaseModel): 
    time: str = None
    info: str = None
    busyness_score: int = None
    typical_time_spent: str = None

class PopularTime(BaseModel):
    live: PopularLive = None
    graph_results: Days = None

class UserReviw(BaseModel):
    summary: str = None
    user: Dict = None

class WebReview(BaseModel):
    company: str = None
    link: str = None
    rating: float = None
    review_count: int = None

class KnowledgeGraph(BaseModel):
    title: str = None
    type: str = None
    image: str = None
    website_link: str = None
    description: str = None
    source: Any = None
    stock_price: str = None
    founders: str = None
    products: str = None
    ceo: str = None
    headquarters: str = None
    susbsidiaries: str = None
    founded : str = None
    rating: float = None
    review_count: int = None
    service_options: List[str]|str = None
    address: str = None
    raw_hours: str = None
    popular_times: PopularTime = None
    merchant_description: str = None
    user_reviews: List[UserReviw] = None
    reviews_from_the_web: List[WebReview] = None

class SearchMetadata(BaseModel):
    id: str = None
    status: str = None
    json_endpoint: str = None
    created_at: str = None
    processed_at: str = None
    google_url: str = None
    raw_html_file: str = None
    total_time_taken: float = None

class GameTeam(BaseModel):
    name: str = None
    thumbnail: str = None

class GameSpotlight(BaseModel):
    stadium: str = None
    date: str = None
    teams: List[GameTeam] = None

class Games(BaseModel):
    tournament: str = None
    stadium: str = None
    date: str = None
    teams: List[GameTeam] = None
    status: str = None

class SportsResult(BaseModel):
    title: str = None
    thumbnail: str = None
    game_spotlight: GameSpotlight = None
    games: List[Games] = None

class GpsCordinates(BaseModel):
    latitude: float = None
    longitude: float = None

class Places(BaseModel):
    title: str = None
    place_id_search: str = None
    rating: Any = None
    phone: str = None
    address: str = None
    hours: str = None
    gps_coordinates: GpsCordinates = None

class LocalResult(BaseModel):
    title : str = None
    more_locations_link : str = None
    places : List[Places] = None

class ReciepeResult(BaseModel):
   
    title: str = None
    source: str = None
    rating: float = None
    reviews: int = None
    total_time: str = None
    ingredients: List[str] = None
    thumbnail: str = None

class TweetAuthor(BaseModel):
    thumbnail: str = None
    twitter_blue: bool = None
    title: str = None
    account: str = None

class Tweets(BaseModel):
    link: str = None
    snippet: str = None
    published_date: str = None
    thumbnail: str = None
    author: TweetAuthor = None

class TwitterResult(BaseModel):
    title: str = None
    link: str = None
    displayed_link: str = None
    tweets: List[Tweets] = None

class VisualStories(BaseModel):
    title: str = None
    source: str = None

class Events(BaseModel):
    title: str = None
    date: str = None
    address: List[str] = None
    link: str = None
    thumbnail: str = None
    
class Destination(BaseModel):
    title: str = None
    link: str = None
    description: str = None
    flight_price: str = None
    extracted_flight_price: float = None
    hotel_price: str = None
    extracted_hotel_price: float = None
    thumbnail: str = None

class Destinations(BaseModel):
    destinations: List[Destination] = None

class QuestionAndAnswer(BaseModel):
    source: str = None
    question: str = None
    answer: str = None
    votes: int = None

class Showing(BaseModel):
    time: List[str] = None
    type: str = None

class Theaters(BaseModel):
    name: str = None
    link: str = None
    distance: str = None
    address: str = None
    showing: List[Showing] = None

class ShowTime(BaseModel):
    date: str = None
    theaters: List[Theaters] = None

class Sights(BaseModel):
    title: str = None
    link: str = None
    description: str = None
    rating: float = None
    reviews: int = None
    thumbnail: str = None

class TopSights(BaseModel):
    sights: List[Destination] = None

class TopStories(BaseModel):
    title: str = None
    link: str = None
    source: str = None
    date: str = None

class ImmersiveSnippet(BaseModel):
    text: str = None

class ImmersiveProducts(BaseModel):
    source: str = None
    title: str = None
    snippets: List[str]|List[ImmersiveSnippet] = None
    price: str = None
    extracted_price: float = None

class InlineImages(BaseModel):
    source: str = None
    thumbnail: str = None
    original: str = None
    title: str = None
    source_name: str = None

class SerpAPIResponse(BaseModel):
    search_metadata: SearchMetadata = None
    organic_results: List[OrganicResultItem] = None
    related_questions: List[RelatedQuestion] = None
    answer_box: AnswerBox = None
    image_results: List[ImageResult] = None
    visual_matches: List[VisualMatch] = None
    available_on: List[AvailableOn] = None
    shopping_results: List[ShoppingResult] = None
    inline_products: List[InlineProduct] = None
    jobs_results: JobsResults = None
    news_results: List[NewsResult] = None
    perspectives: List[Perspective] = None
    knowledge_graph: KnowledgeGraph = None
    sports_results: SportsResult = None
    local_results: LocalResult = None
    recipes_results: List[ReciepeResult] = None
    twitter_results: TwitterResult = None
    visual_stories: VisualStories = None
    event_results: List[Events] = None
    popular_destinations: Destinations = None
    questions_and_answers: List[QuestionAndAnswer] = None
    showtimes: List[ShowTime] = None
    top_sights: TopSights = None
    immersive_products: List[ImmersiveProducts] = None
    inline_images: List[InlineImages] = None
    top_stories: List[TopStories] = None

    def summarise(self, max_search_results: int = 5):
        """
        summarise the search result
        """
        # TODO: add more summarisation and prioritisation logic based on most useful information
        content_lines = []
        # add in try except block to prevent error
        original_count = max_search_results
        try:
            if self.answer_box:
                if self.answer_box.type =="finance_results":
                    summry =f"{self.answer_box.stock}: price:{self.answer_box.currency} {self.answer_box.price} ({self.answer_box.exchange}\n"
                    summry +=f"{self.answer_box.price_movement.movement} {self.answer_box.price_movement.price} ({self.answer_box.price_movement.percentage} %) for {self.answer_box.price_movement.date}\n\n"
                    summry += "\n".join([f"{row.name}:{row.value}" for row in self.answer_box.table])
                    content_lines.append(summry)
                    max_search_results = max_search_results -1
                if "organic_result" in self.answer_box.type:
                    summry =f"{self.answer_box.title}\n" if self.answer_box.title else ""
                    summry += f"{self.answer_box.snippet}\n" if self.answer_box.snippet else ""
                    summry += f"({self.answer_box.date})\n" if self.answer_box.date else ""
                    summry = summry.rstrip("\n")+"\n\n"
                    content_lines.append(summry)
                    max_search_results = max_search_results -1
        except Exception as e: print(e)
        max_search_results = original_count
        try:
            if self.top_stories:
                for res in self.top_stories:
                    if max_search_results>0:
                        summry =f"{res.title}\n"
                        summry +=f"source :{res.source}\n" if res.source else ""
                        summry +=f"date :{res.date}\n" if res.date else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results = original_count
            if self.organic_results:
                for res in self.organic_results:
                    if max_search_results>0:
                        summry =f"{res.title}:\n" if res.title else ""
                        summry = f"{res.snippet}\n" if res.snippet else ""
                        summry += f"{','.join(res.rich_snippet.bottom.extensions)}\n" if res.rich_snippet  and res.rich_snippet.bottom and res.rich_snippet.bottom.extensions  else ""
                        summry += f"{','.join(res.rich_snippet.top.extensions)}\n" if res.rich_snippet  and res.rich_snippet.top and res.rich_snippet.top.extensions  else ""
                        summry += f"{','.join(res.rich_snippet.left.extensions)}\n" if res.rich_snippet  and res.rich_snippet.left and res.rich_snippet.left.extensions  else ""
                        summry += f"{','.join(res.rich_snippet.right.extensions)}\n" if res.rich_snippet  and res.rich_snippet.right and res.rich_snippet.right.extensions  else ""
                        if res.rich_snippet and res.rich_snippet.bottom and res.rich_snippet.bottom.detected_extensions:
                            for k,v in res.rich_snippet.bottom.detected_extensions.items():
                                summry += f"{k}:{v}\n"
                        if res.rich_snippet and res.rich_snippet.top and res.rich_snippet.top.detected_extensions:
                            for k,v in res.rich_snippet.top.detected_extensions.items():
                                summry += f"{k}:{v}\n"
                        if res.rich_snippet and res.rich_snippet.left and res.rich_snippet.left.detected_extensions:
                            for k,v in res.rich_snippet.left.detected_extensions.items():
                                summry += f"{k}:{v}\n"
                        if res.rich_snippet and res.rich_snippet.right and res.rich_snippet.right.detected_extensions:
                            for k,v in res.rich_snippet.right.detected_extensions.items():
                                summry += f"{k}:{v}\n"
                        summry += f"source:{res.source}\n" if res.source else ""
                        
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results = original_count
            if self.image_results:
                for res in self.image_results:
                    if max_search_results>0:
                        summry = f"{res.title}:\n" if res.title else ""
                        summry += f"{res.snippet}\n" if res.snippet else ""
                        summry += f"source:{res.source}\n\n" if res.source else ""
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results = original_count
            if self.inline_images:
                for res in self.inline_images:
                    if max_search_results>0:
                        summry = f"{res.title}:\n" if res.title else ""
                        summry += f"{res.source_name}\n" if res.source_name else ""
                        summry +="source:"+res.source+"\n" if res.source else ""
                        summry += summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results = original_count
            if self.local_results:
                for res in self.local_results.places:
                    if max_search_results>0:
                        summry =f"{res.title}\n"
                        summry +=f"rating :{res.rating}\n" if res.rating else ""
                        summry +=f"phone :{res.phone}\n" if res.phone else ""
                        summry +=f"address :{res.address}\n" if res.address else ""
                        summry +=f"hours :{res.hours}\n" if res.hours else ""
                        summry +=f"gps_coordinates :{res.gps_coordinates}\n" if res.gps_coordinates else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results = original_count
            if self.showtimes:
                for res in self.showtimes:
                    if max_search_results>0:
                        summry =f"{res.date}\n"
                        for theater in res.theaters:
                            summry +=f"{theater.name}\n"
                            summry +=f"{theater.distance}\n" if theater.distance else ""
                            summry +=f"{theater.address}\n" if theater.address else ""
                            for show in theater.showing:
                                summry +=f"{show.time}\n"
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results = original_count
            if self.event_results:
                for res in self.event_results:
                    if max_search_results>0:
                        summry =f"{res.title}\n"
                        summry +=f"date :{res.date}\n" if res.date else ""
                        summry +=f"address :{res.address}\n" if res.address else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.questions_and_answers:
                for res in self.questions_and_answers:
                    if max_search_results>0:
                        summry =f"question :{res.question}\n" if res.question else ""
                        summry +=f"answer :{res.answer}\n" if res.answer else ""
                        summry +=f"votes :{res.votes}\n" if res.votes else ""
                        summry +=f"source :{res.source}\n"
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.twitter_results:
                for res in self.twitter_results.tweets:
                    if max_search_results>0:
                        summry =f"{res.snippet}\n"
                        summry +=f"published_date :{res.published_date}\n" if res.published_date else ""
                        summry +=f"author :{res.author.title}\n" if res.author and res.author.title else ""
                        summry +=f"account :{res.author.account}\n" if res.author and res.author.account else ""
                        summry +=f"verfiied :{res.author.twitter_blue}\n" if res.author and res.author.twitter_blue else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.visual_stories:
                if max_search_results>0:
                    summry =f"{self.visual_stories.title}\n"
                    summry +=f"source :{self.visual_stories.source}\n" if self.visual_stories.source else ""
                    summry = summry.rstrip("\n")+"\n\n"
                    content_lines.append(summry)
                    max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.popular_destinations:
                for res in self.popular_destinations.destinations:
                    if max_search_results>0:
                        summry =f"{res.title}\n"
                        summry +=f"flight_price :{res.flight_price}\n" if res.flight_price else ""
                        summry +=f"hotel_price :{res.hotel_price}\n" if res.hotel_price else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.top_sights:
                for res in self.top_sights.sights:
                    if max_search_results>0:
                        summry =f"{res.title}\n"
                        summry +=f"rating :{res.rating}\n" if res.rating else ""
                        summry +=f"reviews :{res.reviews}\n" if res.reviews else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.recipes_results:
                for res in self.recipes_results:
                    if max_search_results>0:
                        summry =f"{res.title}\nsource:{res.source}\n"
                        summry +=f"rating :{res.rating}\n" if res.rating else ""
                        summry +=f"reviews :{res.reviews}\n" if res.rating else ""
                        summry +=f"total_time :{res.total_time}\n" if res.total_time else ""
                        summry +=f"ingredients :{','.join(res.ingredients)}\n" if res.ingredients else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.perspectives:
                for res in self.perspectives:
                    if max_search_results>0:
                        summry =f"{res.title}\nsource:{res.source}\n"
                        summry +=f"snippet :{res.snippet}\n" if res.snippet else ""
                        summry +=f"link :{res.link}\n" if res.link else ""
                        summry +=f"date :{res.date}\n" if res.date else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.shopping_results:
                for res in self.shopping_results:
                    if max_search_results>0:
                        summry =f"{res.title}\nsource:{res.source}\n"
                        summry +=f"rating :{res.rating}\n" if res.rating else ""
                        summry +=f"reviews :{res.reviews}\n" if res.rating else ""
                        if res.price:
                            summry +=f"{res.price}" if res.price else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.immersive_products:
                for res in self.immersive_products:
                    if max_search_results>0:
                        summry =f"{res.title}\n"if res.title else ""
                        summry +=f"price :{res.price}\n" if res.price else ""
                        summry +=f"snippets :{','.join(res.snippets)}\n" if res.snippets else ""
                        summry +=f"source :{res.source}\n" if res.source else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.sports_results:
                if self.sports_results.game_spotlight:
                    summry =f"{self.sports_results.game_spotlight.stadium}\n"
                    summry +=f"{self.sports_results.game_spotlight.date}\n"
                    summry +=f"{self.sports_results.game_spotlight.teams[0].name} vs {self.sports_results.game_spotlight.teams[1].name}\n"
                    content_lines.append(summry)
                    max_search_results = max_search_results -1
                if self.sports_results.games:
                    for res in self.sports_results.games:
                        if max_search_results>0:
                            summry =f"{res.tournament}\n"
                            summry +=f"{res.stadium}\n" if res.stadium else ""
                            summry +=f"{res.date}\n" if res.date else ""
                            summry +=f"{res.teams[0].name} vs {res.teams[1].name}\n"
                            content_lines.append(summry)
                            max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.inline_products:
                for res in self.inline_products:
                    if max_search_results>0:
                        summry =f"{res.title}\nsource:{res.source}\n"
                        summry +=f"rating :{res.rating}\n" if res.rating else ""
                        summry +=f"reviews :{res.reviews}\n" if res.rating else ""
                        if res.price:
                            summry +=f"{res.price}" if res.price else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.jobs_results:
                for res in self.jobs_results.jobs:
                    if max_search_results>0:
                        summry =f"{res.title}\ncompany:{res.company_name}\n"
                        summry +=f"location :{res.location}\n" if res.location else ""
                        summry +=f"via :{res.via}\n" if res.via else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.news_results:
                for res in self.news_results:
                    if max_search_results>0:
                        summry =f"{res.title}\nsource:{res.source}\n"
                        summry +=f"date :{res.date}\n" if res.date else ""
                        summry +=f"snippet :{res.snippet}\n" if res.snippet else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.available_on:
                for res in self.available_on:
                    if max_search_results>0:
                        summry =f"{res.name}\nsource:{res.avaliable}\n"
                        summry +=f"price :{res.price}\n" if res.price else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try:
            max_search_results   = original_count
            if self.visual_matches:
                for res in self.visual_matches:
                    if max_search_results>0:
                        summry =f"{res.title}\nsource:{res.source}\n"
                        summry +=f"rating :{res.rating}\n" if res.rating else ""
                        summry +=f"reviews :{res.reviews}\n" if res.rating else ""
                        if res.price:
                            summry +=f"{res.price.value}" if res.price.value else ""
                        summry = summry.rstrip("\n")+"\n\n"
                        content_lines.append(summry)
                        max_search_results = max_search_results -1
        except Exception as e: print(e)
        try: 
            max_search_results   = original_count
            if self.knowledge_graph:
                summry =f"{self.knowledge_graph.title}\nsource:{self.knowledge_graph.source}\n"
                summry +=f"description :{self.knowledge_graph.description}\n" if self.knowledge_graph.description else ""
                summry +=f"stock_price :{self.knowledge_graph.stock_price}\n" if self.knowledge_graph.stock_price else ""
                summry +=f"founders :{self.knowledge_graph.founders}\n" if self.knowledge_graph.founders else ""
                summry +=f"products :{self.knowledge_graph.products}\n" if self.knowledge_graph.products else ""
                summry +=f"ceo :{self.knowledge_graph.ceo}\n" if self.knowledge_graph.ceo else ""
                summry +=f"headquarters :{self.knowledge_graph.headquarters}\n" if self.knowledge_graph.headquarters else ""
                summry +=f"susbsidiaries :{self.knowledge_graph.susbsidiaries}\n" if self.knowledge_graph.susbsidiaries else ""
                summry +=f"founded :{self.knowledge_graph.founded}\n" if self.knowledge_graph.founded else ""
                summry +=f"rating :{self.knowledge_graph.rating}\n" if self.knowledge_graph.rating else ""
                summry +=f"review_count :{self.knowledge_graph.review_count}\n" if self.knowledge_graph.review_count else ""
                if isinstance(self.knowledge_graph.service_options, list):
                    summry +=f"service_options :{','.join(self.knowledge_graph.service_options)}\n" if self.knowledge_graph.service_options else ""
                else:
                    summry +=f"service_options :{self.knowledge_graph.service_options}\n" if self.knowledge_graph.service_options else ""
                summry +=f"address :{self.knowledge_graph.address}\n" if self.knowledge_graph.address else ""
                summry +=f"raw_hours :{self.knowledge_graph.raw_hours}\n" if self.knowledge_graph.raw_hours else ""
                summry +=f"merchant_description :{self.knowledge_graph.merchant_description}\n" if self.knowledge_graph.merchant_description else ""
                summry +=f"live time :{self.knowledge_graph.popular_times.live.time}\n" if self.knowledge_graph.popular_times and self.knowledge_graph.popular_times.live else ""
                summry +=f"live info :{self.knowledge_graph.popular_times.live.info}\n" if self.knowledge_graph.popular_times and self.knowledge_graph.popular_times.live else ""
                summry +=f"live busyness_score :{self.knowledge_graph.popular_times.live.busyness_score}\n" if self.knowledge_graph.popular_times and self.knowledge_graph.popular_times.live else ""
                summry = summry.rstrip("\n")+"\n\n"
                content_lines.append(summry)
        except Exception as e:
            print(e)
        content = "\n".join(content_lines) if len(content_lines) > 0 else "No result found"
        return content
    
def SerpAPISearch(query: str, engine: str="google", use_photo: bool = False, image_url: str | None = None, save_to_file: str | None = None, uule: str | None = None) -> SerpAPIResponse:
    client = serpapi.Client(api_key=SERP_API_KEY)
    if use_photo and image_url:
        if str(engine) != "google_lens":
           engine = "google_reverse_image"
        print(f"Using {engine} for image search")
        if engine == "google_lens":
            response_obj = client.search(engine=engine, url=str(image_url).strip("\n"),  hl="en", uule=uule)
        else:
            response_obj = client.search(q=query,engine=engine, image_url=str(image_url).strip("\n"),  hl="en", uule=uule)
    else:
        response_obj = client.search(q=query, engine=engine,  hl="en", uule=uule)

    if save_to_file:
        with open(save_to_file, "w") as f:
            json.dump(response_obj.as_dict(), f, indent=4)
    try:
        resp: SerpAPIResponse = SerpAPIResponse.model_validate(response_obj)
    except Exception as e:
        print(f"Failed to validate response: {e}")
        print(response_obj)
        return "No result found"
    return resp

def upload_image_to_cdn(image_bytes: bytes) -> str | None:
    response = requests.post(TEST_CDN, files={'file': ('file', BytesIO(image_bytes)), 'expires': str(2)})
    if response.status_code != 200:
        print(f"Failed to upload image to CDN: {response.content}")
        return None
    return response.content.decode()

class SerpWebSearch(WebSearch):
    def __init__(self, save_to_file: str | None = None, engine: str = "google", max_search_results: int = 5):
        super().__init__()
        self._save_to_file = save_to_file
        self._engine = engine
        self._max_search_results = max_search_results
    
    async def search_web(self, query: str, use_photo: bool = False, image_bytes: bytes | None = None, location: str | None = None, uule: str | None = None) -> WebSearchResult:
        uule = uule_grabber.uule(location)
        image_url = upload_image_to_cdn(image_bytes=image_bytes) if image_bytes else None
        serp_response = SerpAPISearch(query, engine=self._engine, use_photo=use_photo, image_url=image_url, save_to_file=self._save_to_file, uule=uule)
        return WebSearchResult(summary=serp_response.summarise(max_search_results=self._max_search_results), search_provider_metadata=serp_response.search_metadata.json_endpoint)

WebSearch.register(SerpWebSearch)