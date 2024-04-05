#
# async_serpapi_client.py
#
# An asynchronous version of the SerpAPI client built on aiohttp. This is based on the serpapi
# package's Client class. If we import a newer version of the pacakage with substantial changes to
# the API, we will need to update this async client.
#

import aiohttp

from serpapi.__version__ import __version__
from serpapi import SerpResults


class AsyncSerpAPIClient:
    BASE_DOMAIN = "https://serpapi.com"
    USER_AGENT = f"serpapi-python, v{__version__}"
    DASHBOARD_URL = "https://serpapi.com/dashboard"

    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        self._api_key = api_key
        self._session = session

    def __del__(self):
        self._session.detach()

    def __repr__(self):
        return "<Async SerpApi Client>"

    async def search(self, params: dict = None, **kwargs) -> SerpResults | str:
        """Fetch a page of results from SerpApi. Returns a :class:`SerpResults <serpapi.client.SerpResults>` object, or unicode text (*e.g.* if ``'output': 'html'`` was passed).

        The following three calls are equivalent:

        .. code-block:: python

            >>> s = serpapi.search(q="Coffee", location="Austin, Texas, United States")

        .. code-block:: python

            >>> params = {"q": "Coffee", "location": "Austin, Texas, United States"}
            >>> s = serpapi.search(**params)

        .. code-block:: python

            >>> params = {"q": "Coffee", "location": "Austin, Texas, United States"}
            >>> s = serpapi.search(params)


        :param q: typically, this is the parameter for the search engine query.
        :param engine: the search engine to use. Defaults to ``google``.
        :param output: the output format desired (``html`` or ``json``). Defaults to ``json``.
        :param api_key: the API Key to use for SerpApi.com.
        :param **: any additional parameters to pass to the API.


        **Learn more**: https://serpapi.com/search-api
        """
        path = "/search"
        assert_200 = True

        if params is None:
            params = {}

        if kwargs:
            params.update(kwargs)
        
        # Inject the API Key into the params.
        if "api_key" not in params:
            params["api_key"] = self._api_key

        # Build the URL, as needed
        if not path.startswith("http"):
            url = self.BASE_DOMAIN + path
        else:
            url = path

        # Make the HTTP request.       
        headers = {"User-Agent": self.USER_AGENT}

        # Perform GET
        async with self._session.get(url=url, params=params, headers=headers) as response:
            if assert_200:
                response.raise_for_status()
            return await self._serp_results_from_json(response=response)
    
    @staticmethod
    async def _serp_results_from_json(response: aiohttp.ClientResponse):
        """Construct a SerpResults object from an HTTP response.

        :param assert_200: if ``True`` (default), raise an exception if the status code is not 200.
        :param client: the Client instance which was used to send this request.

        An instance of this class is returned if the response is a valid JSON object.
        Otherwise, the raw text (as a properly decoded unicode string) is returned.
        """

        try:
            return SerpResults(data=await response.json(), client=None)
        except ValueError:
            # If the response is not JSON, return the raw text.
            return await response.text()