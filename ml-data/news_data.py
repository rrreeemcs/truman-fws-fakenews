# Using newsapi.org to fetch news data on different topics
NEWS_API_KEY = "26451b16ac234ef4870541553c70048f"

from newsapi import NewsApiClient
import json

newsapi = NewsApiClient(api_key=NEWS_API_KEY)

top_headlines = newsapi.get_top_headlines(
    q="technology",
    category="technology",
    sources="bbc-news,cnn",
    language="en",
)

sources = newsapi.get_sources()
print("Sources:", json.dumps(sources, indent=2))
