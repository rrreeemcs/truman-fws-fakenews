# Sameer Ramkissoon - Web Scraping Script
# File used to store functions for web scraping news articles (done in scrape_sites.py)
# Websites planned to scrape: theonion.com, babylonbee.com, beforeitsnews.com

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin
from datetime import datetime

# Headers for the web scraping -> to avoid being blocked by the website
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
session = requests.Session()
session.headers.update(headers)

# Helper functions to extract the same fields as News API (title, author, description, news_category, urlToImage, publishedAt, source_id)
def grab_title(soup):
    """
    Extracting the article title from the websites.
    Title selectors are a list of common HTML tags used for titles.
    If none of the selectors match, it returns 'No Title'.
    """
    title_selectors = ['h1', '.entry-title', '.post-title', '.article-title', 'title']

    for selector in title_selectors:
        title = soup.select_one(selector)
        if title:
            return title.get_text(strip=True)
        
    return 'No Title'

def grab_author(soup):
    """
    Extracting the article author from the websites.
    Author selectors are a list of common HTML tags used for authors.
    If none of the selectors match, it returns 'No Author'.
    """
    author_selectors = ['.author', '.byline', '.post-author', '.article-author']

    for selector in author_selectors:
        author = soup.select_one(selector)
        if author:
            return author.get_text(strip=True)
        
    return 'No Author'

def grab_description(soup):
    """
    Extracting the article description from the websites.
    Description selectors are a list of common HTML tags used for descriptions.
    If none of the selectors match, it returns 'No Description'.
    """
    description_selectors = ['meta[name="description"]', 'meta[property="og:description"]', '.entry-content p', '.post-content p', '.article-content p', '.excerpt', '.post-excerpt', '.entry-summary', 'p']

    for selector in description_selectors:
        if selector.startswith('meta'):
            # For meta tags, we need to get the content attribute
            description_element = soup.select_one(selector)
            if description_element:
                return description_element.get('content', '').strip()
        else:
            description = soup.select_one(selector)
            if description:
                desc = description.get_text(strip=True)
                if len(desc) > 50:
                    return desc[:100] + "..." if len(desc) > 100 else desc
        
    return 'No Description'

def get_news_image(soup, base_url):
    """
    Extracting the article image URL from the websites.
    Image selectors are a list of common HTML tags used for images.
    If none of the selectors match, it returns 'No Image'.
    """
    image_selectors = ['meta[property="og:image"]', 'meta[name="twitter:image"]', 'img', '.featured-image img', '.post-thumnail img', '.article-image img', '.entry-image img']

    for selector in image_selectors:
        if selector.startswith('meta'):
            # For meta tags, we need to get the content attribute
            image_element = soup.select_one(selector)
            if image_element:
                img_url = image_element.get('content', '').strip()
                if img_url:
                    return urljoin(base_url, img_url)
        else:
            image = soup.select_one(selector)
            if image and 'src' in image.attrs:
                return urljoin(base_url, image['src'])
        
    return 'No Image'