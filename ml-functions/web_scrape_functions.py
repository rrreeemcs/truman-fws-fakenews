# Sameer Ramkissoon - Web Scraping Functions
# File used to store functions for web scraping news articles (done in scrape_sites.py)
# Websites planned to scrape: theonion.com, babylonbee.com, beforeitsnews.com

import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin

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

# Functions that scrape the articles from the websites
def scrape_theonion(limit = 30):
    """
    Scraping articles from theonion.com.
    Returns a list of dictionaries containing the article details.
    """
    articles = []
    base_url = 'https://www.theonion.com'

    try:
        response = session.get(base_url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Finding the article links on the homepage
        article_links = soup.find_all('a', href=True)
        article_urls = []

        # Goes through all of the links and checks type of article
        # Once we reach the limit, stop adding articles
        for link in article_links:
            href = link.get('href')
            if href and ('/c/' in href or '/news/' in href or '/politics/' in href):
                full_url = urljoin(base_url, href)
                if full_url not in article_urls:
                    article_urls.append(full_url)
                    if len(article_urls) >= limit:
                        break

        # Going through each article URL and scraping the details
        for url in article_urls[:limit]:
            try:
                article_response = session.get(url, timeout=10)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')

                title = grab_title(article_soup)
                author = grab_author(article_soup)
                description = grab_description(article_soup)
                image_url = get_news_image(article_soup, base_url)

                articles.append({
                    'title': title,
                    'author': author,
                    'description': description,
                    'news_category': 'satire',  # The Onion is a satire site
                    'urlToImage': image_url,
                    'publishedAt': None,  # The Onion does not provide a published date
                    'source_id': 'theonion',
                    'news_related': True
                })
                time.sleep(1)
            except Exception as e:
                print(f"Error scraping article {url}: {e}")
                continue
    except Exception as e:
        print(f"Error scraping The Onion: {e}")
    return articles

def scrape_babylonbee(limit = 30):
    """
    Scraping articles from babylonbee.com.
    Returns a list of dictionaries containing the article details.
    """
    articles = []
    base_url = 'https://www.babylonbee.com'

    try:
        response = session.get(base_url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Finding the article links on the homepage
        article_links = soup.find_all('a', href=True)
        article_urls = []

        # Goes through all of the links and checks type of article
        # Once we reach the limit, stop adding articles
        for link in article_links:
            href = link.get('href')
            if href and ('/news/' in href or '/politics/' in href):
                full_url = urljoin(base_url, href)
                if full_url not in article_urls:
                    article_urls.append(full_url)
                    if len(article_urls) >= limit:
                        break

        # Going through each article URL and scraping the details
        for url in article_urls[:limit]:
            try:
                article_response = session.get(url, timeout=10)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')

                title = grab_title(article_soup)
                author = grab_author(article_soup)
                description = grab_description(article_soup)
                image_url = get_news_image(article_soup, base_url)

                articles.append({
                    'title': title,
                    'author': author,
                    'description': description,
                    'news_category': 'satire',  # BabylonBee is a satire site
                    'urlToImage': image_url,
                    'publishedAt': None,  # BabylonBee does not provide a published date
                    'source_id': 'babylonbee',
                    'news_related': True
                })
                time.sleep(1)
            except Exception as e:
                print(f"Error scraping article {url}: {e}")
                continue
    except Exception as e:
        print(f"Error scraping BabylonBee: {e}")
    return articles

def scrape_beforeitsnews(limit = 30):
    """
    Scraping articles from beforeitsnews.com.
    Returns a list of dictionaries containing the article details.
    """
    articles = []
    base_url = 'https://www.beforeitsnews.com'

    try:
        response = session.get(base_url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Finding the article links on the homepage
        article_links = soup.find_all('a', href=True)
        article_urls = []

        # Goes through all of the links and checks type of article
        # Once we reach the limit, stop adding articles
        for link in article_links:
            href = link.get('href')
            if href and ('/story/' in href or '/politics/' in href):
                full_url = urljoin(base_url, href)
                if full_url not in article_urls:
                    article_urls.append(full_url)
                    if len(article_urls) >= limit:
                        break

        # Going through each article URL and scraping the details
        for url in article_urls[:limit]:
            try:
                article_response = session.get(url, timeout=10)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')

                title = grab_title(article_soup)
                author = grab_author(article_soup)
                description = grab_description(article_soup)
                image_url = get_news_image(article_soup, base_url)

                articles.append({
                    'title': title,
                    'author': author,
                    'description': description,
                    'news_category': 'satire',  # Before Its News is a satire site
                    'urlToImage': image_url,
                    'publishedAt': None,  # Before Its News does not provide a published date
                    'source_id': 'beforeitsnews',
                    'news_related': True
                })
                time.sleep(1)
            except Exception as e:
                print(f"Error scraping article {url}: {e}")
                continue
    except Exception as e:
        print(f"Error scraping BeforeItsNews: {e}")
    return articles

# Executing scraping fucntions
if __name__ == "__main__":
    all_articles = []
    print("Scraping articles...")

    # The Onion
    print("Scraping The Onion...")
    onion_articles = scrape_theonion(limit=30)
    all_articles.extend(onion_articles)
    print(f"Scraped {len(onion_articles)} articles from The Onion.")

    # Babylon Bee
    print("Scraping Babylon Bee...")
    bee_articles = scrape_babylonbee(limit=30)
    all_articles.extend(bee_articles)
    print(f"Scraped {len(bee_articles)} articles from Babylon Bee.")

    # Before Its News
    print("Scraping Before Its News...")
    before_articles = scrape_beforeitsnews(limit=30)
    all_articles.extend(before_articles)
    print(f"Scraped {len(before_articles)} articles from Before Its News.")

    # Saving the articles to a JSON file
    with open('scraped_articles.json', 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=4)
    print("Articles saved to scraped_articles.json")