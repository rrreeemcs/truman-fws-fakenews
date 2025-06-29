# Sameer Ramkissoon - Babylon Bee Scraping Script
# Scraping data from Babylon Bee homepage for articles (gets around 20-30)

import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import re
import time

def scrape_babylon_bee():
    """
    Main function used to scrape Babylon Bee homepage for articles.
    Gathers the same fields as NewsAPI (if possible): title, author, description, news_category, urlToImage, publishedAt, source_id, news_related. 
    """
    
    # Headers to mimic a real browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    url = 'https://babylonbee.com'
    
    # Check if the URL is reachable
    try:
        print(f"Fetching {url}...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []
        
        # Find article containers - The Babylon Bee uses various selectors
        article_selectors = [
            'article',
            '.post',
            '.entry',
            '.story',
            '.bb-story',
            '.post-item',
            '.entry-item',
            '.story-item',
            '[class*="story"]',
            '[class*="post"]',
            '.content-item',
            '.feed-item'
        ]
        
        found_articles = []
        
        # Using the list of article selectors to try and find articles on the homepage
        for selector in article_selectors:
            elements = soup.select(selector)
            if elements:
                found_articles.extend(elements)
                print(f"Found {len(elements)} articles with selector: {selector}")
        
        # Using additional logic to find article links in headlines ('a' and href attributes)
        headline_links = soup.find_all('a', href=True)
        article_links = []
        
        for link in headline_links:
            href = link.get('href', '')
            # Look for article URLs (typically contain year/month or article path)
            if (('babylonbee.com' in href or href.startswith('/')) and 
                any(indicator in href.lower() for indicator in ['/news/', '/202', '/article', '/story']) and
                len(link.get_text(strip=True)) > 10):
                article_links.append(link)
        
        # With article elements and links, we need to ensure uniqueness hence using a set()
        unique_articles = []
        seen_links = set()
        
        # Adding the found articles to the unique_articles list
        for article in found_articles:
            link_elem = article.find('a', href=True)
            if link_elem and link_elem['href'] not in seen_links:
                unique_articles.append(article)
                seen_links.add(link_elem['href'])
        
        # Also accounting for pseudo articles (linked without all of the article elements)
        for link in article_links:
            if link['href'] not in seen_links:
                # Create a pseudo-article element from the link
                pseudo_article = soup.new_tag('div')
                pseudo_article.append(link)
                unique_articles.append(pseudo_article)
                seen_links.add(link['href'])
        
        print(f"Processing {len(unique_articles)} unique articles...")
        
        # Getting the title of each article and appending the data of the article to the articles list
        for article in unique_articles:
            try:
                article_data = extract_article_data(article, headers)
                if article_data:
                    articles.append(article_data)
                    print(f"Extracted: {article_data['title'][:50]}...")
                    
                # Small delay between extractions
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def extract_article_data(article_elem, headers):
    """
    Function used to extract the desired information from the articles.
    """
    
    # Initialize article data with default values
    article_data = {
        'title': '',
        'author': '',
        'description': '',
        'news_category': 'politics',
        'urlToImage': '',
        'publishedAt': '',
        'source_id': 'BabylonBee',
        'news_related': 1
    }
    
    # Extract title
    title_selectors = [
        'h1', 'h2', 'h3', 'h4',
        '.headline', '.title', '.entry-title', '.post-title',
        '.story-title', '.bb-title', '.article-title',
        'a[href*="/news/"]', 'a[href*="/202"]'
    ]
    
    for selector in title_selectors:
        title_elem = article_elem.select_one(selector)
        if title_elem:
            title_text = title_elem.get_text(strip=True)
            if len(title_text) > 5:  # Ensure it's substantial
                article_data['title'] = title_text
                break
    
    # If no title found, try finding it in link text
    if not article_data['title']:
        link_elem = article_elem.find('a', href=True)
        if link_elem:
            link_text = link_elem.get_text(strip=True)
            if len(link_text) > 5:
                article_data['title'] = link_text
    
    # Extract author
    author_selectors = [
        '.author', '.byline', '.post-author', '.entry-author',
        '.story-author', '.bb-author', '.article-author',
        '[class*="author"]', '[class*="byline"]',
        '.meta .author', '.post-meta .author'
    ]
    
    for selector in author_selectors:
        author_elem = article_elem.select_one(selector)
        if author_elem:
            author_text = author_elem.get_text(strip=True)
            # Clean up author text (remove "By" prefix if present) using regex expressions
            author_text = re.sub(r'^(by|author:?)\s*', '', author_text, flags=re.IGNORECASE)
            if author_text:
                article_data['author'] = author_text
                break
    
    # Extract description/excerpt
    desc_selectors = [
        '.excerpt', '.summary', '.description', '.entry-summary',
        '.post-excerpt', '.story-excerpt', '.bb-excerpt',
        '.content', '.entry-content', '.post-content',
        'p', '.meta-description'
    ]
    
    for selector in desc_selectors:
        desc_elem = article_elem.select_one(selector)
        if desc_elem:
            desc_text = desc_elem.get_text(strip=True)
            # Do not use anything too short or similar to metadata (social media links)
            if len(desc_text) > 30 and not any(skip_word in desc_text.lower() 
                                             for skip_word in ['share', 'tweet', 'facebook', 'subscribe']):
                article_data['description'] = desc_text[:200] + '...' if len(desc_text) > 200 else desc_text
                break
    
    # Extract image URL
    img_selectors = ['img', '.featured-image img', '.post-image img', '.story-image img']
    
    for selector in img_selectors:
        img_elem = article_elem.select_one(selector)
        if img_elem:
            img_src = (img_elem.get('src') or 
                      img_elem.get('data-src') or 
                      img_elem.get('data-lazy-src') or
                      img_elem.get('data-original'))
            
            if img_src and not any(skip in img_src.lower() for skip in ['logo', 'icon', 'avatar']):
                # Creating the proper URL if it is relative
                if img_src.startswith('//'):
                    img_src = 'https:' + img_src
                elif img_src.startswith('/'):
                    img_src = 'https://babylonbee.com' + img_src
                article_data['urlToImage'] = img_src
                break
    
    # Extract publication date
    date_selectors = [
        'time', '.date', '.published', '.post-date', '.entry-date',
        '.story-date', '.bb-date', '.article-date',
        '[datetime]', '[class*="date"]', '.meta .date',
        '.post-meta .date', '.timestamp'
    ]
    
    for selector in date_selectors:
        date_elem = article_elem.select_one(selector)
        if date_elem:
            # Try to get datetime attribute first
            date_str = (date_elem.get('datetime') or 
                       date_elem.get('title') or 
                       date_elem.get('data-date') or
                       date_elem.get_text(strip=True))
            
            if date_str:
                parsed_date = parse_date(date_str)
                if parsed_date:
                    article_data['publishedAt'] = parsed_date
                    break
    
    # Another chance to grab data from the article link if no date was found previously
    if not article_data['publishedAt']:
        link_elem = article_elem.find('a', href=True)
        if link_elem:
            url = link_elem['href']
            # Try to extract date from URL pattern like /2024/06/article-name
            date_match = re.search(r'/(\d{4})/(\d{2})', url)
            if date_match:
                year, month = date_match.groups()
                try:
                    dt = datetime(int(year), int(month), 1)
                    article_data['publishedAt'] = dt.isoformat()
                except:
                    article_data['publishedAt'] = datetime.now().isoformat()
            else:
                article_data['publishedAt'] = datetime.now().isoformat()
        else:
            article_data['publishedAt'] = datetime.now().isoformat()
    
    # If we don't have a title, skip this article
    if not article_data['title'] or len(article_data['title']) < 5:
        return None
    
    return article_data

def parse_date(date_str):
    """
    Cleaning up the date to properly match how we have it in the database (ISO Format).
    """
    try:
        # Clean the date string
        date_str = re.sub(r'[^\w\s\-:./,]', '', str(date_str)).strip()
        
        # Common date patterns
        patterns = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%B %d, %Y',
            '%b %d, %Y',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%B %d %Y',
            '%b %d %Y'
        ]
        
        for pattern in patterns:
            try:
                dt = datetime.strptime(date_str, pattern)
                return dt.isoformat()
            except ValueError:
                continue
        
        # Try parsing relative dates like "2 hours ago"
        if 'ago' in date_str.lower():
            return datetime.now().isoformat()
        
        return None
        
    except Exception:
        return None

def save_to_csv(articles, filename='../../ml-data/pre-processed/babylon_articles.csv'):
    """
    Save articles to CSV file located in ml-data/pre-processed directory.
    """
    try:
        if not articles:
            print("No articles to save")
            return
        
        # Define the CSV column headers
        fieldnames = [
            'title', 'author', 'description', 'news_category', 
            'urlToImage', 'publishedAt', 'source_id', 'news_related'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write the header row
            writer.writeheader()
            
            # Write each article as a row
            for article in articles:
                # Ensure all required fields are present
                row = {}
                for field in fieldnames:
                    row[field] = article.get(field, '')
                writer.writerow(row)
        
        print(f"Saved {len(articles)} articles to {filename}")
        
    except Exception as e:
        print(f"Error saving to CSV: {e}")

# Main execution
if __name__ == "__main__":
    print("Starting The Babylon Bee Scraper...")
    
    # Scrape articles
    articles = scrape_babylon_bee()
    
    if articles:
        # Save to the designated CSV File
        save_to_csv(articles)
        
        # Show number of articles scraped
        print(f"\nSuccessfully scraped {len(articles)} articles.")
        print("Data saved to 'ml-data/pre-processed/babylon_articles.csv'") 
    else:
        print("No articles were scraped. The website structure may have changed.")