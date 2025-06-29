import requests
from bs4 import BeautifulSoup
import json
import csv
from datetime import datetime
import re
import time

def scrape_onion_politics():
    """
    Scrape The Onion politics section for article data
    """
    
    # Headers to mimic a real browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    url = 'https://theonion.com/politics'
    
    try:
        print(f"Fetching {url}...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []
        
        # Find article containers - The Onion uses various selectors
        article_selectors = [
            'article',
            '.post-item',
            '.entry-item',
            '.sc-1out364-0',  # Common class pattern for The Onion
            '[data-module="PostList"] > div',
            '.js_post_item'
        ]
        
        found_articles = []
        
        # Try different selectors to find articles
        for selector in article_selectors:
            elements = soup.select(selector)
            if elements:
                found_articles.extend(elements)
                print(f"Found {len(elements)} articles with selector: {selector}")
        
        # Remove duplicates by converting to set using article links
        unique_articles = []
        seen_links = set()
        
        for article in found_articles:
            link_elem = article.find('a', href=True)
            if link_elem and link_elem['href'] not in seen_links:
                unique_articles.append(article)
                seen_links.add(link_elem['href'])
        
        print(f"Processing {len(unique_articles)} unique articles...")
        
        for article in unique_articles:
            try:
                article_data = extract_article_data(article, headers)
                if article_data:
                    articles.append(article_data)
                    print(f"Extracted: {article_data['title'][:50]}...")
                    
                # Be respectful - add small delay
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
    Extract data from individual article element
    """
    
    # Initialize article data with default values
    article_data = {
        'title': '',
        'author': '',
        'description': '',
        'news_category': 'politics',
        'urlToImage': '',
        'publishedAt': '',
        'source_id': 'TheOnion',
        'news_related': 1
    }
    
    # Extract title
    title_selectors = ['h1', 'h2', 'h3', '.headline', '.entry-title', '.post-title']
    for selector in title_selectors:
        title_elem = article_elem.select_one(selector)
        if title_elem:
            article_data['title'] = title_elem.get_text(strip=True)
            break
    
    # If no title found, try finding it in link text
    if not article_data['title']:
        link_elem = article_elem.find('a')
        if link_elem:
            article_data['title'] = link_elem.get_text(strip=True)
    
    # Extract author
    author_selectors = [
        '.author', '.byline', '.post-author', '.entry-author',
        '[data-module="Byline"]', '.sc-1ef7kkm-0'
    ]
    for selector in author_selectors:
        author_elem = article_elem.select_one(selector)
        if author_elem:
            article_data['author'] = author_elem.get_text(strip=True)
            break
    
    # Extract description/excerpt
    desc_selectors = [
        '.excerpt', '.summary', '.description', '.entry-summary',
        '.post-excerpt', 'p', '.sc-1out364-1'
    ]
    for selector in desc_selectors:
        desc_elem = article_elem.select_one(selector)
        if desc_elem:
            desc_text = desc_elem.get_text(strip=True)
            if len(desc_text) > 20:  # Ensure it's substantial content
                article_data['description'] = desc_text[:200] + '...' if len(desc_text) > 200 else desc_text
                break
    
    # Extract image URL
    img_elem = article_elem.find('img')
    if img_elem:
        img_src = img_elem.get('src') or img_elem.get('data-src') or img_elem.get('data-lazy-src')
        if img_src:
            # Handle relative URLs
            if img_src.startswith('//'):
                img_src = 'https:' + img_src
            elif img_src.startswith('/'):
                img_src = 'https://theonion.com' + img_src
            article_data['urlToImage'] = img_src
    
    # Extract publication date
    date_selectors = [
        'time', '.date', '.published', '.post-date', '.entry-date',
        '[datetime]', '.sc-1ef7kkm-1'
    ]
    for selector in date_selectors:
        date_elem = article_elem.select_one(selector)
        if date_elem:
            # Try to get datetime attribute first
            date_str = date_elem.get('datetime') or date_elem.get('title') or date_elem.get_text(strip=True)
            if date_str:
                article_data['publishedAt'] = parse_date(date_str)
                break
    
    # If we don't have a title, skip this article
    if not article_data['title']:
        return None
    
    return article_data

def parse_date(date_str):
    """
    Parse various date formats and return ISO format
    """
    try:
        # Common date patterns
        patterns = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%B %d, %Y',
            '%b %d, %Y',
            '%m/%d/%Y',
            '%d/%m/%Y'
        ]
        
        # Clean the date string
        date_str = re.sub(r'[^\w\s\-:./,]', '', date_str).strip()
        
        for pattern in patterns:
            try:
                dt = datetime.strptime(date_str, pattern)
                return dt.isoformat()
            except ValueError:
                continue
        
        # If no pattern matches, return current date
        return datetime.now().isoformat()
        
    except Exception:
        return datetime.now().isoformat()

def save_to_csv(articles, filename='../ml-data/onion_articles.csv'):
    """
    Save articles to CSV file
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

def print_articles(articles):
    """
    Print articles in a readable format
    """
    print(f"\n{'='*60}")
    print(f"SCRAPED {len(articles)} ARTICLES FROM THE ONION POLITICS")
    print(f"{'='*60}")
    
    for i, article in enumerate(articles, 1):
        print(f"\n--- Article {i} ---")
        print(f"Title: {article['title']}")
        print(f"Author: {article['author']}")
        print(f"Description: {article['description'][:100]}...")
        print(f"Category: {article['news_category']}")
        print(f"Image URL: {article['urlToImage']}")
        print(f"Published: {article['publishedAt']}")
        print(f"Source: {article['source_id']}")
        print(f"News Related: {article['news_related']}")

# Main execution
if __name__ == "__main__":
    print("Starting The Onion Politics Scraper...")
    
    # Scrape articles
    articles = scrape_onion_politics()
    
    if articles:
        # Display results
        print_articles(articles)
        
        # Save to CSV file
        save_to_csv(articles)
        
        # Print summary
        print(f"\n✅ Successfully scraped {len(articles)} articles!")
        print("Data saved to 'onion_articles.csv'")
        
    else:
        print("❌ No articles were scraped. The website structure may have changed.")
        print("You may need to inspect the HTML and update the selectors.")