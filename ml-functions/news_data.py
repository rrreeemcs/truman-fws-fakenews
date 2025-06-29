# Sameer Ramkissoon - News Data Fetching Script
# Using newsapi.org to fetch (currently) 20 different news articles on various topics

import requests
import pprint
import pandas as pd
import os

NEWS_API_KEY = "26451b16ac234ef4870541553c70048f"
base_url = "https://newsapi.org/v2/everything"

# Topics can be adjusted depending on the focus of the news articles
list_topics = ["business", "politics", "entertainment", "technology"]

# Function to Fetch News Articles
def get_news_articles(topics):
    """Goes through each topic and uses the News API to fetch articles.
    Looks for everything in the US for each topic (gets 20 for each).
    Performs a GET request to the News API for the following fields:
    - Title
    - Author
    - Description
    - Image URL
    - Published At
    - Content

    After fetching the articles, it creates a dictionary for each and stores in all_articles for proper storage as DataFrame.
    """

    all_articles = []
    for topic in topics:
        params = {
            "apiKey": NEWS_API_KEY,
            "q": topic,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 20,
        }
        
        response = requests.get(base_url, params=params)
        
        # Check if the request was successful
        print(f"Fetching news for topic: {topic.upper()}...")
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles")
            
            # Making new dictionaries for each article then appending to all_articles
            for article in articles:
                new_dict = {}
                new_dict['title'] = article.get('title', 'No Title')
                new_dict['author'] = article.get('author', 'No Author')
                new_dict['description'] = article.get('description', 'No Description')
                new_dict['news_category'] = topic
                new_dict['urlToImage'] = article.get('urlToImage', 'No Image URL')
                new_dict['publishedAt'] = article.get('publishedAt', 'No Published Date')
                new_dict['source_id'] = article['source'].get('name', 'No Source ID')
                new_dict['news_related'] = 1

                all_articles.append(new_dict)
        # If the request was not successful, print the status code
        else:
            print(f"Failed to fetch news for topic: {topic}, Status Code: {response.status_code}")

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(all_articles)

if __name__ == "__main__":
    print("Fetching news articles...")
    new_df = get_news_articles(list_topics)
    print("Saving news articles to 'news_articles.csv'...")

    # Always save relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', 'ml-data', 'news_articles.csv')
    new_df.to_csv(output_path, index=False)
    print(f"News articles saved to '{output_path}'.")