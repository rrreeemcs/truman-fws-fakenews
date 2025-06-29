# Sameer Ramkissoon - Calculating Reliability of News Articles
# Calculating the reliability of news articles based on various factors (news_articles.csv)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

# Reading the news articles data
filename = input("Enter the path to the news articles CSV file: ")
df = pd.read_csv(f'../ml-data/pre-processed/{filename}.csv')

# Calculating the reliability of each news article
def calculate_reliability(row):
    """
    Calculates a reliability score (0-100) based on the following criteria:
    - Source credibility (most weight)
    - Author presence in the article
    - Age of the article
    - Description
    - URL Structure for Image
    The numbered scores are assigned manually. Using the Hugging Face model created some severe issues when it came to simulating a pipeline for reliability scores.
    """
    score = 0

    # (1) Checking Source - contains all of the sources in the dataset
    source_scores = {
        # Tier 1: Highest Credibility
        'BBC News': 45,
        'NPR': 40,
        'The Atlantic': 39,
        'Harvard Business Review': 36,
        'Time': 35,
        
        # Tier 2: Very Credible Sources
        'Wired': 34,
        'The Verge': 32,
        'Business Insider': 30,
        'MacRumors': 29,
        'VentureBeat': 28,
        
        # Tier 3: Good, but not amazing/top-tier
        'Gizmodo.com': 25,
        'Android Central': 24,
        'Kotaku': 22,
        'Rolling Stone': 21,
        'ABC News': 20,
        
        # Tier 4: More blog heavy
        'Slate Magazine': 18,
        'Seths.blog': 15,
        'Wordpress.com': 12,
        'Hackaday': 10,
        
        # Tier 5: Not as established
        'Slashdot.org': 14,
        'Yahoo Entertainment': 8,
        'BabylonBee': 7,
        'TheOnion': 6,
        '': 5  # Missing source
    }

    source = row['source_id']
    score += source_scores.get(source, 10)  # Default score if source not found

    # (2) Author Presence In Article (more authors the better)
    author = str(row['author'])
    # No Author Listed
    if pd.isna(row['author']) or author.strip() == '' or author == 'nan':
        score += 2
    # Author is an email address as opposed to a name
    elif '@' in author or 'gmail.com' in author or 'yahoo.com' in author or 'hotmail.com' in author:
        score += 6
    # Author contains multiple names (e.g. "John Doe, Jane Smith")
    elif ',' in author or 'and' in author:
        score += 15
    # Author is a single name (e.g. "John Doe")
    else:
        score += 10

    # (3) Age of the Article (how recent is the article)
    try:
        published_date = pd.to_datetime(row['publishedAt'])
        days_old = (datetime.now() - published_date).days
        # Very recent articles (0-7 days old)
        if days_old <= 7:
            score += 20
        # Recent articles (8-30 days old)
        elif days_old <= 30:
            score += 15
        # Moderately old articles (31-90 days old)
        elif days_old <= 90:
            score += 10
        # Old articles (90+ days old)
        else:
            score += 5
    # Invalid date format or missing date
    except:
        score += 3

    # (4) Description of the Article
    description = str(row['description'])
    if pd.isna(row['description']) or description.strip() == '' or description == 'nan':
        score += 2
    else:
        # <50 = too short, >200 = too long, 50-200 = just right
        desc_length = len(description)
        if desc_length < 50:
            score += 5
        elif desc_length > 200:
            score += 9
        else:
            score += 14

        # Any clickbait words in the description deducts points
        clickbait_words = ['shocking', 'unbelievable', 'you won\'t believe', 'must see', 'incredible', 'amazing', 'this will blow your mind', 'try this now']
        if any(phrase in description.lower() for phrase in clickbait_words):
            score -= 7

    # (5) URL Structure of Image
    image_url = str(row['urlToImage'])
    # No image associated means descrease in score
    if pd.isna(row['urlToImage']) or image_url.strip() == '' or image_url == 'nan':
        score -= 10
    # Professional image hosting services
    elif 'cdn' in image_url or 'imgur' in image_url or 'static' in image_url or 'media' in image_url:
        score += 15
    else:
        score += 7

    # Returning final score (between 0 and 100)
    return min(max(score, 0), 100)

# Applying this to each row in the DataFrame
df['reliability_score'] = df.apply(calculate_reliability, axis=1)

# Creating reliability labels based on the score
def reliability_label(score):
    """Assigns a reliability label based on the score."""
    if score >= 90:
        return 'Highly Reliable'
    elif score >= 75:
        return 'Reliable'
    elif score >= 50:
        return 'Moderately Reliable'
    elif score >= 30:
        return 'Low Reliability'
    else:
        return 'Unreliable'
df['reliability_label'] = df['reliability_score'].apply(reliability_label)

df.to_csv(f'../ml-data/reliabile-score-version/{filename}_with_reliability.csv', index=False)
print(f"Reliability scores calculated and saved to '/ml-data/reliabile-score-version/{filename}_with_reliability.csv'.")