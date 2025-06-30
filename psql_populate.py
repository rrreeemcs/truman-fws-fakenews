# Sameer Ramkissoon - PSQL Populate Script
# Recreating the PSQL Database for Future Users to Continue Development 
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os

# Database connection parameters
HOST = "localhost"
PORT = "5432"
USER = "postgres"
PASSWORD = os.environ.get("PGPASSWORD")  # Ensure you set this environment variable before running the script
DB_NAME = "truthtide_ml"

def create_database():
    """Creates a PostgreSQL database for the ML data"""
    # Connecting to PostgreSQL server (initially made to 'postgres' database)
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        print("Connected to PostgreSQL server successfully.")

        # Creating the New Database
        cursor.execute(f"DROP DATABASE IF EXISTS {DB_NAME};")
        cursor.execute(f"CREATE DATABASE {DB_NAME};")
        print(f"Database '{DB_NAME}' created successfully.")

        cursor.close()
        conn.close()

        # Connecting to the newly created database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT
        )
        cursor = conn.cursor()

        # Making tables for the database
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Articles (
                article_id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                description TEXT,
                news_category TEXT,
                image_url TEXT,
                publishedAt TIMESTAMP,
                source_id TEXT NOT NULL,
                news_related BOOLEAN NOT NULL DEFAULT FALSE,
                reliability_score INT DEFAULT 0,
                reliability_label TEXT DEFAULT 'Unknown'
            );
        """)
        print("Table 'Articles' created successfully.")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Posts (
                post_id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                caption TEXT,
                likes INT,
                image_url TEXT
            );
        """)
        conn.commit()
        print("Table 'Posts' created successfully.")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error creating database: {e}")

def add_articles():
    """Adds the articles gathered from news_data.py and web_scrape_functions.py and puts into the Articles table."""
    # Connecting to truthtide_ml database
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT
        )
        cursor = conn.cursor()
        print("Connected to the 'truthtide_ml' database successfully.")

        # Sample data to insert into Articles table
        print("Adding articles to the 'Articles' table...")

        # Use copy_expert to avoid server-side file permission issues
        with open('C:/Users/ramsa/Desktop/NYIT/FWS/truthtide-fws/ml-data/all_articles.csv', 'r', encoding='utf-8') as f:
            cursor.copy_expert("""
                COPY Articles(title, author, description, news_category, image_url, publishedat, source_id, news_related, reliability_score, reliability_label)
                FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')
            """, f)

        conn.commit()
        print("Articles added successfully to the 'Articles' table.")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error adding articles: {e}")

# Run this script to create the database and add articles
if __name__ == "__main__":
    create_database()
    add_articles()
