# Sameer Ramkissoon - PSQL Populate Script
# Recreating the PSQL Database for Future Users to Continue Development 
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database():
    """Creates a PostgreSQL database for the ML data"""

    # Database connection parameters
    HOST = "localhost"
    PORT = "5432"
    USER = "postgres"
    PASSWORD = input("ENTER YOUR POSTGRES PASSWORD: ")
    DB_NAME = "truthtide_ml"

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
                author TEXT NOT NULL,
                description TEXT NOT NULL,
                news_category TEXT,
                image_url TEXT,
                publishedAt TIMESTAMP,
                source_id TEXT NOT NULL
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

if __name__ == "__main__":
    create_database()
    print("Database setup complete. You can now use the 'truthtide_ml' database.")
