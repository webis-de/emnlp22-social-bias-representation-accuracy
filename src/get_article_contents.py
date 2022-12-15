import cdx_toolkit
import json
import logging
import pandas as pd
import sqlite3

from datetime import datetime
from multiprocessing import Pool
from newsplease import NewsPlease
from os import path
from pathlib import Path
from sqlite3 import IntegrityError

from embedding_bias.config import LOGGING_CONFIG_FILE, NEWS_ARTICLE_DB_NAME

PARENT_DIR = Path(__file__).parent.parent
DATA_DIR = path.join(PARENT_DIR, "data")
LOG_DIR = path.join(PARENT_DIR, "logs")
URL_FILE_DIR = path.join(DATA_DIR, "raw", "url-files")
OUTLET_CONFIG_FILE = path.join(DATA_DIR, "raw", "outlet-config.json")

log_file = path.join(
    LOG_DIR, f"get-article-contents_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.log")
logging.basicConfig(**{**LOGGING_CONFIG_FILE, "filename": log_file, "level": logging.WARNING})


def get_warc_record(url: str):
    """Fetch the WARC record from common crawl for a given URL.

    Return the WARC file, if fetching is successful, None otherwise.

    Parameters:
    ----------
    url : str
          The URL for which to fetch the WARC record.
    """
    cdx = cdx_toolkit.CDXFetcher(source='cc')

    for obj in cdx.iter(url):
        # Sometimes we cannot fetch the requested WARC file; in those cases we want to return None.
        if obj["status"] == "200":
            return obj.fetch_warc_record()
    return None


def retrieve_content(url: str) -> tuple:
    """Fetch the text content of a given URL.

    Fetch the WARC file from CommonCrawl of a given URL and uses the news-please library to parse
    the textual content and the date of publication (if available) from the WARC file.

    Return a tuple containing the main text of the URL and the date of publication. If no date of
    publication is available, return an empty string in its position. Finally, if text could not
    be extracted or WARC file not retrieved, return a tuple of two None objects.

    Parameters:
    ----------
    url : str
          The URL for which to extract the text and publication date.
    """
    logging.warning(f"URL: {url}")
    warc_record = get_warc_record(url)
    if not warc_record:
        return (None, None)

    try:
        np_warc = NewsPlease.from_warc(warc_record)
        article_date = np_warc.date_publish
        if article_date:
            return (np_warc.maintext, article_date.strftime("%Y-%m-%d"))
        else:
            logging.error(f"Parsing date for {url} not possible. Returning without a date.")
            return (np_warc.maintext, "")
    except ValueError as e:
        logging.error(f"While parsing content for {url}, got: '{e}'. Skipping.")
        return (None, None)


def get_content_for_row(row) -> tuple:
    """Fetch the content of a URL specified in the given pandas row.

    Return a tuple containing the UUID of the url, the date of publication, the extracted text
    content and the preprocessed text content (the latter is just a placeholder for later).

    Parameters:
    ----------
    row : str
          The pandas row containing the URL and UUID for which the information should be extracted.
    """
    content, date = retrieve_content(row[1].url)
    return (row[1].uuid, date, content, "no-preprocessing-for-now")


# Data loading and preparation
# Outlet config file
with open(OUTLET_CONFIG_FILE, "r") as f:
    outlet_config = json.load(f)

# Target sqlite database
target_db_connection = sqlite3.connect(path.join(DATA_DIR, "raw", NEWS_ARTICLE_DB_NAME))
target_db_cursor = target_db_connection.cursor()

# Create a new article content table if it doesn't already exist
target_db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
target_db_tables = [t[0] for t in target_db_cursor.fetchall()]
if "article_contents" not in target_db_tables:
    logging.warning("Table 'article_contents' doesn't seem to exist yet. Creating it.")
    target_db_cursor.execute(
        """
        CREATE TABLE
        article_contents (uuid TEXT UNIQUE, date TEXT, content TEXT, content_preprocessed TEXT)
        """)

# For each outlet we specified in our outlet configuration file...
for outlet in outlet_config:
    logging.warning("=" * 80)
    logging.warning(f"Collecting article contents for {outlet['tld']}")

    # Retrieve all news article URLs for the current media outlet from the database
    urls = pd.read_sql(
        f"SELECT * FROM article_urls WHERE outlet_name='{outlet['name']}'",
        target_db_connection)

    # Retrieve existing article content UUIDs and remove existing URLs
    # This allows to collect the data in either multiple sessions or in a distributed manner.
    target_db_cursor.execute("SELECT uuid FROM article_contents")
    existing_content_ids = [t[0] for t in target_db_cursor.fetchall()]
    logging.warning(f"Found {len(existing_content_ids)} entries. Skipping those.")
    filtered_urls = urls[~urls.uuid.isin(existing_content_ids)]

    if len(filtered_urls) == 0:
        logging.warning(f"No URLs left for {outlet['tld']}. Skipping.")
        continue

    # Process dataframe in chunks due to memory constraints
    chunk_size = 1000
    url_chunks = [
        filtered_urls[i:i + chunk_size] for i in range(0, filtered_urls.shape[0], chunk_size)]
    for i, chunk in enumerate(url_chunks):
        # Processing multiple URLs at the same time
        logging.warning("-" * 80)
        logging.warning(f"Processing URL chunk {i} of {len(url_chunks)}")
        pool = Pool(8)
        contents = pool.imap(get_content_for_row, chunk.iterrows())

        logging.warning("Writing collected articles to database.")
        for content in contents:
            try:
                query = "INSERT INTO article_contents VALUES (?, ?, ?, ?);"
                target_db_cursor.execute(query, content)
            except IntegrityError:
                logging.warning(
                    f"Found potential duplicate article with id {content[0]}; won't insert.")
        target_db_connection.commit()

print("Closing all database connections.")
target_db_connection.close()
logging.warning("Done.")
