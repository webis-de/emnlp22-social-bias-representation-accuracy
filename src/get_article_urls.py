import json
import logging
import pyhash
import re
import sqlite3
import subprocess

from datetime import datetime
from os import listdir, path
from urllib.parse import urlparse
from sqlite3 import IntegrityError

from embedding_bias.config import LOGGING_CONFIG_FILE, NEWS_ARTICLE_DB_NAME

PARENT_DIR = Path(__file__).parent.parent
DATA_DIR = path.join(PARENT_DIR, "data")
LOG_DIR = path.join(PARENT_DIR, "logs")
URL_FILE_DIR = path.join(DATA_DIR, "raw", "url-files")
OUTLET_CONFIG_FILE = path.join(DATA_DIR, "raw", "outlet-config.json")
HASHER = pyhash.farm_64()

log_file = path.join(
    LOG_DIR, f"get-article-urls_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.log")
logging.basicConfig(**{**LOGGING_CONFIG_FILE, "filename": log_file, "level": logging.WARNING})


def run_gau(output_file: str, domain: str) -> None:
    """Run the gau command to collect available URLs in commoncrawl using specific settings.

    Parameters:
    ----------
    output_file : str
                  Path to the file to which the URLs should be saved to.
    domain : str
             The TLD for which URLs should be searched for.
    """
    gau_command = [
        "bin/gau",
        "--o", output_file,
        "--threads", "20",
        "--blacklist", "ttf,woff,svg,png",
        "--providers", "commoncrawl",
        "--mc", "200",
        "--subs",
        domain
    ]
    subprocess.run(gau_command, capture_output=True)


# Data loading and preparation
# Outlet config file
with open(OUTLET_CONFIG_FILE, "r") as f:
    outlet_config = json.load(f)

# Target sqlite database
target_db_connection = sqlite3.connect(path.join(DATA_DIR, "raw", NEWS_ARTICLE_DB_NAME))
target_db_cursor = target_db_connection.cursor()

# Create a new URLs table if it doesn't already exist
target_db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
target_db_tables = [t[0] for t in target_db_cursor.fetchall()]
if "article_urls" not in target_db_tables:
    logging.warning("Table 'article_urls' doesn't seem to exist yet. Creating it.")
    target_db_cursor.execute(
        "CREATE TABLE article_urls (uuid TEXT UNIQUE, url TEXT, outlet_name TEXT)")

# Retrieving and filtering URLs for all outlets
# In the URL file directory, each file represents the URLs available for a specific media outlet.
collected_url_outlets = listdir(URL_FILE_DIR)
for outlet in outlet_config:
    logging.warning(f"Retrieving URLs for {outlet['tld']}.")
    filename = f"{outlet['tld'].replace('/', '_')}.urls"
    filepath = path.join(URL_FILE_DIR, f"{outlet['tld'].replace('/', '_')}.urls")
    if filename not in collected_url_outlets:
        run_gau(filepath, outlet["tld"])
    else:
        logging.warning("Already collected URLs. Skipping GAU process.")
        continue

    # Filter collected URLs, if necessary
    logging.warning(f"Filtering URLs for {outlet['tld']}.")
    with open(filepath, "r") as f:
        urls = f.read().split("\n")

    if outlet["filter"]:
        filtered_urls = list(filter(re.compile(outlet["filter"]).search, urls))
    else:
        logging.warning(f"No filter defined for {outlet['name']}. Using all URLs.")
        filtered_urls = urls

    # Cleaning URLs from HTTP queries (we don't need them)
    urls_without_query = []
    for url in filtered_urls:
        u = urlparse(url)
        urls_without_query.append(f"{u.scheme}://{u.netloc}{u.path}")

    # Add all URLs for the current media outlet into the database
    logging.warning("Inserting into database.")
    url_rows = [(str(HASHER(url)), url, outlet["name"]) for url in set(urls_without_query)]

    try:
        query = "INSERT INTO article_urls VALUES (?, ?, ?);"
        target_db_cursor.executemany(query, url_rows)
    except IntegrityError:
        logging.warning("Found potential duplicate URL; won't insert.")
    target_db_connection.commit()

print("Closing all database connections.")
target_db_connection.close()
logging.warning("Done.")
