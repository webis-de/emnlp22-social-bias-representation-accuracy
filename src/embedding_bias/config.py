import logging
import sys

# Basic logging configuration
LOGGING_CONFIG = {
    "stream": sys.stdout,
    "format": "%(levelname)s:%(asctime)s:%(message)s",
    "level": logging.INFO,
    "datefmt": "%Y-%m-%d %H:%M:%S"}
LOGGING_CONFIG_FILE = {
    "format": "%(levelname)s:%(asctime)s:%(message)s",
    "level": logging.INFO,
    "datefmt": "%Y-%m-%d %H:%M:%S"}

# Special tokens used througout the pre-/post-processing
SENTENCE_ENDING_TOKEN = "<SENT_END>"

# Crawl date limits
CRAWL_FIRST_YEAR = 2010
CRAWL_LAST_YEAR = 2021

NEWS_ARTICLE_DB_NAME = "news_articles.db"
