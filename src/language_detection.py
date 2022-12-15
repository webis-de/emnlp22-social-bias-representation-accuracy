import pandas as pd
import sqlite3

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from os import path
from pathlib import Path
from tqdm import tqdm

from embedding_bias.config import NEWS_ARTICLE_DB_NAME

tqdm.pandas()

PARENT_DIR = Path(__file__).parent.parent
DATA_DIR = path.join(PARENT_DIR, "data")
DB_PATH = path.join(DATA_DIR, "raw", NEWS_ARTICLE_DB_NAME)


def detect_language(text: str):
    """Detect the langauge of the given text. If langauge detection is not possible, return None.

    Parameters:
    text -- The text of which the langauge should be detected.
    """
    try:
        return detect(text)
    except LangDetectException as e:
        print(f"Issue with detecting language for '{text}'. Got {e}.")
        return None


# Prepare target database
print("Reading database from disk.")
target_db_connection = sqlite3.connect(DB_PATH)
target_db_cursor = target_db_connection.cursor()

target_table = target_db_cursor.execute("SELECT * from article_contents").description
target_db_columns = list(map(lambda x: x[0], target_table))
if "language" not in target_db_columns:
    print("Column 'language' doesn't seem to exist yet. Creating it.")
    target_db_cursor.execute("""
        ALTER TABLE article_contents
        ADD COLUMN language TEXT
    """)

# Retrieve article contents from database
original_contents = pd.read_sql(
    sql="SELECT * FROM article_contents",
    con=target_db_connection)
original_contents = original_contents.dropna(subset=["content"])

print("Starting language detection.")
original_contents["language"] = original_contents.content.progress_apply(detect_language)

# Saving the detected language of each article to the database
print("Saving detected languages to database.")
for i, article in tqdm(original_contents.iterrows(), total=len(original_contents)):
    query = """
    UPDATE article_contents
    SET language = ?
    WHERE uuid = ?
    """
    query_parameters = (
        article.language,
        article.uuid)
    target_db_connection.execute(query, query_parameters)

target_db_connection.commit()

print("Closing all database connections.")
target_db_connection.close()
print("Done.")
