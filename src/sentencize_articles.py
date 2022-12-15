import pandas as pd
import spacy
import sqlite3

from os import path
from tqdm import tqdm

from embedding_bias.config import NEWS_ARTICLE_DB_NAME, SENTENCE_ENDING_TOKEN


tqdm.pandas()

PARENT_DIR = Path(__file__).parent.parent
DATA_DIR = path.join(PARENT_DIR, "data")
DB_PATH = path.join(DATA_DIR, "raw", NEWS_ARTICLE_DB_NAME)
OUTPUT_PATH = path.join(DATA_DIR, "processed", "article-contents-sentencized.pkl.zst")
BATCH_SIZE = 50000

# Initializing spacy
spacy.prefer_gpu()
nlp = spacy.load(name="en_core_web_sm", disable=["parser", "ner", "lemmatizer", "textcat"])
nlp.add_pipe("sentencizer")

# Prepare target database
print("Reading database from disk.")
target_db_connection = sqlite3.connect(DB_PATH)
original_contents = pd.read_sql(
    sql="SELECT * FROM article_contents",
    con=target_db_connection)
original_contents = original_contents.dropna(subset=["content"])

# Define batches to work with to limit memory use by spacy
batch_indices = list(range(0, len(original_contents), BATCH_SIZE))
if len(original_contents) % BATCH_SIZE > 0:
    batch_indices.append(len(original_contents) + 1)

print("Starting text processing.")
for i, batch in enumerate(batch_indices):
    # If we are at the last batch index, we already processed all the text, so we skip this step
    if batch == batch_indices[-1]:
        continue

    article_batch = original_contents[batch:batch_indices[i + 1]]

    print(f"Batch index {batch} to {batch_indices[i+1]} ({i + 1} of {len(batch_indices) + 1}).")

    # Process all article texts into lists of sentences. As the `n_process` parameter seems to yield
    # significantly worse results (even though it is faster), we don't use it here.
    articles_pipe = nlp.pipe(tqdm(article_batch.content, leave=False))

    # Saving the sentencized version of each article to the database
    for i, article in enumerate(articles_pipe):
        query = """
        UPDATE article_contents
        SET content_preprocessed = ?
        WHERE uuid = ?
        """
        query_parameters = (
            SENTENCE_ENDING_TOKEN.join([s.text for s in article.sents]),
            article_batch.uuid.iloc[i])
        target_db_connection.execute(query, query_parameters)

    target_db_connection.commit()

print("Closing all database connections.")
target_db_connection.close()
print("Done.")
