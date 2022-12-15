import logging
import pandas as pd
import pathlib
import sqlite3
import sys
import torch

from gensim.models import KeyedVectors
from os import path
from pathlib import Path
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.query import Query
from wefe.preprocessing import get_embeddings_from_query

PARENT_DIR = Path(__file__).parent.parent
sys.path.append(path.join(PARENT_DIR, "embedding_evaluation"))
from embedding_evaluation.evaluate import Evaluation


def get_articles_as_df(
        db_connection: sqlite3.Connection,
        outlet_selection: pd.DataFrame,
        allsides_ranking: dict,
        preprocessed: bool = True) -> pd.DataFrame():
    """Retrieve all articles from the database and return them in a dataframe.

    Parameters:
    -----------
    db_connection : sqlite3.Connection
                    Connection to the source database, from which to fetch the articles.
    outlet_selection : pd.DataFrame
                       The selection of media outlets for which to retrieve articles from the
                       database. Each row should at least have a 'name' and a 'allsides_name'
                       property, where the latter maps to the names of the media outlet specified in
                       the allsides rating file.
    allsides_ranking : dict
                       A dictionary that maps the name of a media outlet to its respective allsides
                       media bias rating.
    preprocessed : Defines whether to retrieve the preprocessed or raw news article text from the
                   database.
    """
    # Load article data and collect (text, outlet, orientation) pairs
    outlet_article_ids = {}
    articles = pd.DataFrame(columns=["text", "date", "outlet", "orientation"])
    original_contents = pd.read_sql(
        sql="SELECT * FROM article_contents",
        con=db_connection)

    for i, outlet in outlet_selection.iterrows():
        print("Collecting articles for", outlet["name"])

        # Collecting all URL UUIDs from database
        uuids = pd.read_sql(
            sql="SELECT uuid FROM article_urls WHERE outlet_name=?",
            con=db_connection,
            params=(outlet["name"],))
        outlet_article_ids[outlet["name"]] = list(uuids.uuid)

        # Retrieving the outlets orientation as defined by allsides.com
        outlet_orientation = allsides_ranking[
            allsides_ranking["outlet"] == outlet["allsides_name"]].bias.iloc[0]

        # Find all articles from the current outlet for which we have content and for which the
        # specified language is English (and most importantly not None)
        outlet_articles = original_contents[
            original_contents.uuid.isin(uuids.uuid)].dropna(subset=["content"])
        outlet_articles = outlet_articles[outlet_articles.language == "en"]

        # Use the preprocessed article texts, if specified; otherwise, return unprocessed content
        if preprocessed:
            outlet_texts = outlet_articles.content_preprocessed
        else:
            outlet_texts = outlet_articles.content

        # Add articles from the current outlet to the dataframe
        articles = pd.concat([articles, pd.DataFrame(
            data={
                "text": outlet_texts,
                "date": outlet_articles.date,
                "outlet": outlet["name"],
                "orientation": outlet_orientation})])

    return articles


def evaluate_models(
        models: list,
        word_sets: dict,
        metrics: list,
        threshold: float = 0.25,
        log: bool = True) -> pd.DataFrame:
    """Evaluate the given models on the specified bias metrics. Return a dataframe with the results.

    Parameters:
    -----------
    models : list
             A list of word embedding models that should be evaluated. Each model object in the list
             is expected to be subscriptable and return a vector, when given the respective word as
             key, i.e. `model[0]["foobar"]` shold return the word vector for "foobar".
    word_sets : dict
                A set of words to be used in the bias evaluation. It is expected to have the same
                structure as the file `data/raw/word-sets.json`.
    metrics : list
              A list of metrics that should be evaluated for. They are expected to be WEFE metric
              instances.
    threshold : float
                The threshold for out-of-vocabulary tokens up to which bias evaluations should still
                be conducted. Check [1] for more details.
    log : bool
          Whether to log warning messages or not.

    [1]: https://wefe.readthedocs.io/en/latest/user_guide/measurement_user_guide.html
    """
    # Evaluate all models using WEFE and WEAT (other metrics later)
    # Bias types: gender, ethnicity, religion (our word lists)
    results = pd.DataFrame(columns=["model", "bias_type", "metric", "result"])

    # Define word set queries
    queries = {
        "gender": Query(
            target_sets=[
                [w.lower() for w in word_sets["target_sets"]["gender"]["male"]["set"]],
                [w.lower() for w in word_sets["target_sets"]["gender"]["female"]["set"]]],
            attribute_sets=[
                [w.lower() for w in word_sets["attribute_sets"]["career"]["set"]],
                [w.lower() for w in word_sets["attribute_sets"]["family"]["set"]]],
            target_sets_names=[
                "MaleTerms", "FemaleTerms"],
            attribute_sets_names=[
                "Career", "Family"]),
        "ethnicity": Query(
            target_sets=[
                [w.lower()
                    for w in word_sets["target_sets"]["ethnicity"]["european_american"]["set"]],
                [w.lower()
                    for w in word_sets["target_sets"]["ethnicity"]["african_american"]["set"]]],
            attribute_sets=[
                [w.lower() for w in word_sets["attribute_sets"]["pleasant"]["set"]],
                [w.lower() for w in word_sets["attribute_sets"]["unpleasant"]["set"]]],
            target_sets_names=[
                "EuroAmNames", "AfriAmNames"],
            attribute_sets_names=[
                "Pleasant", "Unpleasant"]),
        "religion": Query(
            target_sets=[
                [w.lower() for w in word_sets["target_sets"]["religion"]["christianity"]["set"]],
                [w.lower() for w in word_sets["target_sets"]["religion"]["islam"]["set"]]],
            attribute_sets=[
                [w.lower() for w in word_sets["attribute_sets"]["pleasant"]["set"]],
                [w.lower() for w in word_sets["attribute_sets"]["unpleasant"]["set"]]],
            target_sets_names=[
                "ChristTerms", "IslamTerms"],
            attribute_sets_names=[
                "Pleasant", "Unpleasant"])}

    # Conduct bias evaluations for each of the specified metrics
    for metric in metrics:
        logging.warning("=" * 80) if log else None
        logging.warning(f"Evaluating with {metric.metric_name}.") if log else None

        for query_name, query in queries.items():
            logging.warning(f"{' ' * 4}Using {query_name} query.") if log else None

            for model_name, model_wv in models.items():
                logging.warning(f"{' ' * 8}Evaluating {model_name} model.") if log else None
                model_wefe = WordEmbeddingModel(model_wv)
                result = metric.run_query(
                    query,
                    model_wefe,
                    warn_not_found_words=log,
                    lost_vocabulary_threshold=threshold)
                results.loc[len(results)] = [
                    model_name, query_name, metric.metric_short_name, result["effect_size"]]

    return results


def get_test_vocabulary(word_sets: list, similarity_eval_data_path: str) -> set:
    """Retrieve the vocabulary necessary to conduct social bias and similarity tests.

    Return a set of all unique tokens.

    Some models don't have a fixed vocabulary (e.g. when trained as a more general sub-token model).
    To test those models with our regular testing framework, we need to create a mapping from test
    vocabulary and respective embeddings. This function helps to collect all the necessary words to
    to so.

    Parameters:
    -----------
    word_sets : list
                A multi-dimensional list of dictionaries with words that should be included in the
                vocabulary. First dimension describes the bias type, the second the target group,
                which then as an attribute "set" that contains a list of all words.
                See `data/raw/word-sets.json` for an example structure.
    similarity_eval_data_path : str
                                Path to a directory that contains the word similarity test files
                                for the ws353 and men tests.

    """
    vocabulary = set()

    for bias in word_sets["target_sets"].keys():
        target_words = []
        for target_group in word_sets["target_sets"][bias].keys():
            target_words.extend(
                [w.lower() for w in word_sets["target_sets"][bias][target_group]["set"]])
        vocabulary.update(target_words)

    for attribute in word_sets["attribute_sets"].keys():
        vocabulary.update([w.lower() for w in word_sets["attribute_sets"][attribute]["set"]])

    # Retrieve tokens from disk
    with open(f"{similarity_eval_data_path}/men/MEN_dataset_natural_form_full", "r") as f:
        lines = f.read().splitlines()
        men_pairs = [line.split(" ") for line in lines][1:]
        men_tokens = [token for pair in men_pairs for token in pair[:-1]]

    with open(f"{similarity_eval_data_path}/wordsim/combined.csv", "r") as f:
        lines = f.read().splitlines()
        ws353_pairs = [line.split(",") for line in lines][1:]
        ws353_tokens = [token for pair in ws353_pairs for token in pair[:-1]]

    return set([*vocabulary, *ws353_tokens, *men_tokens])


def benchmark_models(models: dict, return_full_json=False) -> pd.DataFrame:
    """Run word similarity benchmarks for the given models. Return a dataframe with the results.

    Parameters:
    -----------
    models : dict
             A dictionary that contains the models to be evaluated. The key of each item should be
             the name of the model, the value the model object itself.
    return_full_json : bool
                       If true, return a tuple where the first element is a dictionary containing
                       all details of the results. The second item of the tuple is then the regular
                       dataframe containing the results.
    """
    results_df = pd.DataFrame(columns=[
        "model_name", 'usf', 'ws353', "ws353-rare-left", "ws353-rare-center", "ws353-rare-right",
        'men', "men-rare-left", "men-rare-center", "men-rare-right", 'vis_sim', 'sem_sim', 'simlex',
        'simlex-q1', 'simlex-q2', 'simlex-q3', 'simlex-q4', 'mturk771', 'rw'])
    evaluation = Evaluation()

    full_eval_results = {}

    for name, model in models.items():
        eval_results = evaluation.evaluate(model)
        full_eval_results[name] = eval_results

        results_dict = {
            "model_name": name}
        for bm_name, similarities in eval_results["similarity"].items():
            results_dict[bm_name] = similarities["all_entities"]

        # results_df = results_df.append(results_dict, ignore_index=True)
        results_df.loc[len(results_df)] = results_dict

    if return_full_json:
        return full_eval_results, results_df

    return results_df


def get_embedding_dict_from_pytorch(emb_model, dictionary) -> dict:
    """Retrieve an embedding dictionary from a given PyTorch model. Return the dictionary.

    The dictionary is then a map from tokens (the keys) to word vectors (the values).

    Parameters:
    -----------
    emb_model : PyTorch model object
                A PyTorch model from which word vectors can extracted using the encoder function.
    dictionary : dict
                 The token -> idx map of the given trained model.
    """
    embeddings_dict = {}
    for token, idx in dictionary.items():
        try:
            lookup_tensor = torch.tensor([idx], dtype=torch.long).cuda()
            embeddings_dict[token] = emb_model.encoder(lookup_tensor)[0].tolist()
        except:
            pass

    return embeddings_dict


def dict_to_word2vec_file(embedding_dict: dict, output_file: str):
    """Convert an embedding dictionary to a word2vec-format file and save it to disk.

    Return a gensim.models.KeyedVectors instance that already loaded the embedding file again.

    Parameters:
    -----------
    embedding_dict : dict
                     A dictionary that maps words to word vectors.
    output_file : str
                  Path to which the word2vec-format file should be saved.
    """
    print("Writing embeddings to file.")
    with open(f"{output_file}", "w") as f:
        f.write(f"{len(embedding_dict)} {len(embedding_dict['woman'])}\n")
        for token, vector in embedding_dict.items():
            try:
                token_vector_str = ' '.join([str(d) for d in vector])
                f.write(f"{token} {token_vector_str}\n")
            except KeyError:
                print(f"Token {token} not in dictionary.")

    print("Loading embeddings as KeyedVectors.")
    return KeyedVectors.load_word2vec_format(output_file)
