import contractions
import logging
import re

from nltk import download as nltk_download
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from unidecode import unidecode


def _prepare_preprocessing() -> None:
    """Ensure that all required NLTK resources are actually available locally."""
    logging.info("Downloading required nltk resources.")
    nltk_download("punkt")
    nltk_download("averaged_perceptron_tagger")
    nltk_download("wordnet")
    nltk_download('omw-1.4')


def _remove_urls(text: str) -> str:
    """Remove all HTTP(S) URLs from the given text. Return the cleaned text.

    The regular expression presented in [1] is applied to detect URLs in the text.


    Parameters
    ----------
    text : str
           The text that should be cleaned from URLs.

    [1]: https://urlregex.com/
    """
    url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    return re.sub(url_regex, "", text, flags=re.MULTILINE)


def _split_sentences(text: str) -> str:
    """Split the given text into single sentences. Return a list of sentences.

    This function uses NLTK to do the sentence splitting.


    Parameters
    ----------
    text : str
           The text that should be split into sentences.
    """
    return sent_tokenize(text)


def remove_symbols(text: str) -> str:
    """Remove symbols from the given text that we don't want to have in the final text.

    Return the cleaned text.


    Parameters
    ----------
    text : str
           The text from which symbols should be removed.
    """
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_excess_whitespace(text: str) -> str:
    """Remove additional whitespace tokens that were probably introduced during crawling.

    Return the cleaned text.


    Parameters
    ----------
    text : str
           The text from which the excess whitespaces should be removed.
    """
    return " ".join(text.split())


def nltk_tag_to_wordnet_tag(nltk_tag: str) -> str:
    """Convert the given NLTK part of speech tag to WordNet tags. Return the WordNet tag.

    If the given tag is uncommon or not in our selection, return `None`.


    Parameters
    ----------
    nltk_tag : str
               The NLTK tag that should be converted.
    """
    if nltk_tag.startswith("J"):
        return wordnet.ADJ
    elif nltk_tag.startswith("V"):
        return wordnet.VERB
    elif nltk_tag.startswith("N"):
        return wordnet.NOUN
    elif nltk_tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def _lemmatize_sentence(sentence: str, lemmatizer: WordNetLemmatizer) -> str:
    """Lemmative each token of the given sentence using the given lemmatizer.

    Return the lemmatized sentence.


    Parameters
    ----------
    sentence : str
               The sentence in which each token should be lemmatized.
    lemmatizer : nltk.stem.WordNetLemmatizer
                 The lemmatizer that should be used to lemmatize each token.
    """
    # Tokenize the sentence and find the POS tag for each token
    nltk_tagged = pos_tag(word_tokenize(sentence))

    # Convert the tagged sentences to get a tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)

    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        # Check for each token if a WordNet tag exists. If this is not the case, append the token
        # as is, since we cannot form a lemmatized form of it.
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    return " ".join(lemmatized_sentence)


def preprocess_text(text: str, run_preparation: bool = False) -> list:
    """Preprocess and clean the given text with different techniques. Return the preprocessed text.

    This function does takes the following preprocessing steps:
    - Removing URLs
    - Splitting the text into sentences
    - Expanding contractions
    - Decoding unicode characters (if necessary)
    - Lemmatize the text
    - Remove unwanted symbols
    - Remove excess whitespace
    - Lowercase the text


    Parameters
    ----------
    text : str
           The text that should be preprocessed.
    run_preparation : bool, optional
                      If this parameter is `True`, this function will run a preparation script to
                      make sure that all necessary (NLTK) resources are present on the local
                      machine. As this might take a bit of time, it is suggested to run this only
                      on the first run. Default value is `False`.
    """
    if run_preparation:
        _prepare_preprocessing()

    if not text:
        raise ValueError("Given text is empty. Cannot process empty text.")

    lemmatizer = WordNetLemmatizer()

    # Run all preprocessing steps to prepare the text
    no_urls_text = _remove_urls(text)
    text_sentences = _split_sentences(no_urls_text)
    cleaned_sentences = [unidecode(s) for s in text_sentences]
    cleaned_sentences = [contractions.fix(s) for s in cleaned_sentences]
    cleaned_sentences = [_lemmatize_sentence(s, lemmatizer) for s in cleaned_sentences]
    cleaned_sentences = [remove_symbols(s) for s in cleaned_sentences]
    cleaned_sentences = [remove_excess_whitespace(s) for s in cleaned_sentences]
    cleaned_sentences = [str.lower(s) for s in cleaned_sentences]

    return cleaned_sentences
