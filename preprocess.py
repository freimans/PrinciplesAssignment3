import pandas as pd
import numpy as np
import regex as re
import multiprocessing
import nltk
from multiprocessing import Pool


# from segmenter import Analyzer
# from nltk.stem import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer

def normalize_case(tweet):
    """
    normalize text case - to lower case
    """
    return tweet.lower()


def replace_dots_with_space(tweet):
    """
    Replace multiple dots with space
    """
    return re.sub(r'\.\.+', ' ', tweet).replace('.', '')


def remove_html_entities(tweet):
    tweet = re.sub(r"&gt;", "", tweet)
    tweet = re.sub(r"&lt;", "", tweet)
    tweet = re.sub(r"&amp;", "", tweet)
    tweet = re.sub(r"&quot;", "", tweet)
    return tweet


def remove_quot(tweet):
    tweet = re.sub(r"\'", "", tweet)
    tweet = re.sub(r"\"", "", tweet)
    return tweet


def reduce_duplicate_last_letter(tweet):
    return re.sub(r'(\w)\1+\b', r'\1', tweet)


def remove_urls(tweet):
    return re.sub(r'((www\.[\S]+)|(https?://[\S]+)).', "", tweet)


def remove_mentions(tweet):
    return re.sub( r'@(\S+)', ' MENTION', tweet )


def extract_hashtags(tweet):
    return tweet.replace("#", "").replace("_", " ")

def replace_abbreviation(tweet):
    tweet = re.sub(r"i\'m", "i am", tweet)
    tweet = re.sub(r"\'re", "are", tweet)
    tweet = re.sub(r"he\'s", "he is", tweet)
    tweet = re.sub(r"it\'s", "it is", tweet)
    tweet = re.sub(r"that\'s", "that is", tweet)
    tweet = re.sub(r"who\'s", "who is", tweet)
    tweet = re.sub(r"what\'s", "what is", tweet)
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    return tweet


def remove_non_letters(tweet):
    return


def temp_func(inputt):
    x = replace_dots_with_space(inputt)

    x = replace_abbreviation(x)
    x = reduce_duplicate_last_letter(x)
    x = remove_html_entities(x)
    x = remove_urls(x)
    x = remove_mentions(x)
    return extract_hashtags(x)

# inputt = ' @BonesFan021 #hashhg_of_words  I can\'t @BonesFan hear it! &quot;Big &amp; Rich - Between Raising Hell and Amazing Grace&quot;  The story of my life  â™« http://blip.fm/~7hdp9'
# inputt = '&quot;pigs didn\'t start the swine flu...&quot; &quot;we didn\'t do anythingggg wrong!&quot; ... wie sï¿½ï¿½. '
# x = normalize_case(input)
# x= replace_dots_with_space(inputt)
# x = remove_quot(x)
# x = replace_abbreviation(x)
# x = reduce_duplicate_last_letter(x)
# x = remove_html_entities(x)
# x = remove_urls(x)
# x = remove_mentions(x)
# x = extract_hashtags(x)
# print(x)
# Strip spaces and quotes (" and ’) from the ends of tweet
# Replace 2 or more spaces with a single space

# TODO twitter feature pre process


# replace URLs with 'URL'
# regex for url - s ((www\.[\S]+)|(https?://[\S]+)).

# remove numbers

# replace mentions with 'USER_MENTION'
# regex  -  @[\S]+. 2

# replace emotions with EMO_POS EMO_NEG
# regex table in https://arxiv.org/ftp/arxiv/papers/1807/1807.07752.pdf

# remove hashtag  # stmbol and split words of it
# regex -  #(\S+).

# remove Retweet RT
# regex - \brt\b.

# TODO NLTK function

# remove stop words

# lemmatize

# stem
# nltk.download('punkt')
# tokens = nltk.word_tokenize(x)