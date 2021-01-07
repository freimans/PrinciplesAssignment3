import pandas as pd
import numpy as np
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import params

nltk.download('stopwords')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stoplist = stopwords.words('english')


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
    tweet = re.sub(r"&(\w)+;", "", tweet)
    return tweet


def remove_quot(tweet):
    tweet = re.sub(r"\'", "", tweet)
    tweet = re.sub(r"\"", "", tweet)
    return tweet


def reduce_duplicate_last_letter(tweet):
    return re.sub(r'(\w)\1+\b', r'\1', tweet)


def remove_urls(tweet):
    return re.sub(r'((www\.[\S]+)|(https?://[\S]+)).', "", tweet)


def replace_mentions(tweet):
    return re.sub(r'@(\S+)', ' MENTION', tweet)


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


def emphasize_exclamation(tweet):
    tweet = re.sub("(\w+)!+", r'\1 \1', tweet)
    return tweet


def remove_stopwords(tweet):
    words = tweet.split()
    for i,w in enumerate(words):
        if w in stoplist:
            words[i] = ''
    return ' '.join(x for x in words if x)


def remove_non_letters(tweet):
    tweet = re.sub('[^a-zA-Z]+', ' ', tweet)
    return tweet


def lemmatize_word(word):
    try:
        lemmatized = lemmatizer.lemmatize(word).lower()
        return lemmatized
    except Exception as e:
        return word


def lemmatize_tweet(tweet):
    lemmatized = [lemmatize_word(w.lower()) for w in tweet.split()]
    return " ".join(lemmatized)


def stem_tweet(tweet):
    stemmed = [stemmer.stem(w) for w in tweet.split()]
    return " ".join(stemmed)


def preprocess(tweet):
    clean_tweet = normalize_case(tweet)
    clean_tweet = replace_dots_with_space(clean_tweet)
    clean_tweet = remove_html_entities(clean_tweet)
    clean_tweet = reduce_duplicate_last_letter(clean_tweet)
    clean_tweet = remove_urls(clean_tweet)
    clean_tweet = replace_mentions(clean_tweet)
    clean_tweet = extract_hashtags(clean_tweet)
    clean_tweet = replace_abbreviation(clean_tweet)
    clean_tweet = remove_quot(clean_tweet)
    clean_tweet = emphasize_exclamation(clean_tweet)
    clean_tweet = remove_stopwords(clean_tweet)
    clean_tweet = remove_non_letters(clean_tweet)
    if params.is_stem:
        clean_tweet = stem_tweet(clean_tweet)
    if params.is_lemmatize:
        clean_tweet = lemmatize_tweet(clean_tweet)

    return clean_tweet


# inputt = ' @BonesFan021 #hashhg_of_words  I can\'t @BonesFan hear it! &quot;Big &amp; Rich - Between Raising Hell and Amazing Grace&quot;  The story of my life  â™« http://blip.fm/~7hdp9'
# inputt = '&quot;pigs didn\'t start the swine flu...&quot; &quot; we didn\'t do anythingggg wrong!&quot; ... wie sï¿½ï¿½. '

# print(preprocess(inputt, False, False))

# stem
# nltk.download('punkt')
# tokens = nltk.word_tokenize(x)
