import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import params
import preprocess
import numpy as np


def create_features_from_tweet(tweet, emot_analysis_intensity, emotion_analysis, twitter_word_intensity, good_list, bad_list):

    tweet_dict = {
        'anger': 0,
        'anticipation': 0,
        'disgust': 0,
        'fear': 0,
        'joy': 0,
        'negative': 0,
        'positive': 0,
        'sadness': 0,
        'surprise': 0,
        'trust': 0
    }

    tweet_tokens = nltk.word_tokenize(tweet)
    for word in tweet_tokens:
        word_emot = emotion_analysis[emotion_analysis['word'] == word]
        if len(word_emot) != 0:
            tweet_dict['positive'] += (word_emot['positive'].values[0] / len(tweet_tokens))
            tweet_dict['negative'] += (word_emot['negative'].values[0] / len(tweet_tokens))
            word_emot = emot_analysis_intensity[emot_analysis_intensity['word'] == word]
            for row in word_emot.iterrows():
                temp = row[1]['emotion-intensity-score']
                tweet_dict[row[1]['emotion']] += temp
        elif word + '\n' in good_list:
            tweet_dict['positive'] += 1 / len(tweet_tokens)
        elif word + '\n' in bad_list:
            tweet_dict['negative'] += 1 / len(tweet_tokens)
        word_intensity = twitter_word_intensity[twitter_word_intensity['word'] == word]
        if len(word_intensity) != 0:
            if word_intensity['intensity'].values[0] > 0:
                tweet_dict['positive'] += word_intensity['intensity'].values[0]
            else:
                tweet_dict['negative'] -= word_intensity['intensity'].values[0]

    tweet_dict['length'] = len(tweet_tokens)

    return pd.DataFrame(tweet_dict, index=['i',])


def create_features_for_tweet_df(tweet_df):
    emot_analysis_intensity = pd.read_csv('/Users/shaharfreiman/Desktop/Degree/Y4S1/Principles of Programming Languages/Assignments/Assignment3/words_dictionary/words emotion intensity.txt', sep='\t')
    emot_analysis = pd.read_csv('/Users/shaharfreiman/Desktop/Degree/Y4S1/Principles of Programming Languages/Assignments/Assignment3/words_dictionary/word_sentiment.csv')
    good_list = set(open('/Users/shaharfreiman/Desktop/Degree/Y4S1/Principles of Programming Languages/Assignments/Assignment3/words_dictionary/positive-words.txt', encoding="ISO-8859-1").readlines())
    bad_list = set(open('/Users/shaharfreiman/Desktop/Degree/Y4S1/Principles of Programming Languages/Assignments/Assignment3/words_dictionary/negative-words.txt', encoding="ISO-8859-1").readlines())
    twitter_word_intensity = pd.read_csv('/Users/shaharfreiman/Desktop/Degree/Y4S1/Principles of Programming Languages/Assignments/Assignment3/words_dictionary/twitter words to sentiment.txt', sep='\t', names=['intensity', 'word'])


    tweets_features = pd.DataFrame(columns=('anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust'))

    proccessed_tweets = []
    i = 0
    for tweet in tweet_df:
        if i % 1000 == 0:
            print('{}/{}'.format(i, len(tweet_df)))
        tweet_text = preprocess.preprocess(tweet)
        proccessed_tweets.append(tweet_text)
        tweet_feature = create_features_from_tweet(tweet_text, emot_analysis_intensity, emot_analysis, twitter_word_intensity, good_list, bad_list)
        tweets_features = tweets_features.append(tweet_feature)
        i+=1

    # build TFIDF features on train reviews
    tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1, 2), sublinear_tf=True, max_features=2000)
    tf_idf = tv.fit_transform(tweet_df)
    tfidf_features = pd.DataFrame(tf_idf.toarray())

    tweets_features.index = np.arange(0, len(tweets_features))

    for column in tweets_features:
        tfidf_features[column] = tweets_features[column]

    if len(tfidf_features) > 60000:
        tfidf_features.to_csv(params.processed_train_data_path, index=False)
    else:
        tfidf_features.to_csv(params.processed_test_data_path, index=False)

    return tfidf_features



