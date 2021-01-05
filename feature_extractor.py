import nltk
import pandas as pd

import preprocess


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

    tweets_features = pd.DataFrame(columns=('anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'tweet', 'sentiment'))
    df_length = len(tweet_df)
    print(df_length)
    for index, tweet in tweet_df.iterrows():
        if index % 1000 == 0:
            print('{}/{}'.format(index,df_length))
        tweet_text = tweet['SentimentText']
        tweet_sentiment = tweet['Sentiment']
        tweet_text = preprocess.temp_func(tweet_text)
        tweet_feature = create_features_from_tweet(tweet_text, emot_analysis_intensity, emot_analysis, twitter_word_intensity, good_list, bad_list)
        tweet_feature['tweet'] = tweet_text
        tweet_feature['sentiment'] = tweet_sentiment
        tweets_features = tweets_features.append(tweet_feature)
    return tweets_features


tweets_df = pd.read_csv('/Users/shaharfreiman/Desktop/Degree/Y4S1/Principles of Programming Languages/Assignments/Assignment3/2020-ppl3/Train.csv', encoding="ISO-8859-1")
df = create_features_for_tweet_df(tweets_df)
print(df)


