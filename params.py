from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score

# Path of data after preprocessing
processed_data_path = 'processed_tweets_df.csv'

# Path of raw train data
raw_data_path = '2020-ppl3/Train.csv'

# Path of test data
test_data_path = '2020-ppl3/Test.csv'

# Features names for emotional feature extraction
features = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'length']

# Path for features of given data
data_features_path = 'tweets_features.csv'

# Boolean flag that indicates if a feature extraction process is necessary, or if it been done before
is_feature_extractor = False

# Metrics for 10 cross validation
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score)}

# Flag indicating if to use stemming
is_stem = False

# Flag indicating if to use lemmatizing
is_lemmatize = False

# Flag inidicating if preprocess is required or if it has been done before
is_preprocess = True

