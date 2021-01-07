import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import feature_extractor as fe
import params
from sklearn import model_selection


def get_models():
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('NB', GaussianNB()))
    return models


data = None
raw_data = pd.read_csv(params.raw_data_path, encoding="ISO-8859-1")
x_train, x_test, y_train, y_test = model_selection.train_test_split(raw_data['SentimentText'], raw_data['Sentiment'], test_size=params.val_size, random_state=params.seed)
if params.is_feature_extractor:
    x_train = fe.create_features_for_tweet_df(x_train)
    x_test = fe.create_features_for_tweet_df(x_test)
else:
    x_train = pd.read_csv(params.processed_train_data_path)
    x_test = pd.read_csv(params.processed_test_data_path)

models = get_models()

results = []
names = []

for name, model in models:
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=10, scoring=params.scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)