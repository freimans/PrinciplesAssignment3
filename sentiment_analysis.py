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


def get_train_test_data(data):

    X = data[params.features]
    Y = list(data['sentiment'])

    return model_selection.train_test_split(X, Y, test_size=params.val_size, random_state=params.seed)



def get_models():
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    return models


data = None
if not params.is_feature_extractor:
    data = pd.read_csv(params.processed_data_path, index_col=False)
else:
    raw_data = pd.read_csv(params.raw_data_path, encoding="ISO-8859-1")
    data = fe.create_features_for_tweet_df(raw_data)

x_train, x_test, y_train, y_test = get_train_test_data(data)

models = get_models()

results = []
names = []

for name, model in models:
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=10, scoring=params.scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)