import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import params
from sklearn import model_selection
import feature_extractor as fe
import preprocess as pp


def get_models():
    """
    Create a list of AI models
    :return: list of models
    """
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    # models.append(('RFC', RandomForestClassifier(n_estimators=20, verbose=2, n_jobs=5)))
    # models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
    return models


raw_data = pd.read_csv(params.raw_data_path, encoding="ISO-8859-1")
kaggle_test = pd.read_csv(params.test_data_path, encoding="ISO-8859-1")

if params.is_preprocess:
    data = pp.preprocess_df(raw_data)
    data.to_csv(params.processed_data_path)
else:
    data = pd.read_csv(params.processed_data_path)

# build TFIDF features on train reviews
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1, 2), sublinear_tf=True)
train_tf_idf = tv.fit_transform(data['SentimentText'])

models = get_models()

results = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_validate(model, train_tf_idf, data['Sentiment'], cv=kfold, scoring=params.scoring)
    results.append((name, cv_results))

for result in results:
  print(result[0] + ":")
  print('\tAccuracy: {}\n\tPrecision: {}\n\tRecall: {}'.format(result[1]['test_accuracy'].mean(), result[1]['test_precision'].mean(), result[1]['test_precision'].mean()))