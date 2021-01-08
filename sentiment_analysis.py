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


