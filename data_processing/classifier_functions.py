import pandas as pd
from typing import Union, Dict, Callable
from numpy import ndarray
from datetime import datetime
import joblib
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV


Classifier = Union[RandomForestClassifier, AdaBoostClassifier,
                   DecisionTreeClassifier]


class SaveModelError(Exception):
    pass


def notifications(func: Callable):
    """
    Decorator function used to time each function
    :param func: function to decorate
    :return: decorated function
    """

    def warp(*args, **kwargs):
        start = datetime.now()
        message = f'{func.__name__} started at {start}'
        print(message)

        output = func(*args, **kwargs)
        end = datetime.now()
        seconds = int((end - start).seconds)

        message = f'{func.__name__} ended at {end};\n' \
                  f'after {seconds // 60} minutes and {seconds % 60} ' \
                  f'seconds'
        print(message)

        return output

    return warp


@notifications
def save_model(model: Union[Classifier, Pipeline], model_filepath):
    """
    Function used to save the model
    :param model: model can be a Classifier or Pipeline object
    :param model_filepath: file path to save the model
    :return: None

    :raises SaveModelError: if an error occurs while saving the model
    """

    try:
        with open(f'{model_filepath}.joblib', 'wb') as file:
            joblib.dump(model, file)
    except:
        raise SaveModelError('An error occurred while saving the model')


def tokenize(text):
    """
        Function used to clean text data:
        - text normalization
        - text tokenization
        - text lemmatization or stemming
        :param text: text to clean
        :return: cleaned text
        """
    # text normalization and tokenization
    tokenizer = RegexpTokenizer(r'[\w]')
    tokens = tokenizer.tokenize(text)

    # removing stopwords
    stop_words = stopwords.words('english')
    tokens = [word.strip() for word in tokens if word not in stop_words]

    # lemmatization or stemming
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in tokens]

    return text


@notifications
def build_model(classifier):
    """
    Function used to create a scikit-learn pipeline
    :param classifier: classifier used in the pipeline
    :return: a scikit-learn pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('cls',
         MultiOutputClassifier(estimator=classifier))
    ])

    return pipeline


@notifications
def evaluate_model(model: Union[Classifier, Pipeline, GridSearchCV], x_test:
                   ndarray, y_test: ndarray):
    """
    Function to evaluate the model: accuracy, precision, recall and f-score
    :param model: a trained Classifier or Pipeline object
    :param x_test: features test dataset (numpy array)
    :param y_test: labels test dataset (numpy array)
    :return: None
    """
    y_predict = model.predict(x_test)

    scores = pd.DataFrame(columns=['accuracy', 'precision', 'recall',
                                   'f_score'])

    for real_val, pred_val in zip(y_test.T, y_predict.T):
        accuracy = accuracy_score(real_val, pred_val)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            real_val, pred_val, average='weighted'
        )

        df = pd.DataFrame(data={
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f_score': [f_score]
        })
        scores = scores.append(df)

    print(scores.mean())


@notifications
def optimize_model(model: Union[Classifier, Pipeline], params: Dict,
                   x: ndarray, y: ndarray) -> GridSearchCV:
    """
    Function used to run GridSearch on model using parameters
    :param model: model can be a Classifier or Pipeline object
    :param params: parameters to use with GridSearch
    :param x: features dataset
    :param y: labels dataset
    :return: fitted GridSearch object
    """
    gs = GridSearchCV(
        estimator=model,
        param_grid=params,
        n_jobs=-1
    )

    gs.fit(x, y)

    return gs.best_estimator_
