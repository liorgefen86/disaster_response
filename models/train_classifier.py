import sys
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
import joblib


class SaveModelError(Exception):
    pass


def load_data(database_filepath):
    """
        Function used to load data from sqlite database
        :param db_name: database file name
        :return: X, Y, categories
        """
    engine = create_engine(f'sqlite:///{database_filepath}.db')
    table_name = engine.table_names()[0]
    df = pd.read_sql(f'{table_name}', engine, index_col='id')
    not_in_Y = ['original', 'genre', 'message']
    X = df['message']
    Y = df[[col for col in df.columns if col not in not_in_Y]]
    categories = Y.columns

    return X.values, Y.values, categories


def tokenize(text):
    """
        Function used to clean text data:
        - text normalization
        - text tokenization
        - text lemmatization or stemming
        :param text: text to clean
        :param reduce_word: lammatizer or stemming object
        :return: clean text
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


def build_model(classifier):
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfids', TfidfTransformer()),
        ('cls',
         MultiOutputClassifier(estimator=classifier))
    ])

    return pipeline


def evaluate_model(model, X_test, y_test):
    """
    Function to evaluate the model: accuracy, precision, recall
    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """
    y_predict = model.predict(X_test)

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


def save_model(model, model_filepath):
    """
    Function used to pickle the model
    :param model: model variable to be pickled
    :param model_filepath: file path to save the pickled model
    :return:
    """

    try:
        with open(model_filepath, 'wb') as file:
            joblib.dump(model, file)
    except:
        raise SaveModelError('An error occurred while saving the model')


def main():
    if len(sys.argv) == 4:
        start = datetime.now()
        print(f'Pipeline start at {start}')
        database_filepath, classifier, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)
        
        print('Building model...')
        if classifier.lower() == 'rf':
            classifier = RandomForestClassifier()
        elif classifier.lower() == 'ad':
            classifier = AdaBoostClassifier()
        elif classifier.lower() == 'dt':
            classifier = DecisionTreeClassifier()
        model = build_model(classifier=classifier)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        end = datetime.now()
        secondes = int((end - start).seconds)
        print(f'''Pipeline finished at {end}\n.
                  Total time: {secondes // 60} minutes and {secondes % 60} 
                  secondes''')
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
