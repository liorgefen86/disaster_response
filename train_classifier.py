import sys
from data_processing.classifier_functions import notifications, build_model, \
    optimize_model, save_model, evaluate_model
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import nltk

nltk.download(['stopwords', 'wordnet'])

@notifications
def load_data(database_filepath):
    """
        Function used to load data from sqlite database
        :param database_filepath: database file name
        :return: X, Y, categories
        """
    engine = create_engine(f'sqlite:///{database_filepath}.db')
    table_name = engine.table_names()[0]
    df = pd.read_sql(f'{table_name}', engine, index_col='id')
    not_in_y = ['original', 'genre', 'message']
    x = df['message']
    y = df[[col for col in df.columns if col not in not_in_y]]
    categories = y.columns

    return x.values, y.values, categories


@notifications
def main():
    if len(sys.argv) == 4:
        database_filepath, classifier, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2)

        print('Building model...')
        if classifier.lower() == 'rf':
            message = f'Model used: RandomForest\n'
            classifier = RandomForestClassifier()
            cls_params = {
                'cls__estimator__n_estimators': [50, 100, 200],
                'cls__estimator__max_depth': [None, 1, 3, 5]
            }
        elif classifier.lower() == 'ad':
            message = f'Model used: AdaBoost\n'
            classifier = AdaBoostClassifier()
            cls_params = {
                'cls__estimator__base_estimator': [DecisionTreeClassifier(
                    max_depth=1
                ), RandomForestClassifier(
                    max_depth=1
                )],
                'cls__estimator__n_estimators': [25, 50, 100],
            }
        elif classifier.lower() == 'dt':
            message = f'Model used: DecisionTree\n'
            classifier = DecisionTreeClassifier()
            cls_params = {
                'cls__estimator__max_depth': [None, 1, 3, 5]
            }
        else:
            raise ValueError(f'{classifier.lower()} is not a valid '
                             f'classifier choice')

        params = {
            'vect__max_features': [None, 20, 50, 100],
            'tfidf__use_idf': [True, False],
            'tfidf__norm': ['l1', 'l2'],
            **cls_params
        }
        message = message + f'Parameters for GridSearch:\n{params}'
        print(message)
        model = build_model(classifier=classifier)

        print('Training model...')
        best_estimator = optimize_model(
            model=model,
            params=params,
            x=x_train,
            y=y_train
        )

        print('Evaluating model...')
        evaluate_model(best_estimator, x_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_estimator, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
