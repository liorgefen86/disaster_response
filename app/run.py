import sys
sys.path.append('.')
from data_processing.classifier_functions import tokenize
import json
import plotly
import pandas as pd
import joblib

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine
from pandas import DataFrame

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
table_name = engine.table_names()[0]
df = pd.read_sql_table(table_name, engine)

# load model
with open('models/be_dt.joblib', 'rb') as file:
    model = joblib.load(file)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Function used to generate the main page of the website
    :return: a rendered main page
    """

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    def count_categories(input: DataFrame, n_cats: int = 10, filter_on: str = None) \
            -> DataFrame:
        """
        Function used to count the number of occurence of a category
        :param input: pandas DataFrame
        :param n_cats: number of categories to maintain
        :param filter_on: value to filter on (filter applied on 'genre' field)
        :return: pandas DataFrame
        """
        categories = input.columns[4:]
        output = input.copy()
        if filter:
            output = output.query(f'genre == "{filter_on}"')

        cat_count = dict()
        for row in output.iloc[:, 4:].values:
            for ind, val in enumerate(row):
                label = categories[ind]
                cat_count[label] = val + cat_count.get(label, 0)

        graph_data = pd.DataFrame({
            'category': cat_count.keys(),
            'count': cat_count.values()
        })
        return graph_data.sort_values(by=['count'], ascending=False).iloc[:n_cats, :]

    graph2_data = count_categories(df, filter_on='news')
    graph3_data = count_categories(df, filter_on='direct')

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    marker={
                        'colors': ['#FDE12D', '#CA895F', '#465362']
                    }
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                    'categoryorder': 'total descending'
                }
            }
        },
        {
            'data': [
                Bar(
                    x=graph2_data['category'],
                    y=graph2_data['count'],
                    marker={
                        'color': '#CA895F'
                    }
                )
            ],

            'layout': {
                'title': 'Distribution of the 10 most used Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Disaster Message Category",
                    'categoryorder': 'total descending'
                }
            }
        },
        {
            'data': [
                Bar(
                    x=graph3_data['category'],
                    y=graph3_data['count'],
                    marker={
                        'color': '#FDE12D'
                    }
                )
            ],

            'layout': {
                'title': 'Distribution of the 10 most used Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Disaster Message Category",
                    'categoryorder': 'total descending'
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Function used to render query results page
    :return: rendered result page
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Function use to create the local server
    :return:
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
