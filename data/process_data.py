import sys
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> DataFrame:
    """
    Function used to read data files and return a pandas DataFrame
    :param messages_filepath: path to csv containing messages
    :param categories_filepath: path to csv containing categories
    :return: pandas DataFrame with merged data from csv files
    """
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')

    # merging both DataFrames on index
    merged = pd.merge(
        left=messages, left_index=True,
        right=categories, right_index=True
    )
    return merged


def clean_data(df: DataFrame) -> DataFrame:
    """
    Function used to clean data
    :param df: pandas DataFrame to clean
    :return: cleaned pandas DataFrame
    """

    # create a categories DataFrame and drop the column from df
    categories = df[['categories']]
    df.drop(labels='categories', axis=1, inplace=True)

    # splitting all categories and creating a column per category
    categories = categories['categories'].str.split(pat=';', expand=True)
    cat_names = [val[:-2] for val in categories.iloc[0, :]]
    categories.columns = cat_names

    # cleaning all category columns to have 0 and 1
    for col in categories:
        categories[col] = categories[col].apply(lambda val: val[-1])
        categories[col] = categories[col].astype(dtype=int)

    # merge categories back to df
    df = df.merge(right=categories, left_index=True, right_index=True)
    df.drop_duplicates(inplace=True)  # remove duplicates

    return df


def save_data(df: DataFrame, database_filename: str, table_name: str = 'tab') \
        -> None:
    """
    Function used to save a DataFrame into a sqlite database
    :param df: DataFrame to save
    :param database_filename: sqlite database file full path
    :param table_name: table name in the database
    :return: None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(table_name, engine, index=True, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath =  \
            sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
