import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    function to load data
    Arguments:
        messages_filepath: holds the path to messages data (.csv)
        categories_filepath: holds the path to categories data (.csv)
    Returns:
        merged: dataframe. merged df of messages and caetegories.
    """
    messages = pd.read_csv(messages_filepath)
    categories  = pd.read_csv(categories_filepath)
    merged = pd.merge(messages, categories, on='id', how='outer')
    return merged


def clean_data(df):
    """
    function to clean the data
    Arguments:
        df: Dataframe. of data to clean.
    Returns:
        df: dataframe. A cleaned version of df.
    """
    categories = df.categories.str.split(';', expand=True)
    firstrow = categories.iloc[0,:]
    category_colnames = firstrow.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    for column in categories:

        categories[column] = categories[column].astype('str')
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = pd.to_numeric(categories[column])

    df.drop('categories', axis=1, inplace=True)
    df = df.join(categories)
    df.drop_duplicates(inplace=True)
    assert df.duplicated().sum() == 0

    return df

def save_data(df, database_filename):
    """
    function to save the data to sqldatabase
    Arguments:
        df: Dataframe. data to save.
        database_filename: String, name saved database.
    """

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('df', engine, index=False)
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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