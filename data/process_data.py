import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    
    """
    Tokenizes incoming text

    Parameters
    ----------
    messages_filepath : str
        Filepath to messages data
    categories_filepath : str
        Filepath to categories data

    Returns
    -------
    DataFrame
        A DataFrame of the merged DataFrames
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return pd.merge(messages, categories, on = "id")


def clean_data(df):
    
    """
    Cleans the given DataFrame

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains all the information to process
    
    Returns
    -------
    DataFrame
        A cleaned DataFrame 
    """
    
    categories = df["categories"].str.split(";", expand = True)
    row = categories.iloc[0,:]
    category_colnames = categories.iloc[0,:].apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x.split("-")[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(axis=1, columns='categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # drop duplicates & return
    return df.drop_duplicates()


def save_data(df, database_filename):
    
    """
    Saves the given DataFrame to a SQL Database

    Parameters
    ----------
    df : DataFrame
        DataFrame to store
    database_filename : str
        Name of the Database
    """  
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Tweets', engine, index=False)

    
    
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