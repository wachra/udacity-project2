import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline, FeatureUnion #, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer #BaseEstimator, 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib

def load_data(database_filepath):

    """
    Loads data from the Database

    Parameters
    ----------
    database_filepath : str
        Filepath to database
        
    Returns
    -------
    X : DataFrame
        A DataFrame of the training data
    Y : DataFrame
        A DataFrame of the test data
    cat_names : str
        Column names of Y
    """
    
    # Create engine & load data
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("Select * From Tweets", engine)
    
    # Split data
    X = df["message"]
    Y = df.loc[:,"related":]
    cat_names = Y.columns
    
    return X, Y, cat_names

def tokenize(text):
    
    """
    Tokenizes incoming text

    Parameters
    ----------
    text : str
        Tweets

    Returns
    -------
    list
        A tokenized text message
    """
    
    text = re.sub("[^A-Za-z0-9]", " ", text.lower())   # replace non-alph&digits
    words = word_tokenize(text)   # tokenize by word
    lemmatizer = WordNetLemmatizer()
    
    fin_list = []
    for w in [word for word in words if word not in stopwords.words("english")]:   # remove stopwords
        fin_list.append(lemmatizer.lemmatize(w, pos = "v"))   # stem and append to list
        
    return fin_list


def build_model():
    
    """
    Builds a model with parameter optimizing (inculding Cross-Validation)
    
    
    Returns
    -------
    GridSearchCV
        A GridSearchCV Object that is ready to fit
    """
    
    # tokenize & tfidf & create RFC
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer = tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(RandomForestClassifier(n_jobs = -1)))
    ]) 
    
    
    parameters = {
        #'tfidf__use_idf': (True, False),
        #'clf__estimator__max_leaf_nodes': ,
        'clf__estimator__n_estimators': np.arange(60, 100, 20),
        'clf__estimator__min_samples_leaf': range(1,3),
        'clf__estimator__criterion': ['gini', 'entropy']
    }
    
    
    return GridSearchCV(pipeline, param_grid = parameters)


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Predicts and evaluates with the fitted model
    
    
    Parameters
    ----------
    model : GridSearchCV
        A GridSearchCV Object that is ready to predict
    X_test : Series
        Features for predicting the labels
    Y_test : DataFrame
        A DataFrame with the labels to predict
    category_names : Series
        Labels of Y_test
    """
    
    y_pred = model.predict(X_test)   # prediction
    for i, col in enumerate(category_names):   # calculate scores
        print(col)
        print(classification_report(Y_test[col].values, y_pred[:,i]))
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    
   
def save_model(model, model_filepath):
    
    """
    Saves the model in a Pickle file
    
    Parameters
    ----------
    model : GridSearchCV
        A fitted GridSearchCV Object
    model_filepath : stt
        The path to store the model
    """
    
    joblib.dump(model, model_filepath, compress = 1)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()