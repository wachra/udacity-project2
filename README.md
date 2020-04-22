# Udacity Data Science Nanodegree - Project 2
This Repository contains the files for Udacity's Data Science Nanodegree Project 2.

### 1. Summary  
In this repository a data engineering pipeline is built to analyze and classify disaster text. The classifier can identify which type of support is needed bases on messages. The trained model is used in a web application.

### 2. Installations
- [Python 3.X](https://www.python.org/downloads/)
- [Numpy](https://pypi.org/project/numpy/)
- [Pandas](https://pypi.org/project/pandas/)
- [SQLAlchemy](https://pypi.org/project/SQLAlchemy/)
- [nltk](https://pypi.org/project/nltk/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)

### 3. File Descriptions
- **/app/run.py**: 
- **/data/DisasterResponse.db**: 
- **/data/process_data.py**:
- **/models/train_classifier.py**:

### 4. Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python3 run.py`

3. Go to http://0.0.0.0:3001/


### 5. Acknowledgements
Thanks to [Figure Eight](https://figure-eight.com) who made their *Multilingual Disaster Response Messages* Dataset freely [availabe](https://appen.com/datasets/combined-disaster-response-data/).
