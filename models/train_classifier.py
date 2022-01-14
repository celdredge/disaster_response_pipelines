import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download(['punkt','wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer




def load_data(database_filepath):
    from sqlalchemy import create_engine

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    
    X = df.message
    y = df.iloc[:,4:]
    
    # additional data cleaning - source https://knowledge.udacity.com/questions/764236
    y.related.replace(2,1,inplace=True)
    
    return X, y, y.columns


def tokenize(text):

    lemmatizer = WordNetLemmatizer()
    
    # remove punctuation, lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize
    tokens = word_tokenize(text)
    
    # lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


def build_model():
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.multiclass import OneVsRestClassifier

    model = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer = tokenize)),
        ('classifier', OneVsRestClassifier(LogisticRegression(max_iter = 200, penalty='l1'))),
                        ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    from sklearn.metrics import classification_report
    
    Y_pred_test = model.predict(X_test)

    print(classification_report(Y_test.values, Y_pred_test, target_names=Y_test.columns.values))
    
    pass


def save_model(model, model_filepath):
    
    import pickle
    
    pickle.dump(model, open('model_filepath.pkl', 'wb'))


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