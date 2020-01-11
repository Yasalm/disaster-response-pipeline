import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sqlalchemy import create_engine
# from sklearn.linear_model import SGDClassifier
# from sklearn.neighbors import KNeighborsClassifier
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    """
    Function to load data
    Arguments:
        database_filepath: String. Holds the path to database (sqllite)
    Return:
        X: Dataframe. Training data (features).
        y: Dataframes. labeles.
        categories: column names of labels.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM df", engine)
    X = df['message']
    y = df.iloc[:, 4:]
    categories = y.columns
    return X, y, categories


def tokenize(text):
    """
    Function to tokenize text.
    Arguments:
        text: text to tokenize
    Return:
        cleaned_tokenize: Toknized text.
    """
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()


    _lemmatize = lambda token: lemmatizer.lemmatize(token).lower().strip()
    cleaned_tokens = [ _lemmatize(token) for token in tokens ]

    return cleaned_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting verb extractor.
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Function to build model.
    Arguments:
        None.
    Return:
        Pipeline: Scikit ML Pipeline.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate the model performance.
    Arguments:
        model: model Pipelline.
        X_test: test features
        Y_test: test labels
        category_names: label names
    """
    y_pre = model.predict(X_test)

    for index, column in enumerate(Y_test.columns):
        print(f'Model Performance with Feature: {column}')
        print(classification_report(Y_test[column],y_pre[:, index]))
    pass


def save_model(model, model_filepath):
    """
    Function to save the model.
    Arguments:
        model: model Pipelline.
        model_filepath: path to save model.
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


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