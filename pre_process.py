import pandas as pd
import re
from nltk.corpus import stopwords, words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


REVIEW_COLUMN = 'review'
SENTIMENT_COLUMN = 'sentiment'

stop_words = set(stopwords.words('english'))


def remove_duplicate_rows(dataset):
    print(f"Number of duplicate rows before: {dataset.duplicated(subset=[REVIEW_COLUMN]).sum()}")
    result = dataset.drop_duplicates(
        subset=[REVIEW_COLUMN],
        keep='first'
    )
    print(f"Number of duplicate rows after: {result.duplicated(subset=[REVIEW_COLUMN]).sum()}")
    return result


def remove_html_tags(text):
    return re.sub(r'<[^>]*>', '', text)


def to_lowercase(text):
    return text.lower()

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def remove_non_letters(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)


def apply_pre_processing(text):
    text = remove_html_tags(text)
    text = to_lowercase(text)
    text = remove_stopwords(text)
    text = remove_non_letters(text)
    return text


def pre_process_dataset():
    dataset = pd.read_csv("data/dataset.csv")

    dataset = remove_duplicate_rows(dataset)

    # Apply preprocessing more efficiently using pandas apply
    dataset[REVIEW_COLUMN] = dataset[REVIEW_COLUMN].apply(apply_pre_processing)

    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(dataset[REVIEW_COLUMN])
    
    le = LabelEncoder()
    Y = le.fit_transform(dataset[SENTIMENT_COLUMN])
   
    return X, Y, vectorizer, le

