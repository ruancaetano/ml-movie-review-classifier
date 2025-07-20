import nltk
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pre_process import pre_process_dataset

def main():
    print("--------------------------------")
    print("Pre-processing dataset...")

    X, Y, vectorizer, le = pre_process_dataset()


    print("--------------------------------")
    print("Training models...")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    logistic_regression_model = train_logistic_regression_model(X_train, Y_train)
    random_forest_model = train_random_forest_model(X_train, Y_train)

    print("--------------------------------")
    print(f"Logistic Regression Model score: {logistic_regression_model.score(X_test, Y_test)}")
    print(f"Random Forest Model score: {random_forest_model.score(X_test, Y_test)}")

    test_review = "The movie is very very very good"
    to_predict = vectorizer.transform([test_review])

    print("--------------------------------")
    print(f"Test review: {test_review}")
    print(f"Logistic Regression Model prediction: {le.inverse_transform(logistic_regression_model.predict(to_predict))}")
    print(f"Random Forest Model prediction: {le.inverse_transform(random_forest_model.predict(to_predict))}")

def train_logistic_regression_model(X_train, Y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    return model

def train_random_forest_model(X_train, Y_train):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    return model




if __name__ == "__main__":
    main()