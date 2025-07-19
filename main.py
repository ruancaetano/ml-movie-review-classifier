import nltk
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from pre_process import pre_process_dataset

def main():
    X, Y, vectorizer, le = pre_process_dataset()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    logistic_regression_model = train_logistic_regression_model(X_train, Y_train)
    linear_svc_model = train_linear_svc_model(X_train, Y_train)

    print("--------------------------------")
    print(f"Logistic Regression Model score: {logistic_regression_model.score(X_test, Y_test)}")
    print(f"Linear SVC Model score: {linear_svc_model.score(X_test, Y_test)}")

    test_review = "The movie is very very very good"
    to_predict = vectorizer.transform([test_review])

    print("--------------------------------")
    print(f"Logistic Regression Model prediction: {le.inverse_transform(logistic_regression_model.predict(to_predict))}")
    print(f"Linear SVC Model prediction: {le.inverse_transform(linear_svc_model.predict(to_predict))}")




def train_logistic_regression_model(X_train, Y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    return model

def train_linear_svc_model(X_train, Y_train):
    model = LinearSVC()
    model.fit(X_train, Y_train)

    return model




if __name__ == "__main__":
    main()