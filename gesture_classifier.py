import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class GestureClassifier:
    def __init__(self, dataset_path, n_neighbors=5):
        self.__scaler = None
        self.__classifier = KNeighborsClassifier(n_neighbors)
        self.__train_model(dataset_path)

    def __train_model(self, dataset_path):
        dataset = pd.read_csv(dataset_path)

        X = dataset.iloc[:, 1:].values
        y = dataset.iloc[:, 0].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        self.__scaler = StandardScaler()
        self.__scaler.fit(X_train)

        X_train = self.__scaler.transform(X_train)

        self.__classifier.fit(X_train, y_train)

    def predict_gesture(self, features_data):
        pred = self.__scaler.transform(features_data)
        return self.__classifier.predict(pred)
