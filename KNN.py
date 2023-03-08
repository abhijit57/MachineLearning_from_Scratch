import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


import warnings
warnings.filterwarnings("ignore")


class KNNClassifier:
    '''
    Initialize k with any value, let's say k = 3
    Compute distances between all the points in the dataset using any type of distance metric depending upon your use case. Euclidean distance method has been used here.
    Arrange the distances in ascending order and select k distances (ascending order arranged)
    Find the most frequent class/label from the k selected distances
    '''

    def __init__(self, K):
        # Initialize K
        self.K = K

    def fit(self, X_train, y_train):
        # Initialize X_train and y_train
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # Initialize X_test and y_pred
        self.X_test = X_test
        y_pred = np.zeros(self.X_test.shape[0])

        for i, x in enumerate(self.X_test):
            # Compute euclidean distances for the points in the dataset
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            # Extract the indices of the k distances arranged in ascending order
            idx = np.argsort(distances)[:self.K]
            # Get the labels based on the indices extracted in the previous step
            kn_labels = self.y_train[idx]
            # Mode of the labels from the k selected distances
            y_pred[i] = np.bincount(kn_labels).argmax()

        return y_pred
    

class KNNRegressor(KNNClassifier):
    '''
    Repeat steps 1-3 of KNN Classifier
    Find the mean of the true target values of those k neighbors
    '''
    def __init__(self, K):
        super().__init__(K)

    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)
    
    def predict(self, X_test):
        self.X_test = X_test
        y_pred = np.zeros(self.X_test.shape[0])

        for i, x in enumerate(self.X_test):
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            idx = np.argsort(distances)[:self.K]
            kn_labels = self.y_train[idx]
            y_pred[i] = np.mean(kn_labels)
        
        return y_pred


# Load digits dataset from sklearn datasets module
Xd, yd = load_digits(return_X_y=True)
# Perform a train and test split on the digits dataset
Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.2, random_state=23)

def main(model, xtrain, ytrain, xtest):
    model = model
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    return ypred


if __name__ == "__main__":

    print()
    yd_pred = main(KNNClassifier(3), Xd_train, yd_train, Xd_test)
    print('Accuracy of KNN Classsifier from scratch: ', round((np.sum(yd_pred == yd_test)/len(yd_test)*100),2))

    ydsk_pred = main(KNeighborsClassifier(n_neighbors=3), Xd_train, yd_train, Xd_test)
    print('Accuracy of KNN Classifier from sklearn: ', round((accuracy_score(ydsk_pred, yd_test)*100),2))

    print()
    yd_predr = main(KNNRegressor(3), Xd_train, yd_train, Xd_test)
    print('Accuracy of KNN Regressor from scratch: ', round((np.sum(yd_predr == yd_test)/len(yd_test)*100),2))

    ydsk_predr = main(KNeighborsRegressor(n_neighbors=3), Xd_train, yd_train, Xd_test)
    print('Accuracy of KNN Regressor from sklearn: ', round((np.sum(ydsk_predr == yd_test)/len(yd_test)*100),2))
    print()





## Time complexity of the program
'''
The time complexity of this code is mainly determined by the predict method. The fit and __init__ methods have constant time complexity O(1), since they just store the training data and hyperparameters.

The predict method has a time complexity of O(qnk), where q is the number of test samples, n is the number of training samples and k is the number of nearest neighbors. Let's break down the time complexity of each step in the predict method:

Loop over each test sample: This takes O(q) time.

Compute distances between the test sample and all training samples: This involves computing the Euclidean distance between the test sample and all n training samples, which takes O(n) time for each test sample. Therefore, this step takes O(qn) time in total.

Sort distances and select the k nearest neighbors: Sorting n distances takes O(n log n) time, and selecting the k smallest distances takes O(k) time. Since we perform this operation for each test sample, this step takes O(q(n log n + k)) time in total.

Compute the most frequent class/label from the k selected distances: This involves computing the mode of the k labels, which takes O(k) time. Since we perform this operation for each test sample, this step takes O(qk) time in total.

Therefore, the overall time complexity of the predict method is O(qnk). Note that this code assumes a fixed number of neighbors k, which can be a hyperparameter that is tuned during the model selection process.
'''