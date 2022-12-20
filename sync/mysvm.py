import os
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import pandas
from .data import DataProcessor


class SVCModel(object):
    def __init__(self, x, y):
        self.clf = SVC(kernel='rbf')
        self.x = x
        self.y = y

    def split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, shuffle=False)

        return [self.x_train, self.x_test, self.y_train, self.y_test]

    def train(self):
        self.split()
        self.clf.fit(self.x_train, self.y_train)
        accuracy = self.clf.score(self.x_test, self.y_test)
        print('accuracy: {}'.format(accuracy))

    def predict(self, x):
        return clf.predict(x)



if __name__ == '__main__':
    dp = DataProcessor()
    svm = SVCModel(dp.train_data[['ts', 'x1']], dp.train_data[['y']])
    svm.train()
