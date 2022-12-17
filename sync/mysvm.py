import os
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import pandas



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


class DataProcessor(object):
    def __init__(self):
        cwd = os.getcwd()
        df_x1_train = pandas.read_csv(os.path.abspath(os.path.join(cwd, "../data/x1_train.csv")))
        df_x2_train = pandas.read_csv(os.path.abspath(os.path.join(cwd, '../data/x2_train.csv')))
        df_y = pandas.read_csv(os.path.abspath(os.path.join(cwd, '../data/y_train.csv')))
        df_x1_test = pandas.read_csv(os.path.abspath(os.path.join(cwd, '../data/x1_test.csv')))
        df_x2_test = pandas.read_csv(os.path.abspath(os.path.join(cwd, '../data/x2_test.csv')))

        dfx1 = self._sort(self._add_ts(df_x1_train, ts_column='ts'), 'ts').dropna()
        dfx2 = self._sort(self._add_ts(df_x2_train, ts_column='ts'), 'ts').dropna()
        dfy = self._sort(self._add_ts(df_y, ts_column='ts'), 'ts').dropna()

        df = (
             dfx1.merge(dfy, on='ts', how='inner', suffixes=('_dfx', '_dfy'))
                 .merge(dfx2, on='ts', how='inner', suffixes=('_x1', '_x2'))
        )

        df = df.rename(columns={
            "Adj Close_dfx": "x1",
            "Adj Close": 'x2',
            "Adj Close_dfy": "y",
        })

        self._df = df[['ts', 'x1', 'x2', 'y']]

    @property
    def df(self):
        return self._df

    @staticmethod
    def _sort(df, column_name):
        return df.sort_values(column_name, ascending=True, inplace=False)

    @staticmethod
    def _add_ts(df, date_column='Date', ts_column='ts'):
        try:
            df[date_column] = pandas.to_datetime(df[date_column], format='%Y-%m-%d')
        except ValueError:
            df[date_column] = pandas.to_datetime(df[date_column], dayfirst=True)

        df[ts_column] = df[date_column].apply(lambda x: x.timestamp())
        return df



if __name__ == '__main__':
    dp = DataProcessor()
    svm = SVCModel(dp.df[['ts', 'x1']], dp.df[['y']])
    svm.train()
