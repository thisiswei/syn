import pandas
import os


CWD = os.path.dirname(__file__)


class DataProcessor(object):
    def __init__(self):
        cwd = os.getcwd()
        self._df_x1_train = pandas.read_csv(os.path.abspath(os.path.join(CWD, "../data/x1_train.csv")))
        self._df_x2_train = pandas.read_csv(os.path.abspath(os.path.join(CWD, '../data/x2_train.csv')))
        self._df_y = pandas.read_csv(os.path.abspath(os.path.join(CWD, '../data/y_train.csv')))
        self._x1_test = pandas.read_csv(os.path.abspath(os.path.join(CWD, '../data/x1_test.csv')))
        self._x2_test = pandas.read_csv(os.path.abspath(os.path.join(CWD, '../data/x2_test.csv')))
        self._train_data = None
        self._test_data = None

    @property
    def train_data(self):
        if self._train_data is not None:
            return self._train_data

        self._train_data = self._preprocess(self._df_x1_train, self._df_x2_train, self._df_y)
        return self._train_data

    @property
    def test_data(self):
        if self._test_data is not None:
            return self._test_data

        self._test_data = self._preprocess(self._x1_test, self._x2_test)
        return self._test_data

    def _prepare_data(self, df):
        return self._sort(self._add_ts(df, ts_column='ts'), 'ts').dropna()

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

    def _preprocess(self, x1, x2, y=None):
        dfx1 = self._prepare_data(x1)
        dfx2 = self._prepare_data(x2)
        df = dfx1.merge(dfx2, on='ts', how='inner', suffixes=('_dfx1', '_dfx2'))
        df = df.rename(columns={
            "Adj Close_dfx1": "x1",
            "Adj Close_dfx2": 'x2',
        })

        if y is not None:
            dfy = self._prepare_data(y)
            df = df.merge(dfy, on='ts', how='inner')
            df = df.rename(columns={
                "Adj Close": "y",
            })
            return df[['ts', 'x1', 'x2', 'y']]
        else:
            return df[['ts', 'x1', 'x2']]
