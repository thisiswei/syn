import os

import pandas
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.varmax import VARMAX


class VARMAXModel(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def train(self):
        size = len(self.x)
        training_size = int(size * 0.8)

        x_train = self.x[:training_size]
        x_test = self.x[training_size:]
        y_test = self.y[training_size:]

        # TODO: how to use more lags without error?
        model = VARMAX(x_train, exog=x_train[['ts', 'x']], order=(1, 0, 0), trend='c')
        self.model_fit = model.fit()

        predictions = self.model_fit.forecast(steps=len(x_test), exog=x_test[['ts', 'x']])
        mse = mean_squared_error(y_test['y'], predictions['y'])
        print('MSE: {}'.format(mse))
        return predictions

    def predict(self, x):
        predicted = self.model_fit.forecast(steps=len(x), exog=x)
        return predicted


def _sort(df, column_name):
    return df.sort_values(column_name, ascending=True, inplace=False)


def _add_ts(df, date_column='Date', ts_column='ts'):
    try:
        df[date_column] = pandas.to_datetime(df[date_column], format='%Y-%m-%d')
    except ValueError:
        df[date_column] = pandas.to_datetime(df[date_column], dayfirst=True)

    df[ts_column] = df[date_column].apply(lambda x: x.timestamp())
    return df


if __name__ == '__main__':
    cwd = os.getcwd()
    df_x1_train = pandas.read_csv(os.path.abspath(os.path.join(cwd, "../data/x1_train.csv")))
    df_x2_train = pandas.read_csv(os.path.abspath(os.path.join(cwd, '../data/x2_train.csv')))
    df_y = pandas.read_csv(os.path.abspath(os.path.join(cwd, '../data/y_train.csv')))
    df_x1_test = pandas.read_csv(os.path.abspath(os.path.join(cwd, '../data/x1_test.csv')))
    df_x2_test = pandas.read_csv(os.path.abspath(os.path.join(cwd, '../data/x2_test.csv')))

    dfx = _sort(_add_ts(df_x1_train, ts_column='ts'), 'ts').dropna()
    dfy = _sort(_add_ts(df_y, ts_column='ts'), 'ts').dropna()
    df = dfx.merge(dfy, on='ts', how='inner', suffixes=('_dfx', '_dfy'))
    df = df.rename(columns={
        "Adj Close_dfx": "x",
        "Adj Close_dfy": "y",
    })
    x = df[['ts', 'x', 'y']]
    y = df[['ts', 'y']]


    def try_varmax():
        var = VARMAXModel(x, y)
        var.train()
        df = _add_ts(df_x1_test)
        predicted = (var.predict(df[['ts', 'Adj Close']]))
        predicted['date'] = df['Date'].values
        path = '/tmp/y.csv'
        predicted.to_csv(path)
        print('csv file saved in here: {}'.format(path))

    try_varmax()
