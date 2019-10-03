# ---------------------------------
# データ等の準備
# ----------------------------------
import numpy as np
import pandas as pd

# train_xは学習データ、train_yは目的変数、test_xはテストデータ
# pandasのDataFrame, Seriesで保持します。（numpyのarrayで保持することもあります）

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

# 学習データを学習データとバリデーションデータに分ける
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# xgboostによる学習・予測を行うクラス
import xgboost as xgb


class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y, va_x, va_y):
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


# -----------------------------------
# 探索するパラメータの空間の指定
# -----------------------------------
# hp.choiceでは、複数の選択肢から選ぶ
# hp.uniformでは、下限・上限を指定した一様分布から抽出する。引数は下限・上限
# hp.quniformでは、下限・上限を指定した一様分布のうち一定の間隔ごとの点から抽出する。引数は下限・上限・間隔
# hp.loguniformでは、下限・上限を指定した対数が一様分布に従う分布から抽出する。引数は下限・上限の対数をとった値

from hyperopt import hp

space = {
    'activation': hp.choice('activation', ['prelu', 'relu']),
    'dropout': hp.uniform('dropout', 0, 0.2),
    'units': hp.quniform('units', 32, 256, 32),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.01)),
}

# -----------------------------------
# hyperoptを使ったパラメータ探索
# -----------------------------------
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import log_loss


def score(params):
    # パラメータを与えたときに最小化する評価指標を指定する
    # 具体的には、モデルにパラメータを指定して学習・予測させた場合のスコアを返すようにする

    # max_depthの型を整数型に修正する
    params['max_depth'] = int(params['max_depth'])

    # Modelクラスを定義しているものとする
    # Modelクラスは、fitで学習し、predictで予測値の確率を出力する
    model = Model(params)
    model.fit(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    print(f'params: {params}, logloss: {score:.4f}')

    # 情報を記録しておく
    history.append((params, score))

    return {'loss': score, 'status': STATUS_OK}


# 探索するパラメータの空間を指定する
space = {
    'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
    'max_depth': hp.quniform('max_depth', 3, 9, 1),
    'gamma': hp.quniform('gamma', 0, 0.4, 0.1),
}

# hyperoptによるパラメータ探索の実行
max_evals = 10
trials = Trials()
history = []
fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

# 記録した情報からパラメータとスコアを出力する
# （trialsからも情報が取得できるが、パラメータの取得がやや行いづらいため）
history = sorted(history, key=lambda tpl: tpl[1])
best = history[0]
print(f'best params:{best[0]}, score:{best[1]:.4f}')
