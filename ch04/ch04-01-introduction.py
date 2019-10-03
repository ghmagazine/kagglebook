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

import xgboost as xgb


# コードの動作を確認するためのモデル
class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y):
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        self.model = xgb.train(params, dtrain, num_round)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


# -----------------------------------
# モデルの学習と予測
# -----------------------------------
# モデルのハイパーパラメータを指定する
params = {'param1': 10, 'param2': 100}

# Modelクラスを定義しているものとする
# Modelクラスは、fitで学習し、predictで予測値の確率を出力する

# モデルを定義する
model = Model(params)

# 学習データに対してモデルを学習させる
model.fit(train_x, train_y)

# テストデータに対して予測結果を出力する
pred = model.predict(test_x)

# -----------------------------------
# バリデーション
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# 学習データ・バリデーションデータを分けるためのインデックスを作成する
# 学習データを4つに分割し、うち1つをバリデーションデータとする
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# 学習データを学習データとバリデーションデータに分ける
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# モデルを定義する
model = Model(params)

# 学習データに対してモデルを学習させる
# モデルによっては、バリデーションデータを同時に与えてスコアをモニタリングすることができる
model.fit(tr_x, tr_y)

# バリデーションデータに対して予測し、評価を行う
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# -----------------------------------
# クロスバリデーション
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# 学習データを4つに分け、うち1つをバリデーションデータとする
# どれをバリデーションデータとするかを変えて学習・評価を4回行う
scores = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    model = Model(params)
    model.fit(tr_x, tr_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    scores.append(score)

# クロスバリデーションの平均のスコアを出力する
print(f'logloss: {np.mean(scores):.4f}')
