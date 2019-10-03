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

from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# 学習データを学習データとバリデーションデータに分ける
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# xgboost におけるカスタム評価指標と目的関数の例
# （参考）https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py
# -----------------------------------
import xgboost as xgb
from sklearn.metrics import log_loss

# 特徴量と目的変数をxgboostのデータ構造に変換する
# 学習データの特徴量と目的変数がtr_x, tr_x、バリデーションデータの特徴量と目的変数がva_x, va_yとする
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)


# カスタム目的関数（この場合はloglossであり、xgboostの'binary:logistic'と等価）
def logregobj(preds, dtrain):
    labels = dtrain.get_label()  # 真の値のラベルを取得
    preds = 1.0 / (1.0 + np.exp(-preds))  # シグモイド関数
    grad = preds - labels  # 勾配
    hess = preds * (1.0 - preds)  # 二階微分値
    return grad, hess


# カスタム評価指標（この場合は誤答率）
def evalerror(preds, dtrain):
    labels = dtrain.get_label()  # 真の値のラベルを取得
    return 'custom-error', float(sum(labels != (preds > 0.0))) / len(labels)


# ハイパーパラメータの設定
params = {'silent': 1, 'random_state': 71}
num_round = 50
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# モデルの学習の実行
bst = xgb.train(params, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror)

# 目的関数にbinary:logisticを指定したときと違い、確率に変換する前の値で予測値が出力されるので変換が必要
pred_val = bst.predict(dvalid)
pred = 1.0 / (1.0 + np.exp(-pred_val))
logloss = log_loss(va_y, pred)
print(logloss)

# （参考）通常の方法で学習を行う場合
params = {'silent': 1, 'random_state': 71, 'objective': 'binary:logistic'}
bst = xgb.train(params, dtrain, num_round, watchlist)

pred = bst.predict(dvalid)
logloss = log_loss(va_y, pred)
print(logloss)
