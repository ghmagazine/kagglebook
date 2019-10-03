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

# -----------------------------------
# xgboostの実装
# -----------------------------------
import xgboost as xgb
from sklearn.metrics import log_loss

# 特徴量と目的変数をxgboostのデータ構造に変換する
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(test_x)

# ハイパーパラメータの設定
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
num_round = 50

# 学習の実行
# バリデーションデータもモデルに渡し、学習の進行とともにスコアがどう変わるかモニタリングする
# watchlistには学習データおよびバリデーションデータをセットする
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist)

# バリデーションデータでのスコアの確認
va_pred = model.predict(dvalid)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# 予測（二値の予測値ではなく、1である確率を出力するようにしている）
pred = model.predict(dtest)

# -----------------------------------
# 学習データとバリデーションデータのスコアのモニタリング
# -----------------------------------
# モニタリングをloglossで行い、アーリーストッピングの観察するroundを20とする
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71,
          'eval_metric': 'logloss'}
num_round = 500
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist,
                  early_stopping_rounds=20)

# 最適な決定木の本数で予測を行う
pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
