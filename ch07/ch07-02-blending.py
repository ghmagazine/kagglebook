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

# neural net用のデータ
train_nn = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x_nn = train_nn.drop(['target'], axis=1)
train_y_nn = train_nn['target']
test_x_nn = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# ---------------------------------
# hold-outデータへの予測値を用いたアンサンブル
# ----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_index = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_index]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_index]
tr_x_nn, va_x_nn = train_x_nn.iloc[tr_idx], train_x_nn.iloc[va_index]

# models.pyにModel1_1, Model1_2, Model2を定義しているものとする
# 各クラスは、fitで学習し、predictで予測値の確率を出力する
from models import Model1Xgb, Model1NN, Model2Linear

# 1層目のモデル
# 学習データで学習し、hold-outデータとテストデータへの予測値を出力する
model_1a = Model1Xgb()
model_1a.fit(tr_x, tr_y, va_x, va_y)
va_pred_1a = model_1a.predict(va_x)
test_pred_1a = model_1a.predict(test_x)

model_1b = Model1NN()
model_1b.fit(tr_x_nn, tr_y, va_x_nn, va_y)
va_pred_1b = model_1b.predict(va_x_nn)
test_pred_1b = model_1b.predict(test_x_nn)

# hold-outデータでの精度を評価する
print(f'logloss: {log_loss(va_y, va_pred_1a, eps=1e-7):.4f}')
print(f'logloss: {log_loss(va_y, va_pred_1b, eps=1e-7):.4f}')

# hold-outデータとテストデータへの予測値を特徴量としてデータフレームを作成
va_x_2 = pd.DataFrame({'pred_1a': va_pred_1a, 'pred_1b': va_pred_1b})
test_x_2 = pd.DataFrame({'pred_1a': test_pred_1a, 'pred_1b': test_pred_1b})

# 2層目のモデル
# Hold-outデータすべてで学習しているので、評価することができない。
# 評価を行うには、Hold-outデータをさらにクロスバリデーションする方法が考えられる。
model2 = Model2Linear()
model2.fit(va_x_2, va_y, None, None)
pred_test_2 = model2.predict(test_x_2)
