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

# 時系列データであり、時間に沿って変数periodを設定したとする
train_x['period'] = np.arange(0, len(train_x)) // (len(train_x) // 4)
train_x['period'] = np.clip(train_x['period'], 0, 3)
test_x['period'] = 4

# -----------------------------------
# 時系列データのhold-out法
# -----------------------------------
# 変数periodを基準に分割することにする（0から3までが学習データ、4がテストデータとする）
# ここでは、学習データのうち、変数periodが3のデータをバリデーションデータとし、0から2までのデータを学習に用いる
is_tr = train_x['period'] < 3
is_va = train_x['period'] == 3
tr_x, va_x = train_x[is_tr], train_x[is_va]
tr_y, va_y = train_y[is_tr], train_y[is_va]

# -----------------------------------
# 時系列データのクロスバリデーション（時系列に沿って行う方法）
# -----------------------------------
# 変数periodを基準に分割することにする（0から3までが学習データ、4がテストデータとする）
# 変数periodが1, 2, 3のデータをそれぞれバリデーションデータとし、それ以前のデータを学習に使う

va_period_list = [1, 2, 3]
for va_period in va_period_list:
    is_tr = train_x['period'] < va_period
    is_va = train_x['period'] == va_period
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]

# （参考）periodSeriesSplitの場合、データの並び順しか使えないため使いづらい
from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=4)
for tr_idx, va_idx in tss.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# 時系列データのクロスバリデーション（単純に時間で分割する方法）
# -----------------------------------
# 変数periodを基準に分割することにする（0から3までが学習データ、4がテストデータとする）
# 変数periodが0, 1, 2, 3のデータをそれぞれバリデーションデータとし、それ以外の学習データを学習に使う

va_period_list = [0, 1, 2, 3]
for va_period in va_period_list:
    is_tr = train_x['period'] != va_period
    is_va = train_x['period'] == va_period
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]
