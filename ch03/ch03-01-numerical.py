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

# 説明用に学習データとテストデータの元の状態を保存しておく
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()


# 学習データとテストデータを返す関数
def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x


# 変換する数値変数をリストに格納
num_cols = ['age', 'height', 'weight', 'amount',
            'medical_info_a1', 'medical_info_a2', 'medical_info_a3', 'medical_info_b1']

# -----------------------------------
# 標準化
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 学習データに基づいて複数列の標準化を定義
scaler = StandardScaler()
scaler.fit(train_x[num_cols])

# 変換後のデータで各列を置換
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 学習データとテストデータを結合したものに基づいて複数列の標準化を定義
scaler = StandardScaler()
scaler.fit(pd.concat([train_x[num_cols], test_x[num_cols]]))

# 変換後のデータで各列を置換
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 学習データとテストデータを別々に標準化（悪い例）
scaler_train = StandardScaler()
scaler_train.fit(train_x[num_cols])
train_x[num_cols] = scaler_train.transform(train_x[num_cols])
scaler_test = StandardScaler()
scaler_test.fit(test_x[num_cols])
test_x[num_cols] = scaler_test.transform(test_x[num_cols])

# -----------------------------------
# Min-Maxスケーリング
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import MinMaxScaler

# 学習データに基づいて複数列のMin-Maxスケーリングを定義
scaler = MinMaxScaler()
scaler.fit(train_x[num_cols])

# 変換後のデータで各列を置換
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# 対数変換
# -----------------------------------
x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

# 単に対数をとる
x1 = np.log(x)

# 1を加えたあとに対数をとる
x2 = np.log1p(x)

# 絶対値の対数をとってから元の符号を付加する
x3 = np.sign(x) * np.log(np.abs(x))

# -----------------------------------
# Box-Cox変換
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------

# 正の値のみをとる変数を変換対象としてリストに格納する
# なお、欠損値も含める場合は、(~(train_x[c] <= 0.0)).all() などとする必要があるので注意
pos_cols = [c for c in num_cols if (train_x[c] > 0.0).all() and (test_x[c] > 0.0).all()]

from sklearn.preprocessing import PowerTransformer

# 学習データに基づいて複数列のBox-Cox変換を定義
pt = PowerTransformer(method='box-cox')
pt.fit(train_x[pos_cols])

# 変換後のデータで各列を置換
train_x[pos_cols] = pt.transform(train_x[pos_cols])
test_x[pos_cols] = pt.transform(test_x[pos_cols])

# -----------------------------------
# Yeo-Johnson変換
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------

from sklearn.preprocessing import PowerTransformer

# 学習データに基づいて複数列のYeo-Johnson変換を定義
pt = PowerTransformer(method='yeo-johnson')
pt.fit(train_x[num_cols])

# 変換後のデータで各列を置換
train_x[num_cols] = pt.transform(train_x[num_cols])
test_x[num_cols] = pt.transform(test_x[num_cols])

# -----------------------------------
# clipping
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
# 列ごとに学習データの1％点、99％点を計算
p01 = train_x[num_cols].quantile(0.01)
p99 = train_x[num_cols].quantile(0.99)

# 1％点以下の値は1％点に、99％点以上の値は99％点にclippingする
train_x[num_cols] = train_x[num_cols].clip(p01, p99, axis=1)
test_x[num_cols] = test_x[num_cols].clip(p01, p99, axis=1)

# -----------------------------------
# binning
# -----------------------------------
x = [1, 7, 5, 4, 6, 3]

# pandasのcut関数でbinningを行う

# binの数を指定する場合
binned = pd.cut(x, 3, labels=False)
print(binned)
# [0 2 1 1 2 0] - 変換された値は3つのbinのどれに入ったかを表す

# binの範囲を指定する場合（3.0以下、3.0より大きく5.0以下、5.0より大きい）
bin_edges = [-float('inf'), 3.0, 5.0, float('inf')]
binned = pd.cut(x, bin_edges, labels=False)
print(binned)
# [0 2 1 1 2 0] - 変換された値は3つのbinのどれに入ったかを表す

# -----------------------------------
# 順位への変換
# -----------------------------------
x = [10, 20, 30, 0, 40, 40]

# pandasのrank関数で順位に変換する
rank = pd.Series(x).rank()
print(rank.values)
# はじまりが1、同順位があった場合は平均の順位となる
# [2. 3. 4. 1. 5.5 5.5]

# numpyのargsort関数を2回適用する方法で順位に変換する
order = np.argsort(x)
rank = np.argsort(order)
print(rank)
# はじまりが0、同順位があった場合はどちらかが上位となる
# [1 2 3 0 4 5]

# -----------------------------------
# RankGauss
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import QuantileTransformer

# 学習データに基づいて複数列のRankGaussによる変換を定義
transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
transformer.fit(train_x[num_cols])

# 変換後のデータで各列を置換
train_x[num_cols] = transformer.transform(train_x[num_cols])
test_x[num_cols] = transformer.transform(test_x[num_cols])
