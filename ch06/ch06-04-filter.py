# ---------------------------------
# データ等の準備
# ----------------------------------
import numpy as np
import pandas as pd

# train_xは学習データ、train_yは目的変数、test_xはテストデータ
# pandasのDataFrame, Seriesで保持します。（numpyのarrayで保持することもあります）

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# ---------------------------------
# argsortによるインデックスのソート
# ---------------------------------
# argsortを使うことで、配列の値が小さい順／大きい順にインデックスをソートできる
ary = np.array([10, 20, 30, 0])
idx = ary.argsort()
print(idx)  # 降順 - [3 0 1 2]
print(idx[::-1])  # 昇順 - [2 1 0 3]

print(ary[idx[::-1][:3]])  # ベスト3を出力 - [30, 20, 10]

# ---------------------------------
# 相関係数
# ---------------------------------
import scipy.stats as st

# 相関係数
corrs = []
for c in train_x.columns:
    corr = np.corrcoef(train_x[c], train_y)[0, 1]
    corrs.append(corr)
corrs = np.array(corrs)

# スピアマンの順位相関係数
corrs_sp = []
for c in train_x.columns:
    corr_sp = st.spearmanr(train_x[c], train_y).correlation
    corrs_sp.append(corr_sp)
corrs_sp = np.array(corrs_sp)

# 重要度の上位を出力する（上位5個まで）
# np.argsortを使うことで、値の順序のとおりに並べたインデックスを取得できる
idx = np.argsort(np.abs(corrs))[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)

idx2 = np.argsort(np.abs(corrs_sp))[::-1]
top_cols2, top_importances2 = train_x.columns.values[idx][:5], corrs_sp[idx][:5]
print(top_cols2, top_importances2)

# ---------------------------------
# カイ二乗統計量
# ---------------------------------
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# カイ二乗統計量
x = MinMaxScaler().fit_transform(train_x)
c2, _ = chi2(x, train_y)

# 重要度の上位を出力する（上位5個まで）
idx = np.argsort(c2)[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)

# ---------------------------------
# 相互情報量
# ---------------------------------
from sklearn.feature_selection import mutual_info_classif

# 相互情報量
mi = mutual_info_classif(train_x, train_y)

# 重要度の上位を出力する（上位5個まで）
idx = np.argsort(mi)[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)
