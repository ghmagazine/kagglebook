# ---------------------------------
# データ等の準備
# ----------------------------------
import numpy as np
import pandas as pd

# train_xは学習データ、train_yは目的変数、test_xはテストデータ
# pandasのDataFrame, Seriesで保持します。（numpyのarrayで保持することもあります）

train = pd.read_csv('../input/sample-data/train.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test.csv')

# 説明用に学習データとテストデータの元の状態を保存しておく
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()


# 学習データとテストデータを返す関数
def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x


# 変換するカテゴリ変数をリストに格納
cat_cols = ['sex', 'product', 'medical_info_b2', 'medical_info_b3']

# -----------------------------------
# one-hot encoding
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------

# 学習データとテストデータを結合してget_dummiesによるone-hot encodingを行う
all_x = pd.concat([train_x, test_x])
all_x = pd.get_dummies(all_x, columns=cat_cols)

# 学習データとテストデータに再分割
train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True)
test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True)

# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# OneHotEncoderでのencoding
ohe = OneHotEncoder(sparse=False, categories='auto')
ohe.fit(train_x[cat_cols])

# ダミー変数の列名の作成
columns = []
for i, c in enumerate(cat_cols):
    columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 生成されたダミー変数をデータフレームに変換
dummy_vals_train = pd.DataFrame(ohe.transform(train_x[cat_cols]), columns=columns)
dummy_vals_test = pd.DataFrame(ohe.transform(test_x[cat_cols]), columns=columns)

# 残りの変数と結合
train_x = pd.concat([train_x.drop(cat_cols, axis=1), dummy_vals_train], axis=1)
test_x = pd.concat([test_x.drop(cat_cols, axis=1), dummy_vals_test], axis=1)

# -----------------------------------
# label encoding
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# カテゴリ変数をループしてlabel encoding
for c in cat_cols:
    # 学習データに基づいて定義する
    le = LabelEncoder()
    le.fit(train_x[c])
    train_x[c] = le.transform(train_x[c])
    test_x[c] = le.transform(test_x[c])

# -----------------------------------
# feature hashing
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
from sklearn.feature_extraction import FeatureHasher

# カテゴリ変数をループしてfeature hashing
for c in cat_cols:
    # FeatureHasherの使い方は、他のencoderとは少し異なる

    fh = FeatureHasher(n_features=5, input_type='string')
    # 変数を文字列に変換してからFeatureHasherを適用
    hash_train = fh.transform(train_x[[c]].astype(str).values)
    hash_test = fh.transform(test_x[[c]].astype(str).values)
    # データフレームに変換
    hash_train = pd.DataFrame(hash_train.todense(), columns=[f'{c}_{i}' for i in range(5)])
    hash_test = pd.DataFrame(hash_test.todense(), columns=[f'{c}_{i}' for i in range(5)])
    # 元のデータフレームと結合
    train_x = pd.concat([train_x, hash_train], axis=1)
    test_x = pd.concat([test_x, hash_test], axis=1)

# 元のカテゴリ変数を削除
train_x.drop(cat_cols, axis=1, inplace=True)
test_x.drop(cat_cols, axis=1, inplace=True)

# -----------------------------------
# frequency encoding
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
# 変数をループしてfrequency encoding
for c in cat_cols:
    freq = train_x[c].value_counts()
    # カテゴリの出現回数で置換
    train_x[c] = train_x[c].map(freq)
    test_x[c] = test_x[c].map(freq)

# -----------------------------------
# target encoding
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# 変数をループしてtarget encoding
for c in cat_cols:
    # 学習データ全体で各カテゴリにおけるtargetの平均を計算
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    target_mean = data_tmp.groupby(c)['target'].mean()
    # テストデータのカテゴリを置換
    test_x[c] = test_x[c].map(target_mean)

    # 学習データの変換後の値を格納する配列を準備
    tmp = np.repeat(np.nan, train_x.shape[0])

    # 学習データを分割
    kf = KFold(n_splits=4, shuffle=True, random_state=72)
    for idx_1, idx_2 in kf.split(train_x):
        # out-of-foldで各カテゴリにおける目的変数の平均を計算
        target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        # 変換後の値を一時配列に格納
        tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

    # 変換後のデータで元の変数を置換
    train_x[c] = tmp

# -----------------------------------
# target encoding - クロスバリデーションのfoldごとの場合
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# クロスバリデーションのfoldごとにtarget encodingをやり直す
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):

    # 学習データからバリデーションデータを分ける
    tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 変数をループしてtarget encoding
    for c in cat_cols:
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # バリデーションデータのカテゴリを置換
        va_x.loc[:, c] = va_x[c].map(target_mean)

        # 学習データの変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, tr_x.shape[0])
        kf_encoding = KFold(n_splits=4, shuffle=True, random_state=72)
        for idx_1, idx_2 in kf_encoding.split(tr_x):
            # out-of-foldで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # 変換後の値を一時配列に格納
            tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)

        tr_x.loc[:, c] = tmp

    # 必要に応じてencodeされた特徴量を保存し、あとで読み込めるようにしておく

# -----------------------------------
# target encoding - クロスバリデーションのfoldとtarget encodingのfoldの分割を合わせる場合
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# クロスバリデーションのfoldを定義
kf = KFold(n_splits=4, shuffle=True, random_state=71)

# 変数をループしてtarget encoding
for c in cat_cols:

    # targetを付加
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    # 変換後の値を格納する配列を準備
    tmp = np.repeat(np.nan, train_x.shape[0])

    # 学習データからバリデーションデータを分ける
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        # 学習データについて、各カテゴリにおける目的変数の平均を計算
        target_mean = data_tmp.iloc[tr_idx].groupby(c)['target'].mean()
        # バリデーションデータについて、変換後の値を一時配列に格納
        tmp[va_idx] = train_x[c].iloc[va_idx].map(target_mean)

    # 変換後のデータで元の変数を置換
    train_x[c] = tmp
