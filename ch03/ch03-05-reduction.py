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

# 説明用に学習データとテストデータの元の状態を保存しておく
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 標準化を行った学習データとテストデータを返す関数
def load_standarized_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    return pd.DataFrame(train_x), pd.DataFrame(test_x)


# MinMaxスケーリングを行った学習データとテストデータを返す関数
def load_minmax_scaled_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()

    # Min-Max Scalingを行う
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([train_x, test_x], axis=0))
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    return pd.DataFrame(train_x), pd.DataFrame(test_x)


# -----------------------------------
# PCA
# -----------------------------------
# 標準化されたデータを用いる
train_x, test_x = load_standarized_data()
# -----------------------------------
# PCA
from sklearn.decomposition import PCA

# データは標準化などのスケールを揃える前処理が行われているものとする

# 学習データに基づいてPCAによる変換を定義
pca = PCA(n_components=5)
pca.fit(train_x)

# 変換の適用
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)

# -----------------------------------
# 標準化されたデータを用いる
train_x, test_x = load_standarized_data()
# -----------------------------------
# TruncatedSVD
from sklearn.decomposition import TruncatedSVD

# データは標準化などのスケールを揃える前処理が行われているものとする

# 学習データに基づいてSVDによる変換を定義
svd = TruncatedSVD(n_components=5, random_state=71)
svd.fit(train_x)

# 変換の適用
train_x = svd.transform(train_x)
test_x = svd.transform(test_x)

# -----------------------------------
# NMF
# -----------------------------------
# 非負の値とするため、MinMaxスケーリングを行ったデータを用いる
train_x, test_x = load_minmax_scaled_data()
# -----------------------------------
from sklearn.decomposition import NMF

# データは非負の値から構成されているとする

# 学習データに基づいてNMFによる変換を定義
model = NMF(n_components=5, init='random', random_state=71)
model.fit(train_x)

# 変換の適用
train_x = model.transform(train_x)
test_x = model.transform(test_x)

# -----------------------------------
# LatentDirichletAllocation
# -----------------------------------
# MinMaxスケーリングを行ったデータを用いる
# カウント行列ではないが、非負の値であれば計算は可能
train_x, test_x = load_minmax_scaled_data()
# -----------------------------------
from sklearn.decomposition import LatentDirichletAllocation

# データは単語文書のカウント行列などとする

# 学習データに基づいてLDAによる変換を定義
model = LatentDirichletAllocation(n_components=5, random_state=71)
model.fit(train_x)

# 変換の適用
train_x = model.transform(train_x)
test_x = model.transform(test_x)

# -----------------------------------
# LinearDiscriminantAnalysis
# -----------------------------------
# 標準化されたデータを用いる
train_x, test_x = load_standarized_data()
# -----------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# データは標準化などのスケールを揃える前処理が行われているものとする

# 学習データに基づいてLDAによる変換を定義
lda = LDA(n_components=1)
lda.fit(train_x, train_y)

# 変換の適用
train_x = lda.transform(train_x)
test_x = lda.transform(test_x)

# -----------------------------------
# t-sne
# -----------------------------------
# 標準化されたデータを用いる
train_x, test_x = load_standarized_data()
# -----------------------------------
import bhtsne

# データは標準化などのスケールを揃える前処理が行われているものとする

# t-sneによる変換
data = pd.concat([train_x, test_x])
embedded = bhtsne.tsne(data.astype(np.float64), dimensions=2, rand_seed=71)

# -----------------------------------
# UMAP
# -----------------------------------
# 標準化されたデータを用いる
train_x, test_x = load_standarized_data()
# -----------------------------------
import umap

# データは標準化などのスケールを揃える前処理が行われているものとする

# 学習データに基づいてUMAPによる変換を定義
um = umap.UMAP()
um.fit(train_x)

# 変換の適用
train_x = um.transform(train_x)
test_x = um.transform(test_x)

# -----------------------------------
# クラスタリング
# -----------------------------------
# 標準化されたデータを用いる
train_x, test_x = load_standarized_data()
# -----------------------------------
from sklearn.cluster import MiniBatchKMeans

# データは標準化などのスケールを揃える前処理が行われているものとする

# 学習データに基づいてMini-Batch K-Meansによる変換を定義
kmeans = MiniBatchKMeans(n_clusters=10, random_state=71)
kmeans.fit(train_x)

# 属するクラスタを出力する
train_clusters = kmeans.predict(train_x)
test_clusters = kmeans.predict(test_x)

# 各クラスタの中心までの距離を出力する
train_distances = kmeans.transform(train_x)
test_distances = kmeans.transform(test_x)
