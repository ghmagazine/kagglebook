import numpy as np
import pandas as pd

# -----------------------------------
# 回帰
# -----------------------------------
# rmse

from sklearn.metrics import mean_squared_error

# y_trueが真の値、y_predが予測値
y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
# 0.5532

# -----------------------------------
# 二値分類
# -----------------------------------
# 混同行列

from sklearn.metrics import confusion_matrix

# 0, 1で表される二値分類の真の値と予測値
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp],
                              [fn, tn]])
print(confusion_matrix1)
# array([[3, 1],
#        [2, 2]])

# scikit-learnのmetricsモジュールのconfusion_matrixでも作成できるが、混同行列の要素の配置が違うので注意が必要
confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)
# array([[2, 1],
#        [2, 3]])

# -----------------------------------
# accuracy

from sklearn.metrics import accuracy_score

# 0, 1で表される二値分類の真の値と予測値
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
# 0.625

# -----------------------------------
# logloss

from sklearn.metrics import log_loss

# 0, 1で表される二値分類の真の値と予測確率
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)
# 0.7136

# -----------------------------------
# マルチクラス分類
# -----------------------------------
# multi-class logloss

from sklearn.metrics import log_loss

# 3クラス分類の真の値と予測値
y_true = np.array([0, 2, 1, 2, 2])
y_pred = np.array([[0.68, 0.32, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.60, 0.40, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.28, 0.12, 0.60]])
logloss = log_loss(y_true, y_pred)
print(logloss)
# 0.3626

# -----------------------------------
# マルチラベル分類
# -----------------------------------
# mean_f1, macro_f1, micro_f1

from sklearn.metrics import f1_score

# マルチラベル分類の真の値・予測値は、評価指標の計算上はレコード×クラスの二値の行列とした方が扱いやすい
# 真の値 - [[1,2], [1], [1,2,3], [2,3], [3]]
y_true = np.array([[1, 1, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [0, 1, 1],
                   [0, 0, 1]])

# 予測値 - [[1,3], [2], [1,3], [3], [3]]
y_pred = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

# mean-f1ではレコードごとにF1-scoreを計算して平均をとる
mean_f1 = np.mean([f1_score(y_true[i, :], y_pred[i, :]) for i in range(len(y_true))])

# macro-f1ではクラスごとにF1-scoreを計算して平均をとる
n_class = 3
macro_f1 = np.mean([f1_score(y_true[:, c], y_pred[:, c]) for c in range(n_class)])

# micro-f1ではレコード×クラスのペアごとにTP/TN/FP/FNを計算し、F1-scoreを求める
micro_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1))

print(mean_f1, macro_f1, micro_f1)
# 0.5933, 0.5524, 0.6250

# scikit-learnのメソッドを使うことでも計算できる
mean_f1 = f1_score(y_true, y_pred, average='samples')
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

# -----------------------------------
# クラス間に順序関係があるマルチクラス分類
# -----------------------------------
# quadratic weighted kappa

from sklearn.metrics import confusion_matrix, cohen_kappa_score


# quadratic weighted kappaを計算する関数
def quadratic_weighted_kappa(c_matrix):
    numer = 0.0
    denom = 0.0

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            n = c_matrix.shape[0]
            wij = ((i - j) ** 2.0)
            oij = c_matrix[i, j]
            eij = c_matrix[i, :].sum() * c_matrix[:, j].sum() / c_matrix.sum()
            numer += wij * oij
            denom += wij * eij

    return 1.0 - numer / denom


# y_true は真の値のクラスのリスト、y_pred は予測値のクラスのリスト
y_true = [1, 2, 3, 4, 3]
y_pred = [2, 2, 4, 4, 5]

# 混同行列を計算する
c_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

# quadratic weighted kappaを計算する
kappa = quadratic_weighted_kappa(c_matrix)
print(kappa)
# 0.6153

# scikit-learnのメソッドを使うことでも計算できる
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

# -----------------------------------
# レコメンデーション
# -----------------------------------
# MAP@K

# K=3、レコード数は5個、クラスは4種類とする
K = 3

# 各レコードの真の値
y_true = [[1, 2], [1, 2], [4], [1, 2, 3, 4], [3, 4]]

# 各レコードに対する予測値 - K=3なので、通常は各レコードにそれぞれ3個まで順位をつけて予測する
y_pred = [[1, 2, 4], [4, 1, 2], [1, 4, 3], [1, 2, 3], [1, 2, 4]]


# 各レコードごとのaverage precisionを計算する関数
def apk(y_i_true, y_i_pred):
    # y_predがK以下の長さで、要素がすべて異なることが必要
    assert (len(y_i_pred) <= K)
    assert (len(np.unique(y_i_pred)) == len(y_i_pred))

    sum_precision = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_i_pred):
        if p in y_i_true:
            num_hits += 1
            precision = num_hits / (i + 1)
            sum_precision += precision

    return sum_precision / min(len(y_i_true), K)


# MAP@K を計算する関数
def mapk(y_true, y_pred):
    return np.mean([apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])


# MAP@Kを求める
print(mapk(y_true, y_pred))
# 0.65

# 正解の数が同じでも、順序が違うとスコアも異なる
print(apk(y_true[0], y_pred[0]))
print(apk(y_true[1], y_pred[1]))
# 1.0, 0.5833
