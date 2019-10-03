# ---------------------------------
# データ等の準備
# ----------------------------------
import numpy as np
import pandas as pd

# データの作成（ランダムなデータとしています）
rand = np.random.RandomState(71)
train_x = pd.DataFrame(rand.uniform(0.0, 1.0, (10000, 2)), columns=['model1', 'model2'])
adv_train = pd.Series(rand.uniform(0.0, 1.0, 10000))
w = np.array([0.3, 0.7]).reshape(1, -1)
train_y = pd.Series((train_x.values * w).sum(axis=1) > 0.5)

# ---------------------------------
# adversarial stochastic blending
# ----------------------------------
# モデルの予測値を加重平均する重みの値をadversarial validationで求める
# train_x: 各モデルによる確率の予測値（実際には順位に変換したものを使用）
# train_y: 目的変数
# adv_train: 学習データのテストデータらしさを確率で表した値

from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

n_sampling = 50  # サンプリングの回数
frac_sampling = 0.5  # サンプリングで学習データから取り出す割合


def score(x, data_x, data_y):
    # 評価指標はAUCとする
    y_prob = data_x['model1'] * x + data_x['model2'] * (1 - x)
    return -roc_auc_score(data_y, y_prob)


# サンプリングにより加重平均の重みの値を求めることを繰り返す
results = []
for i in range(n_sampling):
    # サンプリングを行う
    seed = i
    idx = pd.Series(np.arange(len(train_y))).sample(frac=frac_sampling, replace=False,
                                                    random_state=seed, weights=adv_train)
    x_sample = train_x.iloc[idx]
    y_sample = train_y.iloc[idx]

    # サンプリングしたデータに対して、加重平均の重みの値を最適化により求める
    # 制約式を持たせるようにしたため、アルゴリズムはCOBYLAを選択
    init_x = np.array(0.5)
    constraints = (
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'ineq', 'fun': lambda x: 1.0 - x},
    )
    result = minimize(score, x0=init_x,
                      args=(x_sample, y_sample),
                      constraints=constraints,
                      method='COBYLA')
    results.append((result.x, 1.0 - result.x))

# model1, model2の加重平均の重み
results = np.array(results)
w_model1, w_model2 = results.mean(axis=0)
