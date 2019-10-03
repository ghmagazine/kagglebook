import numpy as np
import pandas as pd

# -----------------------------------
# 閾値の最適化
# -----------------------------------
from sklearn.metrics import f1_score
from scipy.optimize import minimize

# サンプルデータ生成の準備
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000)

# 真の値と予測値が以下のtrain_y, train_pred_probであったとする
train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

# 閾値を0.5とすると、F1は0.722
init_threshold = 0.5
init_score = f1_score(train_y, train_pred_prob >= init_threshold)
print(init_threshold, init_score)


# 最適化の目的関数を設定
def f1_opt(x):
    return -f1_score(train_y, train_pred_prob >= x)


# scipy.optimizeのminimizeメソッドで最適な閾値を求める
# 求めた最適な閾値をもとにF1を求めると、0.756 となる
result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
best_threshold = result['x'].item()
best_score = f1_score(train_y, train_pred_prob >= best_threshold)
print(best_threshold, best_score)
