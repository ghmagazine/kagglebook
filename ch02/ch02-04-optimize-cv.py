import numpy as np
import pandas as pd

# -----------------------------------
# out-of-foldでの閾値の最適化
# -----------------------------------
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# サンプルデータ生成の準備
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000)

# 真の値と予測値が以下のtrain_y, train_pred_probであったとする
train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

# クロスバリデーションの枠組みで閾値を求める
thresholds = []
scores_tr = []
scores_va = []

kf = KFold(n_splits=4, random_state=71, shuffle=True)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_pred_prob)):
    tr_pred_prob, va_pred_prob = train_pred_prob[tr_idx], train_pred_prob[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 最適化の目的関数を設定
    def f1_opt(x):
        return -f1_score(tr_y, tr_pred_prob >= x)

    # 学習データで閾値の最適化を行い、バリデーションデータで評価を行う
    result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
    threshold = result['x'].item()
    score_tr = f1_score(tr_y, tr_pred_prob >= threshold)
    score_va = f1_score(va_y, va_pred_prob >= threshold)
    print(threshold, score_tr, score_va)

    thresholds.append(threshold)
    scores_tr.append(score_tr)
    scores_va.append(score_va)

# 各foldの閾値の平均をテストデータには適用する
threshold_test = np.mean(thresholds)
print(threshold_test)
