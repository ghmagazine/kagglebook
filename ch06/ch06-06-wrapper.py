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

# 学習データを学習データとバリデーションデータに分ける
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# 特徴量のリストに対して精度を評価するevaluate関数の定義
import xgboost as xgb
from sklearn.metrics import log_loss


def evaluate(features):
    dtrain = xgb.DMatrix(tr_x[features], label=tr_y)
    dvalid = xgb.DMatrix(va_x[features], label=va_y)
    params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
    num_round = 10  # 実際にはもっと多いround数が必要
    early_stopping_rounds = 3
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    model = xgb.train(params, dtrain, num_round,
                      evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                      verbose_eval=0)
    va_pred = model.predict(dvalid)
    score = log_loss(va_y, va_pred)

    return score


# ---------------------------------
# Greedy Forward Selection
# ----------------------------------

best_score = 9999.0
selected = set([])

print('start greedy forward selection')

while True:

    if len(selected) == len(train_x.columns):
        # すべての特徴が選ばれて終了
        break

    scores = []
    for feature in train_x.columns:
        if feature not in selected:
            # 特徴量のリストに対して精度を評価するevaluate関数があるものとする
            fs = list(selected) + [feature]
            score = evaluate(fs)
            scores.append((feature, score))

    # スコアは低い方が良いとする
    b_feature, b_score = sorted(scores, key=lambda tpl: tpl[1])[0]
    if b_score < best_score:
        selected.add(b_feature)
        best_score = b_score
        print(f'selected:{b_feature}')
        print(f'score:{b_score}')
    else:
        # どの特徴を追加してもスコアが上がらないので終了
        break

print(f'selected features: {selected}')

# ---------------------------------
# Greedy Forward Selectionを単純化した手法
# ----------------------------------

best_score = 9999.0
candidates = np.random.RandomState(71).permutation(train_x.columns)
selected = set([])

print('start simple selection')
for feature in candidates:
    # 特徴量のリストに対して精度を評価するevaluate関数があるものとする
    fs = list(selected) + [feature]
    score = evaluate(fs)

    # スコアは低い方が良いとする
    if score < best_score:
        selected.add(feature)
        best_score = score
        print(f'selected:{feature}')
        print(f'score:{score}')

print(f'selected features: {selected}')
