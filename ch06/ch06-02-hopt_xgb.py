import numpy as np
from hyperopt import hp

# -----------------------------------
# xgboostのパラメータ空間の例
# -----------------------------------

# ベースラインのパラメータ
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': 0.1,
    'gamma': 0.0,
    'alpha': 0.0,
    'lambda': 1.0,
    'min_child_weight': 1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 71,
}

# パラメータの探索範囲
param_space = {
    'min_child_weight': hp.loguniform('min_child_weight', np.log(0.1), np.log(10)),
    'max_depth': hp.quniform('max_depth', 3, 9, 1),
    'subsample': hp.quniform('subsample', 0.6, 0.95, 0.05),
    'colsample_bytree': hp.quniform('subsample', 0.6, 0.95, 0.05),
    'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
    # 余裕があればalpha, lambdaも調整する
    # 'alpha' : hp.loguniform('alpha', np.log(1e-8), np.log(1.0)),
    # 'lambda' : hp.loguniform('lambda', np.log(1e-6), np.log(10.0)),
}
