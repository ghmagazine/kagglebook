import numpy as np
import pandas as pd


# -----------------------------------
# カスタム目的関数での評価指標の近似によるMAEの最適化
# -----------------------------------

# Fair 関数
def fair(preds, dtrain):
    x = preds - dtrain.get_labels()  # 残差を取得
    c = 1.0  # Fair関数のパラメータ
    den = abs(x) + c  # 勾配の式の分母を計算
    grad = c * x / den  # 勾配
    hess = c * c / den ** 2  # 二階微分値
    return grad, hess


# Pseudo-Huber 関数
def psuedo_huber(preds, dtrain):
    d = preds - dtrain.get_labels()  # 残差を取得
    delta = 1.0  # Pseudo-Huber関数のパラメータ
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt  # 勾配
    hess = 1 / scale / scale_sqrt  # 二階微分値
    return grad, hess
