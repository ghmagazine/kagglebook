import numpy as np
import pandas as pd

# -----------------------------------
# データの結合
# -----------------------------------
# データの読み込み
train = pd.read_csv('../input/ch03/multi_table_train.csv')
product_master = pd.read_csv('../input/ch03/multi_table_product.csv')
user_log = pd.read_csv('../input/ch03/multi_table_log.csv')

# -----------------------------------
# 図の形式のデータフレームがあるとする
# train         : 学習データ（ユーザID, 商品ID, 目的変数などの列がある）
# product_master: 商品マスタ（商品IDと商品の情報を表す列がある）
# user_log      : ユーザの行動のログデータ（ユーザIDと各行動の情報を表す列がある）

# 商品マスタを学習データと結合する
train = train.merge(product_master, on='product_id', how='left')

# ログデータのユーザごとの行数を集計し、学習データと結合する
user_log_agg = user_log.groupby('user_id').size().reset_index().rename(columns={0: 'user_count'})
train = train.merge(user_log_agg, on='user_id', how='left')
