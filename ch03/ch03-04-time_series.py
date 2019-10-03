import numpy as np
import pandas as pd

# -----------------------------------
# ワイドフォーマット、ロングフォーマット
# -----------------------------------

# ワイドフォーマットのデータを読み込む
df_wide = pd.read_csv('../input/ch03/time_series_wide.csv', index_col=0)
# インデックスの型を日付型に変更する
df_wide.index = pd.to_datetime(df_wide.index)

print(df_wide.iloc[:5, :3])
'''
              A     B     C
date
2016-07-01  532  3314  1136
2016-07-02  798  2461  1188
2016-07-03  823  3522  1711
2016-07-04  937  5451  1977
2016-07-05  881  4729  1975
'''

# ロングフォーマットに変換する
df_long = df_wide.stack().reset_index(1)
df_long.columns = ['id', 'value']

print(df_long.head(10))
'''
           id  value
date
2016-07-01  A    532
2016-07-01  B   3314
2016-07-01  C   1136
2016-07-02  A    798
2016-07-02  B   2461
2016-07-02  C   1188
2016-07-03  A    823
2016-07-03  B   3522
2016-07-03  C   1711
2016-07-04  A    937
...
'''

# ワイドフォーマットに戻す
df_wide = df_long.pivot(index=None, columns='id', values='value')

# -----------------------------------
# ラグ変数
# -----------------------------------
# ワイドフォーマットのデータをセットする
x = df_wide
# -----------------------------------
# xはワイドフォーマットのデータフレーム
# インデックスが日付などの時間、列がユーザや店舗などで、値が売上などの注目する変数を表すものとする

# 1期前のlagを取得
x_lag1 = x.shift(1)

# 7期前のlagを取得
x_lag7 = x.shift(7)

# -----------------------------------
# 1期前から3期間の移動平均を算出
x_avg3 = x.shift(1).rolling(window=3).mean()

# -----------------------------------
# 1期前から7期間の最大値を算出
x_max7 = x.shift(1).rolling(window=7).max()

# -----------------------------------
# 7期前, 14期前, 21期前, 28期前の値の平均
x_e7_avg = (x.shift(7) + x.shift(14) + x.shift(21) + x.shift(28)) / 4.0

# -----------------------------------
# 1期先の値を取得
x_lead1 = x.shift(-1)

# -----------------------------------
# ラグ変数
# -----------------------------------
# データの読み込み
train_x = pd.read_csv('../input/ch03/time_series_train.csv')
event_history = pd.read_csv('../input/ch03/time_series_events.csv')
train_x['date'] = pd.to_datetime(train_x['date'])
event_history['date'] = pd.to_datetime(event_history['date'])
# -----------------------------------

# train_xは学習データで、ユーザID, 日付を列として持つDataFrameとする
# event_historyは、過去に開催したイベントの情報で、日付、イベントを列として持つDataFrameとする

# occurrencesは、日付、セールが開催されたか否かを列として持つDataFrameとなる
dates = np.sort(train_x['date'].unique())
occurrences = pd.DataFrame(dates, columns=['date'])
sale_history = event_history[event_history['event'] == 'sale']
occurrences['sale'] = occurrences['date'].isin(sale_history['date'])

# 累積和をとることで、それぞれの日付での累積出現回数を表すようにする
# occurrencesは、日付、セールの累積出現回数を列として持つDataFrameとなる
occurrences['sale'] = occurrences['sale'].cumsum()

# 日付をキーとして学習データと結合する
train_x = train_x.merge(occurrences, on='date', how='left')
