# ---------------------------------
# データ等の準備
# ----------------------------------
import numpy as np
import pandas as pd

# train_xは学習データ、train_yは目的変数、test_xはテストデータ
# pandasのDataFrame, Seriesで保持します。（numpyのarrayで保持することもあります）
# one-hot encodingされたものを読み込む

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# 学習データを学習データとバリデーションデータに分ける
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# tensorflowの警告抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# -----------------------------------
# ニューラルネットの実装
# -----------------------------------
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# データのスケーリング
scaler = StandardScaler()
tr_x = scaler.fit_transform(tr_x)
va_x = scaler.transform(va_x)
test_x = scaler.transform(test_x)

# ニューラルネットモデルの構築
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(train_x.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# 学習の実行
# バリデーションデータもモデルに渡し、学習の進行とともにスコアがどう変わるかモニタリングする
batch_size = 128
epochs = 10
history = model.fit(tr_x, tr_y,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(va_x, va_y))

# バリデーションデータでのスコアの確認
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred, eps=1e-7)
print(f'logloss: {score:.4f}')

# 予測
pred = model.predict(test_x)

# -----------------------------------
# アーリーストッピング
# -----------------------------------
from keras.callbacks import EarlyStopping

# アーリーストッピングの観察するroundを20とする
# restore_best_weightsを設定することで、最適なエポックでのモデルを使用する
epochs = 50
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(tr_x, tr_y,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(va_x, va_y), callbacks=[early_stopping])
pred = model.predict(test_x)
