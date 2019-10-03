import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# データの読み込み、train/testを一度統合する
df_train = pd.read_csv('train.csv')
df_train['is_train'] = True
df_test = pd.read_csv('test.csv')
df_test['target'] = 0
df_test['is_train'] = False

df = pd.concat([df_train, df_test], axis=0)

# 日付の前処理
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['yearmonth'] = df['year'] * 12 + df['month']
df = df.drop(['date'], axis=1)

# 特徴量の種類
numerical_features = ['age', 'height', 'weight', 'amount', 'year', 'month', 'month', 'yearmonth'
                                                                                     'medical_info_a1',
                      'medical_info_a2', 'medical_info_a3', 'medical_info_b1']
binary_features = [f'medical_keyword_{i}' for i in range(10)]
categorical_features = ['sex', 'product', 'medical_info_b2', 'medical_info_b3']

# カテゴリカル変数についてLabel Encodingを行う
for c in categorical_features:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    print(f'{c} - {le.classes_}')

# targetを一番最後の列に持ってくる（見やすさのため）
df = df.reindex(columns=[c for c in df.columns if c != 'target'] + ['target'])

# train/testに分割し直して出力する
train = df[df['is_train']].drop(['is_train'], axis=1).reset_index(drop=True)
test = df[~df['is_train']].drop(['is_train', 'target'], axis=1).reset_index(drop=True)
train.to_csv('train_preprocessed.csv', index=False)
test.to_csv('test_preprocessed.csv', index=False)

# ----------------------
# ニューラルネット・線形モデル用の前処理

# 欠損値の補完
has_nan_features = ['medical_info_c1', 'medical_info_c2']
for c in has_nan_features:
    df[f'{c}_nan'] = df[c].isnull()
    df[c].fillna(df[c].mean(), inplace=True)

# One-hot Encodingを行う
df_onehot = pd.DataFrame(None, index=df.index)
for c in df.columns:
    if c in categorical_features and df[c].nunique() > 2:
        dummies = pd.get_dummies(df[c], prefix=c)
        df_onehot = pd.concat([df_onehot, dummies], axis=1)
        print(f'one-hot encoded - {c}')
    else:
        df_onehot[c] = df[c]


# targetを一番最後の列に持ってくる（見やすさのため）
df_onehot = df_onehot.reindex(columns=[c for c in df_onehot.columns if c != 'target'] + ['target'])

# train/testに分割し直して出力する
train_onehot = df_onehot[df_onehot['is_train']].drop(['is_train'], axis=1).reset_index(drop=True)
test_onehot = df_onehot[~df_onehot['is_train']].drop(['is_train', 'target'], axis=1).reset_index(drop=True)
train_onehot.to_csv('train_preprocessed_onehot.csv', index=False)
test_onehot.to_csv('test_preprocessed_onehot.csv', index=False)
