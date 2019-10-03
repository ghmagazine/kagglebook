## 入力データ

### サンプルデータ(sample-data)

#### サンプルデータ概要

* 第2章以降のサンプルコードの入力データ
* Kaggleの[Prudential Life Insurance Assessment](https://www.kaggle.com/c/prudential-life-insurance-assessment)
を参考にし、保険引受データを想定した人工データ。データの生成はシンプルで、現実のデータと比較してデータの背景にある構造が単純になっている。
* 学習データ・テストデータともに10000行

#### サンプルデータ項目

|列名| 内容 |
|:----|:-------|
| age | 年齢 |
| gender | 性別 |
| height | 身長 |
| weight | 体重 |
| product | 商品種類 |
| amount | 保険金額 |
| date | 申込日 |
| medical_info_a1/a2/a3 | 医療情報 連続変数 |
| medical_info_b1/b2/b3 | 医療情報 連続変数／カテゴリ変数 |
| medical_info_c1/c2 | 医療情報 連続変数／カテゴリ変数 |
| medical_keyword_1-10 | 医療情報 バイナリ変数 |
| target | 目的変数（二値） |


### 第1章で使用する入力データ(ch01-titanic)

* Kaggleの[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) の[データ](https://www.kaggle.com/c/titanic/data)を保存して下さい。  
  （ch01-titanic/train.csv, ch01-titanic/test.csv となるように保存して下さい。）


### 第3章で使用する入力データ(ch03)

* 複数テーブルの結合を説明するための入力データ
* 時系列データの処理を説明するための入力データ

