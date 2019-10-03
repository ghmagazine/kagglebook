## サンプルコード

「Kaggleで勝つデータ分析の技術」([amazon](https://www.amazon.co.jp/dp/4297108437)) のサンプルコードです。

<img src="misc/cover_small.jpg" width="200">

### 各フォルダの内容

|フォルダ| 内容 |
|:----|:-------|
| input | 入力ファイル |
| ch01 | 第1章のサンプルコード |
| ch02 | 第2章のサンプルコード |
| ch03 | 第3章のサンプルコード |
| ch04 | 第4章のサンプルコード |
| ch05 | 第5章のサンプルコード |
| ch06 | 第6章のサンプルコード |
| ch07 | 第7章のサンプルコード |
| ch04-model-interface | 第4章の「分析コンペ用のクラスやフォルダの構成」のコード |

* 各章のディレクトリをカレントディレクトリとしてコードを実行して下さい。
* 第1章のタイタニックのデータは、[input/readme.md](https://github.com/ttakuya/KaggleBook/blob/master/code-public/input) のとおりダウンロード下さい。
* 第4章の「分析コンペ用のクラスやフォルダの構成」のコードについては、[ch04-model-interface/readme.md](ch04-model-interface) を参照下さい。


### Requirements

サンプルコードの動作は、Google Cloud Platform(GCP)で確認しています。  

環境は以下のとおりです。

* Ubuntu 18.04 LTS  
* Anaconda 2019.03 Python 3.7
* 必要なPythonパッケージ（下記スクリプト参照）

以下のスクリプトのとおりにGCPの環境構築を行っています。
```
# utils -----

# 開発に必要なツールをインストール
cd ~/
sudo apt-get update
sudo apt-get install -y git build-essential libatlas-base-dev
sudo apt-get install -y python3-dev

# anaconda -----

# Anacondaをダウンロードしインストール
mkdir lib
wget --quiet https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh -O lib/anaconda.sh
/bin/bash lib/anaconda.sh -b

# PATHを通す
echo export PATH=~/anaconda3/bin:$PATH >> ~/.bashrc
source ~/.bashrc

# python packages -----

# Pythonパッケージのインストール
# numpy, scipy, pandasはAnaconda 2019.03のバージョンのまま
# pip install numpy==1.16.2 
# pip install scipy==1.2.1 
# pip install pandas==0.24.2
pip install scikit-learn==0.21.2

pip install xgboost==0.81
pip install lightgbm==2.2.2
pip install tensorflow==1.14.0
pip install keras==2.2.4
pip install hyperopt==0.1.1
pip install bhtsne==0.1.9
pip install rgf_python==3.4.0
pip install umap-learn==0.3.9

# set backend for matplotlib to Agg -----

# GCP上で実行するため、matplotlibのbackendを指定し直す
matplotlibrc_path=$(python -c "import site, os, fileinput; packages_dir = site.getsitepackages()[0]; print(os.path.join(packages_dir, 'matplotlib', 'mpl-data', 'matplotlibrc'))") && \
sed -i 's/^backend      : qt5agg/backend      : agg/' $matplotlibrc_path
```
