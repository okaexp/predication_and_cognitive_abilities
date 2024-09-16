# 各参加者ごとに最適化した、Predicationモデルのハイパラ推定(JCSS2024)
最終更新: 2024/4/15, 13:57

### 環境構築
1. Pipenvの初期化と必要ライブラリの準備
    - pipenv --python 3.8
    - pipenv install numpy pandas jupyterlab tqdm gensim
1. gensimのkeyedvector（）をダウンロードして`model`ディレクトリに格納
    - [WikiEntVecのリリースページ](https://github.com/singletongue/WikiEntVec/releases)から`jawiki.entity_vectors.200d.txt.bz2`をインストールし解凍。解凍されたテキストファイルを`jawiki.entity_vectors.200d.txt`と名前を変更し、modelディレクトリに格納する。