#Comparisonモデル関連のライブラリ

import pandas as pd
import numpy as np
import tqdm
import gensim
from gensim.models import KeyedVectors

def create_comparison_int(topic, vehicle, k_common_neighbors, n_int, neighbors_limit, _w2v_model):
    """
      topic (str): 主題
      vehicle (str): 喩辞
      k_common_neighbors (int): 主題と喩辞の共通する単語数(Utsumiのk)
      n_int (int): 抽出する解釈数
      neighbors_limit (int): 主題と喩辞で抽出する近傍語数（言語モデル中の全ての単語との類似度）
    
      前提:
        - _w2v_model（gensimのKeyedVector）はある前提
        - neighbors_limitは言語モデル中の全ての単語数（ただし、読み込みは一回だけで、以降はスライスでメモリから読む）
    """
    #1. Compute k common neighbors (kcn) $Ni(W_T) \cap Ni(W_V)$ of  $W_T$ and $W_T$
    # by finding the smallest i that satisfies $|Ni(W_T) \cap Ni(W_V)| | \geq k$.
    topic_neighbors = [elems[0] for elems in _w2v_model.most_similar(positive=topic, topn=neighbors_limit)]
    vehicle_neighbors =  [elems[0] for elems in _w2v_model.most_similar(positive=vehicle, topn=neighbors_limit)]
    
    for cur_neighbor_idx in range(neighbors_limit):
        #主題と喩辞の類似語のリストを作る
        #cur_neighbor_idxまでで止める（内海先生ご助言）
        topic_veihcle_common_neighbors = set(topic_neighbors[:cur_neighbor_idx]) & set(vehicle_neighbors[:cur_neighbor_idx])
    
        #主題と喩辞の類似語の個数がKを超えたら打ち切り
        if len(topic_veihcle_common_neighbors) > (k_common_neighbors-1):
            break

    #2. Compute a metaphor vector $V_{com}(M)$ as the centroid of
    #  $v(W_T)$ and k vectors of the words chosen in Step 1.
    Vm = np.mean(_w2v_model[[topic] + list(topic_veihcle_common_neighbors)], axis=0)

    #3. lists the top 10 nearest neighbors of the metaphor vector.
    #ref: https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.similar_by_vector.html
    INT = [word_and_sim[0] for word_and_sim in _w2v_model.similar_by_vector(vector=Vm, topn=n_int)]
    
    return INT

def create_compaint_all_words_after_feature_only(topic, vehicle, k_common_neighbors, n_int, neighbors_limit,
                                                 _all_w2v_model, _feature_w2v_model):
    """
      topic (str): 主題
      vehicle (str): 喩辞
      k_common_neighbors (int): 主題と喩辞の共通する単語数(Utsumiのk)
      n_int (int): 抽出する解釈数
      neighbors_limit (int): 主題と喩辞で抽出する近傍語数（言語モデル中の全ての単語との類似度）
      _all_w2v_model (object): gensim.modelsのword2vecモデル(単語全て)
      _feature_w2v_model (object): gensim.modelsのword2vecモデル(特徴単語全て)
    
      前提:
        - _all_w2v_model, _feature_w2v_model（gensimのKeyedVector）は読み込み前提
        - _all_w2v_model, _feature_w2v_model（gensimのKeyedVector）は# word2vecのモデルのベクトルを、L2ノームで正則化して置き換え済み
          ex: normalized = np.sqrt((_all_w2v_model.vectors ** 2).sum(-1))[..., np.newaxis]
          ex: _all_w2v_model.vectors = _all_w2v_model.vectors / normalized
        - n_intは、最終的に抽出する形容詞語の数が最大
        - neighbors_limitは言語モデル中(_all_w2v_model)の全ての単語数（ただし、読み込みは一回だけで、以降はスライスでメモリから読む）
    """
    #1. Compute k common neighbors (kcn) $Ni(W_T) \cap Ni(W_V)$ of  $W_T$ and $W_T$
    # by finding the smallest i that satisfies $|Ni(W_T) \cap Ni(W_V)| | \geq k$.
    # 追記(2024/8/1): 主題と喩辞の類似点については名詞単語も含む（つまり、全ての単語で類似度を求める）
    topic_neighbors = [elems[0] for elems in _all_w2v_model.most_similar(positive=topic, topn=neighbors_limit)]
    vehicle_neighbors =  [elems[0] for elems in _all_w2v_model.most_similar(positive=vehicle, topn=neighbors_limit)]
    
    for cur_neighbor_idx in range(neighbors_limit):
        #主題と喩辞の類似語のリストを作る
        #cur_neighbor_idxまでで止める（内海先生ご助言）
        topic_veihcle_common_neighbors = set(topic_neighbors[:cur_neighbor_idx]) & set(vehicle_neighbors[:cur_neighbor_idx])
    
        #主題と喩辞の類似語の個数がKを超えたら打ち切り
        if len(topic_veihcle_common_neighbors) > (k_common_neighbors-1):
            break

    #2. Compute a metaphor vector $V_{com}(M)$ as the centroid of
    #  $v(W_T)$ and k vectors of the words chosen in Step 1.
    Vm = np.mean(_all_w2v_model[[topic] + list(topic_veihcle_common_neighbors)], axis=0)

    #3. lists the top 10 nearest neighbors of the metaphor vector.
    #ref: https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.similar_by_vector.html
    # 追記(2024/8/1): Vmとの類似ベクトルを求めるタイミングでは_featureベクトルを使う
    INT_ONLY_FEATURE = [word_and_sim[0] for word_and_sim in _feature_w2v_model.similar_by_vector(vector=Vm, topn=n_int)]
    
    return INT_ONLY_FEATURE