#Predicationモデル関連のライブラリ

import pandas as pd
import numpy as np
import tqdm
import gensim
from gensim.models import KeyedVectors

def create_categorization_int(topic, vehicle, n_sim_vehicle, n_sim_topic, n_int, _w2v_model):
    """
    Parameters
    ---------
    topic (str): 主題
    vehicle (str): 喩辞
    n_sim_vehicle (int): 喩辞と類似する単語の数（Utsumiのm）
    n_sim_topic (int): 主題と類似する単語の数(Utsumiのk)
    n_int (int): 抽出する解釈数
    
    Note
    ---------
    - _w2v_model（gensimのKeyedVector）は読み込み前提
    """
    
    #1. Compute N<sub>m</sub>(W<sub>V</sub>), that is, m neighbors of the vehicle W<sub>V</sub>.
    NmWv = [k[0] for k in _w2v_model.similar_by_word(vehicle, topn=n_sim_vehicle)]#単語リストだけ

    #2. Choose k words with the highest similarity to the topic wT from among NmWv.
    WT_sim = [(sim_vehicle_word, _w2v_model.similarity(topic, sim_vehicle_word)) for sim_vehicle_word in NmWv]#並び替え前のリストの取得
    WT_sim_sorted = sorted(WT_sim, key = lambda x: x[1], reverse=True)#降順で並び替え
    WT = [word_and_sim[0] for word_and_sim in WT_sim_sorted[:n_sim_topic]]#類似度の高い上位n_sim_topic個の単語のみを抽出
    
    #3. Compute a vector v<sub>cat</sub>(M) as the centroid of v(w<sub>T</sub>), v(w<sub>V</sub>), and k vectors of the words chosen in Step 2.
    #ref: https://yaronvazana.com/2018/09/20/average-word-vectors-generate-document-paragraph-sentence-embeddings/
    Vm = np.mean(_w2v_model[[topic] + [vehicle] + WT], axis=0)
    
    #4. lists the top 10 nearest neighbors of the metaphor vector.
    #ref: https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.similar_by_vector.html
    INT = [word_and_sim[0] for word_and_sim in _w2v_model.similar_by_vector(vector=Vm, topn=n_int)]
    
    return INT


def create_categint_all_words_after_feature_only(topic, vehicle, n_sim_vehicle, n_sim_topic,
                                                 n_int, _all_w2v_model, _feature_w2v_model):
    """
    Parameters
    ---------
    topic (str): 主題
    vehicle (str): 喩辞
    n_sim_vehicle (int): 喩辞と類似する単語の数（Utsumiのm）
    n_sim_topic (int): 主題と類似する単語の数(Utsumiのk)
    n_int (int): 抽出する解釈数
    _all_w2v_model (object): gensim.modelsのword2vecモデル(単語全て)
    _feature_w2v_model (object): gensim.modelsのword2vecモデル(特徴単語全て)
    
    Note
    ---------
    - _all_w2v_model, _feature_w2v_model（gensimのKeyedVector）は読み込み前提
    - _all_w2v_model, _feature_w2v_model（gensimのKeyedVector）は# word2vecのモデルのベクトルを、L2ノームで正則化して置き換え済み
        ex: normalized = np.sqrt((_all_w2v_model.vectors ** 2).sum(-1))[..., np.newaxis]
        ex: _all_w2v_model.vectors = _all_w2v_model.vectors / normalized
    - n_intは、最終的に抽出する形容詞語の数が最大
    - #1~#3まではcreate_categorization_int関数と同じ
    """

    #1. Compute N<sub>m</sub>(W<sub>V</sub>), that is, m neighbors of the vehicle W<sub>V</sub>.
    NmWv = [k[0] for k in _all_w2v_model.similar_by_word(vehicle, topn=n_sim_vehicle)]#単語リストだけ

    #2. Choose k words with the highest similarity to the topic wT from among NmWv.
    WT_sim = [(sim_vehicle_word, _all_w2v_model.similarity(topic, sim_vehicle_word)) for sim_vehicle_word in NmWv]#並び替え前のリストの取得
    WT_sim_sorted = sorted(WT_sim, key = lambda x: x[1], reverse=True)#降順で並び替え
    WT = [word_and_sim[0] for word_and_sim in WT_sim_sorted[:n_sim_topic]]#類似度の高い上位n_sim_topic個の単語のみを抽出
    
    #3. Compute a vector v<sub>cat</sub>(M) as the centroid of v(w<sub>T</sub>), v(w<sub>V</sub>), and k vectors of the words chosen in Step 2.
    #ref: https://yaronvazana.com/2018/09/20/average-word-vectors-generate-document-paragraph-sentence-embeddings/
    Vm = np.mean(_all_w2v_model[[topic] + [vehicle] + WT], axis=0)
    
    #4. 単語のリストで返す
    INT_ONLY_FEATURE = [word_and_sim[0] for word_and_sim in _feature_w2v_model.similar_by_vector(vector=Vm, topn=n_int)]
    
    return INT_ONLY_FEATURE