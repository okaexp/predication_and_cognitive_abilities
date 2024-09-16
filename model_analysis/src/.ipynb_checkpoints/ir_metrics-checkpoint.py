#情報検索におけるモデルの評価指標
#ref: https://zenn.dev/hellorusk/articles/7e336fd3c6be20a8f8d1

import numpy as np

def average_precision(model_output, ans, _w2v_model):
    """
    Parameters
    ---------
    model_output: モデルの出力(["明るい", "美しい", ...])
    ans: 参加者の書き出した解釈(例：["明るい", "美しい",  ...])

    Note
    ---------
    - _w2v_model（gensimのKeyedVector）は読み込み前提
    - 現状の書き方だと、人の解釈にはあるけれどコーパスに出現しない語については解析から除外している
    """

    aru = 0#モデルの出力が参加者の回答にあるか否か
    sum_precision = 0#average precison算出のためのアイテムごとのprecisionの値
    ap = 0#average precision

    #ansから、コーパス内に無い単語を弾く
    ans_rmnv = [word for word in ans if word in _w2v_model.key_to_index.keys()]

    #anstermsがコーパス中に１つもない場合は"NA"を返す
    if ans_rmnv == []:
        return "NA"

    #それ以外の場合は
    for r, output_word in enumerate(model_output, 1):#モデルの出力を回す; r:回答順位, output_word:単語
      if output_word in ans_rmnv:#モデルが提案する単語が正解データ（参加者の回答）にあれば
        #print("output_word: ", output_word, ", rank: ", r)
        aru += 1
        sum_precision += aru / r

    try:
        ap = sum_precision / aru

    except ZeroDivisionError:
        pass
        #print("このタスクはapを算出できませんでした")

    return ap

def average_precision_for_check(model_output, ans, _w2v_model):
    """
    Parameters
    ---------
    model_output: モデルの出力(["明るい", "美しい", ...])
    ans: 参加者の書き出した解釈(例：["明るい", "美しい",  ...])

    Returns
    ---------
    ap: average precision(defaultは0)
    len_ans: 参加者の書き出した解釈は何件か
    aru: 何件の解釈がモデルの出力に含まれていたか

    Note
    ---------
    - average_precision関数のデバッグ用
    - _w2v_model（gensimのKeyedVector）は読み込み前提
    - 現状の書き方だと、人の解釈にはあるけれどコーパスに出現しない語については解析から除外している
    """

    aru = 0#モデルの出力が参加者の回答にあるか否か
    sum_precision = 0#average precison算出のためのアイテムごとのprecisionの値
    ap = 0#average precision

    #ansから、コーパス内に無い単語を弾く
    ans_rmnv = [word for word in ans if word in _w2v_model.key_to_index.keys()]
    len_ans = len(ans_rmnv)

    #anstermsがコーパス中に１つもない場合は"NA"を返す
    if ans_rmnv == []:
        return "NA", len_ans, aru

    #それ以外の場合は
    for r, output_word in enumerate(model_output, 1):#モデルの出力を回す; r:回答順位, output_word:単語
      if output_word in ans_rmnv:#モデルが提案する単語が正解データ（参加者の回答）にあれば
        #print("output_word: ", output_word, ", rank: ", r)
        aru += 1
        sum_precision += aru / r

    try:
        ap = sum_precision / aru

    except ZeroDivisionError:
        pass
        #print("このタスクはapを算出できませんでした")

    return ap, len_ans, aru

def reciprocal_rank(model_output, ans):
    """
    model_output: モデルの出力(["明るい", "美しい", ...])
    ans: 参加者の書き出した解釈(例：["明るい", "美しい",  ...])

    https://blog.brainpad.co.jp/entry/2017/08/25/140000
    MRRはレコメンドリストを上位から見て、最初の適合アイテムの順位をそのまま計算に利用したシンプルな指標で、以下の手順で算出されます。

    1. 全対象ユーザに対してレコメンドリストを作成する
    2. ユーザが嗜好する適合アイテムが上位何番目で現れたか調べる
    3. この順位の逆数をとる（リストに正解が含まれない場合は0となる）
    4. 全ユーザで平均をとる
    """
    rr = 0#sum of reciprocal_rank

    for r, output_word in enumerate(model_output, 1):#outputの単語を一つずつ見る r:回答順位, output_word:単語
      if output_word in ans:#参加者の回答にoutput_wordが含まれていたらそこで打ち切り
        #print("output_word: ", output_word, ", rank: ", r)
        rr = 1 / r
        break

    return rr

def discounted_cumulative_gain(model_output, ans, at_ndcg):
    """
    model_output: モデルの出力(["明るい", "美しい", ...])
    ans: 参加者の書き出した解釈(例：["明るい", "美しい",  ...])
    at_ndcg: 何位までの順位を見るか。暫定的に10

    注意:
      ・DCGはansで関連度を必要とするが、今回の研究では参加者ごとに関連度を得るのは困難
      ★今回の研究では、暫定的に、「関連度は回答順位の逆数」として算出

    順位を含めて正解データのランキングをどれだけ再現できるのかの評価の指標
    参考：https://blog.brainpad.co.jp/entry/2017/08/25/140000
    """
    dcg = 0

    for r, output_word in enumerate(model_output, 1): #モデルの出力（形容詞ランキング）を回す; output_word: 単語, r: 順位
      if r <= at_ndcg:#足切の順位以下なら
        if output_word in ans: #モデルが提案する形容詞が正解データ（参加者のやつ）にあれば
          dcg += ((1/(ans.index(output_word) + 1)) / np.log2(r + 1))
      else:
        break

    return dcg

def ideal_discounted_cumulative_gain(ans, at_ndcg):
    """
    nDCGを求めるときに使う，理想的リストにおけるDCG
    出力の評価をn番目までで止める
    """
    idcg = 0

    for r, ans_word in enumerate(ans, 1): #正解でデータ（参加者のやつ）を回す
      if r <= at_ndcg:
        idcg += (1/r) / np.log2(r + 1)#間違っていないか確認
      else:
        break

    return idcg

def normalized_discouted_cumulative_gain(model_output, ans, at_ndcg):
    """
    nDCGを求める関数
    出力の評価をn番目までで止める
    """
    return discounted_cumulative_gain(model_output, ans, at_ndcg) / ideal_discounted_cumulative_gain(ans, at_ndcg)