#最終更新: 2024年8月3日 14:19
#喩辞の類似単語は全語彙で、解釈出力時は形容語全語（26,136語）で評価

library(tidyr)
library(dplyr)
library(ggplot2)
library(openxlsx)
library(psych)

source("http://aoki2.si.gunma-u.ac.jp/R/src/mycor.R", encoding="euc-jp") #mycorを持ってくる

# 1. 解析用データの作成（dat_aggregate）----

# 各種データの読み込み ----
dat_maxmap_param <- read.csv("../data/maxmap_param_df_at_26136_pred_all_after_feature.csv")
dat_maxmrr_param <- read.csv("../data/maxmrr_param_df_at_26136_pred_all_after_feature.csv")

dat_rpms <- read.csv("../data/rpms_sum_score.csv")
dat_sst <- read.csv("../data/sst_sum_score.csv")
dat_job <- read.csv("../data/job_sum_score.csv")

# ターゲットキーのみを集計したデータフレーム(dat_aggregate)を作る ----
dat_agg_map <- dat_maxmap_param %>%
  dplyr::inner_join(dat_rpms, by = "cwid") %>%
  dplyr::inner_join(dat_sst, by = "cwid") %>%
  dplyr::inner_join(dat_job, by = "cwid")

dat_agg_mrr <- dat_maxmrr_param %>%
  dplyr::inner_join(dat_rpms, by = "cwid") %>%
  dplyr::inner_join(dat_sst, by = "cwid") %>%
  dplyr::inner_join(dat_job, by = "cwid")

# 2. 相関分析 ----
# m,kについては対数変換をとる

# 2.1. map@26136 ----
dat_cor_mklog_map_at_26136 <- dat_agg_map %>%
  dplyr::select(-cwid) %>%
  dplyr::mutate(log_m = log(m),
                log_k = log(k)) %>%
  dplyr::select(log_m, log_k, maxmap,
                SumRPSMCorrect, SumSSTFinalRating, SumJobScore)
mycor(1:6, dat_cor_mklog_map_at_26136, latex = FALSE)

dat_cor_mklog_mrr_at_26136 <- dat_agg_mrr %>%
  dplyr::select(-cwid) %>%
  dplyr::mutate(log_m = log(m),
                log_k = log(k)) %>%
  dplyr::select(log_m, log_k, maxmrr,
                SumRPSMCorrect, SumSSTFinalRating, SumJobScore)
mycor(1:6, dat_cor_mklog_mrr_at_26136, latex = FALSE)

# 3. 記述統計量と必要な相関分析 ----
#記述統計量
describe(dat_cor_mklog_map_at_26136)
describe(dat_cor_mklog_mrr_at_26136)

# 4. 喩辞の高低ごとに分けた場合 ----

# 各種データの読み込み ----
dat_maxmap_param_conv <- read.csv("../data/maxmap_param_df_at_26136_by_conv_pred_all_after_feature.csv")
dat_maxmrr_param_conv <- read.csv("../data/maxmrr_param_df_at_26136_by_conv_pred_all_after_feature.csv")

dat_rpms <- read.csv("../data/rpms_sum_score.csv")
dat_sst <- read.csv("../data/sst_sum_score.csv")
dat_job <- read.csv("../data/job_sum_score.csv")

#相関を求めるためのデータの整形
#wide型に変換
dat_maxmap_conv_wide <- dat_maxmap_param_conv %>%
  tidyr::pivot_wider(names_from=conv, values_from=c(m, k, maxmap)) %>%
  dplyr::filter(maxmap_H != 0 & maxmap_L !=0)#maxmapが0（つまり、算出できていない）は削除

dat_maxmrr_conv_wide <- dat_maxmrr_param_conv %>%
  tidyr::pivot_wider(names_from=conv, values_from=c(m, k, maxmrr)) %>%
  dplyr::filter(maxmrr_H != 0 & maxmrr_L !=0)#maxmrrが0（つまり、算出できていない）は削除

#相関係数ようのデータに整形
dat_maxmap_cor <- dat_maxmap_conv_wide %>%
  dplyr::inner_join(dat_rpms, by = "cwid") %>%
  dplyr::inner_join(dat_sst, by = "cwid") %>%
  dplyr::inner_join(dat_job, by = "cwid")

dat_maxmrr_cor <- dat_maxmrr_conv_wide %>%
  dplyr::inner_join(dat_rpms, by = "cwid") %>%
  dplyr::inner_join(dat_sst, by = "cwid") %>%
  dplyr::inner_join(dat_job, by = "cwid")


#H,Lで分割して、相関係数用のデータにする
dat_maxmap_cor_mklog <- dat_maxmap_cor %>%
  dplyr::select(-cwid) %>%
  dplyr::mutate(log_m_H = log(m_H),
                log_k_H = log(k_H),
                log_m_L = log(m_L),
                log_k_L = log(k_L)) %>%
  dplyr::select(log_m_H, log_k_H, log_m_L, log_k_L, maxmap_L, maxmap_H, SumRPSMCorrect, SumSSTFinalRating, SumJobScore)

dat_maxmrr_cor_mklog <- dat_maxmrr_cor %>%
  dplyr::select(-cwid) %>%
  dplyr::mutate(log_m_H = log(m_H),
                log_k_H = log(k_H),
                log_m_L = log(m_L),
                log_k_L = log(k_L)) %>%
  dplyr::select(log_m_H, log_k_H, log_m_L, log_k_L, maxmrr_L, maxmrr_H, SumRPSMCorrect, SumSSTFinalRating, SumJobScore)


#相関係数の算出
#maxmap, maxmrr, maxndcg
mycor(1:9, dat_maxmap_cor_mklog, latex = FALSE)
mycor(1:9, dat_maxmrr_cor_mklog, latex = FALSE)

#記述統計量
psych::describe(dat_maxmap_cor_mklog)
psych::describe(dat_maxmrr_cor_mklog)
