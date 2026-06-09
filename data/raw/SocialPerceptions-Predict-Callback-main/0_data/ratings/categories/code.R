"warmth and competence ratings"

rm(list=ls())
library(readr)
library(tidyverse)

script_path <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(script_path)


############################### data collection: April 2023
prp_df<-function(which_col, which_rating, df){
  df %>% 
    select(c(ResponseId, ends_with(which_col))) %>%
    pivot_longer(cols = ends_with(which_col), names_to = "study", values_to = which_rating, values_drop_na = TRUE) %>%
    mutate(study = str_replace(study, "nonparent_1", "nonparent")) %>%
    mutate(study = str_replace(study, "nonparent_2", "nonparent")) %>%
    separate(study, c("study", "level", "rm" ), sep = "_") %>%
    select(!rm) ->df
  return(df)
}

read_delim("CategorySignals-Kline-Flake-Leasure-Prolific.tsv", 
           delim = "\t", escape_double = FALSE, locale = locale(encoding = "UTF-16"),
           trim_ws = TRUE, col_types = cols())->df_temp

cbind(df_temp %>% select(!contains("kline")),
      df_temp %>% reframe(kline_gay_79 = reduce(select(df_temp, matches("kline_gay.*_79")), coalesce)), #kline contains 3 signals for straight/gay"
      df_temp %>% reframe(kline_gay_1 = reduce(select(df_temp, matches("kline_gay.*_1")), coalesce)),
      df_temp %>% reframe(kline_straight_79 = reduce(select(df_temp, matches("kline_straight.*_79")), coalesce)),
      df_temp %>% reframe(kline_straight_1 = reduce(select(df_temp, matches("kline_straight.*_1")), coalesce))) %>%
  
  filter(!row_number() %in% c(1, 2)) %>%
  select(!c(contains("race_1"), contains("disability_1"))) ->df_temp


prp_df(which_col = "_79", which_rating = "competence", df=df_temp) %>%
  left_join(prp_df(which_col = "_1", which_rating = "warm", df=df_temp),
            by=c("ResponseId", "study", "level"), multiple = "all") %>% #multiple bc ishizuka has two nonparent
  mutate(competence=as.numeric(competence),
         warm=as.numeric(warm)) %>%
  group_by(ResponseId, study, level) %>%
  summarise(warm=mean(warm),
            competence=mean(competence))->df_apr

############################### data collection: November 2022
prp_df<-function(which_col, which_rating, df){
  df %>% 
    select(c(ResponseId, ends_with(which_col))) %>%
    pivot_longer(cols = ends_with(which_col), names_to = "study", values_to = which_rating, values_drop_na = TRUE) %>%
    mutate(study = str_replace(study, "farber60", "farber_60")) %>%
    mutate(study = str_replace(study, "farber51", "farber_51")) %>%
    separate(study, c("rm", "study", "level" , "rm2"), sep = "_") %>%
    select(!c(rm, rm2)) ->df
  return(df)
}


read_delim("CategorySignals-Prolific.tsv", delim = "\t", escape_double = FALSE, locale = locale(encoding = "UTF-16"), trim_ws = TRUE, col_types = cols()) %>%
  filter(!row_number() %in% c(1, 2)) %>%
  select(!c(contains("race_1"), contains("disability_1"))) ->df_temp

prp_df(which_col = "_79", which_rating = "competence", df=df_temp) %>%
  left_join(prp_df(which_col = "_1", which_rating = "warm", df=df_temp),
            by=c("ResponseId", "study", "level")) %>%
  mutate(competence=as.numeric(competence),
         warm=as.numeric(warm)) -> df_nov


############################### write csv
rbind(df_nov, df_apr) %>%
  arrange(study) %>%
  
  mutate(level = str_replace(level, "straight ", "straight")) %>%
  mutate(study = str_replace(study, "mishel2021", "mishel")) %>%
  
  mutate(category = if_else(study == "wrigth", "religion", 
                    if_else(study == "yemane", "race and national origin",
                    if_else(study == "correll", "parenthood",
                    if_else(study == "ishizuka", "parenthood",
                    if_else(study == "hipes", "health",
                    if_else(study == "ameri", "health",
                    if_else(study == "namingit", "unemployed",
                    if_else(study == "neumark", "age",
                    if_else(study == "farber", "age",
                    if_else(study == "figinski", "military service or affiliation",
                    if_else(study == "rivera", "wealth",
                    if_else(study == "thomas", "wealth",
                    if_else(study == "tilcsik", "sexual orientation",
                    if_else(study == "bailey", "sexual orientation",
                    if_else(study == "kline", "sexual orientation",
                    if_else(study == "mishel2021", "sexual orientation",
                    if_else(study == "mishel", "sexual orientation",NA)))))))))))))))))) ->df_all

write.csv(df_all,file = "categories.csv", row.names=FALSE)

################

