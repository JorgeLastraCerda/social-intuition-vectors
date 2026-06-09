"extracting ratings on names" 

# in most df, the same set of raters did not rate both warmth and competence; therefore I need to generate an alternative responseid such that i can number rating 1:10

rm(list=ls())
library(tidyverse)
library(haven)
library(tools)

script_path <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(script_path)

split_df<- function(which_rating, which_study, df){
  
  pattern<-paste(which_rating, "|Response ID", sep="")
  df %>% select_if(~ any(grepl(pattern, .))) -> df
  
  split_text <- str_split(df[1,], "- ", n = 3)
  colnames(df) <- str_replace_all(unlist(lapply(split_text, function(x) {trimws(x[3])})), "\t", " ")
  colnames(df)[1]<-"ResponseId"
  
  df %>% 
    filter(!row_number() %in% c(1, 2)) %>%
    pivot_longer(names_to = "name", values_to = which_rating, cols = !ResponseId) %>%
    mutate(study=which_study) %>%
    na.omit() -> df
  
  return(df)
}

### oreopoulos
readr::read_csv("Oreopoulos2011-Jenkins.csv") %>%
  select(ResponseId,c("1_warmth_1":"44_comp_q_1"))-> df_temp

split_df(which_rating = "warm", which_study = "oreopoulos", df=df_temp) %>%
  left_join(split_df(which_rating = "competent", which_study = "oreopoulos", df=df_temp), 
            by=c("name", "ResponseId", "study")) %>%
  mutate(name = toTitleCase(tolower(gsub(" - 1", "", name)))) %>%
  filter(!is.na(competent)) -> df_oreopoulos

### bertrand
readr::read_tsv("Bertrand2004-Prolific.tsv",locale = readr::locale(encoding = "UTF-16LE"), col_types = cols()) %>%
  select(ResponseId,c("1_Q1_299":"15_Q92_461"),-c("1_Q96_1", "1_Q97_1"))-> df_temp

split_df(which_rating = "warm", which_study = "bertrand", df=df_temp) %>%
  mutate(name = sapply(strsplit(name, " "), `[`, 1)) %>%
  left_join(split_df(which_rating = "competent", which_study = "bertrand", df=df_temp) %>%
              mutate(name = sapply(strsplit(name, " "), `[`, 1)) , 
            by=c("name", "ResponseId", "study")) %>%
  filter(!is.na(competent)) ->df_bertrand

### Jacquemet
readr::read_tsv("Jacquemet2012-Mturk.tsv",locale = readr::locale(encoding = "UTF-16LE"), col_types = cols()) %>% 
  select(ResponseId,"1_Q67_299":"2_Q67_467")->df_temp

split_df(which_rating = "warm", which_study = "jacquemet", df=df_temp) %>%
  left_join(split_df(which_rating = "competent", which_study = "jacquemet", df=df_temp), 
            by=c("name", "ResponseId", "study"))%>%
  filter(!is.na(competent)) ->df_jacquemet

### Nunley-Farber
readr::read_tsv("Nunley-Farber-Prolific.tsv",locale = readr::locale(encoding = "UTF-16LE"), col_types = cols()) %>% 
  select(ResponseId,c("1_Q1_299":"1_Q1_385","2_Q1_299":"2_Q1_385"))->df_temp
split_df(which_rating = "warm", which_study = "farber", df=df_temp) %>%
  left_join(split_df(which_rating = "competent", which_study = "farber", df=df_temp), 
            by=c("name", "ResponseId", "study"))->df_farber

readr::read_tsv("Nunley-Farber-Prolific.tsv",locale = readr::locale(encoding = "UTF-16LE"), col_types = cols()) %>% 
  select(ResponseId,c("1_Q1_386":"1_Q1_421","2_Q1_386":"2_Q1_421"))->df_temp                   
split_df(which_rating = "warm", which_study = "nunley", df=df_temp) %>%
  left_join(split_df(which_rating = "competent", which_study = "nunley", df=df_temp), 
            by=c("name", "ResponseId", "study"))->df_nunley

### widner-jacquemet-gorzig
readr::read_tsv("Widner-Jacquemet-Gorzig-Prolific.tsv",locale = readr::locale(encoding = "UTF-16LE"), 
                col_types = cols()) ->df_temp

df_temp %>% select(ResponseId,c("1_Q1_299":"1_Q1_388", "2_Q1_299":"2_Q1_388"))->df_jacquemet_2
df_temp %>% select(ResponseId,c("1_Q1_389":"1_Q1_400", "2_Q1_389":"2_Q1_400"))->df_widner
df_temp %>% select(ResponseId,c("1_Q1_401":"1_Q1_414", "2_Q1_401":"2_Q1_414"))->df_gorzig

split_df(which_rating = "warm", which_study = "jacquemet", df=df_jacquemet_2) %>%
  left_join(split_df(which_rating = "competent", which_study = "jacquemet", df=df_jacquemet_2), 
            by=c("name", "ResponseId", "study"))%>%
  filter(!is.na(competent))->df_jacquemet_2

split_df(which_rating = "warm", which_study = "widner", df=df_widner) %>%
  left_join(split_df(which_rating = "competent", which_study = "widner", df=df_widner), 
            by=c("name", "ResponseId", "study")) %>%
  filter(!is.na(competent))->df_widner

split_df(which_rating = "warm", which_study = "gorzig", df=df_gorzig) %>%
  left_join(split_df(which_rating = "competent", which_study = "gorzig", df=df_gorzig), 
            by=c("name", "ResponseId", "study"))%>%
  filter(!is.na(competent))->df_gorzig

### neumark
readr::read_tsv("Neumark2016-Prolific.tsv",locale = readr::locale(encoding = "UTF-16LE"), col_types = cols()) %>%
  select(ResponseId,"1_Q58_1":"154_Q57_38") -> df_temp

pattern <- "-\\s*([^-]+)-\\s*\\[Field-1\\]\\s*([^\\s]+)"

# warmth
df_temp_w <- df_temp[, grepl("warm", df_temp[1,])]
matches <- str_match(df_temp_w[1,], pattern)
colnames(df_temp_w) <- paste(matches[, 2], matches[, 3], sep = "")
df_temp_w <- df_temp_w[ , !duplicated(colnames(df_temp_w))]
df_temp_w <- cbind(df_temp$ResponseId,df_temp_w)
colnames(df_temp_w)[1]<-"ResponseId"

# competence
df_temp_c <- df_temp[, grepl("competent", df_temp[1,])]
matches <- str_match(df_temp_c[1,], pattern)
colnames(df_temp_c) <- paste(matches[, 2], matches[, 3], sep = "")
df_temp_c <- df_temp_c[ , !duplicated(colnames(df_temp_c))]
df_temp_c <- cbind(df_temp$ResponseId,df_temp_c)
colnames(df_temp_c)[1]<-"ResponseId"

df_temp_w %>%
  filter(!row_number() %in% c(1, 2)) %>%
  pivot_longer(names_to = "name", values_to = "warm", cols = !ResponseId) %>%
  filter(!name=="NANA") %>%
  mutate(study="neumark") %>%
  mutate(name = sapply(strsplit(name, " "), `[`, 1)) %>%
  mutate(name = tolower(str_trim(name))) %>%
  filter(!is.na(warm)) -> df_temp_w

df_temp_c %>%
  filter(!row_number() %in% c(1, 2)) %>%
  pivot_longer(names_to = "name", values_to = "competent", cols = !ResponseId) %>%
  filter(!name=="NANA") %>%
  mutate(study="neumark") %>%
  mutate(name = sapply(strsplit(name, " "), `[`, 1)) %>%
  mutate(name = tolower(str_trim(name))) %>%
  filter(!is.na(competent)) -> df_temp_c

df_temp_w %>%
  left_join(df_temp_c,by=c("ResponseId", "name", "study")) %>%
  filter(!is.na(competent)) -> df_neumark


### CategorySignals-Kline-Flake-Leasure-Prolific

split_df<- function(which_rating, which_study, df){
  
  pattern<-paste(which_rating, "|Response ID", sep="")
  df %>% select_if(~ any(grepl(pattern, .))) ->df
  colnames(df)<-str_extract(df[1,], "(?<=- )\\w+")
  colnames(df)[1]<-"ResponseId"
  
  df %>% 
    filter(!row_number() %in% c(1, 2)) %>%
    pivot_longer(names_to = "name", values_to = which_rating, cols = !ResponseId) %>%
    mutate(study=which_study) %>%
    na.omit() -> df
  
  return(df)
}

readr::read_tsv("../categories/CategorySignals-Kline-Flake-Leasure-Prolific.tsv",
                locale = readr::locale(encoding = "UTF-16LE"), col_types = cols()) -> df_temp

df_temp %>% select(ResponseId, contains("kline_name"))->df_temp_k
split_df(which_rating = "Warm", which_study = "kline", df=df_temp_k) %>%
  rename("warm"="Warm") %>%
  left_join(split_df(which_rating = "Competent", which_study = "kline", df=df_temp_k) %>%
              rename("competent"="Competent"), 
            by=c("name", "ResponseId", "study"))->df_kline

df_temp %>% select(ResponseId, contains("leasure_name"))->df_temp
split_df(which_rating = "Warm", which_study = "flake_leasure", df=df_temp_k) %>%
  rename("warm"="Warm") %>%
  left_join(split_df(which_rating = "Competent", which_study = "flake_leasure", df=df_temp_k) %>%
              rename("competent"="Competent"), 
            by=c("name", "ResponseId", "study"))->df_flake_leasure

### combine 
rm(df_temp,df_temp_k,df_temp_c,df_temp_w,matches,pattern,split_df)
df_names <- ls()[sapply(mget(ls()), is.data.frame)] # Identify data frames in the environment
df_combined <- do.call(rbind, lapply(df_names, get)) # Bind all data frames together

## Pivot wider
df_combined %>%
  mutate(study=tolower(study)) %>%
  mutate(name = tolower(str_trim(name))) %>%
  write.csv(file = "df_all.csv", row.names = FALSE)


### Saving .dta file for prediction analysis
# PCA
df_combined %>%
  ungroup()%>%
  group_by(name, study) %>%
  mutate(warmth     = mean(as.numeric(warm)),
         competence = mean(as.numeric(competent))) %>%
  filter(!is.na(warmth)) %>%
  filter(!is.na(competence)) %>%
  ungroup()-> df_temp
pc <- prcomp(df_temp %>% select(warmth, competence) ,
             center = TRUE, scale. = TRUE)

df_temp %>%
  add_column(pc1 = pc[["x"]][,1]*-1 ,
             pc2 = pc[["x"]][,2]*-1)  -> df_temp

write_dta(df_temp, "df_all.dta")
