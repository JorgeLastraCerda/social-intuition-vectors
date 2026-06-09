"extracting cb for names"

rm(list=ls())
library(tidyverse)
library(haven)
library(readxl)

script_path <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(script_path)

# 1. Bertrand -------------------------------------------------------------

df_bertrand <- read_dta("bertrand2004.dta")

df_bertrand %>%
  rename(name = firstname,
         gender = sex,
         race = race_str) %>%
  select(name, call,
         race, gender) %>%
  group_by(name, race, gender) %>%
  summarise(callback = mean(call, na.rm = TRUE),
            callback_n = sum(call, na.rm = TRUE),
            n = n()) %>%
  drop_na() %>%
  mutate(gender = case_when(gender == "m" ~ "Male",
                            gender == "f" ~ "Female"),
         race = case_when(race == "b" ~ "Black",
                          race == "w" ~ "White")) %>%
  mutate(study = "Bertrand") -> df_bertrand


# 2. Neumark --------------------------------------------------------------

df_neumark <- read_dta("neumark2019.dta")

df_neumark %>%
  rename(name = firstname) %>%
#  mutate(name=paste(name, lastname, sep=" ")) %>%
  select(name, callback,age, gender) %>%
  group_by(name, age, gender) %>%
  summarise(callback = mean(callback, na.rm = TRUE),
            callback_n = sum(callback, na.rm = TRUE),
            n = n()) %>%
  drop_na() %>%
  mutate(race = "White",
         study = "Neumark") -> df_neumark


# 3. Farber ---------------------------------------------------------------

df_farber <- read_dta("farber2016.dta")

df_farber %>%
  mutate(person=factor(person)) %>%
  mutate(person=recode_factor(person, 
                       "1" = "Angela",
                       "2"=    "Donna",
                       "3"=  "Heather",
                       "4"=   "Janice",
                       "5"= "Jennifer",
                       "6"=     "Joan",
                       "7"=   "Lauren",
                       "8"=    "Linda",
                       "9"=     "Mary",
                       "10"=     "Rose",
                       "11"=  "Shannon",
                       "12"=    "Susan")) %>%
  rename(name = person) %>%
  select(name, cback, age) %>%
  group_by(name, age) %>%
  summarise(callback = mean(cback, na.rm = TRUE),
            callback_n = sum(cback, na.rm = TRUE),
            n = n()) %>%
  drop_na() %>%
  mutate(race = "White",
         gender = "Female",
         study = "Farber") -> df_farber



# 4. Nunley ---------------------------------------------------------------

df_nunley<-read_dta("nunley2015.dta")

df_nunley %>%
  rename(name = Name) %>%
  mutate(gender = case_when(female == 1 ~ "Female",
                            male == 1 ~ "Male"),
         race = case_when(black == 1 ~ "Black",
                          white == 1 ~ "White")) %>%
  select(name, call_back, race, gender) %>%
  group_by(name, race, gender) %>%
  summarise(callback = mean(call_back, na.rm = TRUE),
            callback_n = sum(call_back, na.rm = TRUE),
            n = n()) %>%
  drop_na() %>%
  mutate(study = "Nunley") -> df_nunley


# 5. Kline ----------------------------------------------------------------

df_kline<-read_dta("kline2022.dta")

df_kline %>%
  rename(name = firstname) %>%
  select(name, cb, race, gender) %>%
  group_by(name, race, gender) %>%
  summarise(callback = mean(cb, na.rm = TRUE),
            callback_n = sum(cb, na.rm = TRUE),
            n = n()) %>%
  drop_na() %>%
  mutate(study = "Kline") -> df_kline


# 6. Oreopoulos -----------------------------------------------------------

df_oreopoulos<-read_dta("oreopoulos2011.dta")

df_oreopoulos %>%
  rename(race = name_ethnicity,
         gender = female) %>%
  select(name, callback, race, gender) %>%
  mutate(name=str_replace(name, "([a-z])([A-Z])", "\\1 \\2")) %>% 
  group_by(name, race, gender) %>%
  summarise(callback = mean(callback, na.rm = TRUE),
            callback_n = sum(callback, na.rm = TRUE),
            n = n()) %>%
  drop_na() %>%
  mutate(gender = case_when(gender == 1 ~ "Female",.default = "Male"),
          race = case_when(race == "British" ~ "White",
                           race == "Canada" ~ "White",
                           .default = NA)) %>%
  mutate(study = "Oreopoulos") -> df_oreopoulos
  

# 7. Extracted Names ------------------------------------------------------

extracted <- read_excel("../extracted_data/names.xlsx") %>%
  select(-...8) %>%
  mutate(callback_n=as.integer(callback*n))

# 8.  Save csv's ----------------------------------------------------------


# Merge
df_bertrand %>%
  bind_rows(df_farber,
            df_kline,
            df_neumark,
            df_nunley,
            df_oreopoulos,
            extracted) %>%
  mutate(study=tolower(study)) %>%
  mutate(name = tolower(str_trim(name))) %>%
  write.csv(file = "df_all.csv", row.names = FALSE)

