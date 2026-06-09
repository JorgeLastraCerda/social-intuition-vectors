"callback from lippens, covidence, and subgroups"

#rm(list=ls())
library(tidyverse)
library(readxl)

script_path <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(script_path)

################################################# LIPPENS
lippens <- read_excel("lippens.xlsx") %>%
  filter(subregion == "Northern America") %>%
  select("ground",
         "study",
         "treatment_group_high",
         "treatment_group_low",
         #"treat_id",
         "treatment_class",
         "class_short",
         "control_group",
         "country",
         "year_off_pub",
         "year_first_pub",
         "year_research",
         "tested_maj",
         "tested_min",
         "callback_maj",
         "callback_min",
         "sensu",
         "doi")
# 10.1257/aer.p20161008 should be 10.1086/701029
lippens[lippens$doi == "10.1257/aer.p20161008" &
          !is.na(lippens$doi),]$doi <- "10.1086/701029"
# 10.1177/2329496514524541 should be 10.1016/j.rssm.2013.10.002
lippens[lippens$doi == "10.1177/2329496514524541" &
          !is.na(lippens$doi),]$doi = "10.1016/j.rssm.2013.10.002"
# 10.1177/0019793916679601 should be 10.1080/10242694.2017.1357521
lippens[lippens$doi == "10.1177/0019793916679601" &
          !is.na(lippens$doi),]$doi = "10.1080/10242694.2017.1357521"
# 10.1086/700184 should be 10.1257/aer.p20161010
lippens[lippens$doi == "10.1086/700184" &
          !is.na(lippens$doi),]$doi = "10.1257/aer.p20161010"
# Mark Flake as doi = Flake
lippens[lippens$study == "Flake (2019)" &
          !is.na(lippens$study),]$doi = "Flake"

rbind(lippens %>%
        select(study, ground, treatment_group_low, callback_min, tested_min) %>%
        rename(level=treatment_group_low) %>%
        rename(callback=callback_min) %>%
        rename(n=tested_min),
      
      lippens %>%
        select(study, ground, control_group, callback_maj, tested_maj) %>%
        rename(level=control_group) %>%
        rename(callback=callback_maj) %>%
        rename(n=tested_maj)
) %>%
  mutate(callback=callback/n)%>%
#  mutate(condition="") %>%
  rename(category=ground) %>%
  mutate(study=tolower(str_extract(study, pattern = "^[^\\s]+"))) %>%
  filter(!is.na(callback))->lippens

################################################# COVIDENCE 

doi <-  read_csv("covidence_doi.csv",show_col_types = FALSE) %>%
  mutate(`Publication Year` = as.character(`Publication Year`))
doi[doi$DOI == "10.1086/704008" & !is.na(doi$DOI),]$`Publication Year` <- "2019a"
doi[doi$DOI == "10.1086/701029" & !is.na(doi$DOI),]$`Publication Year` <- "2019b"
doi[doi$DOI == "10.1177/00197939211036444" & !is.na(doi$DOI),]$`Publication Year` <- "2021"

covidence <- read_csv("covidence_raw.csv", show_col_types = FALSE) %>%
  mutate(`Study ID...13` = ifelse(is.na(`Study ID...13`),
                                  `Study ID...2`,
                                  `Study ID...13`)) %>%
  separate(`Study ID...13`, c('studyname', 'studyyear'), sep = " ") %>%
  select('Lippens et al.?',
         'studyname',
         'studyyear',
         "Signal",
         "Location",
         "Paired",
         "Subgroup Analysis",
         "Restaurant",
         "Blue Collar",
         "Office Focus",
         "Degree Level",
         #"Document Types",
         "Start date",
         "End date",
         "Total applications",
         "Method of resume submission",
         "Group 1 Treatment", "Group 1 Control", "Group 1 Condition",
         "Group 2 Treatment", "Group 2 Control", "Group 2 Condition",
         "Group 3 Treatment", "Group 3 Control", "Group 3 Condition",
         "Group 4 Treatment", "Group 4 Control", "Group 4 Condition",
         "Group 5 Treatment", "Group 5 Control", "Group 5 Condition",
         "Group 6 Treatment", "Group 6 Control", "Group 6 Condition",
         "Group 1 # resumes (treat)", "Group 1 % callbacks (treat)",
         "Group 1 # resumes (control)", "Group 1 % callbacks (control)",
         "Group 2 # resumes (treat)", "Group 2 % callbacks (treat)",
         "Group 2 # resumes (control)", "Group 2 % callbacks (control)",
         "Group 3 # resumes (treat)", "Group 3 % callbacks (treat)",
         "Group 3 # resumes (control)", "Group 3 % callbacks (control)",
         "Group 4 # resumes (treat)", "Group 4 % callbacks (treat)",
         "Group 4 # resumes (control)", "Group 4 % callbacks (control)",
         "Group 5 # resumes (treat)", "Group 5 % callbacks (treat)",
         "Group 5 # resumes (control)", "Group 5 % callbacks (control)",
         "Group 6 # resumes (treat)", "Group 6 % callbacks (treat)",
         "Group 6 # resumes (control)", "Group 6 % callbacks (control)")


covidence[covidence$studyname == "Figinski",]$studyyear <- "2019" # Figinski is 2019
covidence <- covidence %>% left_join(doi, join_by(closest(studyname <= Author), studyyear == "Publication Year")) %>% rename(doi = DOI)
covidence[covidence$studyname == "Lahey",]$doi = "10.3368/jhr.43.1.30" # Lahey should be 10.3368/jhr.43.1.30
covidence[covidence$studyname == "Flake",]$doi = "Flake" # Mark Flake as doi = Flake


covidence_unique <- covidence %>%
  filter(`Lippens et al.?` == "No") %>%
  pivot_longer(cols = ends_with(")"),names_prefix = "Group ", names_pattern = "(\\d+)(.*)", names_to = c("group","name"), values_drop_na = FALSE) %>%
  pivot_longer(cols = starts_with("Group "), names_prefix = "Group ", names_pattern = "(\\d+)(.*)", names_to = c("group2","name2"), values_to = "value2", values_drop_na = TRUE) %>%
  filter(group == group2) %>%
  select(!"group2") %>%
  pivot_wider(names_from = "name", values_from = "value") %>%
  pivot_wider(names_from = "name2", values_from = "value2")

covidence <- covidence %>%
  filter(`Lippens et al.?` == "Yes") %>%
  select(!c( "Group 1 Treatment":"Group 6 % callbacks (control)")) %>%
  bind_rows(covidence_unique) %>%
  mutate(Paired = case_when(grepl("Yes", Paired) ~ 1, grepl("No",  Paired) ~ 0),
         `Subgroup Analysis` = case_when(grepl("Yes", `Subgroup Analysis`) ~ 1, grepl("No",  `Subgroup Analysis`) ~ 0),
         Restaurant = case_when(grepl("Yes", Restaurant) ~ 1, grepl("No",  Restaurant) ~ 0),
         `Blue Collar` = case_when(grepl("Yes", `Blue Collar`) ~ 1, grepl("No",  `Blue Collar`) ~ 0),
         `Office Focus` = case_when(grepl("Yes", `Office Focus`) ~ 1, grepl("No",  `Office Focus`) ~ 0),
         high_school = case_when(grepl("High school", `Degree Level`) ~ 1, .default = 0),
         some_college = case_when(grepl("Some college", `Degree Level`) ~ 1, .default = 0),
         college = case_when(grepl("College degree", `Degree Level`) ~ 1, .default = 0),
         graduate_degree = case_when(grepl("Graduate degree", `Degree Level`) ~ 1, .default = 0)) %>%
  select(!c("Degree Level", "group")) %>%
  rename(tested_maj = ` # resumes (control)`,
         tested_min = ` # resumes (treat)`,
         proportion_maj = ` % callbacks (control)`,
         proportion_min = ` % callbacks (treat)`) %>%
  mutate(tested_maj = as.numeric(tested_maj),
         tested_min = as.numeric(tested_min))



rbind(covidence %>%
        select(Author,` Treatment`,` Condition`, proportion_min,tested_min) %>%
        rename(study=Author) %>%
        rename(level=` Treatment`) %>%
        rename(condition=` Condition`) %>%
        rename(callback=proportion_min) %>%
        rename(n=tested_min) ,
      
      covidence %>%
        select(Author,` Control`,` Condition`, proportion_maj,tested_maj) %>%
        rename(study=Author) %>%
        rename(level=` Control`) %>%
        rename(condition=` Condition`) %>%
        rename(callback=proportion_maj) %>%
        rename(n=tested_maj)) %>%
  
  mutate(study=tolower(str_extract(study, pattern = "^[^,]+"))) %>%
  mutate(level=tolower(level)) %>%
  
  mutate(level = ifelse(study == "bailey" & level == "straight men", trimws(gsub("men", "", level)), level)) %>%
  mutate(level = ifelse(study == "bailey" & level == "straight women", trimws(gsub("women", "", level)), level)) %>%
  mutate(level=trimws(gsub("philly", "", level))) %>%
  mutate(level=trimws(gsub("chicago", "", level))) %>%
  mutate(level=trimws(gsub("dallas", "", level))) %>%
  
  mutate(level = str_replace(level, "illness-explained gap", "illness")) %>%
  mutate(level = str_replace(level, "newly unemployed", "shortgap")) %>%
  mutate(level = str_replace(level, "unexplained gap", "longgap")) %>%
  
  mutate(level = str_replace(level, "lgbtq club member", "gay")) %>%
  mutate(level = str_replace(level, "other club", "straight")) %>%
  
  mutate(level = str_replace(level, "childless women", "nonparent")) %>%
  mutate(level = str_replace(level, "mother", "parent")) %>%
  
  
  mutate(category = if_else(study == "jolson", "religion", 
                    if_else(study == "gaulke", "education", 
                    if_else(study == "kreisberg", "union", 
                    if_else(study == "ishizuka", "parenthood",
                    if_else(study == "beauregard" | study == "bertrand" | study == "deming" | study == "pedulla" | study == "gorzig" , "race and national origin",
                    if_else(study == "kroft" | study == "namingit" , "unemployed",  
                    if_else(study == "yemane" & level == "black" | level == "white" , "race",
                    if_else(study == "yemane" & level == "men" | level == "women" , "gender",
                    if_else(study == "yemane" & level == "mother" | level == "father" , "parent",
                    if_else(study == "tilcsik" | study == "mishel"| study == "bailey" | study == "kline", "sexual orientation",NA))))))))))) %>%
  
  group_by(study, category, level) %>%
  mutate(n=sum(n)) %>%
  mutate(callback=mean(callback)) %>%
  select(!condition) %>% 
  unique() %>%
  filter(!is.na(callback))%>%
  mutate(level = ifelse(study == "ishizuka" & level == "parents", "parent", level))->covidence

rm(covidence_unique,doi)
################################################# SUBGROUPS

read_excel("subgroups.xlsx") %>%
  select(Study, callback,total) %>%
  mutate(Study = str_replace(Study, "farber60", "farber_60")) %>%
  mutate(Study = str_replace(Study, "farber51", "farber_51")) %>%
  separate(Study, c("rm", "study", "level"), sep = "_") %>%
  select(!"rm") %>%
  mutate(category = if_else(study == "wrigth", "religion", 
                    if_else(study == "yemane", "race and national origin",
                    if_else(study == "correll", "parenthood",
                    if_else(study == "hipes" | study == "ameri" , "health",
                    if_else(study == "neumark" | study == "farber", "age",
                    if_else(study == "gaulke" , "gender",
                    if_else(study == "figinski", "military service or affiliation",
                    if_else(study == "rivera" | study == "thomas", "wealth",
                    if_else(study == "tilcsik" | study == "mishel", "sexual orientation",NA)))))))))) %>%
  rename(n=total) %>%
  mutate(level=tolower(level)) %>%
  mutate(category=tolower(category)) %>%
  
  group_by(study, category, level) %>%
  mutate(n=sum(n)) %>%
  mutate(callback=mean(callback)) %>%
  unique() %>%
  filter(!is.na(callback)) %>%
  select(study, category, level, callback,n)->subgroups

################################################# bind and write

rbind(subgroups, lippens, covidence) %>%
  mutate(category=tolower(category),
         level=tolower(level))%>%
 # filter(!is.na(callback)) %>%
  unique() %>%
  arrange(study, level)->df

write.csv(df, file = "df_all.csv", row.names = FALSE)

