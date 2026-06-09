#https://cran.r-project.org/web/packages/metaSEM/vignettes/Examples.html#one-stage-meta-analytic-structural-equation-modeling-osmasem
library(semPlot)
library(metaSEM)

# prepare df
df %>%
  ungroup() %>%
  dplyr::select(warm, competent) ->df_temp2
pc <- prcomp(df_temp2,center = TRUE, scale. = TRUE)
s_pc<-summary(pc)

PC1<-pc[["x"]][,1] *-1 
PC2<-pc[["x"]][,2] *-1 


df %>% 
  add_column(PC1) %>%
  group_by(study, name, race, n) %>% 
  mutate(race=ifelse(race == "White", 1, 0)) %>%
#  mutate(gender=ifelse(gender == "Female", 1, 0)) %>%
  summarise(PC1=mean(PC1),
            callback=logit(mean(callback)))->df_avg


#studies<-c( "bertrand","kline","neumark","nunley" ,"oreopoulos")
studies <- c("bertrand","kline","nunley","jacquemet")
cor_matrices <- list()
for (s in studies) {
  study_data <- df_avg[df_avg$study == s, ]
  cor_matrix <- cor(study_data[, c("callback", "race", "PC1")], use = "complete.obs", method="pearson")
  #cor_matrix <- cor(study_data[, c("callback", "gender")], use = "complete.obs")
  cor_matrices[[s]] <- cor_matrix
}

df_avg %>%
  filter(study %in% studies) %>%
  group_by(study)%>%
  #summarise(n=sum(n)) %>%
  summarise(n=n()) %>%
  pull(n)->study_n

# step I
fixed1 <- tssem1(cor_matrices, study_n, method="FEM")

model2 <- "## Math is modeled by Spatial and Verbal
           callback ~ c1*race + c2*PC1
           
           ## Variances of predictors are fixed at 1
           race ~~ 1*race
           PC1 ~~ e2*PC1
           
           ## Correlation between the predictors
           PC1 ~ a1*race
           
           ## Error variance
           callback ~~ e1*callback"

model_cb <- "## Math is modeled by Spatial and Verbal
           callback ~ c1*gender 
           
           ## Variances of predictors are fixed at 1
           gender ~~ 1*gender
           
           
           ## Error variance
           callback ~~ e1*callback"

plot(model2)
RAM2 <- lavaan2RAM(model2, obs.variables=c("callback", "race", "PC1"))
RAM2 <- lavaan2RAM(model_cb, obs.variables=c("callback", "gender"))

fixed2 <- tssem2(fixed1, RAM=RAM2, intervals="LB")
summary(fixed2)
plot(fixed2)

