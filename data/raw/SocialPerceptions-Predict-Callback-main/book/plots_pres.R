" Forest plot of correlations for names"

tibble(
  "variable"=c("pooled effect", m.cor_wc$studlab),
  "correlation" = c(rcor, m.cor_wc$cor),
  "lower" = c(m.cor_wc$lower.random, m.cor_wc$lower),
  "upper" = c(m.cor_wc$upper.random,m.cor_wc$upper),
  "pvalue" = c(m.cor_wc$pval.random, m.cor_wc$pval),
  "SE" = c(m.cor_wc$seTE.random,m.cor_wc$seTE)) %>%
  mutate_if(is.numeric, ~round(., 3)) -> df_temp2


library(ggforestplot)

forestplot(
  df = df_temp2,
  name=variable,
  estimate = correlation,
  pvalue = pvalue,
  se=SE,
  colour=variable,
  logodds = FALSE, 
) +
  geom_vline(xintercept=rcor, size=.5,color = "#4169E1")+
  theme(legend.position = "none",
        aspect.ratio = 1/2,
        text = element_text(size = 9.4))+
 # xlab(expression(hat(rho)))+
  scale_x_continuous(breaks = c(0, 0.78, 1)) ->plt3


ggsave(plt3, filename = "/Users/carinaines/Desktop/cor.png", dpi = 600)

