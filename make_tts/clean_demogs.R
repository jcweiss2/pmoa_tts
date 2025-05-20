library(tidyverse)

dat = read_delim("<OUT_DIR>/demogs250417_final.bsv", delim="|", col_names=F)

dat2 = dat %>% 
    mutate(X1 = as.numeric(str_replace_all(X1,".*: ","")))

dat2 %>% select(X2) %>% table()
dat2 %>% select(X2) %>% table() /length(dat2)
dat2 %>% select(X1) %>% summary()
pdf(paste0(tempdir(),"/figures.pdf"), width=5, height=5)
dat2 %>% select(X1) %>% # sample_n(10000,replace=F) %>%
    ggplot(data=., aes(x=X1)) +
    geom_histogram(binwidth=1) + xlab("Age at Presentation (y)") + ylab("Count") +
    scale_x_continuous(breaks = seq(0, 100, 10))
dev.off()

dat2 %>% select(X3) %>% table()
