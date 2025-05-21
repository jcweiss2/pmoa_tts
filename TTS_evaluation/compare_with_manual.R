# ### Input, folders where annotations are, manual is the referent, refs are folder you want to compare
# 
# ### Output: data frames of match comparisons: event matching plot, concordance plots, time discrepancy plots 
# 
##  # bsv --> csv helper
# ann.dir =  c(
#   "/Users/kumars33/Desktop/CHARM_MIMIC/replicates/dsr1_annotations/dsr1_without_ablation/",
#   "/Users/kumars33/Desktop/CHARM_MIMIC/replicates/dsr1_annotations/0/clean2/",
#   "/Users/kumars33/Desktop/CHARM_MIMIC/replicates/dsr1_annotations/1/clean2/",
#   "/Users/kumars33/Desktop/CHARM_MIMIC/replicates/dsr1_annotations/2/clean2/",
#   "/Users/kumars33/Desktop/CHARM_MIMIC/replicates/dsr1_annotations/3/clean2/",
#   "/Users/kumars33/Desktop/CHARM_MIMIC/replicates/dsr1_annotations/4/clean2/"
# )
# 
# #ann.dir = c("/Users/kumars33/Desktop/CHARM_MIMIC/dsr1_ablates_n20/dsr1_without_ablation/")
# df = tibble(ad=ann.dir, fns = map(ann.dir, list.files)) %>%
#   unnest(everything()) %>%
#   filter(str_detect(fns, "bsv$")) %>%
#   #slice(19:21) %>% #as.data.frame()
#   mutate (
#     result = map2(
#       ad,
#       fns,
#       ~ read_delim(paste0(.x, .y), delim="|", col_names=F) %>% mutate(empty=F) %>% select(event=1,time=2) %>% write_csv(paste0(.x, str_replace_all(.y, "\\.bsv", ".csv")))
#       # ~ fread(paste0(.x, .y), sep="|") %>% select(1,2) %>% write_csv(paste0(.x, str_replace_all(.y, "\\.bsv", ".csv")))
#     ) %>% list() #(function(.) {str(.); print(".")})(.)
#   )

####################################################################################
####################################################################################


### PARAMS ###
library(tidyverse)
library(glmnet)
library(survival)
work.dir = "/Users/kumars33/Desktop/CHARM_MIMIC/dsr1_ablates_n20/"
conda.r.env = "Rtest" # need a conda R environment to run comparer.R (see library(...) at top for reqs: (tidyverse, reticulate, stringdist)
setwd(work.dir)

flags = list(
            event.match = T,
            event.match.files = "",
            # event.match = F,
            # event.match.files = 
            #     c(
            #         "<path_to_best_matches_files_file_1>",
            #         "<path_to_best_matches_files_file_2>",
            #     ),
            out.relfolder = "matches",
            #out.folder = paste0(tempdir(),"/"),
            out.folder = "/Users/kumars33/Desktop/CHARM_MIMIC/dsr1_ablates_n20/outputs/",
            event.out.locs = tempdir(),  # if !event.match, use the list here to load
            distance.method = "embedding_cosine",
            distance.threshold = 0.1,
            # pilot.names = c("GPT-4", "GPT-4 w/feedback")
            #pilot.names = c("o1-preview", "l3.1_8b", "l3.1_70b", "l3.3_70b")
            # pilot.names = c("o1-preview", "l3.1_8b", "l3.1_70b", "text-order")
            # pilot.names = c("text-order")
            # pilot.names = c("l3.1_8b", "l3.1_70b", "l3.3_70b")
            
            # pilot.names = c("L33-70B","DeepSeek-R1","DeepSeek-R1-UD-IQ1_S","DeepSeek-V3-UD-IQ1_S","o1-241217-azure")
            pilot.names = c("DSR1", "Zero-shot", "No conjunction", "No role", "Interval", "Interval+Type")
            #pilot.names = c("L33-70B", "0","1","2","3","4")
        )

man.loc = "/Users/kumars33/Desktop/CHARM_MIMIC/dsr1_ablates_n20/manual/"

ref.locs = 
    c(
        "/Users/kumars33/Desktop/CHARM_MIMIC/dsr1_ablates_n20/dsr1_without_ablation/",
        "/Users/kumars33/Desktop/CHARM_MIMIC/dsr1_ablates_n20/0shot/clean/",
        "/Users/kumars33/Desktop/CHARM_MIMIC/dsr1_ablates_n20/noconjinst/clean/",
        "/Users/kumars33/Desktop/CHARM_MIMIC/dsr1_ablates_n20/norole/clean/",
        "/Users/kumars33/Desktop/CHARM_MIMIC/dsr1_ablates_n20/winterval/clean/",
        "/Users/kumars33/Desktop/CHARM_MIMIC/dsr1_ablates_n20/wintervaltyping/clean/"
    )

get_file_paths = function(loc, suffix=".*", dir.append=T) {
    tibble(files = list.files(loc)) %>%
    mutate(paths = ifelse(rep(dir.append,nrow(.)), paste0(loc,files), files)) %>%
    filter(str_detect(paths, paste0(suffix, "$"))) %>%
    .[["paths"]]
}


sources = 
  tibble(locs = c(man.loc, ref.locs), 
         loctype = c("reference", rep("pilot", length(ref.locs))),
         loc.names = c("manual", flags$pilot.names)
  ) %>%
  mutate(files = map(locs, ~ get_file_paths(.x, ".csv"))) %>%
  mutate(wo_paths = map(locs, ~ get_file_paths(.x, ".csv", dir.append=F))) %>% unnest(everything())

### END PARAMS ###


### First do event matching with man.loc, and save them (if dne)
if(flags$event.match) {
    source("/Users/kumars33/Desktop/Clinical_Case_Resport/Sayantan/pmoa241217_man_n20/codes/comparer.R")
    match.tbl = 
        sources %>% filter(loctype=="reference") %>% select(-loc.names) %>%
        # get referent vs each pilot
        inner_join(
            sources %>% filter(loctype=="pilot") %>%
            select(wo_paths, pilot.files = files, pilot.locs = locs, pilot.names=loc.names),
            by = c("wo_paths")
        ) %>%
        select(-loctype, -locs) %>%
        rename(common.files=wo_paths) %>%
        # nest(data = -pilot.locs) %>%
        # do event matching and save them
        # sample_n(2) %>%
        mutate(referent.events = map(files, ~ read_csv(.x)[["event"]])) %>%
        mutate(pilot.events = map(pilot.files, ~ read_csv(.x)[["event"]])) %>%
        mutate(event.match = map2(referent.events, pilot.events, 
            ~ get_match_table(.x, .y, method = flags$distance.method)
        ))

    match.tbl %>% select(pilot.locs) %>%
        mutate(map(pilot.locs, ~ dir.create(paste0(.x,"/", flags$out.relfolder))))
    match.tbl %>% select(pilot.locs, common.files, files, pilot.files, event.match) %>%
        unnest(everything()) %>%
        nest(data = -pilot.locs) %>%
        mutate(complete = map2(pilot.locs, data, ~ .y %>% write_csv(
            paste0(.x, "/", flags$out.relfolder, "/best_matches", as.character(lubridate::today()), ".csv") 
        )))
} else {
    # Reload saved match.tbl
    loaded.match.tbl = 
        tibble(pilot.locs = flags$event.match.files) %>%
        mutate(data = map(pilot.locs, ~ read_csv(.x))) %>%
        unnest(everything())

    # Join it to the source data for table re-creation (from which the match.tbl was created)
    # If there is a mismatch, the inner join should fail.
    match.tbl = 
        sources %>% filter(loctype=="reference") %>% select(-loc.names) %>%
        # get referent vs each pilot
        inner_join(
            sources %>% filter(loctype=="pilot") %>%
            select(wo_paths, pilot.files = files, pilot.locs = locs, pilot.names=loc.names),
            by = c("wo_paths")
        ) %>%
        select(-loctype, -locs) %>%
        rename(common.files=wo_paths) %>%
        inner_join(loaded.match.tbl %>% select(-pilot.locs), by=c("common.files","pilot.files","files")) %>%
        nest(event.match = c(v1,v2,error.rate,keep))  # match.tbl is presented at file-match level
}

### Then, among event matches, do concordance and temporal component
### Get aligned times: referent -join- match -join- pilot, then plot.
match.tbl = match.tbl %>%
    mutate(man.tuples = map(files, ~ read_csv(paste0(.x),
                                        skip_empty_rows = F,
                                        col_types=cols("c","c")))) %>%
    mutate(man.join.match = map2(man.tuples, event.match,
        ~ inner_join(.x, .y, by=c("event"="v1")))
    ) %>%
    mutate(pilot.tuples = map(pilot.files, ~ read_csv(paste0(.x),
                                        skip_empty_rows = F,
                                        col_types=cols("c","c")))) %>%
    mutate(man.join.match.join.pilot = map2(pilot.tuples, man.join.match,
        ~ inner_join(.x %>% 
            mutate(time = str_replace_all(time, "\\\\","")) %>%
            rename(time.pilot=time, event.pilot=event), .y, by=c("event.pilot"="v2")))
    )


##########################
##### Now make plots #####
##########################

### Event match plot (you need the event match info (w/lack of matching))
max.error.rate = match.tbl %>% select(common.files, pilot.files, pilot.names, man.join.match) %>%
    unnest(everything()) %>% select(error.rate) %>% 
    filter(is.finite(error.rate)) %>% max()

### Tables ###
match.tbl %>% select(common.files, pilot.files, pilot.names, man.join.match) %>%
    unnest(everything()) %>%
    mutate(keep=error.rate < flags$distance.threshold, threshold = flags$distance.threshold) %>%
    rename(Version=pilot.names) %>%
    mutate(error.rate = ifelse(is.na(error.rate),Inf, error.rate)) %>%
    group_by(Version) %>%
    summarise(`Match Rate`=mean(error.rate<threshold), count=n())


### Apples-to-apples concordance (unmatched in common set get mapped to Inf)

### Do you want to compare on a consensus event set? If yes, need to make/load a consensus set.
### Else, omit the full.consensus.matches df and omit the join to full.consensus.matches
# (1) get consensus matches
if(!dir.exists(paste0(flags$out.folder, flags$out.relfolder))) { dir.create(paste0(flags$out.folder, flags$out.relfolder))}
consensus.matches = match.tbl %>% select(common.files, pilot.files, pilot.names, man.join.match.join.pilot,) %>%
    unnest(everything()) 
# consensus.matches %>% write_csv(paste0(flags$out.folder, flags$out.relfolder,'/',paste(flags$pilot.names,collapse="_"),"consensus.csv"))
# # Option to load full. here from csv
# full.consensus.matches = consensus.matches %>%
#     # bind_rows(
#     #     read_csv("/var/folders/dq/9wgy3t592zl5ydylz6gvvjy9px7jzr/T/RtmpojC3VE/matches/L33-70B_DeepSeek-R1_DeepSeek-R1-UD-IQ1_S_DeepSeek-V3-UD-IQ1_Sconsensus.csv") %>% mutate(time=as.character(time)) %>%
#     #       filter(pilot.names!="L33-70B")
#     # ) %>%
#     filter(error.rate<flags$distance.threshold) %>%
#     group_by(common.files, event) %>%
#     reframe(count=n(), time, error.rate) %>%
#     filter(count>7) %>%
#     select(-count) %>%
#     distinct(common.files, event, .keep_all=T) %>% 
#     crossing(match.tbl %>% select(pilot.names) %>% distinct())

match.tbl %>% select(common.files, pilot.files, pilot.names, man.join.match.join.pilot) %>%
    unnest(everything()) %>%
    # select(-error.rate) %>%
    # full_join(full.consensus.matches,
    #     by=c("common.files","event","pilot.names","time")
    # ) %>%
    # mutate(dur = as.duration(ifelse(time=="0","0 seconds",time)) %>% as.numeric("hours")) %>%
    # mutate(time = ifelse(str_starts(str_trim(time), "-"), -dur, dur)) %>%
    mutate(time=as.numeric(time)) %>%
    # select(-dur) %>%
    mutate(time.pilot = as.numeric(time.pilot)) %>% mutate(time.pilot = ifelse(is.na(time.pilot),1e8,time.pilot)) %>% # filter(!is.na(time.pilot)) %>%
    # mutate(time.pilot.num = as.numeric(time.pilot)) %>% filter(is.na(time.pilot.num)) %>% View
    filter(error.rate<flags$distance.threshold) %>% # filter(is.na(time.pilot)) %>% View
    nest(data=-c(common.files, pilot.names)) %>%
    filter(map_dbl(data, nrow)>1) %>%
    mutate(
        concordance = map_dbl(data,
            ~ 1 - Cindex(pred=.x$time.pilot, y=Surv(.x$time, rep(1,nrow(.x))))
        )
    ) %>%
    select(-data) %>% 
    group_by(pilot.names) %>% 
    summarise(
        Concordance = median(concordance, na.rm=T),
        C75 = quantile(concordance, .75, na.rm=T),
        C25 = quantile(concordance, .25, na.rm=T)
    )

# AULTC
ecdf_auc2 = function(df, def=0, upper=log(60*24*365.26)) {
    df %>%
    arrange(field) %>% 
    mutate(
        field2 = lag(field, default=def),
        nth = (1:n())/n(),
        dx = field - field2,
        box = dx*nth 
    ) %>%
    filter(field<upper) %>%
    summarise(auc = sum(box)/(min(max(field, na.rm=T),upper)-min(field, na.rm=T))) %>% t() %>% c()
}
match.tbl %>% select(common.files, pilot.files, pilot.names, man.join.match.join.pilot) %>%
    unnest(everything()) %>%
    # mutate(dur = as.duration(ifelse(time=="0","0 seconds",time)) %>% as.numeric("hours")) %>%
    # mutate(time = ifelse(str_starts(str_trim(time), "-"), -dur, dur)) %>%
    mutate(time=as.numeric(time)) %>% 
    # select(-dur) %>%
    mutate(time.pilot = as.numeric(time.pilot)) %>%
    mutate(keep=error.rate < flags$distance.threshold, threshold = flags$distance.threshold) %>%
    rename(Version=pilot.names) %>% 
    filter(keep) %>% #nest(data=-file.name)
    mutate(ae = abs(as.numeric(time.pilot)-as.numeric(time))) %>%
    mutate(lae = ifelse(is.na(ae), max(ae,na.rm=T)+1,ae+1)) %>%
    select(Version, lae) %>%
    nest(data=-Version) %>%
    mutate(auc = map_dbl(data,
        ~ .x %>% select(field=lae) %>% mutate(field=log(field)) %>%
          ecdf_auc2()
    ))

### FIGURES ###

#pilot.names = c("L33-70B","DeepSeek-R1","DeepSeek-R1-UD-IQ1_S","DeepSeek-V3-UD-IQ1_S")
#pilot.names = c("DSR1", "Zero-shot", "No conjunction", "No role", "Interval", "Interval+Type")
#pilot.names = c("L33-70B", "0","1","2","3","4")


pdf(paste0(flags$out.folder,"ablation_figures.pdf"), width=3.2, height=6)
# svg(paste0(tempdir(),"/figures.svg"), width=4, height=7)
match.tbl %>% select(common.files, pilot.files, pilot.names, man.join.match) %>%
    unnest(everything()) %>%
    mutate(keep=error.rate < flags$distance.threshold, threshold = flags$distance.threshold) %>%
    rename(Version=pilot.names) %>%
    mutate(Version = fct_relevel(Version, sort(levels(Version)))) %>%
    mutate(Version = fct_relevel(Version, levels(Version)[c(2,1,3,4)])) %>%
    mutate(Version = fct_recode(Version,
        `DSR1` = "DSR1",
        `Zero-shot` = "Zero-shot",
        `No conjunction` = "No conjunction",
        `No role` = "No role",
        `Interval` = "Interval",
        `Interval+Type` = "Interval+Type"
    )) %>%
    ggplot(data = ., aes(x=ifelse(is.na(error.rate) | is.infinite(error.rate),max(error.rate[!is.infinite(error.rate)],na.rm=T)+0.01, error.rate),
                         color=Version)) +
    stat_ecdf() + 
    # geom_histogram(alpha=0.15, fill="red", aes(y=0.01*after_stat(density)), binwidth=0.01) +
    geom_segment(aes(x = 0.1, xend=0.1, y=0, yend=1), color="grey", alpha=0.5) + xlab("Error rate") +
    scale_y_continuous(n.breaks=10) + theme_minimal() + 
    scale_color_manual(values = c("#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#F0E442", "#0072B2", "#D55E00"))+
    scale_fill_manual(values = c("#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#F0E442", "#0072B2", "#D55E00"))+
    theme(panel.grid.minor = element_blank(), 
        legend.position = c(0.98, 0.25),
        legend.justification = c("right", "bottom"),
        legend.box.background = element_rect(color="#f3f3f3", fill="#f3f3f3", size=2),
        legend.text = element_text(size = 8),
        legend.title=element_text(size=10)
    ) +
    coord_cartesian(xlim=c(0, max.error.rate)) +
    ylab("Cumulative density") + xlab("Cosine distance") # + 
    # labs(title="Manual event match rate")

## Concordance plot
options(vsc.dev.args = list(width=600, height=1500, pointsize=10, res=300))
match.tbl %>% select(common.files, pilot.files, pilot.names, man.join.match.join.pilot) %>%
    unnest(everything()) %>%
    #select(-error.rate) %>%
    # full_join(full.consensus.matches,
    #     by=c("common.files","event","pilot.names","time")
    # ) %>%
    # mutate(dur = as.duration(ifelse(time=="0","0 seconds",time)) %>% as.numeric("hours")) %>%
    # mutate(time = ifelse(str_starts(str_trim(time), "-"), -dur, dur)) %>%
    mutate(time=as.numeric(time)) %>%
    # select(-dur) %>%
    mutate(time.pilot = as.numeric(time.pilot)) %>% mutate(time.pilot = ifelse(is.na(time.pilot),1e8,time.pilot)) %>% # filter(!is.na(time.pilot)) %>%
    # mutate(time.pilot.num = as.numeric(time.pilot)) %>% filter(is.na(time.pilot.num)) %>% View
    filter(error.rate<flags$distance.threshold) %>% # filter(is.na(time.pilot)) %>% View
    nest(data=-c(common.files, pilot.names)) %>%
    mutate(
        concordance = map_dbl(data,
            ~ 1 - Cindex(pred=.x$time.pilot, y=Surv(.x$time, rep(1,nrow(.x))))
        )
    )%>% # filter(pilot.names=="l3.3_70b") %>% .[["concordance"]] %>% summary()
    select(-data) %>%
    mutate(Version = fct_relevel(pilot.names, sort(levels(pilot.names)))) %>%
    mutate(Version = fct_relevel(Version, levels(Version)[c(2,1,3,4)])) %>%
    mutate(Version = fct_recode(Version,
        `DSR1` = "DSR1",
        `Zero-shot` = "Zero-shot",
        `No conjunction` = "No conjunction",
        `No role` = "No role",
        `Interval` = "Interval",
        `Interval+Type` = "Interval+Type"
    )) %>%
    ggplot(data=.) + geom_boxplot(aes(y=concordance, x=Version)) + xlab("") + ylab("Concordance") +
    scale_y_continuous(n.breaks=10,limits = c(0.5,1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    
### Time discrepancy plot
options(vsc.dev.args = list(width=1400, height=1850, pointsize=10, res=300))
match.tbl %>% select(common.files, pilot.files, pilot.names, man.join.match.join.pilot) %>%
    unnest(everything()) %>%
    # mutate(dur = as.duration(ifelse(time=="0","0 seconds",time)) %>% as.numeric("hours")) %>%
    # mutate(time = ifelse(str_starts(str_trim(time), "-"), -dur, dur)) %>%
    mutate(time=as.numeric(time)) %>% 
    # select(-dur) %>%
    mutate(time.pilot = as.numeric(time.pilot)) %>%
    mutate(keep=error.rate < flags$distance.threshold, threshold = flags$distance.threshold) %>%
    rename(Version=pilot.names) %>% 
    filter(keep) %>% #nest(data=-file.name)
    mutate(ae = abs(as.numeric(time.pilot)-as.numeric(time))) %>%
    mutate(lae = ifelse(is.na(ae), max(ae,na.rm=T)+1,ae+1)) %>%
    mutate(Version = fct_relevel(Version, sort(levels(Version)))) %>%
    mutate(Version = fct_relevel(Version, levels(Version)[c(2,1,3,4)])) %>%
    mutate(Version = fct_recode(Version,
        `DSR1` = "DSR1",
        `Zero-shot` = "Zero-shot",
        `No conjunction` = "No conjunction",
        `No role` = "No role",
        `Interval` = "Interval",
        `Interval+Type` = "Interval+Type"
    )) %>%
    # mutate(Version = forcats::fct_relevel(levels=c("GPT-4", "GPT-4 w/feedback","O1-preview"))) %>%
    # mutate(lae = ae+1) %>%
    # mutate(forend = is.na(plot.df$ae)|!plot.df$keep) %>%
    ggplot(data=., aes(x=lae, color=Version)) + 
    # facet_grid(. ~ Version) +
    stat_ecdf() +
    geom_histogram(alpha=0.15, aes(y=0.2*after_stat(density), fill=Version), position="identity",color=NA, binwidth=0.2) +
    scale_color_manual(values = c("#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))+
    scale_fill_manual(values = c("#E69F00", "#FFFFFF", "#56B4E9", "#CC79A7", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))+
         # scale_color_manual(values = c("GPT-4"="#E41A1C", "GPT-4 w/feedback"="#377EB8", "O1-preview"="#4DAF4A")) +
    # scale_fill_manual(values = c("GPT-4"="#E41A1C", "GPT-4 w/feedback"="#377EB8", "O1-preview"="#4DAF4A")) + 
    scale_x_log10(breaks=1+c(0,1, 24, 24*7, 24*365.25), labels=c("exact","hour","day","week","year"),guide=guide_axis(angle=90)) +
    scale_y_continuous(n.breaks=10) +
    geom_vline(xintercept = 1+c(0,1, 24, 24*7, 24*365.25), alpha=0.2) +
    theme_minimal() +
    theme(panel.grid.minor = element_blank(),
        legend.position = c(0.98, 0.25),
        legend.justification = c("right", "bottom"),
        legend.box.background = element_rect(color="#f3f3f3", fill="#f3f3f3", size=0),
        # legend.text = element_text(size = 5),
        # legend.title=element_text(size=7)
        # legend.spacing.x = unit(0, 'cm')
    ) +
    xlab("Time difference (log)") + ylab("Cumulative probability") # + labs(title="Annotation time error")


### Time discrepancy plot by manual subgroup
match.tbl %>% select(common.files, pilot.files, pilot.names, man.join.match.join.pilot) %>%
    unnest(everything()) %>%
    # mutate(dur = as.duration(ifelse(time=="0","0 seconds",time)) %>% as.numeric("hours")) %>%
    # mutate(time = ifelse(str_starts(str_trim(time), "-"), -dur, dur)) %>%
    mutate(time=as.numeric(time)) %>% 
    # select(-dur) %>%
    mutate(time.pilot = as.numeric(time.pilot)) %>%
    mutate(keep=error.rate < flags$distance.threshold, threshold = flags$distance.threshold) %>%
    rename(Version=pilot.names) %>% 
    filter(keep) %>% #nest(data=-file.name)
    mutate(ae = abs(as.numeric(time.pilot)-as.numeric(time))) %>%
    mutate(lae = ifelse(is.na(ae), max(ae,na.rm=T)+1,ae+1)) %>%
    mutate(time.group = cut(abs(time), c(-Inf, 0, 1, 24, 24*7, 24*365.25,Inf),
                                     labels=c("Presentation","1 hour","1 day","1 week","1 year","ever"))) %>%
    mutate(lae = ifelse(is.na(ae), max(ae,na.rm=T)+1,ae+1)) %>%
    filter(!is.na(time.group)) %>%
    mutate(Version = fct_relevel(Version, sort(levels(Version)))) %>%
    mutate(Version = fct_relevel(Version, levels(Version)[c(2,1,3,4)])) %>%
    mutate(Version = fct_recode(Version,
        `DSR1` = "DSR1",
        `Zero-shot` = "Zero-shot",
        `No conjunction` = "No conjunction",
        `No role` = "No role",
        `Interval` = "Interval",
        `Interval+Type` = "Interval+Type"
    )) %>%
    # mutate(forend = is.na(plot.df$ae)|!plot.df$keep) %>%
    ggplot(data=., aes(x=lae, color=Version)) + 
    # facet_grid(. ~ version) +
    stat_ecdf() +
    facet_grid(time.group ~ .) +
    # scale_color_brewer(palette="Set1") + 
    scale_color_manual(values = c("#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))+
    scale_fill_manual(values = c("#E69F00", "#FFFFFF", "#56B4E9", "#CC79A7", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))+
    geom_histogram(alpha=0.15, aes(y=0.2*after_stat(density), fill=Version), position="identity",color=NA, binwidth=0.2) +
    scale_x_log10(breaks=1+c(0,1, 24, 24*7, 24*365.25), labels=c("exact","hour","day","week","year"),guide=guide_axis(angle=90)) +
    scale_y_continuous(n.breaks=6) +
    geom_vline(xintercept = 1+c(0,1, 24, 24*7, 24*365.25), alpha=0.2) +
    theme_minimal() +
    theme(panel.grid.minor = element_blank(),
        legend.position = "None"  
        # legend.position=c(0.25,0.85),
        # legend.position="bottom",
        # legend.box.background = element_rect(color="#f3f3f3", fill="#f3f3f3", size=0),
        # legend.text = element_text(size = 5),
        # legend.title=element_text(size=7)
        # legend.spacing.x = unit(0, 'cm')
    ) +
    xlab("Time difference (log)") + ylab("Cumulative probability") # + labs(title="Annotation time error")

dev.off()