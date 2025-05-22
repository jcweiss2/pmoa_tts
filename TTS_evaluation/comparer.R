library(tidyverse)
library(stringdist)

### Find the best match of v2 (cols) to each v1 (rows), where 1-to-1 match is assumed
### return a tibble of matches, which round the match was made on, and the distance
recursive_match = function(dists, tbl = NULL, rids = NULL, cids = NULL, iter=0) {
    if(nrow(dists)==0) {
        return(tbl)
    }
    if(is.null(rids)) {
        rids = 1:nrow(dists)
    }
    if(is.null(cids)) {
        cids = 1:ncol(dists)
    }
    best.match = tibble(
        temp.rid = 1:nrow(dists),
        rowmins = dists %>% apply(1, min),
        wrowmins = dists %>% apply(1, which.min)
    ) %>% group_by(wrowmins) %>% arrange(rowmins) %>%
        slice(1) %>% ungroup()
    rids.left = rids[-best.match$temp.rid]
    cids.left = cids[-best.match$wrowmins]

    found.tbl = tibble(dist=best.match$rowmins, rid=rids[best.match$temp.rid], cid=cids[best.match$wrowmins], iter=iter)
    if(is.null(tbl)) {
        tbl = found.tbl
    } else {
        tbl = tbl %>% bind_rows(found.tbl)
    }

    if(length(rids.left) == 0) {
        return(tbl)
    }
    if(length(cids.left) == 0) { # There are no cols left but there are rows, map them to Inf
        return(tbl %>% bind_rows(
            tibble(
                dist = Inf,
                rid = rids.left,
                cid = NA,
                iter = iter + 1
            )
        ))
    }
    return(recursive_match(dists[-best.match$temp.rid,-best.match$wrowmins, drop=F],
                           tbl,
                           rids.left,
                           cids.left,
                           iter+1)
    )
}


#' get_match_table expects two-column tables ("event", "time") and returns event match tables of v2 to v1
get_match_table = function(events1, events2, threshold=0.6, method="lv") {
    events1 = ifelse(is.na(events1),"",events1)
    events2 = ifelse(is.na(events2),"",events2)

    if(method=="embedding_cosine") {
        library(reticulate)
        use_condaenv(conda.r.env)
        source_python("distance_helper.py")
        write_csv(tibble(event=events1, noskip=0), paste0(tempdir(),"1.csv"))
        write_csv(tibble(event=events2, noskip=0), paste0(tempdir(),"2.csv"))
        # system2()
        get_and_write_embeddings(paste0(tempdir(),"1.csv"), paste0(tempdir(),"2.csv"), paste0(tempdir(),"out.csv"))
        dists = read_csv(paste0(tempdir(),"out.csv"), skip_empty_rows = F) %>% as.matrix()
        print(dim(dists))

    } else {
        dists = stringdist::stringdistmatrix(events1, events2,method=method)
    }

    tbl = recursive_match(dists)

    return(tbl %>% mutate(v1 = events1[rid], v2 = events2[cid]) %>%
           select(v1, v2, error.rate=dist) %>%
           mutate(keep = error.rate < threshold))
}