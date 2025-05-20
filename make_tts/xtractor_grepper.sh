#!/bin/bash
DPD=$1

find ${DPD}/body/ -name "*.txt.gz" -exec zgrep -ilE "(case report|case presenta)" {} \; | \
    xargs zgrep -ilE "year-? ?old" > \
    ${DPD}/body/grepcase/found.txt
