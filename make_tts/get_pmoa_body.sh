#!/bin/bash

DIR=$1

echo "filling /body/grepcase/ in ${DIR}"
for i in ${DIR}/*.txt.gz; do
    INNER=$(basename "$i")
    INNERFN="${INNER%.*}"
    gzip -dc "$i" | awk '/==== Body/{a=1;next}/==== Ref/{a=0}a' | gzip > "${DIR}/body/${INNERFN%.*}_body.txt.gz"
done
