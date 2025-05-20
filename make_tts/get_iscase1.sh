#!/bin/bash
SUBDIR=$1

grep -lE '^[[:space:]]*1[[:space:]]*$' ${SUBDIR}/anns/iscase/*.csv > ${SUBDIR}/body/grepcase/found1.txt

