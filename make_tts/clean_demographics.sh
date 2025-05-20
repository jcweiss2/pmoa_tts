#!/bin/bash
DATA_PATH=$1
OUT_DIR=$2
for k in $(seq -w 00 15); do
    find "${DATA_PATH}/PMC0${k}xxxxxx/anns/demographics/" -type f -name "*.gz" | while read file; do
        # echo "$file: " >> demogs250420.bsv; zcat $file | awk '/<\/think>/ {found=1; next} found && gsub(/\|/, "&") >= 2 {print} END {print ""}' >> demogs250420.bsv
        echo "$file: " >> "${OUT_DIR}/demogs250420.bsv"; zcat $file | awk '{
            if (!think_seen && /<\/think>/) { think_seen = 1; next }
            if (think_seen && gsub(/\|/, "&") >= 2) print
            else if (!think_seen) buffer[NR] = $0
        }
        END {
            if (!think_seen) {
                for (i = 1; i <= NR; i++) if (gsub(/\|/, "&", buffer[i]) >= 2) print buffer[i]
            }
        }' >> "${OUT_DIR}/demogs250420.bsv"
        
    done
done
sed ':a;N;$!ba;s/: \n/: /g' "${OUT_DIR}/demogs250420.bsv" > "${OUT_DIR}/demogs250420_clean.bsv"
grep "|" "${OUT_DIR}/demogs250420_clean.bsv" > "${OUT_DIR}/demogs250420_clean2.bsv"