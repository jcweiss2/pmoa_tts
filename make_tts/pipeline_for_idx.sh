#!/bin/bash
DATA_PATH=$1
NUM=$2
STOPAT=${3:-$NUM}
# Assume that we have the PMC000* folders with the .txt.gz files (that is from 'get_pmoa.sh')
DIRFIX="PMC0${NUM}xxxxxx"
DPD="${DATA_PATH}/${DIRFIX}"

# # 1. make body files
mkdir -p "${DATA_PATH}/${DIRFIX}/body/grepcase"
./get_pmoa_body.sh ${DPD}

# # 2. make found.txt from the body contents
./xtractor_grepper.sh ${DPD}

#---NOW THE GPU MACHINE---
# 3. use found.txt to determine if each file is a single case report (apply_llm_iscase_annotate_pmoa.py)
python apply_llm_iscase_annotate_pmoa.py --subdir ${DIRFIX} --datadir ${DATA_PATH}

# # 4. make a list of files (found1.txt) that the LLM identifies as having a single case
./get_iscase1.sh ${DPD}

# 5. make time series annotations for those cases in found1.txt
python apply_llm_annotate_ts_pmoa.py --subdir ${DIRFIX} --datadir ${DATA_PATH}

increment_mod_k() {
    local num=$1
    local k=12
    local next_num=$(( (10#$num + 1) % k ))  # Convert to decimal, increment, mod 50
    printf "%02d\n" "$next_num"  # Ensure two-digit formatting
}

NEWNUM=$(increment_mod_k $NUM)
if [[ "$NEWNUM" != "$STOPAT" ]]; then
    ./pipeline_for_idx.sh $NEWNUM $STOPAT
fi

bash