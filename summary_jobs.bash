#!/bin/bash

LIST=$(find results/ -name *.pth | sed 's/.*net_//g' | sed 's/\.pth//g')
# echo $LIST
echo "Found $(echo $LIST | sed 's/ /\n/g' | wc -l) entries..."

OUTFILE="summary.csv"
TESTCOLS="net*pth log*csv prd*csv all*csv out*log err*log opr*log epr*log"

# print the CSV header
echo "JOB_ID "$TESTCOLS > summary.csv

for kk in $LIST; do
    DIRPATH=$(bash scripts/id2path.bash $kk)
    echo "Processing [$kk] @ $DIRPATH"
    echo -n ${kk}" " >> $OUTFILE
    for col in $TESTCOLS; do
        FILENAME=$(echo ${col} | sed "s/\*/_${kk}\./g")
        if [[ -f ${DIRPATH}${FILENAME} ]]; then
            echo -n "YES " >> $OUTFILE   
        else
            echo -n "NOT " >> $OUTFILE
        fi
    done
    echo -e "" >> $OUTFILE
done
