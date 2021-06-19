#!/bin/bash
echo $# arguments 


# check
if test "$#" -ne 3; then
    echo "Illegal number of parameters"
    echo "usage:"
    echo ">> bash scripts/diacritize.sh d3 dataset/tashkeela/test/test.txt dataset/tashkeela/preds/predicts.txt"
    exit 1
fi


cp $1 dataset/tashkeela/test/test.txt

# run python code
python predict_$1.py -c configs/config_$1_diacritize.yaml

cp dataset/tashkeela/preds/predictions_$1.txt $2

