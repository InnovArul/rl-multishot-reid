#!/usr/bin/env bash

epochs=50
sets=0
gpus=$1
data_dir=$4
mode='SAVE'
case $2 in
  iLiDS-VID)
    base=ilds_$3_$sets
    num_id=150
    train_set=image_train
    set_index=$sets
    valid_set=image_valid
    ;;
  PRID-2011)
    base=prid_$3_$sets
    num_id=100
    train_set=image_train
    set_index=$sets
    valid_set=image_valid
    ;;
  MARS)
    base=mars_$3
    num_id=624
    train_set=image_train
    valid_set=image_valid
    set_index=''
    ;;
  *)
    echo "No valid dataset"
    exit
    ;;
esac

case $3 in
  alexnet)
    python baseline.py --gpus $gpus --data-dir $data_dir \
        --num-id $num_id \
        --train-file $train_set --test-file $valid_set --set-index $set_index\
        --lr 1e-4 --num-epoches $epochs --mode $mode --save-dir $base \
        --network alexnet --model-load-prefix alexnet --model-load-epoch 1
    ;;
  inception-bn)
     python baseline.py --gpus $gpus --data-dir $data_dir \
        --num-id $num_id \
        --train-file $train_set --test-file $valid_set --set-index $set_index\
        --lr 1e-2 --num-epoches $epochs --mode $mode --save-dir $base --lmnn # --lsoftmax 
    ;;
  *)
    echo "No valid basenet"
    exit
    ;;
esac
