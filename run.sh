#!/bin/bash
## local
#ROOT_PATH=/Users/tonywy/Desktop/Xode/crossformer
## runpod
ROOT_PATH=/workspace/futs

TRAIN=data/ZCE_CH_UR/train/daily_frame.*.parquet
VAL=data/ZCE_CH_UR/val/daily_frame.*.parquet
TEST=data/ZCE_CH_UR/test/daily_frame.*.parquet

# DEBUG ONLY: Overfitting 1 day of data
#TRAIN=data/ZCE_CH_UR/train/daily_frame.20231205.parquet
#VAL=data/ZCE_CH_UR/train/daily_frame.20231205.parquet

# Train
#python main_crossformer.py --data futs --root_path $ROOT_PATH --train_path $TRAIN --val_path $VAL --in_len 4096 --out_len 1 --seg_len 64 --input_dim 24 --output_dim 1 --num_workers 8 --batch_size 128 --train_epoch 20 --learning_rate 1e-3 --lradj none --mode train 

# Eval
python main_crossformer.py --data futs --root_path $ROOT_PATH --train_path $TRAIN --val_path $VAL --test_path $TEST --in_len 4096 --out_len 1 --seg_len 64 --input_dim 24 --output_dim 1 --num_workers 8 --batch_size 256 --train_epoch 20 --learning_rate 1e-3 --lradj none --mode test --save_pred --load_model futs_01_31_21_42_56_il4096_ol1_sl64_win2_fa10_dm256_nh4_el3_itr0/checkpoint.pth


echo "Idling... Counting down 60 seconds before stopping the pod:"
read -t 60 userInput

if [ -n "$userInput" ]; then
    echo "Exiting stopped"
else
    echo "Stopping pod..."
    # stop the pod when the job is finshed
    runpodctl stop pod $RUNPOD_POD_ID
fi