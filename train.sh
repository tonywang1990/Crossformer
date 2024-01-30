#!/bin/bash
## local
ROOT_PATH=/Users/tonywy/Desktop/Xode/crossformer
## runpod
ROOT_PATH=/workspace/futs

# DEBUG ONLY: Overfitting 1 day of data
python main_crossformer.py --data futs --root_path $ROOT_PATH  --train_path data/ZCE_CH_UR/daily_frame.20231211.parquet --val_path data/ZCE_CH_UR/daily_frame.20231211.parquet --in_len 1024 --out_len 4 --data_dim 24 --d_model 64 --d_ff 128 --batch_size 32 --train_epoch 20 --save_pred 