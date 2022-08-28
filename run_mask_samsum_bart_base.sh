#!/bin/bash
exec > ./run_mask_info_samsum_bart_base.log 2>&1
#SBATCH --output=./run_sumarization_samsum_bart_base.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --exclude=hlt01,hlt02,ttnusa1,ttnusa4,ttnusa5,ttnusa6,ttnusa7
#SBATCH -w ttnusa3

echo "= = = = = = = = = = = = = ="
echo "The project is running..."
currentDate=`date`
echo $currentDate
echo "= = = = = = = = = = = = = ="


python train.py \
    --output_dir ./output/run_mask_info_samsum_bart_base \
    --train_file ./data/samsum/train.csv \
    --validation_file ./data/samsum/val.csv \
    --test_file ./data/samsum/test.csv \
    --text_column dialogue \
    --summary_column summary \
    --model_name_or_path facebook/bart-base \
    --model_type bart \
    --max_source_length 1024 \
    --min_target_length 1 \
    --max_target_length 128 \
    --learning_rate 3e-5 \
    --weight_decay 1e-3 \
    --label_smoothing 0.1 \
    --length_penalty 1.0 \
    --num_train_epochs 12 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --per_device_test_batch_size 1 \
    --num_warmup_steps 0 \
    --cache_dir ./output/cache \
    --overwrite_cache True \
    --seed 12345

