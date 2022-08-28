#!/bin/bash
exec > ./run_sumarization_dialogsum_bart_base_augmented1k.log 2>&1
#SBATCH --output=./run_sumarization_dialogsum_bart_base.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --exclude=hlt01,hlt02,ttnusa1,ttnusa4,ttnusa6,ttnusa7,hlt06,ttnusa9,ttnusa10
#SBATCH -w ttnusa5

echo "= = = = = = = = = = = = = ="
echo "The project is running..."
currentDate=`date`
echo $currentDate
echo "= = = = = = = = = = = = = ="


python train.py \
    --output_dir ./output/run_sumarization_dialogsum_bart_base \
    --train_file ./data/dialogsum/dialogsum.augmented1k.jsonl \
    --validation_file ./data/dialogsum/dialogsum.dev.jsonl \
    --test_file ./data/dialogsum/dialogsum.test.jsonl \
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
    --num_train_epochs 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 4 \
    --per_device_test_batch_size 4 \
    --num_warmup_steps 0 \
    --cache_dir ./output/cache \
    --overwrite_cache True \
    --seed 12345

