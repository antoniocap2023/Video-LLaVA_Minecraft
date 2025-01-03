#!/bin/bash

#########################
#   Slurm directives
#########################
#SBATCH --job-name=finetune_minecraft     # Give your job a name
#SBATCH --account=mdatascience_team                  # Use the account from my_accounts
#SBATCH --partition=gpu
#SBATCH --time=12:00:00                   # Example: 12 hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # 8 CPU cores for data loading/preprocessing
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=32G                        
#SBATCH --output=finetune_mc.%j.out
#SBATCH --error=finetune_mc.%j.err

#########################
#   Environment setup
#########################
#module purge

# Activate your Conda environment
#########################
#   Environment setup
#########################
module purge                      # Clear all loaded modules
module load cuda/11.7
source /sw/pkgs/arc/python3.10-anaconda/2023.03/bin/activate
conda activate videollava         # Activate the correct Conda environment
pip install transformers==4.31.0 --no-cache-dir


# (Optionally) load any additional modules you need
# module load cuda/11.7.1

#########################
#   Data / path setup
#########################
# Update these paths to match where you keep your data and Video-LLaVA repo:
JSON_FOLDER="/home/acapdevi/Video-LLaVA/minecraft_data/train_data_4.json"
VIDEO_FOLDER="/home/acapdevi/Video-LLaVA/minecraft_data/smaller_videos" # This is smaller videos but all of the jsons point to outside of this
TRAINING_SCRIPT="/home/acapdevi/Video-LLaVA/videollava/train/train_mem.py"
IMAGE_FOLDER="/home/acapdevi/Video-LLaVA/minecraft_data/empty_image_folder"
DEEPSPEED_CONFIG="/home/acapdevi/Video-LLaVA/scripts/zero2_offload.json"
OUTPUT_DIR="/home/acapdevi/Video-LLaVA/videollava-7b-minecraft_v2"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
  "$TRAINING_SCRIPT" \
  --lora_enable True --lora_r 4 --lora_alpha 16 --mm_projector_lr 2e-5 \
  --deepspeed "$DEEPSPEED_CONFIG" \
  \
  --model_name_or_path /home/acapdevi/Video-LLaVA/fine_tuned_models/pre_trained_model \
  --version v1 \
  --data_path ${JSON_FOLDER} \
  --image_folder ${IMAGE_FOLDER} \
  --image_tower LanguageBind/LanguageBind_Image \
  --video_folder ${VIDEO_FOLDER} \
  --video_tower LanguageBind/LanguageBind_Video_merge \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --fp16 True \
  --output_dir ${OUTPUT_DIR} \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 50000 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --model_max_length 2048  --tokenizer_model_max_length 3072 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to tensorboard \
  --cache_dir "./cache_dir"

#########################
#   Wrap-up
#########################
echo "Finished at: $(date)"

# Potential Problems:
# CUDA out of memory