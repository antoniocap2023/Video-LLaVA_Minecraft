
module purge                     
module load cuda/11.7  
pip install transformers==4.31.0 --no-cache-dir


# (Optionally) load any additional modules you need
# module load cuda/11.7.1

#########################
#   Data / path setup
#########################
# Update these paths to match where you keep your data and Video-LLaVA repo:
JSON_FOLDER="/content/Video-LLaVA_Minecraft/scripts/v1_5/train_data_3.json"
VIDEO_FOLDER="content/drive/My Drive/train" 
TRAINING_SCRIPT="/content/Video-LLaVA_Minecraft/videollava/train/train_mem.py"
IMAGE_FOLDER="/content/Video-LLaVA_Minecraft/scripts/v1_5/empty_image_folder"
DEEPSPEED_CONFIG="/content/Video-LLaVA_Minecraft/scripts/zero2_offload.json"
OUTPUT_DIR="/content/Video-LLaVA_Minecraft/videollava-7b-minecraft_v2"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
  "$TRAINING_SCRIPT" \
  --lora_enable True --lora_r 4 --lora_alpha 16 --mm_projector_lr 2e-5 \
  --deepspeed "$DEEPSPEED_CONFIG" \
  \
  --model_name_or_path lmsys/vicuna-7b-v1.5 \
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
  --bf16 True \ 
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