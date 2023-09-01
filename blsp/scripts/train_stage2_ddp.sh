export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

export PATH=/usr/local/cuda/bin:$PATH


llama_path=$llama_model
whisper_path=$whisper_model

DATA_ROOT=$DATA_ROOT
SAVE_ROOT=$SAVE_ROOT

mkdir -p $SAVE_ROOT

python -m torch.distributed.run --nproc_per_node=8 blsp/train_stage2.py \
    --deepspeed blsp/config/dp_config_zero1.json \
    --data $DATA_ROOT \
    --output_dir ${SAVE_ROOT} \
    --manifest_files "*.jsonl" \
    --instruction "Continue the following text in a coherent and engaging style with less than 40 words." \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  True \
    \
    --learning_rate 5e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 1000 \
    \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 12 \
    --num_train_epochs 1 \
    \
    --llama_model $llama_path \
    --whisper_model $whisper_path \
    \
    --disable_tqdm True \
    \
    --logging_steps 20 \
    --save_steps 200 \
    --save_total_limit 1