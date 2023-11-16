model=$1
lr=$2
batch_size=$3
dataset_path=$4
lora_rank=$5
epoch=$6
split_batch_size=$7
accumulation_steps=`expr ${batch_size} / ${split_batch_size}`
dir_path=$8
CUDA_VISIBLE_DEVICES=0 python LLM_LORA_FP16.py --model_name_or_path ${model} \
--output_dir ${dir_path} \
--do_train --do_eval --do_predict --per_device_train_batch_size ${split_batch_size} \
--learning_rate ${lr} --gradient_accumulation_steps ${accumulation_steps} \
--num_train_epochs ${epoch} --report_to wandb \
--dataset ${dataset_path} \
--save_strategy steps --save_steps 400 --evaluation_strategy steps --logging_steps 400 \
--load_best_model_at_end True --metric_for_best_model sum --save_total_limit 1 \
--per_device_eval_batch_size ${split_batch_size} --eval_accumulation_steps 1 --greater_is_better True \
--lora_rank ${lora_rank} --run_name LLM_LORA_${model}_ep${epoch}_lr${lr}_bs${batch_size}_r${lora_rank}
