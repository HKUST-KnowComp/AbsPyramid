model=$1
lr=$2
batch_size=$3
dataset_path=$4
lora_rank=$5
split_batch_size=$6
gpu_num=$7
epoch=$8
accumulation_steps=`expr ${batch_size} / ${split_batch_size} / ${gpu_num}`
dir_path=$9
torchrun --nproc-per-node ${gpu_num} generator_main.py \
--model_name_or_path ${model} \
--output_dir ${dir_path} \
--do_train --do_eval --do_predict --per_device_train_batch_size ${split_batch_size} \
--learning_rate ${lr} --gradient_accumulation_steps ${accumulation_steps} \
--num_train_epochs ${epoch} --report_to none \
--dataset ${dataset_path} \
--evaluation_strategy no --save_strategy no \
--per_device_eval_batch_size 1 --eval_accumulation_steps 1 \
--lora_rank ${lora_rank} --max_source_length 30 --max_target_length 10 \
--run_name LLM_GEN_${model}_ep${epoch}_lr${lr}_bs${batch_size}_r${lora_rank} \
--torch_dtype bfloat16 --predict_with_generate