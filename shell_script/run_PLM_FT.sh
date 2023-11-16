model=$1
lr=$2
batch_size=$3
dataset_path=$4
epoch=$5
dir_path=$6
CUDA_VISIBLE_DEVICES=0 python PLM_FT_main.py --model_name_or_path ${model} \
--output_dir ${dir_path} \
--do_train --do_eval --do_predict --per_device_train_batch_size ${batch_size} --learning_rate ${lr} \
--num_train_epochs ${epoch} --report_to wandb \
--dataset ${dataset_path} \
--save_strategy steps --save_steps 400 --evaluation_strategy steps --logging_steps 400 \
--load_best_model_at_end True --metric_for_best_model sum --save_total_limit 1 \
--per_device_eval_batch_size ${batch_size} --eval_accumulation_steps 2 --greater_is_better True \
--run_name ${model}_ep${epoch}_lr${lr}_bs${batch_size}