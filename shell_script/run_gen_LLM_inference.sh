target_len=$1
gpu_num=$2
input_model=$3
dataset_path=$4
dir_path=$5
data_rel=$6
torchrun --nproc-per-node ${gpu_num} generator_main.py \
--model_name_or_path ${input_model} \
--output_dir ${dir_path} --load_from_pretrain False \
--do_eval --do_predict \
--report_to none \
--dataset ${dataset_path} --data_type ${data_rel} \
--evaluation_strategy no --save_strategy no \
--per_device_eval_batch_size 1 --eval_accumulation_steps 1 \
--max_source_length 30 --max_target_length ${target_len} \
--torch_dtype bfloat16 --predict_with_generate