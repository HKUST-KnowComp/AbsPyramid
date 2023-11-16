model=$1
dir_path=$2
data_path=$3
CUDA_VISIBLE_DEVICES=0 python NLI_ZERO_SHOT.py --model_name_or_path ${model} \
--output_dir ${dir_path} \
--do_eval --do_predict --report_to none \
--dataset ${data_path} \
--per_device_eval_batch_size 64 --eval_accumulation_steps 2
