# Generation Fine-tuning
sh shell_script/run_gen_LLM.sh tiiuae/falcon-7b 1e-5 128 [dataset_path] 64 64 2 1 [output_dir]
sh shell_script/run_gen_LLM.sh tiiuae/falcon-7b-instruct 1e-5 128 [dataset_path] 64 64 2 1 [output_dir]
sh shell_script/run_gen_LLM.sh meta-llama/Llama-2-7b-hf 1e-5 128 [dataset_path] 64 64 2 1 [output_dir]
sh shell_script/run_gen_LLM.sh meta-llama/Llama-2-7b-chat-hf 1e-5 128 [dataset_path] 64 64 2 1 [output_dir]
sh shell_script/run_gen_LLM.sh meta-llama/Llama-2-13b-hf 1e-5 128 [dataset_path] 64 32 2 1 [output_dir]
sh shell_script/run_gen_LLM.sh meta-llama/Llama-2-13b-chat-hf 1e-5 128 [dataset_path] 64 32 2 1 [output_dir]
sh shell_script/run_gen_LLM.sh EleutherAI/gpt-j-6b 1e-5 128 [dataset_path] 64 64 2 1 [output_dir]

sh shell_script/run_gen_LLM.sh gpt2 1e-5 128 [dataset_path] 0 64 2 10 [output_dir]
sh shell_script/run_gen_LLM.sh gpt2-medium 1e-5 128 [dataset_path] 0 64 2 10 [output_dir]
sh shell_script/run_gen_LLM.sh gpt2-large 3e-5 128 [dataset_path] 0 64 2 5 [output_dir]
sh shell_script/run_gen_LLM.sh gpt2-xl 3e-5 128 [dataset_path] 0 64 2 5 [output_dir]

# Generation Inference
# we just provide an example since this is easy
# data relation are noun, verb, and event
sh shell_script/run_gen_LLM_inference.sh 10 1 [input_model] [dataset_path] [output_dir] noun
