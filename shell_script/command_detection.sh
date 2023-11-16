## PLM + Fine-tuning
sh shell_script/run_PLM_FT.sh bert-base-uncased 1e-5 128 [dataset_path] 5 [output_dir]
sh shell_script/run_PLM_FT.sh bert-large-uncased 1e-5 128 [dataset_path] 5 [output_dir]
sh shell_script/run_PLM_FT.sh roberta-base 1e-5 128 [dataset_path] 5 [output_dir]
sh shell_script/run_PLM_FT.sh roberta-large 5e-5 128 [dataset_path] 5 [output_dir]
sh shell_script/run_PLM_FT.sh microsoft/deberta-base 1e-5 128 [dataset_path] 5 [output_dir]
sh shell_script/run_PLM_FT.sh microsoft/deberta-large 1e-5 128 [dataset_path] 5 [output_dir]

## LLM + LoRA
sh shell_script/run_LLM_LORA.sh tiiuae/falcon-7b 5e-5 128 [dataset_path] 64 5 64 [output_dir]
sh shell_script/run_LLM_LORA.sh tiiuae/falcon-7b-instruct 5e-5 128 [dataset_path] 64 5 64 [output_dir]
sh shell_script/run_LLM_LORA.sh mistralai/Mistral-7B-v0.1 5e-5 128 [dataset_path] 64 5 64 [output_dir]
sh shell_script/run_LLM_LORA.sh mistralai/Mistral-7B-Instruct-v0.1 5e-5 128 [dataset_path] 64 5 64 [output_dir]
sh shell_script/run_LLM_LORA.sh meta-llama/Llama-2-7b-hf 5e-5 128 [dataset_path] 64 5 64 [output_dir]
sh shell_script/run_LLM_LORA.sh meta-llama/Llama-2-7b-chat-hf 5e-5 128 [dataset_path] 64 5 64 [output_dir]
sh shell_script/run_LLM_LORA.sh meta-llama/Llama-2-13b-hf 5e-5 128 [dataset_path] 64 5 32 [output_dir]
sh shell_script/run_LLM_LORA.sh meta-llama/Llama-2-13b-chat-hf 5e-5 128 [dataset_path] 64 5 32 [output_dir]

## NLI + Zero Shot
sh shell_script/run_NLI_ZERO.sh facebook/bart-large-mnli [output_dir] [dataset_path]
sh shell_script/run_NLI_ZERO.sh roberta-large-mnli [output_dir] [dataset_path]
sh shell_script/run_NLI_ZERO.sh microsoft/deberta-base-mnli [output_dir] [dataset_path]
sh shell_script/run_NLI_ZERO.sh microsoft/deberta-large-mnli [output_dir] [dataset_path]

## NLI + Fine-tuning
sh shell_script/run_NLI_FT.sh facebook/bart-large-mnli 5e-5 128 [dataset_path] 5 [output_dir]
sh shell_script/run_NLI_FT.sh roberta-large-mnli 5e-5 128 [dataset_path] 5 [output_dir]
sh shell_script/run_NLI_FT.sh microsoft/deberta-base-mnli 1e-5 128 [dataset_path] 5 [output_dir]
sh shell_script/run_NLI_FT.sh microsoft/deberta-large-mnli 5e-5 128 [dataset_path] 5 [output_dir]

