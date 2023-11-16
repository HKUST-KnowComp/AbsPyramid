#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import evaluate
from generation_metric.unify_metrics_api import AutoScorer

from datasets import load_dataset, Dataset
import json

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from peft import LoraConfig, TaskType, PeftConfig, PeftModel
from peft import get_peft_model
from utils import is_main_process, init_logger, ds_init_output_dir, format_args
from tqdm import tqdm
from utils import store_generation
from collections import defaultdict


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    load_from_pretrain: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "whether load the model from pre-traind or fine-tuned models"
            )
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    max_source_length: Optional[int] = field(
        default=15,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=10,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    lora_rank: int = field(
        default=64, metadata={"help": "the LoRA rank"}
    )
    num_beams: int = field(
        default=1, metadata={"help": "beam search"}
    )
    length_awards: float = field(
        default=1, metadata={"help": "weight for length. higher for longer, lower for shorter."}
    )
    data_type: Optional[str] = field(
        default=None, metadata={"help": "type"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1
    if is_main_process(local_rank):
        ds_init_output_dir(training_args)

    with training_args.main_process_first(desc="getting logger"):
        log_level = logging.INFO
        logger = init_logger(training_args, log_level)
        logger.setLevel(log_level)
    logger.info(f"LOCAL RANK of current process: {local_rank}")

    # Log on each process the small summary:
    if is_main_process(local_rank):
        logger.info(
            f"Process rank: {local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(format_args(training_args))
        logger.info(format_args(data_args))
        logger.info(format_args(model_args))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if training_args.do_train is not None:
        data_files["train"] = os.path.join(data_args.dataset_name, "train.json")
    if training_args.do_eval is not None:
        data_files["valid"] = os.path.join(data_args.dataset_name, "valid.json")
    if training_args.do_predict is not None:
        data_files["test"] = os.path.join(data_args.dataset_name, "test.json")
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    train_dataset, valid_dataset, test_dataset = raw_datasets["train"], raw_datasets["valid"], raw_datasets["test"]

    model_args.torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    if "gpt2" in model_args.model_name_or_path:
        model_name_or_path = model_args.model_name_or_path
    else:
        if model_args.load_from_pretrain:
            model_name_or_path = model_args.model_name_or_path
        else:
            peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path)
            model_name_or_path = peft_config.base_model_name_or_path

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=model_args.cache_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=model_args.torch_dtype,
    )

    gen_kwargs = {
        "max_length": data_args.max_source_length +
                      data_args.max_target_length,
        "max_new_tokens": data_args.max_target_length,
        "min_new_tokens": 1,
        "num_beams": data_args.num_beams,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1,
        "length_penalty": data_args.length_awards,
        "pad_token_id": config.eos_token_id
    }
    if is_main_process(local_rank):
        logger.info(str(gen_kwargs))
    # "min_length": data_args.max_source_length + 1 This is wrong

    model.generation_config.update(**gen_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    # tokenizer.add_tokens(["<c>", "</c>"])
    model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        if is_main_process(local_rank):
            logger.info("There is not pad token. Use eos token instead.")
        tokenizer.pad_token, tokenizer.cls_token = tokenizer.eos_token, tokenizer.eos_token
        config.pad_token_id, config.cls_token_id = config.eos_token_id, config.eos_token_id

        tokenizer.sep_token, tokenizer.mask_token = tokenizer.eos_token, tokenizer.eos_token
        config.sep_token_id, config.mask_token_id = config.eos_token_id, config.eos_token_id
    tokenizer.padding_side = 'right'  # the padding side of tokenizer of Mistral 7b is left. Wierd.

    if "gpt2" in model_args.model_name_or_path:
        pass
    else:
        if model_args.load_from_pretrain:
            if ("falcon" in model_args.model_name_or_path or "Llama-2" in model_args.model_name_or_path or
                    "gpt-j" in model_args.model_name_or_path):
                kwargs = {}
            elif "Mistral" in model_args.model_name_or_path:
                kwargs = {"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
            else:
                raise ValueError("Model type not included.")
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                     r=data_args.lora_rank, lora_alpha=2 * data_args.lora_rank,
                                     lora_dropout=0.1, **kwargs)
            model = get_peft_model(model, peft_config)
        else:
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path, is_trainable=training_args.do_train)

    if "gpt2" in model_args.model_name_or_path:
        pass
    else:
        trainable_param, all_param = model.get_nb_trainable_parameters()
        if is_main_process(local_rank):
            logger.info(f"The model is loaded into {model.dtype}")
            param_info = f"trainable params: {trainable_param} || all params: " \
                         f"{all_param} || trainable%: {100 * trainable_param / all_param}"
            logger.info(param_info)
            logger.info("data size: train {}, valid {}, test {}".format(
                len(train_dataset), len(valid_dataset),
                len(test_dataset)))

    if training_args.do_train:
        column_names = list(train_dataset.features)
    else:
        column_names = list(valid_dataset.features)

    def tokenize_function(examples, is_eval=False):
        template_map = {"noun": 'In the sentence "{}," {} is an instance of',
                        "verb": 'In the sentence "{}," {} is an instance of',
                        "event": 'The sentence "{}" is an instance of'}
        input_list, full_list = [], []
        for i in range(len(examples['event'])):
            data_type = data_args.data_type if (data_args.data_type
                                                in {"noun", "verb", "event"}) else examples["type"][i]
            event = examples["event"][i]
            cur_template, concept = template_map[data_type], examples["concept"][i]
            # preprocess
            start_idx, end_idx = event.index("<"), event.index(">")
            instance = event[start_idx + 1: end_idx]
            event = event.replace("<", "").replace(">", "")
            # init template
            if data_type in {"noun", "verb"}:
                cur_example = cur_template.format(event, instance)
            elif data_type in {"event"}:
                cur_example = cur_template.format(instance)
            else:
                raise ValueError(f"Wrong type value ({data_type}) when preprocessing data")
            input_list.append(cur_example)
            if not is_eval:
                full_list.append(cur_example + " " + concept + tokenizer.pad_token)

        if is_eval:
            if training_args.per_device_eval_batch_size == 1:
                input_list = tokenizer.batch_encode_plus(input_list,
                                                         padding=False, max_length=data_args.max_source_length,
                                                         truncation=True)
            else:
                tokenizer.padding_side = 'left'
                input_list = tokenizer.batch_encode_plus(input_list, padding='max_length',
                                                         max_length=data_args.max_source_length,
                                                         truncation=True)
                tokenizer.padding_side = 'right'
            return input_list
        else:
            full_list = tokenizer.batch_encode_plus(full_list, padding='max_length',
                                                    max_length=data_args.max_source_length +
                                                               data_args.max_target_length,
                                                    return_tensors='pt', truncation=True)
            input_list = tokenizer.batch_encode_plus(input_list, padding='max_length',
                                                     max_length=data_args.max_source_length +
                                                                data_args.max_target_length,
                                                     return_tensors='pt', truncation=True)
            # for computing loss on tail,
            # reset label tokens from head and relation by -100, so GPT2LMHealModel will not compute loss on that
            labels = full_list['input_ids'] * (1 - input_list['attention_mask']) - 100 * input_list['attention_mask']
            # reset [PAD] token by -100, so GPT2LMHealModel will not compute loss on that
            # but not the first [PAD] token, which is [EOS]
            pad_token_mask = (labels == tokenizer.pad_token_id) * (1 - full_list["attention_mask"])
            pad_token_mask = pad_token_mask.bool()
            labels[pad_token_mask] = -100

            full_list["labels"] = labels.tolist()
            full_list["input_ids"] = full_list["input_ids"].tolist()
            full_list["attention_mask"] = full_list["attention_mask"].tolist()
            return full_list

    if training_args.do_train:
        with training_args.main_process_first(desc="dataset map tokenization"):
            train_dataset = train_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False,
                remove_columns=column_names,
                desc="Running tokenizer on dataset",
            )
        if is_main_process(local_rank):
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the train set: {train_dataset[index]}.")
                logger.info(tokenizer.convert_ids_to_tokens(train_dataset[index]["input_ids"]))

    if training_args.do_eval:
        with training_args.main_process_first(desc="dataset map tokenization"):
            valid_dataset = valid_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                fn_kwargs={"is_eval": True}
            )
        if is_main_process(local_rank):
            for index in random.sample(range(len(valid_dataset)), 3):
                logger.info(f"Sample {index} of the validation set: {valid_dataset[index]}.")
                logger.info(tokenizer.convert_ids_to_tokens(valid_dataset[index]["input_ids"]))

    if training_args.do_predict:
        with training_args.main_process_first(desc="dataset map tokenization"):
            test_dataset = test_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                fn_kwargs={"is_eval": True}
            )

    metric_set = {"bleu", "rouge", "meteor"}
    metric_kwargs = {"bleu": {"max_order": 4}, "rouge": {"use_stemmer": True}, "meteor": {}}
    auto_scorer = AutoScorer(metric_set, reload=False)
    print("finish metric loading")

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    # training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def compute_metrics(inputs, labels, preds):
        # Replace -100s used for padding as we can't decode them
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        full_decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_labels = labels

        # Some simple post-processing
        full_decoded_preds = [d.strip() for d in full_decoded_preds]
        decoded_labels = [[d.strip() for d in d_list] for d_list in decoded_labels]
        decoded_inputs = [d.strip() for d in decoded_inputs]
        # remove input
        decoded_preds = []
        assert len(full_decoded_preds) == len(decoded_labels)
        assert len(full_decoded_preds) == len(decoded_inputs)
        for cur_i, cur_p in zip(decoded_inputs, full_decoded_preds):
            decoded_preds.append(cur_p[len(cur_i):])
        decoded_preds = [[line.strip() for line in d.split("\n") if line.strip()] for d in decoded_preds]
        decoded_preds = [line[0] for line in decoded_preds]

        result = auto_scorer.compute(inputs=decoded_inputs, preds=decoded_preds,
                                     labels=decoded_labels, metric_kwargs=metric_kwargs)
        for key, value in result.items():
            if isinstance(value, dict):
                result[key] = json.dumps(value)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result, decoded_inputs, decoded_labels, full_decoded_preds, decoded_preds

    # evaluation
    if training_args.do_eval:
        logger.info("*** Validation ***")
        eval_results = trainer.predict(test_dataset=valid_dataset,
                                       metric_key_prefix="valid")
        pred_ids = eval_results.predictions
        input_ids, label_text = valid_dataset["input_ids"], valid_dataset["concept"]
        label_text = [l.split("[SEP]") for l in label_text]
        (metrics, decoded_inputs, decoded_labels,
         full_decoded_preds, decoded_preds) = compute_metrics(input_ids, label_text, pred_ids)
        metrics["valid_samples"] = len(valid_dataset)
        trainer.log_metrics("valid", metrics)
        trainer.save_metrics("valid", metrics)
        store_generation(training_args, [input_ids, pred_ids.tolist(), full_decoded_preds,
                                         decoded_inputs, decoded_labels, decoded_preds], split_name="valid")

    if training_args.do_predict:
        logger.info("*** Test ***")
        test_results = trainer.predict(test_dataset=test_dataset,
                                       metric_key_prefix="test")
        pred_ids = test_results.predictions
        input_ids, label_text = test_dataset["input_ids"], test_dataset["concept"]
        label_text = [l.split("[SEP]") for l in label_text]
        (metrics, decoded_inputs, decoded_labels,
         full_decoded_preds, decoded_preds) = compute_metrics(input_ids, label_text, pred_ids)
        metrics["test_samples"] = len(test_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        store_generation(training_args, [input_ids, pred_ids.tolist(), full_decoded_preds,
                                         decoded_inputs, decoded_labels, decoded_preds], split_name="test")

    # write finish file
    if is_main_process(local_rank):
        with open(os.path.join(training_args.output_dir, "checkpoint_finish"), "a") as fout:
            fout.write("training Finished\n")


if __name__ == "__main__":
    main()
