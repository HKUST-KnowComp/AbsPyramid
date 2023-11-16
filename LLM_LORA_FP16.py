import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset
from sklearn import metrics as skmetrics
from utils import average_precision_score

# Before run: install ruamel_yaml==0.11.14, transformers==4.11.0, datasets; uninstall apex

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType
from peft import PeftModel, PeftConfig
from peft import get_peft_model
import torch

from utils import ds_init_output_dir, init_logger, format_args
from utils import is_main_process


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.11.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=48,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_proportion: Optional[float] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    dataset: str = field(
        default=None
    )
    do_final_evaluations: Optional[bool] = field(
        default=False, metadata={"help": "Whether do evaluations after training."}
    )
    lora_rank: Optional[int] = field(
        default=64, metadata={"help": "the lora rank"}
    )
    type_template: Optional[str] = field(
        default=None, metadata={"help": "the prompt template for add a type"}
    )
    type: Optional[str] = field(
        default=None, metadata={"help": "type info"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    load_from_pretrain: bool = field(
        default=True,
        metadata={"help": "Whether to use pre-trained model or trained PEFT model"}
    )
    abs_samples: int = field(
        default=4, metadata={"help": "Number of abstractions used in ConceptMax for training."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this local_script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1
    # init folder
    if is_main_process(local_rank):
        ds_init_output_dir(training_args)

    # Setup logging
    with training_args.main_process_first(desc="getting logger"):
        log_level = logging.INFO
        logger = init_logger(training_args, log_level)
        logger.setLevel(log_level)

    # reset training_args
    if data_args.max_train_proportion == 0:
        training_args.do_train = False
        if is_main_process(local_rank):
            logger.info("Since the training proportion is zero. Argument \"do_train\" is set to False.")

    # Log on each process the small summary:`
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

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    data_files = {}
    if training_args.do_train is not None:
        data_files["train"] = os.path.join(data_args.dataset, "train.json")
    if training_args.do_eval is not None:
        data_files["validation"] = os.path.join(data_args.dataset, "valid.json")
    if training_args.do_predict is not None:
        data_files["test"] = os.path.join(data_args.dataset, "test.json")
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    train_dataset, eval_dataset, predict_dataset = raw_datasets["train"], raw_datasets["validation"], raw_datasets[
        "test"]

    # Labels
    num_labels = 2

    # init/load your base models
    if model_args.load_from_pretrain:
        model_name_or_path = model_args.model_name_or_path
    else:
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        model_name_or_path = peft_config.base_model_name_or_path
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        do_lower_case=model_args.do_lower_case,
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

    # init/load your peft model
    if model_args.load_from_pretrain:
        if ("falcon" in model_args.model_name_or_path or "Llama-2" in model_args.model_name_or_path or
                "gpt" in model_args.model_name_or_path):
            kwargs = {}
        elif "Mistral" in model_args.model_name_or_path:
            kwargs = {"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
        else:
            raise ValueError("Model type not included.")
        modules_to_save = ["score"]
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False,
                                 r=data_args.lora_rank, lora_alpha=2 * data_args.lora_rank,
                                 lora_dropout=0.1, modules_to_save=modules_to_save, **kwargs)
        if is_main_process(local_rank):
            logger.info(f'Peft will save additional modules: {modules_to_save}')
        model = get_peft_model(model, peft_config)
    else:
        print(model.score.weight)
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path, is_trainable=training_args.do_train)
        print(model.base_model.model.score.modules_to_save.default.weight)
        # if not training, the linear layer is loaded as trainable
        if not training_args.do_train:
            for name, param in model.named_parameters():
                if 'score' in name:
                    param.requires_grad = False

    trainable_param, all_param = model.get_nb_trainable_parameters()
    if is_main_process(local_rank):
        logger.info(f"The model is loaded into {model.dtype}")
        param_info = f"trainable params: {trainable_param} || all params: " \
                     f"{all_param} || trainable%: {100 * trainable_param / all_param}"
        logger.info(param_info)
        logger.info("data size: train {}, valid {}, test {}".format(
            len(train_dataset), len(eval_dataset), len(predict_dataset)))

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    def preprocess_function(examples):
        # Tokenize the texts
        for i in range(len(examples['event'])):
            examples['event'][i] = examples['event'][i].replace('<', '[').replace(">", "]")
            # examples['event'][i] = examples['event'][i].replace('[', '<c>').replace(']', '</c>')
            if data_args.type_template is not None:
                data_type = examples['type'][i] if "type" in column_names else data_args.type
                examples['concept'][i] = data_args.type_template.format(
                    examples['concept'][i], data_type)

        return tokenizer(
            examples["event"],
            examples["concept"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

    if training_args.do_train:
        if data_args.max_train_proportion is not None:
            data_args.max_train_proportion = int(len(train_dataset) * data_args.max_train_proportion)
            selected_train_idx = random.sample(range(len(train_dataset)), data_args.max_train_proportion)
            train_dataset = train_dataset.select(selected_train_idx)
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        if is_main_process(local_rank):
            logger.info(f"Few-shot experiments with train data size: {len(train_dataset)}")
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
                logger.info(tokenizer.convert_ids_to_tokens(train_dataset[index]["input_ids"]))

    if training_args.do_eval:
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on validation dataset",
            )
        if is_main_process(local_rank):
            for index in random.sample(range(len(eval_dataset)), 3):
                logger.info(f"Sample {index} of the eval set: {eval_dataset[index]}.")
                logger.info(tokenizer.convert_ids_to_tokens(eval_dataset[index]["input_ids"]))

    if training_args.do_predict:
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on prediction dataset",
            )

    # Get the metric function
    metric_fns = [('accuracy', skmetrics.accuracy_score), ('auc', skmetrics.roc_auc_score), ('f1', skmetrics.f1_score),
                  ('precision', skmetrics.precision_score), ('recall', skmetrics.recall_score),
                  ('ma-f1', skmetrics.f1_score), ('aps', average_precision_score)]

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        probs = torch.softmax(torch.tensor(preds), dim=-1)[:, 1]
        preds = np.argmax(preds, axis=1)
        labels = p.label_ids
        results = {}
        for name, fn in metric_fns:
            if name in {'auc', 'aps'}:
                results[name] = fn(labels, probs)
            elif name == 'ma-f1':
                results[name] = fn(labels, preds, average="macro")
            else:
                results[name] = fn(labels, preds)
        results["sum"] = results["ma-f1"] + results["auc"]
        return results  # macro-f1 + auc

    data_collator = DataCollatorWithPadding(tokenizer,
                                            'max_length' if data_args.pad_to_max_length else 'longest',
                                            pad_to_multiple_of=8)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
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

    # evaluation
    if training_args.do_eval:
        logger.info("*** Validation ***")
        eval_results = trainer.predict(test_dataset=eval_dataset, metric_key_prefix="valid")
        metrics, label_ids, pred_prob = eval_results.metrics, eval_results.label_ids, eval_results.predictions
        pred_prob = pred_prob[0] if isinstance(pred_prob, tuple) else pred_prob
        metrics["valid_samples"] = len(eval_dataset)

        trainer.log_metrics("valid", metrics)
        trainer.save_metrics("valid", metrics)

        range_idx = np.arange(len(eval_dataset)).reshape(-1, 1)
        pred_label = np.argmax(pred_prob, axis=-1).reshape(-1, 1)
        pred_prob = np.concatenate([range_idx, pred_prob, label_ids.reshape(-1, 1), pred_label], axis=-1).round(3)
        np.savetxt(os.path.join(training_args.output_dir, "valid_label.txt"), pred_prob, fmt='%.3f')

    # Test
    if training_args.do_predict:
        logger.info("*** Test ***")
        eval_results = trainer.predict(test_dataset=predict_dataset, metric_key_prefix="test")
        metrics, label_ids, pred_prob = eval_results.metrics, eval_results.label_ids, eval_results.predictions
        pred_prob = pred_prob[0] if isinstance(pred_prob, tuple) else pred_prob
        metrics["test_samples"] = len(predict_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        range_idx = np.arange(len(predict_dataset)).reshape(-1, 1)
        pred_label = np.argmax(pred_prob, axis=-1).reshape(-1, 1)
        pred_prob = np.concatenate([range_idx, pred_prob, label_ids.reshape(-1, 1), pred_label], axis=-1)
        np.savetxt(os.path.join(training_args.output_dir, "test_label.txt"), pred_prob, fmt='%.3f')

    # write finish file
    if is_main_process(local_rank):
        with open(os.path.join(training_args.output_dir, "checkpoint_finish"), "a") as fout:
            logger.info("finished")
            fout.write("training Finished\n")


if __name__ == "__main__":
    main()
