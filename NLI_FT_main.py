import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset
from sklearn import metrics as skmetrics

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
import torch

from utils import ds_init_output_dir, init_logger, format_args, revise_mnli_models


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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )

    dataset: str = field(
        default=None
    )

    do_final_evaluations: Optional[bool] = field(
        default=False, metadata={"help": "Whether do evaluations after training."}
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
    abs_samples: int = field(
        default=4, metadata={"help": "Number of abstractions used in ConceptMax for training."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this local_script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # init folder
    ds_init_output_dir(training_args)

    # Setup logging
    log_level = logging.INFO
    logger = init_logger(training_args, log_level)
    logger.setLevel(log_level)

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
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
    train_dataset, eval_dataset, predict_dataset = raw_datasets["train"], raw_datasets["validation"], raw_datasets["test"]

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=3,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )

    if "NEUTRAL" in model.config.label2id:
        neutral_id = model.config.label2id["NEUTRAL"]
    else:
        neutral_id = model.config.label2id["neutral"]

    if "ENTAILMENT" in model.config.label2id:
        entail_id = model.config.label2id["ENTAILMENT"]
    else:
        entail_id = model.config.label2id["entailment"]

    model = revise_mnli_models(model_args.model_name_or_path, model, neutral_id, entail_id)

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    tokenizer.add_tokens(["<c>", "</c>"])
    model.resize_token_embeddings(len(tokenizer))

    def preprocess_function(examples):
        # Tokenize the texts
        for i in range(len(examples["event"])):
            event, concept = examples["event"][i], examples["concept"][i]
            start_idx = event.index("<")
            end_idx = event.index(">")
            sent_concept = event[:start_idx] + concept + event[end_idx + 1:]
            examples["concept"][i] = sent_concept
            examples["event"][i] = event.replace("<", "").replace(">", "")
            # I eat apple -> I eat fruit.
            # [cls] premise [sep] hypothesis [sep]

        return tokenizer(
            examples["event"],
            examples["concept"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True)

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            logger.info(tokenizer.convert_ids_to_tokens(train_dataset[index]["input_ids"]))

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")
            logger.info(tokenizer.convert_ids_to_tokens(eval_dataset[index]["input_ids"]))

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )

    # Get the metric function
    metric_fns = [('accuracy', skmetrics.accuracy_score), ('auc', skmetrics.roc_auc_score),
                  ('f1', skmetrics.f1_score), ('precision', skmetrics.precision_score),
                  ('recall', skmetrics.recall_score), ('ma-f1', skmetrics.f1_score)]

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        probs = torch.softmax(torch.tensor(preds), dim=-1)[:, 1]
        preds = np.argmax(preds, axis=1)
        labels = p.label_ids
        results = {}
        for name, fn in metric_fns:
            if name == 'auc':
                results[name] = fn(labels, probs)
            elif name == 'ma-f1':
                results[name] = fn(labels, preds, average="macro")
            else:
                results[name] = fn(labels, preds)
        results["sum"] = results["ma-f1"] + results["auc"]
        return results

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
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

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
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["valid_samples"] = min(max_eval_samples, len(eval_dataset))

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
        max_test_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        metrics["test_samples"] = min(max_test_samples, len(predict_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        range_idx = np.arange(len(predict_dataset)).reshape(-1, 1)
        pred_label = np.argmax(pred_prob, axis=-1).reshape(-1, 1)
        pred_prob = np.concatenate([range_idx, pred_prob, label_ids.reshape(-1, 1), pred_label], axis=-1)
        np.savetxt(os.path.join(training_args.output_dir, "test_label.txt"), pred_prob, fmt='%.3f')

    # write finish file
    with open(os.path.join(training_args.output_dir, "checkpoint_finish"), "a") as fout:
        fout.write("training Finished\n")


if __name__ == "__main__":
    main()
