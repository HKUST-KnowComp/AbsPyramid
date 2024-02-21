import logging
import os
import sys
import datasets
import transformers
from dataclasses import asdict
import shutil
from torch import nn
import torch
from sklearn.metrics import precision_recall_curve
import numpy as np
import json
from tqdm import tqdm


def init_logger(training_args, log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # init a formatter to add date information
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # init a file handler and a stream handler
    fh = logging.FileHandler(os.path.join(training_args.output_dir, "train.log"), encoding="utf-8", mode="a")
    fh.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    # set formatter to handlers
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add those handlers to the root logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # the logger level of huggingface packages
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_propagation()

    return logger


def format_args(args):
    args_as_dict = asdict(args)
    args_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in args_as_dict.items()}
    attrs_as_str = [f"{k}={v}," for k, v in sorted(args_as_dict.items())]
    return f"{args.__class__.__name__}\n({' '.join(attrs_as_str)})"


def ds_init_output_dir(training_args):
    if training_args.do_train and os.path.exists(training_args.output_dir):
        if os.path.exists(os.path.join(training_args.output_dir, "checkpoint_finish")) > 0:
            raise ValueError(
                "training process in dir {} is finished, plz clear it manually.".format(training_args.output_dir))
        shutil.rmtree(training_args.output_dir, ignore_errors=True)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    os.system("touch {}".format(os.path.join(training_args.output_dir, "train.log")))


def revise_mnli_models(model_name_or_path, mnli_model, neutral_id, entail_id):
    if "bart" in model_name_or_path:
        head = mnli_model.classification_head
        linear = head.out_proj  # n x 3
    elif "roberta" in model_name_or_path:
        head = mnli_model.classifier
        linear = head.out_proj
    elif "deberta" in model_name_or_path:
        linear = mnli_model.classifier
    else:
        raise ValueError

    # copy weight and bias
    hidden_size = linear.weight.shape[-1]
    new_linear = nn.Linear(hidden_size, 2)  # n x 2
    with torch.no_grad():
        linear_weight = torch.stack([linear.weight[neutral_id, :], linear.weight[entail_id, :]], dim=0)
        linear_bias = torch.stack([linear.bias[neutral_id], linear.bias[entail_id]])
        new_linear.weight.data = linear_weight
        new_linear.bias.data = linear_bias

    if "bart" in model_name_or_path:
        mnli_model.classification_head.out_proj = new_linear
    elif "roberta" in model_name_or_path:
        mnli_model.classifier.out_proj = new_linear
    elif "deberta" in model_name_or_path:
        mnli_model.classifier = new_linear

    # change config
    mnli_model.config.num_labels = 2

    if hasattr(mnli_model, "num_labels"):
        mnli_model.num_labels = 2

    mnli_model.eval()

    return mnli_model

def is_main_process(local_rank):
    return local_rank == 0 or local_rank == -1


def average_precision_score(y_true, y_score, pos_label=1):
    precision, recall, _ = precision_recall_curve(
        y_true, y_score, pos_label=pos_label
    )
    # print(len(precision), precision)
    # print(len(recall), recall)
    recall_diff, precision = np.diff(recall), np.array(precision)[:-1]
    high_precision_mask = precision > 0.5
    recall_diff, precision = recall_diff[high_precision_mask], precision[high_precision_mask]
    # print(len(recall_diff), recall_diff)
    # print(len(precision), precision)
    return -np.sum(recall_diff * precision)


def store_generation(training_args, text_list, split_name):
    with open(os.path.join(training_args.output_dir, "{}.jsonl".format(split_name)), "w") as fout:
        for ri, rp, tp, i, l, p in tqdm(zip(*text_list), "output generations"):
            fout.write(json.dumps({"input": i, "label": l, "pred": p,
                                   "raw_input": ri, "raw_pred": rp, "text_pred": tp}) + "\n")
