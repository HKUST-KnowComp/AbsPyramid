import os
import evaluate
from collections import defaultdict


class AutoScorer:
    def __init__(self, metric_names, reload=True):
        self.rouge = None
        self.bleu = None
        self.meteor = None
        self.bertscore = None
        self.reload = reload

        self._load_metric(metric_names)

    def _load_metric(self, metric_names):
        metric_names = set(metric_names)
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.dir_path = dir_path
        if "rouge" in metric_names:
            self.rouge = evaluate.load(os.path.join(dir_path, "my_rouge.py"))
            metric_names.remove("rouge")
        if "bleu" in metric_names:
            self.bleu = evaluate.load(os.path.join(dir_path, "my_bleu.py"))
            metric_names.remove("bleu")
        if "meteor" in metric_names:
            self.meteor = evaluate.load(os.path.join(dir_path, "meteor.py"))
            metric_names.remove("meteor")
        if "bertscore" in metric_names:
            if self.reload:
                self.bertscore = "unloaded_metric"
            else:
                self.bertscore = evaluate.load(os.path.join(self.dir_path, "bertscore.py"))
            metric_names.remove("bertscore")
        assert len(metric_names) == 0, "there are not found metric names: {}".format(metric_names)

    def compute(self, inputs, preds, labels, metric_kwargs):
        result = {}

        preds_sentences = preds
        labels_sentences = labels

        if self.rouge is not None:
            result["rouge"] = self.rouge.compute(predictions=preds_sentences, references=labels_sentences, **metric_kwargs["rouge"])
        if self.bleu is not None:
            result["bleu"] = self.bleu.compute(predictions=preds_sentences, references=labels_sentences, **metric_kwargs["bleu"])
        if self.meteor is not None:
            result["meteor"] = self.meteor.compute(predictions=preds_sentences, references=labels_sentences, **metric_kwargs["meteor"])
        if self.bertscore is not None:
            if self.reload:
                self.bertscore = evaluate.load(os.path.join(self.dir_path, "bertscore.py"))
                result["bertscore"] = self.bertscore.compute(predictions=preds_sentences, references=labels_sentences, **metric_kwargs["bertscore"])
                self.bertscore = "unloaded_metric"
            else:
                result["bertscore"] = self.bertscore.compute(predictions=preds_sentences, references=labels_sentences,
                                                             **metric_kwargs["bertscore"])

        return result
