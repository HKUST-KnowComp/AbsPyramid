import json
from tqdm import tqdm
from collections import defaultdict

inputs, preds, labels = [], [], []
data_path = "/home/data/zwanggy/event_outputs/test_iterative_t5_finetune/generation/full_script_test.json"
with open(data_path) as fin:
    for line in tqdm(fin, "loading data"):
        line = json.loads(line)
        question, pred, label = line["input"], line["pred"], line["label"]
        inputs.append(question)
        preds.append(pred)
        labels.append(label)

input2labels = defaultdict(set)
# group label here
for i, l in tqdm(zip(inputs, labels), "processing data"):
    input2labels[i].add(tuple(l))

labels = []
for i in tqdm(inputs, "reordering"):
    labels.append([list(ref) for ref in input2labels[i]])

# # test ebertscore
# from ebertscore import EBERTScore
# scorer = EBERTScore()
# scorer.compute(predictions=preds, references=labels, lang="en")


# test ebertprf
# from ebertprf import EBERTPRF
# scorer = EBERTPRF()
# scorer.compute(predictions=preds, references=labels, lang="en")

# test meteor
# from meteor import Meteor
# scorer = Meteor()
# score = scorer.compute(predictions=[" ".join(p) for p in preds], references=[[" ".join(r) for r in r_list] for r_list in labels])
# print(score)

# test bertscore
from bertscore import BERTScore
scorer = BERTScore()
score = scorer.compute(batch_size=128,
                       predictions=[" ".join(p) for p in preds], references=[[" ".join(r) for r in r_list] for r_list in labels])
print(score)

# test bleurt
# from my_bleurt import BLEURT
# scorer = BLEURT()
# scorer.download_and_prepare()
# score = scorer.compute(predictions=[" ".join(p) for p in preds], references=[[" ".join(r) for r in r_list] for r_list in labels])
# print(score)

# test erouge
# from erouge import ERouge
# scorer = ERouge()
# scorer.download_and_prepare()
# score = scorer.compute(predictions=preds, references=labels)
# print(score)

# test EBleu
# from ebleu import EBleu
# scorer = EBleu()
# scorer.download_and_prepare()
# score = scorer.compute(predictions=preds, references=labels)
# print(score)

# test EMeteor
# from emetoer import EMeteor
# scorer = EMeteor()
# scorer.download_and_prepare()
# score = scorer.compute(predictions=preds, references=labels)
# print(score)

# test EBleurt
# from ebleurt import EBLEURT
# scorer = EBLEURT()
# scorer.download_and_prepare()
# score = scorer.compute(predictions=preds, references=labels)
# print(score)
