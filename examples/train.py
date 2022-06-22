# -*- coding: utf-8 -*-
import sys
import pathlib

sys.path.append("../")
from dssm.model import DSSM
import pandas as pd

DATA_PATH: pathlib.Path = pathlib.Path(__file__).parent / "data" / "quora_duplicate_questions.tsv"

# df = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
df = pd.read_csv(DATA_PATH, sep="\t").sample(frac=0.1)
question1 = []
question2 = []
for i, row in df.iterrows():
    if row["is_duplicate"] == 0:
        continue
    question1.append(row["question1"])
    question2.append(row["question2"])

model = DSSM(
    "dssm-model",
    device="cpu",
    lang="en",
    vocab_ngram_range=(3, 3),
    vocab_analyzer="char_wb",
    vocab_binary=False,
)
model.fit(question1, question2, lr=0.0001)
