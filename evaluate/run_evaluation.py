#!/usr/bin/env python
# coding: utf-8

import os
from tqdm.notebook import tqdm
from pathlib import Path
import pandas as pd
from nltk.lm import Vocabulary
import pickle
import sys
import torch
import importlib
import glob
from timeit import default_timer as t
import argparse
sys.path.append("../lib")
from metrics import levenshtein
import ocr_correction
from pytorch_decoding import seq2seq

parser = argparse.ArgumentParser(description = "Launch experiments")
parser.add_argument("--language", type = str)
parser.add_argument("--model_id", type = str)

args = parser.parse_args()
model_id = args.model_id
language = args.language

print(f"Evaluating {language}...")
data = pd.read_pickle(f"../data/models/{language}/data/test.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(f"../data/models/{language}/{model_id}.arch", "rb") as file:
    arch = pickle.load(file)
arch.pop("model")
arch.pop("parameters")
arch["in_vocabulary"] = list(arch["in_vocabulary"].keys())[:-3]
arch["out_vocabulary"] = list(arch["out_vocabulary"].values())[:-3]
model = seq2seq.Transformer(**arch)
model.to(device)
model.eval()
model.load_state_dict(torch.load(f"../data/models/{language}/{model_id}.pt", map_location=torch.device('cpu')))
with open(f"../data/models/{language}/data/vocabulary.pkl", "rb") as file:
    vocabulary = pickle.load(file)
len(vocabulary)
evaluation = ocr_correction.evaluate_model(raw = data.ocr_to_input, 
                                           gs = data.gs_aligned,
                                           model = model,
                                           vocabulary = vocabulary,
                                           window_size = 50,
                                           save_path = f"csv/evaluation_{language}.csv")
evaluation.to_csv(f"csv/evaluation_{language}.csv", index = False)
