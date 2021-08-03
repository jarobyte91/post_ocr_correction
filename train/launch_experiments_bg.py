#!/usr/bin/env python
# coding: utf-8

import torch
import torch.utils.data as tud
import torch.nn as nn
import pickle
import datetime
import argparse
from random import randint as r
from random import choice
import sys
import pandas as pd
sys.path.append("/home/jarobyte/guemes/lib/")
from pytorch_decoding.seq2seq import Transformer
# from metrics import levenshtein
from timeit import default_timer as t
from ocr_correction import evaluate_model
    
parser = argparse.ArgumentParser(description = "Launch experiments")
parser.add_argument("--experiment_id", type = str)
parser.add_argument('--full', action = "store_true")
parser.add_argument('--random', action = "store_true")

main_folder = "/home/jarobyte/scratch/guemes/icdar/bg/"
output_folder = "baseline"

print(f"main folder:{main_folder}\noutput folder:{output_folder}\n")

device = torch.device("cuda")
args = parser.parse_args()
experiment_id = args.experiment_id

# fit parameters
    
learning_rate = 10**-4
batch_size = 100

if args.full:
    epochs = 42
    train_size = 1000000
    dev_size = 1000000
else:
    epochs = 10
    train_size = 1000
    dev_size = 100
    

# model hyperparameters

if args.random:
    encoder_layers = 4
    decoder_layers = encoder_layers
    attention_heads = 8
    embedding_dimension = 128
    feedforward_dimension = embedding_dimension * 4
    dropout = r(1, 5) / 10
    weight_decay = 10**-r(2, 4)
else:
    encoder_layers = 4
    decoder_layers = 4
    attention_heads = 8
    embedding_dimension = 512
    feedforward_dimension = 2048
    dropout = 0.1 

# loading data
    
input_vocabulary = pickle.load(open(main_folder + "data/char2i.pkl", "rb"))
train_source = torch.load(main_folder + "data/train_source.pt")[:train_size].to(device)
dev_source = torch.load(main_folder + "data/dev_source.pt")[:dev_size].to(device)

output_vocabulary = pickle.load(open(main_folder + "data/i2char.pkl", "rb"))
train_target = torch.load(main_folder + "data/train_target.pt")[:train_size].to(device)
dev_target = torch.load(main_folder + "data/dev_target.pt")[:dev_size].to(device)
    
# creating the model
    
net = Transformer(in_vocabulary = input_vocabulary, 
                  out_vocabulary = output_vocabulary, 
                  embedding_dimension = embedding_dimension,
                  encoder_layers = encoder_layers,
                  decoder_layers = decoder_layers,
                  attention_heads = attention_heads,
                  feedforward_dimension = feedforward_dimension,
                  dropout = dropout,
                  max_sequence_length = 110)

net.to(device)

# fitting the model

performance = net.fit(X_train = train_source,
                      Y_train = train_target,
                      X_dev = dev_source,
                      Y_dev = dev_target,
                      epochs = epochs,
                      batch_size = batch_size,
                      learning_rate = learning_rate, 
                      weight_decay = weight_decay, 
                      progress_bar = 0, 
                      save_path = f"{main_folder}{output_folder}/checkpoints/{experiment_id}.pt")

# saving the log and the model architecture

performance\
.assign(encoder_tokens = len(input_vocabulary), 
        decoder_tokens = len(output_vocabulary),
        experiment_id = experiment_id)\
.to_csv(f"{main_folder}{output_folder}/experiments/{experiment_id}.csv", index = False)

net.save_architecture(f"{main_folder}{output_folder}/models/{experiment_id}.arch")

# computing performance

print("\nEvaluating model..")
net.load_state_dict(torch.load(f"{main_folder}{output_folder}/checkpoints/{experiment_id}.pt"))
with open(main_folder + "data/vocabulary.pkl", "rb") as file:
    vocabulary = pickle.load(file)
dev = pd.read_pickle(main_folder + "data/dev.pkl")
if args.full:
    window_size = 50
else:
    window_size = 5
evaluation = evaluate_model(raw = dev.ocr_to_input,
                            gs = dev.gs_aligned,
                            model = net,
                            vocabulary = vocabulary,
                            window_size = window_size)
evaluation = evaluation.assign(experiment_id = experiment_id)[["experiment_id", "improvement"] + list(evaluation.columns)[:-1]]
print(evaluation)
evaluation.to_csv(f"{main_folder}{output_folder}/evaluation/{experiment_id}.csv", index = False)