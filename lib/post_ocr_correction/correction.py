from collections import Counter
import re
from math import exp
from .metrics import levenshtein
import pandas as pd
from timeit import default_timer as t
from tqdm import tqdm
from pytorch_beam_search import seq2seq
import torch

######################
# weighting functions
######################

def uniform(j, window_size):
    return 1.0

def triangle(j, window_size):
    m = window_size//2
    return m - 0.5 * abs(m - j)

def bell(j, window_size):
    m = window_size // 2
    s = window_size // 2
    return exp(-((m-j)/s)**2)

######################
# correction functions
######################

def disjoint(
    string,
    model, 
    source_index,
    target_index,
    window_size = 50,
    decoding_method = "greedy_search", 
    document_progress_bar = False, 
    document_batch_progress_bar = 0, 
    *arcorrect
):
    model.eval()
    windows = [string[i:i+window_size] 
        for i in range(0, len(string), window_size)]
    windows = ["".join([source_index.vocabulary.lookup(c) for c in s])\
        .replace("<UNK>", " ") for s in windows]
    X = source_index.text2tensor(windows, progress_bar = False)
    if decoding_method == "greedy_search":
        predictions, probs = seq2seq.greedy_search(
            model,
            X, 
            predictions = window_size, 
            progress_bar = document_batch_progress_bar, 
            *arcorrect
        )
    elif decoding_method == "beam_search":
        predictions, probs = seq2seq.beam_search(
            model,
            X, 
            predictions = window_size, 
            progress_bar = document_batch_progress_bar, 
            *arcorrect
        )   
        predictions = predictions[:, 0, :]
    output = target_index.tensor2text(predictions)
    output = [re.sub(r"<START>|<PAD>|<UNK>|<END>.*", "", s) for s in output]
    return "".join(output)

def n_grams(
    string,
    model,
    source_index,
    target_index,
    window_size = 50, 
    decoding_method = "greedy_search", 
    weighting = "uniform",
    document_progress_bar = False, 
    document_batch_progress_bar = 0,      
    main_batch_size = 1024,
    *arcorrect
):
    model.eval()
    if len(string) <= window_size:
        windows = [string]
    else:
        windows = [string[i:i + window_size] 
            for i in range(len(string) - window_size + 1)]
    windows = ["".join([source_index.vocabulary.lookup(c) for c in s])\
    .replace("<UNK>", " ") for s in windows]
    X = source_index.text2tensor(windows, progress_bar = False)
    if decoding_method == "greedy_search":
        predictions, probs = seq2seq.greedy_search(
            model,
            X, 
            predictions = window_size, 
            progress_bar = False, 
            *arcorrect
        )
    if decoding_method == "beam_search":
        predictions, probs = seq2seq.beam_search(
            model,
            X, 
            predictions = window_size, 
            progress_bar = document_batch_progress_bar, 
            *arcorrect
        )   
        predictions = predictions[:, 0, :]   
    output = target_index.tensor2text(predictions)
    output = [re.sub(r"<START>|<PAD>|<UNK>|<END>.*", "", s) for s in output]
    if weighting == "uniform":
        weighting = uniform
    elif weighting == "triangle":
        weighting = triangle
    elif weighting == "bell":
        weighting = bell
    votes = [
        {k:0.0 for k in target_index.vocabulary} 
        for c in string
    ]
    for i, s in enumerate(output):
        for j, (counter, char)\
        in enumerate(zip(votes[i:i + window_size], s)):
            counter[char] += weighting(j, window_size)
    output = [max(c.keys(), key = lambda x: c[x]) for c in votes]
    output = "".join(output)
    return votes, output

def full_evaluation(
    raw, 
    gs, 
    model, 
    source_index,
    target_index,
    save_path = None, 
    window_size = 10, 
    document_progress_bar = False
):
    print("evaluating all methods...")
    metrics = []
    old = levenshtein(reference = gs, hypothesis = raw).cer.mean()
    # disjoint
    print("  disjoint window...")
    print("    greedy_search...")
    start = t()
    corrections = [
        disjoint(
            s, 
            model, 
            source_index,
            target_index,
            document_progress_bar = document_progress_bar, 
            window_size = window_size
        ) for s in raw]
    metrics.append(
        {
            "window":"disjoint", 
            "decoding":"greedy",
            "window_size":window_size * 2,
            "weighting":pd.NA,
            "inference_seconds":t() - start,
            "cer_before":old,
            "cer_after":levenshtein(gs, corrections).cer.mean()
        }
    )
    if save_path:
        pd.DataFrame(metrics).assign(
            improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)
        ).to_csv(save_path, index = False)
    start = t()
    corrections = [
        disjoint(
            s, 
            model, 
            source_index,
            target_index,
            document_progress_bar = document_progress_bar, 
            window_size = window_size * 2
        ) 
        for s in raw
    ]
    metrics.append(
        {
            "window":"disjoint", 
            "decoding":"greedy",
            "window_size":window_size,
            "weighting":pd.NA,
            "inference_seconds":t() - start,
            "cer_before":old,
            "cer_after":levenshtein(gs, corrections).cer.mean()
        }
    )
    if save_path:
        pd.DataFrame(metrics).assign(
            improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)
        ).to_csv(save_path, index = False)
    print("    beam_search...")
    start = t()
    corrections = [
        disjoint(
            s, 
            model, 
            source_index,
            target_index,
            decoding_method = "beam_search", 
            document_progress_bar = document_progress_bar, 
            window_size = window_size
        ) 
        for s in raw
    ]
    metrics.append(
        {
            "window":"disjoint", 
            "decoding":"beam", 
            "window_size":window_size * 2,
            "weighting":pd.NA,
            "inference_seconds":t() - start,
            "cer_before":old,
            "cer_after":levenshtein(gs, corrections).cer.mean()
        }
    )
    if save_path:
        pd.DataFrame(metrics).assign(
            improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)
        ).to_csv(save_path, index = False)
    start = t()
    corrections = [
        disjoint(
            s, 
            model, 
            source_index,
            target_index,
            decoding_method = "beam_search", 
            document_progress_bar = document_progress_bar, 
            window_size = window_size * 2
        ) 
        for s in raw
    ]
    metrics.append(
        {
            "window":"disjoint", 
            "decoding":"beam", 
            "window_size":window_size,
            "weighting":pd.NA,
            "inference_seconds":t() - start,
            "cer_before":old,
            "cer_after":levenshtein(gs, corrections).cer.mean()
        }
    )
    if save_path:
        pd.DataFrame(metrics).assign(
            improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)
        ).to_csv(save_path, index = False)
    # sliding
    print("  sliding")
    print("    greedy...")
    ## greedy search
    print("      uniform...")
    start = t()
    corrections = [
        n_grams(
            s, 
            model, 
            source_index,
            target_index,
            weighting = uniform,
            document_progress_bar = document_progress_bar,
            window_size = window_size
        )[1] 
        for s in raw
    ]
    metrics.append(
        {
            "window":"sliding", 
            "decoding":"greedy", 
            "window_size":window_size,
            "weighting":"uniform",
            "inference_seconds":t() - start,
            "cer_before":old,
            "cer_after":levenshtein(gs, corrections).cer.mean()
        }
    )
    if save_path:
        pd.DataFrame(metrics).assign(
            improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)
        ).to_csv(save_path, index = False)

    print("      triangle...")
    start = t()
    corrections = [
        n_grams(
            s, 
            model, 
            source_index,
            target_index,
            weighting = triangle, 
            document_progress_bar = document_progress_bar,
            window_size = window_size
        )[1] 
        for s in raw
    ]
    metrics.append(
        {
            "window":"sliding", 
            "decoding":"greedy", 
            "window_size":window_size,
            "weighting":"triangle",
            "inference_seconds":t() - start,
            "cer_before":old,
            "cer_after":levenshtein(gs, corrections).cer.mean()
        }
    )
    if save_path:
        pd.DataFrame(metrics).assign(
            improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)
        ).to_csv(save_path, index = False)

    print("      bell...")
    start = t()
    corrections = [
        n_grams(
            s, 
            model, 
            source_index,
            target_index,
            weighting = bell, 
            document_progress_bar = document_progress_bar,
            window_size = window_size
        )[1] 
        for s in raw
    ]
    metrics.append(
        {
            "window":"sliding", 
            "decoding":"greedy", 
            "window_size":window_size,
            "weighting":"bell",
            "inference_seconds":t() - start,
            "cer_before":old,
            "cer_after":levenshtein(gs, corrections).cer.mean()
        }
    )
    if save_path:
        pd.DataFrame(metrics).assign(
            improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)
        ).to_csv(save_path, index = False)

    ## beam search
    print("    beam...")
    print("      uniform...")
    start = t()
    corrections = [
        n_grams(
            s, 
            model, 
            source_index,
            target_index,
            decoding_method = "beam_search", 
            weighting = uniform, 
            document_progress_bar = document_progress_bar,
            window_size = window_size
        )[1] 
        for s in raw
    ]
    metrics.append(
        {
            "window":"sliding", 
            "decoding":"beam", 
            "window_size":window_size,
            "weighting":"uniform",
            "inference_seconds":t() - start,
            "cer_before":old,
            "cer_after":levenshtein(gs, corrections).cer.mean()
        }
    )
    if save_path:
        pd.DataFrame(metrics).assign(
            improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)
        ).to_csv(save_path, index = False)

    print("      triangle...")
    start = t()
    corrections = [
        n_grams(
            s, 
            model, 
            source_index,
            target_index,
            decoding_method = "beam_search", 
            weighting = triangle, 
            document_progress_bar = document_progress_bar,
            window_size = window_size
        )[1] 
        for s in raw
    ]
    metrics.append(
        {
            "window":"sliding", 
            "decoding":"beam", 
            "window_size":window_size,
            "weighting":"triangle",
            "inference_seconds":t() - start,
            "cer_before":old,
            "cer_after":levenshtein(gs, corrections).cer.mean()
        }
    )
    if save_path:
        pd.DataFrame(metrics).assign(
            improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)
        ).to_csv(save_path, index = False)

    print("      bell...")
    start = t()
    corrections = [
        n_grams(
            s, 
            model, 
            source_index,
            target_index,
            decoding_method = "beam_search", 
            weighting = bell, 
            document_progress_bar = document_progress_bar,
            window_size = window_size
        )[1] 
        for s in raw
    ]
    metrics.append(
        {
            "window":"sliding", 
            "decoding":"beam", 
            "window_size":window_size,
            "weighting":"bell",
            "inference_seconds":t() - start,
            "cer_before":old,
            "cer_after":levenshtein(gs, corrections).cer.mean()
        }
    )
    if save_path:
        pd.DataFrame(metrics).assign(
            improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)
        ).to_csv(save_path, index = False)
    print()
    return pd.DataFrame(metrics).assign(
        improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)
    )
