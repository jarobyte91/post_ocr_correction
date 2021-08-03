from collections import Counter
import re
from math import exp
from metrics import levenshtein
import pandas as pd
from timeit import default_timer as t
from tqdm.notebook import tqdm

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

def correct_by_disjoint_window(string,
                               model, 
                               vocabulary,
                               window_size = 50,
                               decoding_method = "greedy_search", 
                               document_progress_bar = False, 
                               document_batch_progress_bar = 0, 
                               *arcorrect):
    model.eval()
    windows = [string[i:i+window_size] for i in range(0, len(string), window_size)]
    windows = ["".join([vocabulary.lookup(c) for c in s]).replace("<UNK>", " ") for s in windows]
    X = model.text2tensor(windows)
    predictions, probs = model.predict(X, 
                                       predictions = window_size, 
                                       method = decoding_method, 
                                       progress_bar = document_batch_progress_bar, 
                                       main_progress_bar = document_progress_bar, 
                                       *arcorrect)
    return re.sub(r"<START>|<END>|<PAD>", "", "".join(model.tensor2text(predictions)))

def correct_by_sliding_window(string,
                              model,
                              vocabulary,
                              window_size = 50, 
                              weighting = uniform,
                              decoding_method = "greedy_search", 
                              document_progress_bar = False, 
                              document_batch_progress_bar = 0,      
                              main_batch_size = 1024,
                              *arcorrect):
    model.eval()
    if len(string) <= window_size:
        windows = [string]
    else:
        windows = [string[i:i + window_size] for i in range(len(string) - window_size + 1)]
    windows = ["".join([vocabulary.lookup(c) for c in s]).replace("<UNK>", " ") for s in windows]
    X = model.text2tensor(windows)
    predictions, probs = model.predict(X, 
                                       predictions = window_size, 
                                       method = decoding_method, 
                                       main_progress_bar = document_progress_bar, 
                                       progress_bar = document_batch_progress_bar, 
                                       *arcorrect)
    output = [re.sub(r"<START>|<END>|<PAD>", "", s) for s in model.tensor2text(predictions)]
    votes = [{k:0.0 for k in vocabulary} for c in string]
    for i, s in enumerate(output):
        for j, (counter, char) in enumerate(zip(votes[i:i + window_size], s)):
            counter[char] += weighting(j, window_size)
    return votes, "".join([max(c.keys(), key = lambda x: c[x]) for c in votes])

def evaluate_model(raw, gs, model, vocabulary, save_path, window_size = 10, 
                   document_progress_bar = False):
    metrics = []
    old = levenshtein(reference = gs, hypothesis = raw).cer.mean()
    # disjoint
    print("disjoint window...")
    print("greedy_search...")
    start = t()
    corrections = [correct_by_disjoint_window(s, 
                                              model, 
                                              vocabulary, 
                                              document_progress_bar = document_progress_bar, 
                                              window_size = window_size) 
                   for s in raw]
    metrics.append({"window":"disjoint", 
                    "decoding":"greedy",
                    "window_size":window_size * 2,
                    "inference_seconds":t() - start,
                    "cer_before":old,
                    "cer_after":levenshtein(gs, corrections).cer.mean()})
    pd.DataFrame(metrics).assign(improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)).to_csv(save_path, index = False)
    start = t()
    corrections = [correct_by_disjoint_window(s, 
                                              model, 
                                              vocabulary, 
                                              document_progress_bar = document_progress_bar, 
                                              window_size = window_size * 2) 
                   for s in raw]
    metrics.append({"window":"disjoint", 
                    "decoding":"greedy",
                    "window_size":window_size,
                    "inference_seconds":t() - start,
                    "cer_before":old,
                    "cer_after":levenshtein(gs, corrections).cer.mean()})
    pd.DataFrame(metrics).assign(improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)).to_csv(save_path, index = False)
    
    print("beam_search...")
    start = t()
    corrections = [correct_by_disjoint_window(s, 
                                              model, 
                                              vocabulary, 
                                              decoding_method = "beam_search", 
                                              document_progress_bar = document_progress_bar, 
                                              window_size = window_size) 
                   for s in raw]
    metrics.append({"window":"disjoint", 
                    "decoding":"beam", 
                    "window_size":window_size * 2,
                    "inference_seconds":t() - start,
                    "cer_before":old,
                    "cer_after":levenshtein(gs, corrections).cer.mean()})
    pd.DataFrame(metrics).assign(improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)).to_csv(save_path, index = False)

    start = t()
    corrections = [correct_by_disjoint_window(s, 
                                              model, 
                                              vocabulary, 
                                              decoding_method = "beam_search", 
                                              document_progress_bar = document_progress_bar, 
                                              window_size = window_size * 2) 
                   for s in raw]
    metrics.append({"window":"disjoint", 
                    "decoding":"beam", 
                    "window_size":window_size,
                    "inference_seconds":t() - start,
                    "cer_before":old,
                    "cer_after":levenshtein(gs, corrections).cer.mean()})
    pd.DataFrame(metrics).assign(improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)).to_csv(save_path, index = False)

    # sliding
    print("sliding, greedy...")
    ## greedy search
    print("uniform...")
    start = t()
    corrections = [correct_by_sliding_window(s, model, vocabulary, 
                                             weighting = uniform,
                                             document_progress_bar = document_progress_bar,
                                             window_size = window_size)[1] 
                   for s in raw]
    metrics.append({"window":"sliding", 
                    "decoding":"greedy", 
                    "weighting":"uniform",
                    "window_size":window_size,
                    "inference_seconds":t() - start,
                    "cer_before":old,
                    "cer_after":levenshtein(gs, corrections).cer.mean()})
    pd.DataFrame(metrics).assign(improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)).to_csv(save_path, index = False)

    print("triangle...")
    start = t()
    corrections = [correct_by_sliding_window(s, model, vocabulary, 
                                             weighting = triangle, 
                                             document_progress_bar = document_progress_bar,
                                             window_size = window_size)[1] 
                   for s in raw]
    metrics.append({"window":"sliding", 
                    "decoding":"greedy", 
                    "weighting":"triangle",
                    "window_size":window_size,
                    "inference_seconds":t() - start,
                    "cer_before":old,
                    "cer_after":levenshtein(gs, corrections).cer.mean()})
    pd.DataFrame(metrics).assign(improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)).to_csv(save_path, index = False)

    print("bell...")
    start = t()
    corrections = [correct_by_sliding_window(s, model, vocabulary, weighting = bell, 
                                             document_progress_bar = document_progress_bar,
                                             window_size = window_size)[1] 
                   for s in raw]
    metrics.append({"window":"sliding", 
                    "decoding":"greedy", 
                    "weighting":"bell",
                    "window_size":window_size,
                    "inference_seconds":t() - start,
                    "cer_before":old,
                    "cer_after":levenshtein(gs, corrections).cer.mean()})
    pd.DataFrame(metrics).assign(improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)).to_csv(save_path, index = False)

    ## beam search
    print("sliding, beam...")
    print("uniform...")
    start = t()
    corrections = [correct_by_sliding_window(s, model, vocabulary, 
                                             decoding_method = "beam_search", 
                                             weighting = uniform, 
                                             document_progress_bar = document_progress_bar,
                                             window_size = window_size)[1] 
                   for s in raw]
    metrics.append({"window":"sliding", 
                    "decoding":"beam", 
                    "weighting":"uniform",
                    "window_size":window_size,
                    "inference_seconds":t() - start,
                    "cer_before":old,
                    "cer_after":levenshtein(gs, corrections).cer.mean()})
    pd.DataFrame(metrics).assign(improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)).to_csv(save_path, index = False)

    print("triangle...")
    start = t()
    corrections = [correct_by_sliding_window(s, model, vocabulary, 
                                             decoding_method = "beam_search", 
                                             weighting = triangle, 
                                             document_progress_bar = document_progress_bar,
                                             window_size = window_size)[1] 
                   for s in raw]
    metrics.append({"window":"sliding", 
                    "decoding":"beam", 
                    "weighting":"triangle",
                    "window_size":window_size,
                    "inference_seconds":t() - start,
                    "cer_before":old,
                    "cer_after":levenshtein(gs, corrections).cer.mean()})
    pd.DataFrame(metrics).assign(improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)).to_csv(save_path, index = False)

    print("bell...")
    start = t()
    corrections = [correct_by_sliding_window(s, model, vocabulary, 
                                             decoding_method = "beam_search", 
                                             weighting = bell, 
                                             document_progress_bar = document_progress_bar,
                                             window_size = window_size)[1] 
                   for s in raw]
    metrics.append({"window":"sliding", 
                    "decoding":"beam", 
                    "weighting":"bell",
                    "window_size":window_size,
                    "inference_seconds":t() - start,
                    "cer_before":old,
                    "cer_after":levenshtein(gs, corrections).cer.mean()})
    pd.DataFrame(metrics).assign(improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before)).to_csv(save_path, index = False)

    return pd.DataFrame(metrics).assign(improvement = lambda df: 100 * (1 - df.cer_after / df.cer_before))
