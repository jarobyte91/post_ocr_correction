import pandas as pd
from Levenshtein import distance
import re
from tqdm.auto import tqdm
       
def levenshtein(reference, hypothesis, progress_bar = False):
    assert len(reference) == len(hypothesis)
    text = zip(reference, hypothesis)
    if progress_bar:
        text = tqdm(text, total = len(reference))
    d = [distance(r, h) for r, h in text]
    output = pd.DataFrame({"reference":reference, "hypothesis":hypothesis})\
    .assign(distance = lambda df: d)\
    .assign(
        cer = lambda df: df.apply(
            lambda r: 100 * r["distance"] / max(len(r["reference"]), 1), 
            axis = 1
        )
    )
    return output
