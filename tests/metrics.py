print("testing metrics module...")
from post_ocr_correction import metrics
from pprint import pprint

print("reference")
reference = ["abc", "def"]
pprint(reference)
print("hypothesis")
hypothesis = ["abc", "d"]
pprint(hypothesis)
print("computing metrics...")
distance = metrics.levenshtein(reference, hypothesis, progress_bar = True)
print("metrics")
print(distance)

