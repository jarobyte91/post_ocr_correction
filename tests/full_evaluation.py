from pprint import pprint
from pytorch_beam_search import seq2seq
from post_ocr_correction import correction
import re
import os

print("train data")
source = [list("abcdefghijkl"), list("mnopqrstwxyz")]
target = [list("abcdefghijk"), list("mnopqrstwxy")]
# target = [list("ABCDEFGHIJKL"), list("MNOPQRSTWXYZ")]
print("source")
pprint(source)
print("target")
pprint(target)
print("creating indexes...")
source_index = seq2seq.Index(source, progress_bar = False)
target_index = seq2seq.Index(target, progress_bar = False)
print("creating tensors...")
X = source_index.text2tensor(source, progress_bar = False)
Y = target_index.text2tensor(target, progress_bar = False)
print("creating model...")
model = seq2seq.Transformer(source_index, target_index)
print("training model...")
model.train()
model.fit(X, Y, epochs = 100, progress_bar = 0)
model.eval()
print("\ntest data")
test = "ghijklmnopqrst"
new_source = [list(test)]
print("new source")
pprint(new_source)
print("creating tensor...")
X_new = source_index.text2tensor(new_source, progress_bar = False)

print("\nplain beam search...")
predictions, log_probabilities = seq2seq.beam_search(
    model, 
    X_new,
    progress_bar = 0)
# print(predictions.shape)
just_beam = target_index.tensor2text(predictions[:, 0, :])[0]
just_beam = re.sub(r"<START>|<PAD>|<END>.*", "", just_beam)

print("\ncorrecting by disjoint windows:")
disjoint_greedy = correction.disjoint(
    test,
    model,
    source_index,
    target_index,
    5,
    "greedy_search",
)
disjoint_beam = correction.disjoint(
    test,
    model,
    source_index,
    target_index,
    5,
    "beam_search",
)
print("correcting by n-grams:")
_, n_grams_greedy = correction.n_grams(
    test,
    model,
    source_index,
    target_index,
    5,
    "greedy_search",
)
_, n_grams_beam = correction.n_grams(
    test,
    model,
    source_index,
    target_index,
    5,
    "beam_search",
    "triangle"
)

print("\nevaluating all correction methods...")
evaluation = correction.full_evaluation(
    [test],
    [test[:-1]],
    model,
    source_index,
    target_index,
)

print("\n--results--")
print("test data:", test)
print("plain beam search:", just_beam)
print("disjoint")
print("  greedy search:", disjoint_greedy)
print("  beam search:", disjoint_beam)
print("n_grams")
print("  greedy search:", n_grams_greedy)
print("  beam search:", n_grams_beam)

print("\n", evaluation)

