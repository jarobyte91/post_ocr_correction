import pickle
import torch
from pytorch_beam_search import seq2seq
from post_ocr_correction import correction
import re
from pprint import pprint

# load vocabularies and model
with open("data/models/en/model_en.arch", "rb") as file:
    architecture = pickle.load(file)
source = list(architecture["in_vocabulary"].keys())
target = list(architecture["out_vocabulary"].values())
source_index = seq2seq.Index(source)
target_index = seq2seq.Index(target)

# remove keys from old API of pytorch_beam_search
for k in [
   "in_vocabulary",
   "out_vocabulary",
   "model",
   "parameters"
]:
    architecture.pop(k)
model = seq2seq.Transformer(source_index, target_index, **architecture)
state_dict = torch.load(
    "data/models/en/model_en.pt",
    map_location = torch.device("cpu") # comment this line if you have a GPU
)

# change names from old API of pytorch_beam_search
state_dict["source_embeddings.weight"] = state_dict.pop("in_embeddings.weight")
state_dict["target_embeddings.weight"] = state_dict.pop("out_embeddings.weight")
model.load_state_dict(state_dict)
model.eval()

# test data
test = "th1s 1s a c0rrupted str1ng"
reference = "this is a corrupted string"
new_source = [list(test)]
X_new = source_index.text2tensor(new_source)

# plain beam search
predictions, log_probabilities = seq2seq.beam_search(
    model, 
    X_new,
    progress_bar = 0)
just_beam = target_index.tensor2text(predictions[:, 0, :])[0]
just_beam = re.sub(r"<START>|<PAD>|<UNK>|<END>.*", "", just_beam)

# post ocr correction
disjoint_beam= correction.disjoint(
    test,
    model,
    source_index,
    target_index,
    5,
    "beam_search",
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
evaluation = correction.full_evaluation(
    [test],
    [reference],
    model,
    source_index,
    target_index,
)

print("results")
print("  reference                      ", reference)
print("  test data                      ", test)
print("  plain beam search              ", just_beam)
print("  disjoint windows, beam search  ", disjoint_beam)
print("  n-grams, beam search, triangle ", n_grams_beam)
print()
print(evaluation)
