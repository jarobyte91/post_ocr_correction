from pytorch_beam_search import seq2seq
from post_ocr_correction import correction
import re

# training data
source = [list("abcdefghijkl"), list("mnopqrstwxyz")]
target = [list("abcdefghijk"), list("mnopqrstwxy")]

# preprocessing
source_index = seq2seq.Index(source)
target_index = seq2seq.Index(target)
X = source_index.text2tensor(source)
Y = target_index.text2tensor(target)

# model
model = seq2seq.Transformer(source_index, target_index)
model.train()
model.fit(X, Y, epochs = 100, progress_bar = 0)
model.eval()

# test data
test = "ghijklmnopqrst"
test_source = [list(test)]
X_test = source_index.text2tensor(test_source)

# plain beam search
predictions, log_probabilities = seq2seq.beam_search(
    model, 
    X_test,
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

print("\nresults")
print("  test data                      ", test)
print("  plain beam search              ", just_beam)
print("  disjoint windows, beam search  ", disjoint_beam)
print("  n-grams, beam search, triangle ", n_grams_beam)

