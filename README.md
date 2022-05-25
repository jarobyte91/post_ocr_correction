# Post-OCR Document Correction with large Ensembles of Character Sequence Models

This is the code for the paper [Post-OCR Document Correction with large Ensembles of Character Sequence Models](https://arxiv.org/abs/2109.06264) by Ramirez-Orta et al., (2021).

Python package [https://pypi.org/project/post-ocr-correction/](https://pypi.org/project/post-ocr-correction/)

## Usage

### Minimal example

    from pytorch_beam_search import seq2seq
    from post_ocr_correction import correction
    import re
    
    # train data and model
    source = [list("abcdefghijkl"), list("mnopqrstwxyz")]
    target = [list("abcdefghijk"), list("mnopqrstwxy")]
    source_index = seq2seq.Index(source)
    target_index = seq2seq.Index(target)
    X = source_index.text2tensor(source)
    Y = target_index.text2tensor(target)
    model = seq2seq.Transformer(source_index, target_index)
    model.fit(X, Y, epochs = 100, progress_bar = 0)
    
    # test data
    test = "ghijklmnopqrst"
    new_source = [list(test)]
    X_new = source_index.text2tensor(new_source)
    
    # plain beam search
    predictions, log_probabilities = seq2seq.beam_search(
        model, 
        X_new,
        progress_bar = 0
    )
    just_beam = target_index.tensor2text(predictions[:, 0, :])[0]
    just_beam = re.sub(r"<START>|<PAD>|<UNK>|<END>.*", "", just_beam)
    
    # post ocr correction
    disjoint_beam = correction.disjoint(
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

### Load one of the pre-trained models

First download the models
 
    python download_data.py

Now you can use them

    import pickle
    import torch
    from pytorch_beam_search import seq2seq
    from post_ocr_correction import correction
    import re
    from pprint import pprint
    
    # load vocabularies and model
    architecture = pickle.load(open("data/models/en/model_en.arch", "rb"))
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
    state_dict = torch.load("data/models/en/model_en.pt")
    
    # change names from old API of pytorch_beam_search
    state_dict["source_embeddings.weight"] = state_dict.pop("in_embeddings.weight")
    state_dict["target_embeddings.weight"] = state_dict.pop("out_embeddings.weight")
    model.load_state_dict(state_dict)
    
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
    print("  test data                      ", test)
    print("  reference                      ", reference)
    print("  plain beam search              ", just_beam)
    print("  disjoint windows, beam search  ", disjoint_beam)
    print("  n-grams, beam search, triangle ", n_grams_beam)
    print()
    print(evaluation)

## Features

* The **data** folder contains the model parameters and architectural specifications to recover the models for each language.
* The **evaluate** folder contains the scripts to reproduce the evaluation results from the paper.
* The **lib** folder contains the code to use the sequence-to-sequence models to correct very long strings of characters, to compute the metrics used in the paper and the source code of the sequence-to-sequence models.
* The **notebooks** folder contains the Jupyter Notebooks to build the datasets required to train the sequence-to-sequence models, as well as the exploratory data analysis of the data from the [ICDAR 2019 competition](https://sites.google.com/view/icdar2019-postcorrectionocr).
* The **train** folder contains the scripts with hyper-parameters to train the models shown in the paper.

## Installation

    pip install post_ocr_correction
    
    # to download in development mode
    git clone https://github.com/jarobyte91/post_ocr_correction.git
    cd post_ocr_correction
    pip install -e .
    
    # To download the datasets and models
    python download_data.py
    
    # To reproduce the results from the paper
    pip install -r requirements.txt

## Contribute

Feel free to use the Issue Tracker and to send Pull Requests.

## Support

Feel free to contact me at jarobyte91@gmail.com

## License

The project is licensed under the MIT License.
