# Post-OCR Document Correction with large Ensembles of Character Sequence Models

This is the code for the paper [Post-OCR Document Correction with large Ensembles of Character Sequence Models](https://arxiv.org/abs/2109.06264) by Ramirez-Orta et al.,(2021).

# Usage

To correct a string of characters using one of the sequence-to-sequence models

    # string is the string of characters to correct
    # model is a PyTorch sequence-to-sequence model
    # vocabulary is a correspondence between integers and tokens 
    # (see the preprocessing notebook in the notebooks folder for each language)
    
    from ocr_correction import correct_by_disjoint_window, correct_by_sliding_window
    corrected_string = correct_by_disjoint_window(string, model, vocabulary)
    # or
    corrected_string = correct_by_sliding_window(string, model, vocabulary)
    corrected_string.replace("@", "") # to remove padding character
    
To evaluate a sequence-to-sequence model on a dataset with all the hyper-parameters

    # data is a Pandas DataFrame with two columns: the raw text (ocr_to_input) and  
    # the correct transcriptions (gs_aligned), aligned using the @ character 
    # (see the paper for more details)
    # model is a PyTorch sequence-to-sequence model
    # vocabulary is a correspondence between integers and tokens 
    # (see the preprocessing notebook in the notebooks folder for each language)
    
    from ocr_correction import evaluate_model
    evaluation = evaluate_model(raw = data.ocr_to_input, 
                                gs = data.gs_aligned,
                                model = model,
                                vocabulary = vocabulary)
# Features

* The **data** folder contains the model parameters and archictural specifications to recover the models for each language.
* The **evaluate** folder contains the scripts to reproduce the evaluation results from the paper.
* The **lib** folder contains the code to use the sequence-to-sequence models to correct very long strings of characters, to compute the metrics used in the paper and the source code of the sequence-to-sequence models.
* The **notebooks** folder contains the Jupyter Notebooks to build the datasets required to train the sequence-to-sequence models, as well as the exploratory data analysis of the data from the [ICDAR 2019 competition](https://sites.google.com/view/icdar2019-postcorrectionocr).
* The **train** folder contains the scripts with hyper-parameters to train the models shown in the paper.

# Installation

This project requires [Git LFS](https://git-lfs.github.com/). You can install it with

    git lfs install

This project relies on other GitHub repositories. The easiest way to clone is with 

    git clone https://github.com/jarobyte91/post_ocr_correction.git --recurse-submodules

If you forgot the --recurse-submodules flag, you can use 

    git submodule init
    git submodule update

to sync all the subrepos. 

## Contribute

Feel free to use the issue tracker on Github and to send Pull Requests to improve the code.

## Support

Feel free to send an email to any of the authors of the paper.

## License

The project is licensed under the MIT License.
