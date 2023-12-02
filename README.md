# Post-OCR Document Correction with Large Ensembles of Character Sequence-to-Sequence Models 

This is the source code for the paper *Post-OCR Document Correction with Large Ensembles of Character Sequence-to-Sequence Models* by Ramirez-Orta et al., (2021).

* [AAAI version](https://ojs.aaai.org/index.php/AAAI/article/view/21369)
* [Arxiv Preprint](https://arxiv.org/abs/2109.06264) 

## Abstract

In this paper, we propose a novel method to extend sequence-to-sequence models to accurately process sequences much longer than the ones used during training while being sample-and resource-efficient, supported by thorough experimentation. To investigate the effectiveness of our method, we apply it to the task of correcting documents already processed with Optical Character Recognition (OCR) systems using sequence-to-sequence models based on characters. We test our method on nine languages of the ICDAR 2019 competition on post-OCR text correction and achieve a new state-of-the-art performance in five of them. The strategy with the best performance involves splitting the input document in character n-grams and combining their individual corrections into the final output using a voting scheme that is equivalent to an ensemble of a large number of sequence models. We further investigate how to weigh the contributions from each one of the members of this ensemble.

## Usage

* [Train a new model from scratch](tutorials/new_model.ipynb)
* [Load one of the pre-trained models](tutorials/load_model.ipynb)

## Contents

* The **data** folder contains the model parameters and architecture specifications to reconstruct the models for each language.
* The **evaluate** folder contains the scripts to reproduce the evaluation results from the paper.
* The **lib** folder contains the code to use the sequence-to-sequence models to correct very long strings of characters, to compute the metrics used in the paper and the source code of the sequence-to-sequence models.
* The **notebooks** folder contains the Jupyter Notebooks to build the datasets required to train the sequence-to-sequence models, as well as the exploratory data analysis of the data from the [ICDAR 2019 competition](https://sites.google.com/view/icdar2019-postcorrectionocr).
* The **train** folder contains the scripts with hyper-parameters to train the models shown in the paper.

## Installation

    git clone https://github.com/jarobyte91/post_ocr_correction.git
    cd post_ocr_correction
    pip install .
    
To download the datasets and models

    python download_data.py
    
To reproduce the results from the paper

    pip install -r requirements.txt
    cd notebooks

To install the Python package

    pip install post_ocr_correction

## Contribute & Support

* [Issue Tracker](https://github.com/jarobyte91/post_ocr_correction/issues)
* [Pull Requests](https://github.com/jarobyte91/post_ocr_correction/pulls)

## License

The project is licensed under the MIT License.
