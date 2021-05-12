# Homonymy and synonymy in context
This repository contains the dataset and code for the paper _Exploring the Representation of Word Meanings in Context: A Case Study on Homonymy and Synonymy_, accepted at ACL-IJCNLP 2021.

## Dataset

## Code

## Models
The paper releases a [_fastText_](https://fasttext.cc/) and two [BERT](https://github.com/google-research/bert) models for Galician (ILG/RAG spelling).

### _fastText_
The _fastText_ model was trained for 15 iterations on a corpus with about 600M tokens using 300 dimensions, a window size of 5, negative sampling of 25, and minimum word frequency of 5. It can be download from [Zenodo](https://zenodo.org/record/4481614).

### BERT
We release two BERT models: one with 6 Transformer layers (_small_) and one with 12 layers (_base_). Both models were trained with the [HuggingFace Transformers library](https://github.com/huggingface/transformers) on a single NVIDIA Titan XP GPU with 12GB with the following hyperparameters: block size of 128, learning rate of 0.0001, masked language modeling probability of 0.15, and 0.01 of weight decay. They have been trained only with the MLM objective.

BERT models can be downloaded from the following links: [_small_](https://zenodo.org/record/4481575), and [_base_](https://zenodo.org/record/4481591).

## Citation
If you use the dataset, code, or models referred in this repository, please cite the following paper:

```
Garcia, Marcos. 2021.
Exploring the Representation of Word Meanings in Context: A Case Study on Homonymy and Synonymy.
In Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021).
Association for Computational Linguistics.
```

## Contact
[Marcos Garcia](https://citius.usc.es/equipo/investigadores-postdoutorais/marcos-garcia-gonzalez?language=en)
