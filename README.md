# Homonymy and synonymy in context
This repository contains the datasets and code for the paper [_Exploring the Representation of Word Meanings in Context: A Case Study on Homonymy and Synonymy_](https://aclanthology.org/2021.acl-long.281/), presented at ACL-IJCNLP 2021.

## Datasets
This resource contains data in four language varieties: Galician (gl), Portuguese (pt), Spanish (es), and English (en).<sup>1</sup>

Each dataset has three variants (see the paper for details):

  1. Triples: three sentences in which the first two target words (inside <b></b> tags) convey the same meaning, and the third a different one. Each triple contains three comparisons (between Sent1 and Sent2, between Sent1 and Sent3, and between Sent2 and Sent3):
      1. POS-tag: Do the two target words belong to the same POS-tag in these sentences? (_same_ or _diff_).
      2. Context: Do they occur in the same context? (_same_ or _diff_).
      3. Overlap: Is there an overlap between both target words? (_true_ or _false_).
  2. Pairs: sentence pairs in which the target words may convey the same (T) of a different meaning (F).
  3. WiC-like pairs: same as _pairs_ but restricted to target words with the same form, as in the [Word in Context (WiC)](https://pilehvar.github.io/wic/) datasets.

The `.conllu` files include the sentences of the datasets parsed in [CoNLL-U format](https://universaldependencies.org/format.html).

<sup>1</sup>Galician is actually a variety of the (Galician-)Portuguese language. They are divided here as the Galician dataset is composed of sentences written with the standard Spanish-based orthograpy (ILG/RAG).

## Code
To replicate the results described in the paper you should run `compare_embeddings_static.py` or `compare_embeddings_transformers.py` selecting the dataset, language, system, and model.

For instance:
`python compare_embeddings_static.py --file datasets/dataset_en.tsv --lang en --system fasttext --model fasttext_model.vec`
will use `fasttext_model.vec` to analyze `dataset_en.tsv`.

For transformers:
`python compare_embeddings_transformers.py --file datasets/dataset_en.tsv --lang en --system bert --model 1`
will use bert-base-multilingual-uncased (model 1). Run `compare_embeddings_transformers.py -h` to see other models. To use the Galician models (see below) please add their path in lines 98 and 99.

## Models
The paper releases a [_fastText_](https://fasttext.cc/) and two [BERT](https://github.com/google-research/bert) models for Galician (ILG/RAG spelling).

### _fastText_
The _fastText_ model was trained for 15 iterations on a corpus with about 600M tokens using 300 dimensions, a window size of 5, negative sampling of 25, and minimum word frequency of 5. It can be download from [Zenodo](https://zenodo.org/record/4481614).

### BERT
We release two BERT models: one with 6 Transformer layers (_small_) and one with 12 layers (_base_). Both models were trained with the [HuggingFace Transformers library](https://github.com/huggingface/transformers) on a single NVIDIA Titan XP GPU with 12GB with the following hyperparameters: block size of 128, learning rate of 0.0001, masked language modeling probability of 0.15, and 0.01 of weight decay. They have been trained only with the MLM objective.

BERT models can be downloaded from the following links: [_small_](https://huggingface.co/marcosgg/bert-small-gl-cased), and [_base_](https://huggingface.co/marcosgg/bert-base-gl-cased).

## Citation
If you use the datasets, code, or models referred in this repository, please cite the following [paper](https://aclanthology.org/2021.acl-long.281/):

```
@inproceedings{garcia-2021-exploring,
    title = "Exploring the Representation of Word Meanings in Context: {A} Case Study on Homonymy and Synonymy",
    author = "Garcia, Marcos",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.281",
    doi = "10.18653/v1/2021.acl-long.281",
    pages = "3625--3640"
}
```
