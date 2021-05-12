#!/usr/bin/python3

# Extracts and compares the embeddings of a given word/MWE in three sentences

import sys
import os
import re
from sklearn.metrics.pairwise import cosine_similarity

from utils.utils_transformers import process_data, get_layers, get_mean_vector, preprocess_mwe, get_sent_layers, get_mwe_vector, get_sent_vector, get_syntax_vector
from utils.utils_conllu import read_conllu

from argparse import ArgumentParser

arg_parser = ArgumentParser(
    description='Compare vectors from transformer models. This script gets three sentences, each of them with a labeled <b>word/MWE</b>, in which the first and second ones are synonyms, and the third one conveys a different meaning. It analyzes whether vectors obtained from different layers of transformer models reflect this distinction, i.e., the expressions in sentences 1 and 2 should be closer (cosine similarity) than those of sentences 1 and 2 to 3.'
)

arg_parser.add_argument(
    '--file',
    '-f',
    type=str, help='Input file (TSV): tab separated values with 7 fields: Target word/MWE, POS, context, overlap, Sent1, Sent2, Sent3. Each sentence contains the target expression inside <b></b> labels. This script ignores the first four fields (it only uses the last three: the sentences).',
    required=True,
    default=None
)

arg_parser.add_argument(
    '--system',
    '-s',
    type=str,
    help='System: bert, dbert, xlm, roberta',
    choices=['bert', 'xlm', 'dbert', 'roberta'],
    default='bert'
)

arg_parser.add_argument(
    '--model',
    '-m',
    type=int,
    help='Model (integer). For BERT (1-4): 1=mBERT-cased, 2-4: Galician models. For XLM, 1=mlm-100-1280. For (XLM-)RoBERTA (1-2): 1=base, 2=large. For DistilBERT (1-2): 1=base-multilingual-cased',
    choices=[1, 2, 3, 4],
    default=1
)

arg_parser.add_argument(
    '--lang',
    '-l',
    type=str,
    choices=['en', 'gl', 'pt', 'es'],
    help="Language of the dataset (and of the model).",
    required=True,
    default=False
)

arg_parser.add_argument(
    '--comp',
    '-c',
    help="Makes two comparisons (sent1 vs. sent2 and sent3) or three (sent1 vs. sent2 and sent3; sent2 vs. sent3)",
    type=int,
    choices=[2, 3],
    default=3
)

arg_parser.add_argument(
    '--syntax',
    type=int,
    help='Instead of representing the target word/MWE using its embeddings, combine it with the syntactic head(s) and/or dependent(s). 1=head (if any, or dep: obj/nsubj/obl/nmod). 2=head+dep. 3=head+dep+co-head. 4=head+dep+co-head+2nd-dep.',
    choices=[1, 2, 3, 4],
    required=False
)

args = arg_parser.parse_args()

inputfile = args.file
sistema = args.system
load_model = args.model
lang = args.lang
comp= args.comp

# Names/Paths to models:
if sistema == 'xlm':
    from transformers import XLMTokenizer, XLMModel
    model_select = {1:'xlm-mlm-100-1280'}
elif sistema == 'roberta':
    from transformers import XLMRobertaTokenizer, XLMRobertaModel
    model_select = {1:'xlm-roberta-base', 2:'xlm-roberta-large'}
elif sistema == 'bert':
    from transformers import BertTokenizer, BertModel, BertForMaskedLM
    if lang == 'en':
        model_select = {
            1:'bert-base-multilingual-cased',
            2:'bert-base-uncased',
            3:'bert-large-uncased'
        }
    elif lang == 'gl':
        model_select = {
            1:'bert-base-multilingual-cased',
            2:'', # Path to Galician BERT (6 layers)
            3:'' # Path to Galician BERT (12 layers)
        }
    elif lang == 'pt':
        model_select = {
            1:'bert-base-multilingual-cased',
            2: 'neuralmind/bert-base-portuguese-cased', # bertimbau base
            3: 'neuralmind/bert-large-portuguese-cased' # bertimbau large
        }
    elif lang == 'es':
        model_select = {
            1:'bert-base-multilingual-cased',
            2:'dccuchile/bert-base-spanish-wwm-cased', # beto base
        }
        
elif sistema == 'dbert':
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForMaskedLM
    if lang == 'en':
        model_select = {1:'distilbert-base-multilingual-cased', 2:'distilbert-base-uncased'}
    else:
        model_select = {1:'distilbert-base-multilingual-cased'}

# Load pre-trained tokenizer and model (weights)
if sistema == 'xlm':
    tokenizer = XLMTokenizer.from_pretrained(model_select[load_model])
    model = XLMModel.from_pretrained(model_select[load_model], output_hidden_states=True)
    layers_size = len(model.attentions)
elif sistema == 'roberta':
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_select[load_model])
    model = XLMRobertaModel.from_pretrained(model_select[load_model], output_hidden_states=True)
    layers_size = len(model.encoder.layer)
elif sistema == 'bert':
    tokenizer = BertTokenizer.from_pretrained(model_select[load_model])
    model = BertModel.from_pretrained(model_select[load_model], output_hidden_states=True)
    layers_size = len(model.encoder.layer)
elif sistema == 'dbert':
    tokenizer = DistilBertTokenizer.from_pretrained(model_select[load_model])
    model = DistilBertModel.from_pretrained(model_select[load_model], output_hidden_states=True)
    layers_size = len(model.transformer.layer)


# Computes similarities and get results (pair)
##############################################
'''
Sent2 is closer to sent1 than sent3 (to sent1): correct
'''
def get_results(id1, id2, id3, vecs):
    cos1 = cosine_similarity(vecs[id1].reshape(1,-1), vecs[id2].reshape(1,-1))[0][0]
    cos2 = cosine_similarity(vecs[id1].reshape(1,-1), vecs[id3].reshape(1,-1))[0][0]    
    if cos1 > cos2:
        result = "correct"
    else:
        result = "wrong"
    return(cos1, cos2, result)

# Computes similarities and get results (triangle)
##################################################
'''
Sentence 3 should be more distant to both sent1 and sent3
'''
def get_results_all(id1, id2, id3, vecs):
    cos1 = cosine_similarity(vecs[id1].reshape(1,-1), vecs[id2].reshape(1,-1))[0][0]
    cos2 = cosine_similarity(vecs[id1].reshape(1,-1), vecs[id3].reshape(1,-1))[0][0]
    cos3 = cosine_similarity(vecs[id2].reshape(1,-1), vecs[id3].reshape(1,-1))[0][0]    
    if cos1 > cos2 and cos1 > cos3:
        result = "correct"
    else:
        result = "wrong"
    return(cos1, cos2, cos3, result)



# Input data
############

if __name__ == "__main__":

    # Parse sentences (for syntax-mode)
    if type(args.syntax) == int:
        conllu_file = re.sub('^.+/', '', inputfile)
        conllu_file = os.path.join('datasets', re.sub('\.[^.]+$', '', conllu_file) + '_sentences.conllu')
        if os.path.isfile(conllu_file):
            sents_conllu = read_conllu(conllu_file)
        else:
            exit(1)

    # Reads from csv
    with open(inputfile, 'r') as inputdata:
        for i in inputdata:
            i = i.rstrip()
            inputinfo = i.split('\t')
            if inputinfo[0] != 'Target':
                tgt = inputinfo[0]

                vectors_concat = {}
                vectors_sum = {}
                sents_avg = {}
                mwes = {}
                sents = {}
                vectors_layers = {}
                for layer in range(layers_size):
                    vectors_layers[layer] = {}

                # 3 sents = 4,5,6
                for s in range(4, len(inputinfo)):
                    sent = re.sub(' $', '', inputinfo[s])
                    sent_punct = re.sub('([\.\?!;,])', r' \1', sent)
                    sent_punct = re.sub('</b>', '</b> ', sent_punct) # To avoid punctuation inside the MWE
                    sent_punct = re.sub('<b>', ' <b>', sent_punct) # To avoid punctuation inside the MWE
                    sent_punct = re.sub('^ ', '', re.sub(' $', '', sent_punct))
                    concat_layers, sum_layers, all_layers, tokens, map_tk = get_sent_layers(sent_punct, sistema, tokenizer, model)
                    sents_avg[s] = get_sent_vector(concat_layers)
                    mwe, sub_mwe, vec_mwe_concat, vec_mwe_sum, vecs_mwe = get_mwe_vector(sent_punct, sistema, concat_layers, sum_layers, all_layers, tokens, map_tk)
                    mwes[s] = mwe
                    sents[s] = sent

                    # Syntax mode
                    if type(args.syntax) == int:
                        deep = args.syntax
                        sent_clean = re.sub('</?b>', '', sent)
                        sent_punct = re.sub('</?b>', '', sent_punct.replace('  ', ' '))
                        target = ' '.join(mwe)
                        combined_layers, combined_concat, combined_sum = get_syntax_vector(target, sent_punct, sents_conllu[sent_clean], sistema, concat_layers, sum_layers, all_layers, tokens, map_tk, vec_mwe_concat, vec_mwe_sum, vecs_mwe, deep)
                        vectors_concat[s] = combined_concat
                        vectors_sum[s] = combined_sum
                        # NC vector on each layer
                        for l in vectors_layers:
                            vectors_layers[l][s] = combined_layers[l]
                    else:
                        vectors_concat[s] = vec_mwe_concat
                        vectors_sum[s] = vec_mwe_sum
                        for l in vectors_layers:
                            vectors_layers[l][s] = vecs_mwe[l]

                # Results
                if comp == 2:
                    cos1_concat, cos2_concat, result_concat = get_results(4, 5, 6, vectors_concat)
                    cos1_sum, cos2_sum, result_sum = get_results(4, 5, 6, vectors_sum)
                    cos1_avg, cos2_avg, result_avg = get_results(4, 5, 6, sents_avg)
                    results_layers = {} # key: layer, value: result
                    for layer in vectors_layers:
                        cos1, cos2, result = get_results(4, 5, 6, vectors_layers[layer])
                        results_layers[layer+1] = result
                # Get results all
                elif comp == 3:
                    cos1_concat, cos2_concat, cos3_concat, result_concat = get_results_all(4, 5, 6, vectors_concat)
                    cos1_sum, cos2_sum, cos3_sum, result_sum = get_results_all(4, 5, 6, vectors_sum)
                    cos1_avg, cos2_avg, cos3_avg, result_avg = get_results_all(4, 5, 6, sents_avg)
                    results_layers = {} # key: layer, value: result
                    for layer in vectors_layers:
                        cos1, cos2, cos3, result = get_results_all(4, 5, 6, vectors_layers[layer])
                        results_layers[layer+1] = result

                # Print static
                print(tgt, inputinfo[1], inputinfo[2], inputinfo[3], sents[4], sents[5], sents[6], result_avg, result_concat, result_sum, sep='\t', end='\t')
                # E por layer
                for l in results_layers:
                    print(results_layers[l], end='\t')
                print('')
            else:
                print("Target", "POS", "Context", "Overlap", "Sent1", "Sent2", "Sent3", "ResultsSentAvg[-4]", "ResultsConcat[-4]", "ResultsSum[-4]", sep='\t', end='\t')

                # Number of layers
                for l in range(1, layers_size+1):
                    print("Lay_%i" %(l), end='\t')
                print('')
