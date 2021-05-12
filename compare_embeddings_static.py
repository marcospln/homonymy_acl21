#!/usr/bin/python3

# Compares different methods to obtain the embeddings of a given word/MWE in a particular sentence.
#
# Baseline1: embedding of the word [average embeddings of the MWE].
# Baseline2: average embeddings of the whole sentence.
# Baseline3: sum of the embeddings of the whole sentence.
# Baseline3: multiplication of the embeddings of the whole sentence.
# 
# Method1: combination (average/sum/mult) of: embedding of the word/MWE and head+dependent [core relations] | For MWEs: head+deps of the nucleus.
# Method2: combination (average/sum/mult) of: embedding of the word/MWE selectional preferences of the head+dependent [core relations] | For MWEs: head+deps of the nucleus.
#

import sys
import os.path
import re
from argparse import ArgumentParser
from sklearn.metrics.pairwise import cosine_similarity

from utils.utils_static import load_model, get_target, get_word_vectors, get_sentence_vector, get_head_dep_vector, get_select_prefs_vector
from utils.utils_conllu import sentences_to_conllu, read_conllu


arg_parser = ArgumentParser(
    description='Compare word vectors from static models. This script gets three sentences, each of them with a labeled <b>word/MWE</b>, in which the first and second ones are synonyms, and the third one conveys a different meaning. It analyzes whether vectors obtained from static embeddings using different methods reflect this distinction, i.e., the representations of sentences 1 and 2 should be closer (cosine similarity) than those of sentences 1 and 3 (2 vs. 3 is not performed here).'
)

arg_parser.add_argument(
    '--file',
    '-f',
    type=str,
    help='Input file (TSV): tab separated values with 7 fields: Target word/MWE, POS, context, overlap, Sent1, Sent2, Sent3. Each sentence contains the target expression inside <b></b> labels. This script ignores the first four fields (it only uses the last three: the sentences).',
    required=True,
    default=None
)
arg_parser.add_argument(
    '--system',
    '-s',
    type=str,
    help='System: word2vec, glove, fasttext, or dep2vec (word2vecf)',
    choices=['w2v', 'fasttext', 'glove', 'dep2vec'],
    default='w2v',
    required=True
)
arg_parser.add_argument(
    '--model',
    '-m',
    type=str,
    help='Model (path to file). Format is gensim KeyedVectors (plain text with one vector per line).',
    required=True
)
arg_parser.add_argument(
    '--parse',
    '-p',
    type=str,
    choices=['y', 'n'],
    help="Read parsed sentences from .conllu file or parse the intput TSV (FreeLing and UDPipe). Input .conllu is the name of the input file (without extensions + '_sentences.conllu'.",
    default=False
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
    '--prefs',
    type=str,
    help="File with the selectional preferences for each dependency (previously extracted from large corpora). Default: 'datasets/preferences.txt'. Field separator ';', and ',' to split lema,deprel, and lema,POS.",
    required=None
)

args = arg_parser.parse_args()

inputfile = args.file
system = args.system
model_file = args.model
lang = args.lang
comp= args.comp

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


# Starting
##########
if __name__ == '__main__':

    # Parse sentences
    if args.parse == 'y':
        print("Please edit 'utils/utils_conllu.py' (lines 51 to 57) and add the path to your NLP models that generate a .conllu file")
        exit(1)
        conllu_file = os.path.join('datasets', sentences_to_conllu(inputfile, lang))
    else:
        conllu_file = re.sub('^.+/', '', inputfile)
        conllu_file = os.path.join('datasets', re.sub('\.[^.]+$', '', conllu_file) + '_sentences.conllu')
        
    if os.path.isfile(conllu_file):
        sents_conllu = read_conllu(conllu_file)
    else:
        exit('\nPlease parse the sentences first (--parse y)')

    # Read selectional preferences
    if args.prefs == None:
        f = 'preferences_mi_' + lang + '.txt'
        prefs_file = os.path.join('datasets', f)
    else:
        prefs_file = args.prefs
    if os.path.isfile(prefs_file):
        preferences = {'h':dict(), 'd':dict()} # key: H|D (direction of deprel), value=dict: key: tuple (head, deprel*), value: list of tuples (lema, NOUN)
        with open(prefs_file, 'r') as pf:
            for line in pf:
                line = line.rstrip()
                info = line.split(';')
                key = (info[1].split(',')[0], info[1].split(',')[1])
                vals = []
                preferences[info[0]][key] = []
                for i in range(2,len(info)):
                    tup = (info[i].split(',')[0], info[i].split(',')[1])
                    preferences[info[0]][key].append(tup)
    else:
        exit("File '%s' (selectional preferences) not found." %prefs_file)

    # Model and vocab
    model, vocab = load_model(system, model_file)
    
    # Reads from csv
    with open(inputfile, 'r') as inputdata:
        for i in inputdata:
            i = i.rstrip()
            inputinfo = i.split('\t')
            if inputinfo[0] != 'Target':
                tgt = inputinfo[0]
                sents = {}

                # Vectors: dicts with three keys (sentences)
                baseline1 = {}
                baseline2 = {}
                baseline3 = {}
                baseline4 = {}
                method1a = {}
                method1b = {}
                method1c = {}
                method1d = {}
                method2a = {}
                method2b = {}
                method2c = {}

                # 3 sents = 4,5,6
                for s in range(4, len(inputinfo)):
                    sent = inputinfo[s]
                    sents[s] = sent
                    
                    target, sent_clean, position = get_target(sent)
                    if sent_clean in sents_conllu:
                        sent_conllu = sents_conllu[sent_clean]
                        baseline1[s] = get_word_vectors(target, model, vocab)
                        baseline2[s] = get_sentence_vector(sent_clean, sent_conllu, model, vocab, 'avg')
                        baseline3[s] = get_sentence_vector(sent_clean, sent_conllu, model, vocab, 'sum')
                        baseline4[s] = get_sentence_vector(sent_clean, sent_conllu, model, vocab, 'mult')
                        method1a[s] = get_head_dep_vector(target, position, sent_clean, sent_conllu, model, vocab, 'sum', 1)
                        method1b[s] = get_head_dep_vector(target, position, sent_clean, sent_conllu, model, vocab, 'sum', 2)
                        method1c[s] = get_head_dep_vector(target, position, sent_clean, sent_conllu, model, vocab, 'sum', 3)
                        method1d[s] = get_head_dep_vector(target, position, sent_clean, sent_conllu, model, vocab, 'sum', 4)
                        method2a[s] = get_select_prefs_vector(target, preferences, sent_conllu, model, vocab, 'sum', 10, 1)
                        method2b[s] = get_select_prefs_vector(target, preferences, sent_conllu, model, vocab, 'sum', 10, 2)
                        method2c[s] = get_select_prefs_vector(target, preferences, sent_conllu, model, vocab, 'sum', 10, 3)
                    else:
                        print("ERROR", sent_clean, sent)
                        exit(1)

                # Results
                if comp == 2:
                    cos1_bas1, cos2_bas1, result_bas1 = get_results(4, 5, 6, baseline1)
                    cos1_bas2, cos2_bas2, result_bas2 = get_results(4, 5, 6, baseline2)
                    cos1_bas3, cos2_bas3, result_bas3 = get_results(4, 5, 6, baseline3)
                    cos1_bas4, cos2_bas4, result_bas4 = get_results(4, 5, 6, baseline4)
                    cos1_met1a, cos2_met1a, result_met1a = get_results(4, 5, 6, method1a)
                    cos1_met1b, cos2_met1b, result_met1b = get_results(4, 5, 6, method1b)
                    cos1_met1c, cos2_met1c, result_met1c = get_results(4, 5, 6, method1c)
                    cos1_met1d, cos2_met1d, result_met1d = get_results(4, 5, 6, method1d)
                    cos1_met2a, cos2_met2a, result_met2a = get_results(4, 5, 6, method2a)
                    cos1_met2b, cos2_met2b, result_met2b = get_results(4, 5, 6, method2b)
                    cos1_met2c, cos2_met2c, result_met2c = get_results(4, 5, 6, method2c)
                elif comp == 3:
                    cos1_bas1, cos2_bas1, cos3_bas1, result_bas1 = get_results_all(4, 5, 6, baseline1)
                    cos1_bas2, cos2_bas2, cos3_bas2, result_bas2 = get_results_all(4, 5, 6, baseline2)
                    cos1_bas3, cos2_bas3, cos3_bas3, result_bas3 = get_results_all(4, 5, 6, baseline3)
                    cos1_bas4, cos2_bas4, cos3_bas4, result_bas4 = get_results_all(4, 5, 6, baseline4)
                    cos1_met1a, cos2_met1a, cos3_met1a, result_met1a = get_results_all(4, 5, 6, method1a)
                    cos1_met1b, cos2_met1b, cos3_met1b, result_met1b = get_results_all(4, 5, 6, method1b)
                    cos1_met1c, cos2_met1c, cos3_met1c, result_met1c = get_results_all(4, 5, 6, method1c)
                    cos1_met1d, cos2_met1d, cos3_met1d, result_met1d = get_results_all(4, 5, 6, method1d)
                    cos1_met2a, cos2_met2a, cos3_met2a, result_met2a = get_results_all(4, 5, 6, method2a)
                    cos1_met2b, cos2_met2b, cos3_met2b, result_met2b = get_results_all(4, 5, 6, method2b)
                    cos1_met2c, cos2_met2c, cos3_met2c, result_met2c = get_results_all(4, 5, 6, method2c)


                # Print results
                print(tgt, inputinfo[1], inputinfo[2], inputinfo[3], sents[4], sents[5], sents[6],
                      result_bas1, result_bas2, result_bas3, result_bas4,
                      result_met1a, result_met1b, result_met1c, result_met1d,
                      result_met2a, result_met2b, result_met2c,
                      sep='\t')

            else:
                # Print head
                print("Target", "POS", "Context", "Overlap", "Sent1", "Sent2", "Sent3",
                      "Bas1", "Bas2", "Bas3", "Bas4",
                      "Meth1a", "Meth1b", "Meth1c", "Meth1d",
                      "Meth2a", "Meth2b", "Meth2c",
                      sep='\t')
