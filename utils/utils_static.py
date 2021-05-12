# Utilities for static word-embeddings

import gensim
import re
import numpy as np
from gensim.models import word2vec
from gensim.models import KeyedVectors
import conllu
from conllu import parse, parse_tree
from utils.utils_conllu import get_heads_deps, get_root
from sklearn.metrics.pairwise import cosine_similarity

# Load the embeddings
#####################
def load_model(system, model_file):

    # Word2Vec
    if system == 'w2v':
        if model_file.endswith('txt'):
            model = KeyedVectors.load_word2vec_format(model_file, binary=False)
            vocab = set(model.index2word)
        elif model_file.endswith('w2v'):
            model = word2vec.Word2Vec.load(model_file)
            vocab = set(model.wv.index2word)
    # Other models [only .txt KeyedVectors]
    elif system == 'glove':
        model = KeyedVectors.load_word2vec_format(model_file, binary=False)
        vocab = set(model.index2word)
    elif system == 'fasttext':
        model = KeyedVectors.load_word2vec_format(model_file, binary=False)
        vocab = set(model.index2word)
    elif system == 'dep2vec':
        model = KeyedVectors.load_word2vec_format(model_file, binary=False)
        vocab = set(model.index2word)

    return(model, vocab)


# Gets the labeled target <b>word/MWE</b> from a sentence
#########################################
# Outputs target and clean sentence (without tags)
# Position: list of start and end postion.
# For MWEs, 3: start, end of first word, end of MWE
def get_target(sentence):
    position = []
    target = ''
    clean_sentence = re.sub('</?b>', '', sentence.replace('  ', ' '))
    clean_sentence = re.sub(' +$', '', clean_sentence)
    target = re.sub('</b>', '</b> ', sentence) # To avoid punctuation inside the MWE
    target = re.sub('<b>', ' <b>', target) # To avoid punctuation inside the MWE
    target = re.sub('^.+<b>', '', target)
    target = re.sub('</b>.+$', '', target)
    start = 0
    end = 0
    found = 0
    sentence = sentence.replace('  ', ' ')
    for i in range(len(sentence)):
        if sentence[i] == '<':
            if sentence[i+1] == 'b' and sentence[i+2] == '>':
                start = i
                position.append(start)
            elif sentence[i+1] == '/' and sentence[i+2] == 'b' and sentence[i+3] == '>':
                end = i
                position.append(end-3)
                found = 1
        elif start>0 and i>start and i>end and found == 0:
            if sentence[i] == ' ':
                mid = i
                position.append(mid-3)
                found = 1
    return(target, clean_sentence, position)


# Gets a vector for a single word
#################################
# If not in vocab: None | [1 * len(dimensions)]
def get_vector(word, model, vocab):
    if word in vocab:
        vector = model.wv[word]
    elif word.lower() in vocab:
        vector = model.wv[word.lower()]
    else:
        dimensions = model.vector_size
        vector = model.wv['a']
        for d in vector:
            d = 0
    return(vector)


# Gets the vector of a given word|MWE
#####################################
def get_word_vectors(words, model, vocab):
    
    # Single words
    if not ' ' in words:
        return(get_vector(words, model, vocab))
    # MWEs
    else:
        mwe = words.split(' ')
        vecs = []
        # Average vector
        for w in mwe:
            vec = get_vector(w, model, vocab)
            if vec is not None:
                vecs.append(vec)
        return(sum(vecs)/len(vecs))


# Gets the vector of a sentence:
# Similar to MWE, but using .conllu
################################
def get_sentence_vector(sentence, conll, model, vocab, funct):

    vecs = []
    for token in parse(conll)[0]:
        vec = get_vector(token['form'], model, vocab)
        if vec is not None:
            vecs.append(vec)

    if funct == 'avg':
        return(sum(vecs)/len(vecs))
    elif funct == 'sum':
        return(sum(vecs))
    elif funct == 'mult':
        out_vec = vecs[0]
        for i in range(1,len(vecs)):
            out_vec = out_vec * vecs[i]
        return(out_vec)


# Method1 (from conllu)
# For a given word/MWE in a sentence, gets the combined vector
# Of the word + head + dep.
# numb is the number of elements:
# 1=head1, 2=head1+dep1/head1+head2, 3=h1+d1+h2/h1+d1+d2, 4:all
def get_head_dep_vector(target, position, sentence, conll, model, vocab, funct, numb):

    heads, deps, tgt_pos, tgt_deprel = get_heads_deps(target, conll, 'lemma')

    vecs = []
    if len(heads)>0:
        vecs.append(get_vector(heads[0][0], model, vocab))
    if len(deps)>0:
        vecs.append(get_vector(deps[0][0], model, vocab))
    if len(heads)>1:
        vecs.append(get_vector(heads[1][0], model, vocab))
    if len(deps)>1:
        vecs.append(get_vector(deps[1][0], model, vocab))

    # If no head/dep found [and target is not root] get the sentence root:
    if len(vecs) == 0:
        root = get_root(conll)
        if root != target:
            vecs.append(get_vector(root, model, vocab))

    if len(vecs)>0:
        vecs_numb = vecs[:numb]

        vec_tgt = get_word_vectors(target, model, vocab)
        vecs_numb.append(vec_tgt)

        # Check for empty vectors
        vecs_f = []
        for v in vecs_numb:
            if v is not None:
                vecs_f.append(v)

        if len(vecs_f)>1:
            if funct == 'avg':
                return(sum(vecs_f)/len(vecs_f))
            elif funct == 'sum':
                return(sum(vecs_f))
            elif funct == 'mult':
                out_vec = vecs_f[0]
                for i in range(1,len(vecs_f)):
                    out_vec = out_vec * vecs_f[i]
                return(out_vec)
        else:
            if vec_tgt is not None:
                return(vec_tgt)
            else:
                return(None)
    else:
        return(None)


# Method 2 (selectional preferences)
# Returns a combined (sum) vector of the target word
# with other representing its top-n selectional preferences
# (previously obtained from corpora), also summed.
# method2[s] = get_select_prefs_vector(target, position, sent_clean, sent_conllu, model, vocab, 'sum', 5)
def get_select_prefs_vector(target, sprefs, conll, model, vocab, funct, size, nprefs):
    heads, deps, tgt_pos, tgt_deprel = get_heads_deps(target, conll, 'lemma')

    '''
    Two lists of selectional preferences: prefs1 and prefs2
    For target words with head+dep, prefs1 = head, prefs2 = dep.
    For target words without head -> order: obj->nsubj->obl->nmod
    '''
    prefs1 = [] 
    prefs2 = []
    # For non-root targets: head1
    prefs = []
    if tgt_deprel != 'root':
        if len(heads)>0:
            if heads[0] in sprefs['h']:
                for p in sprefs['h'][heads[0]]:
                    # Select only preferences with the same POS as target
                    if p[1] == tgt_pos: 
                        prefs1.append(p[0])
        if len(deps)>0:
            if deps[0] in sprefs['d']:
                for p in sprefs['d'][deps[0]]:
                    # Select only preferences with the same POS as target
                    if p[1] == tgt_pos: 
                        prefs2.append(p[0])

    # For root:
    else:
        if len(deps)>0:
            if deps[0] in sprefs['d']:
                for p in sprefs['d'][deps[0]]:
                    # Select only preferences with the same POS as target
                    if p[1] == tgt_pos: 
                        prefs1.append(p[0])
            if len(deps)>1:
                if deps[1] in sprefs['d']:
                    for p in sprefs['d'][deps[1]]:
                        # Select only preferences with the same POS as target
                        if p[1] == tgt_pos:
                            prefs2.append(p[0])

    target_vector = get_word_vectors(target, model, vocab)

    # One set of selectional preference [head (1st dep in root)]
    if nprefs == 1:
        prefs1_vector = combine_vectors(prefs1, model, vocab, target, target_vector, 'sum', size, 'y')
        if prefs1_vector != None:
            combined_vector = target_vector + prefs1_vector
            return(combined_vector)
        else:
            return(target_vector)
    # One set of selectional preferences [dep]
    elif nprefs == 2:
        prefs2_vector = combine_vectors(prefs2, model, vocab, target, target_vector, 'sum', size, 'y')
        if prefs2_vector != None:
            combined_vector = target_vector + prefs2_vector
            return(combined_vector)
        else:
            return(target_vector)
    # Both sets of selectional preferences
    elif nprefs == 3:
        combined_vector = None
        prefs1_vector = combine_vectors(prefs1, model, vocab, target, target_vector, 'sum', size, 'y')
        prefs2_vector = combine_vectors(prefs2, model, vocab, target, target_vector, 'sum', size, 'y')
        if prefs1_vector != None:
            combined_vector = target_vector + prefs1_vector
            if prefs2_vector != None:
                combined_vector = combined_vector + prefs2_vector
        elif prefs2_vector != None:
            combined_vector = target_vector + prefs2_vector
        if combined_vector != None:
            return(combined_vector)
        else:
            return(target_vector)


# Combine vectors
# Given a set of words get a combined vector
def combine_vectors(words, model, vocab, target, target_vector, funct, size, filt):
    
    if len(words)>0:
        vecs = []
        count = 0
        for p in words:
            vec = get_vector(p, model, vocab)
            if vec is not None and count < size:

                if filt == 'y': # Exclude non-related selectional preferences
                    diff = cosine_similarity(target_vector.reshape(1,-1), vec.reshape(1,-1))[0][0]
                    if diff >= 0.25:
                        vecs.append(vec)
                        count+=1
                else:
                    vecs.append(vec)
                    count+=1

        if funct == 'avg':
            words_vec = sum(vecs)/len(vecs)
        elif funct == 'sum':
            words_vec = sum(vecs)
        elif funct == 'mult':
            words_vec = vecs[0]
            for i in range(1,len(vecs)):
                words_vec = words_vec * vecs[i]
        else:
            return(words_vec)
    else:
        return(None)
