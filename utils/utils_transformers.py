# Utils to process data from transformers

import re
import torch
import numpy as np
import unidecode
from utils.utils_conllu import get_heads_deps, get_word_in_sentence

##################
# Process the data
##################
# Input: sentence (+ model + tokenizer).
# Output: tokens and segments tensors, and list of tokens/sub-words
def process_data(input_tokens, system, tokenizer):
    tokens = [] # bert/xlm tokens
    map_tokens = [] # mapping between original tokens and bert/xlm tokenizers

    if system == 'xlm' or system == 'roberta':
        tokens.append("<s>")
        for t in input_tokens:
            map_tokens.append(len(tokens))
            tokens.extend(tokenizer.tokenize(t))
        tokens.append("</s>")

    elif system == 'bert':
        tokens.append("[CLS]")
        for t in input_tokens:
            map_tokens.append(len(tokens))
            tokens.extend(tokenizer.tokenize(t))
        tokens.append("[SEP]")

    elif system == 'dbert':
        tokens.append("[CLS]")
        for t in input_tokens:
            map_tokens.append(len(tokens))
            tokens.extend(tokenizer.tokenize(t))
        tokens.append("[SEP]")

    # ids from models vocab -> id list for each token/sub-word-word.
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

    # segment (sentences) ids
    if system == 'xlm' or system == 'roberta':
        segments_ids = [1] * len(tokens)
    elif system == 'bert' or system == 'dbert':
        segments_ids = [0] * len(tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([tokens_ids])
    segments_tensor = torch.tensor([segments_ids])

    return(tokens_tensor, segments_tensor, tokens, map_tokens)


###############################
# Get the vectors of a sentence
###############################
# Input: tensors and list of tokens from process_data.
# Output: vector of the input: concatenation of 4 last layers, sum of 4 last layers, all layers
def get_layers(tokens_tensor, segments_tensor, tokens, system, model):
    # Predict hidden states features for each layer
    with torch.no_grad():
        # dbert does not have token_type_ids
        if system == 'dbert':
            encoded_layers = model(tokens_tensor)
        else:
            encoded_layers = model(tokens_tensor, segments_tensor)

    token_embeddings = []
    batch_i = 0

    if system == 'xlm':
        for token_i in range(len(tokens)):
            # Hidden layers for each token:
            hidden_layers = []
            # For each layer:
            for layer_i in range(len(encoded_layers[1])):
                # Lookup the vector for token_i in layer_i
                vec = encoded_layers[1][layer_i][batch_i][token_i]        
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
            
    elif system == 'bert' or system == 'roberta':
        for token_i in range(len(tokens)):
            hidden_layers = []
            for layer_i in range(len(encoded_layers[2])):
                vec = encoded_layers[2][layer_i][batch_i][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)

    elif system == 'dbert':
        for token_i in range(len(tokens)):
            hidden_layers = []
            for layer_i in range(len(encoded_layers[1])):
                vec = encoded_layers[1][layer_i][batch_i][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)

    num_layers = len(hidden_layers)

    # outputs: sum and concatenation of the last four layers, and individual layers
    concat_last_4_layers = [torch.cat((torch.stack(layer)[-1], torch.stack(layer)[-2],torch.stack(layer)[-3],torch.stack(layer)[-4]), 0) for layer in token_embeddings]
    sum_last_4_layers = [torch.sum((torch.stack(layer)[-4:]), 0) for layer in token_embeddings]
    layers_n = {} # key: layer; value: list of embedding per token
    for l in range(1, num_layers): # Ignores input layer [0]
        layers_n[l] = []
        for t in token_embeddings:
            layers_n[l].append(t[l])
    return(concat_last_4_layers, sum_last_4_layers, layers_n)


###############################
# Get mean vector (of subwords)
###############################
# Input: layers + position of subwords
# Output: mean vector
def get_mean_vector(layers, subw_posit):
    matrix = np.empty((0,len(layers[0])))
    for p,s in enumerate(subw_posit):
        lista = [element.item() for element in layers[s].flatten()]
        matrix = np.append(matrix, [lista], axis=0) 
    mean_vector = np.mean(matrix, 0)
    return(mean_vector)


############################
# Preprocess target MWE/Word
############################
# Gets position of MWE in the original sentence:
# Input: original sentence with <b>target expression</b> labeled
# Output: tokens, target word/MWE and position
def preprocess_mwe(input_sentence):
    sent_tokens = input_sentence.replace('  ', ' ').split(' ')
    mwe = [] # MWE inside <tags>
    mwe_position = [] # integers: start+end position
    for p,t in enumerate(sent_tokens):
        if t.startswith('<b>'):
            mwe_position.append(p)
            sent_tokens[p] = re.sub('</?b>', '', t)
        elif '</b>' in t:
            mwe_position.append(p)
            sent_tokens[p] = re.sub('</b>', '', t)
    for r in range(mwe_position[0], mwe_position[-1]+1):
        mwe.append(sent_tokens[r])
    return(sent_tokens, mwe, mwe_position)


##########################
# Get vector of a word/MWE
##########################
# Input: input sentence:
# target word/MWE labeled with <b></b>, and layers
# Output: mwe, subwords_mwe, three vector-types: concatenation, addition, all_layers
def get_mwe_vector(input_sentence, system, concat_layers, sum_layers, all_layers, tokens, map_tk):
    sent_tokens, mwe, mwe_position = preprocess_mwe(input_sentence)

    position = []
    sub_mwe = []
    if mwe_position[0] < len(map_tk)-1:
        for z in range(map_tk[mwe_position[0]], map_tk[mwe_position[-1]+1]):
            position.append(z)
            sub_mwe.append(tokens[z])
    else:
        # Several tokens end-position
        if ((system == 'dbert' or system == 'bert') and tokens[map_tk[mwe_position[0]]+1].startswith('##')) or \
           (system == 'xlm' and not tokens[map_tk[mwe_position[0]]].endswith('</w>')) or \
           (system == 'roberta' and not tokens[map_tk[mwe_position[0]]+1].startswith('▁')):
            for z in range(map_tk[mwe_position[0]], len(tokens)-1):
                position.append(z)
                sub_mwe.append(tokens[z])
        else:
            # Single token end-position
            position.append(map_tk[mwe_position[0]])
            sub_mwe.append(tokens[map_tk[mwe_position[0]]])

    # Get vector of MWE (or synonym) using subwords position
    # Not splitted (only for MWE synonyms of one token, not splitted by bert)   
    vecs_mwe = [] # one per layer
    for l in range(1, len(all_layers)+1):
        if len(position) == 1:
            vec_mwe = all_layers[l][position[0]]
        else:
            vec_mwe = get_mean_vector(all_layers[l], position)
        vecs_mwe.append(vec_mwe)

    # Sum and concatenation
    if len(position) == 1:
        vec_mwe_concat = concat_layers[position[0]]
        vec_mwe_sum = sum_layers[position[0]]        
    else:
        vec_mwe_concat = get_mean_vector(concat_layers, position)
        vec_mwe_sum = get_mean_vector(sum_layers, position)

    # Check if the MWE in the sentence is equal to the MWE recovered from the subword tokenization
    str_mwe = unidecode.unidecode(''.join(mwe)).lower()
    str_sub_mwe = unidecode.unidecode(''.join(sub_mwe)).lower()
    if system == 'bert' or system == 'dbert':
        str_sub_mwe = re.sub('##', '', str_sub_mwe)
    elif system == 'xlm':
        str_sub_mwe = re.sub('</w>', '', str_sub_mwe)
    elif system == 'roberta':
        str_sub_mwe = re.sub('#', '', str_sub_mwe) # '▁' character is converted to '#' by unidecode

    check = 0
    if str_mwe == str_sub_mwe:
        check = 1
    else:
        # Exception: some models may represent rare tokens as UNK:
        if '[unk]' in str_sub_mwe:
            trunk = str_sub_mwe.replace('[unk]', '')
            if trunk in str_mwe:
                check = 1

    if check == 1:
        return(mwe, sub_mwe, vec_mwe_concat, vec_mwe_sum, vecs_mwe)
    else:
        print("WARNING", str_mwe, str_sub_mwe)

###############################
# Get subwords of an expression
###############################
# Input: list of tokens and target word
# Output: subwords positions of the target word
# Not used (XLM-RoBERTa not added)
def get_subwords(tokens, target_lg, system):
    if system == 'xlm':
        subwords = []
        target = target_lg + '</w>'
        i = 0
        while i < len(tokens)-1 or target == '':
            size = len(tokens[i])
            # If it is an 'end' subword:
            if tokens[i].endswith('</w>'):
                # If its the end of the target
                if tokens[i] == target:
                    subwords.append(i)
                    target = target[size:]
                    break
                # If not, start
                else:
                    target = target_lg + '</w>'
                    subwords = []
                    # If subword starts with 'target'
            elif tokens[i] == target[0:size]:
                target = target[size:]
                subwords.append(i)
            i += 1

    elif system == 'bert' or system == 'dbert':
        subwords = []
        target = target_lg
        i = 0
        while i < len(tokens)-1:
            size = len(tokens[i])
            # Check if starts with target (and target is not a subword of it)
            if target.startswith(tokens[i]) and len(target) > size:
                target = target[size:]
                subwords.append(i)
            if tokens[i].startswith('##'):
                if target.startswith(tokens[i][2:size]):
                    target = target[size-2:]
                    subwords.append(i)
                if target == '' and not tokens[i+1].startswith('##'):
                    break
                else:
                    if tokens[i+1].startswith('##') and target == '':
                        subwords = []
                        target = target_lg
                    elif target != '' and not tokens[i+1].startswith('##'):
                        subwords=[]
                        target = target_lg
            i += 1
    return(subwords)


##########################
# Get layers of a sentence
##########################
# Input: input sentence (plain)
# Output: sum|concat|all_layers
def get_sent_layers(input_sentence, system, tokenizer, model):
    sent_tokens = re.sub('</?b>', '', input_sentence) # remove tags
    sent_tokens = sent_tokens.replace('  ', ' ').split(' ')
    tokens_tensor, segments_tensor, tokens, map_tk = process_data(sent_tokens, system, tokenizer)
    concat_layers, sum_layers, all_layers = get_layers(tokens_tensor, segments_tensor, tokens, system, model)

    return(concat_layers, sum_layers, all_layers, tokens, map_tk)


##########################
# Get vector of a sentence
##########################
# Input: layers of a sentece (one per (sub)word)
# Output: sentence layer (average of (sub)words)
def get_sent_vector(layers):
    matrix = np.empty((0,len(layers[0])))
    for p,t in enumerate(layers):
        lista = [element.item() for element in layers[p].flatten()]
        matrix = np.append(matrix, [lista], axis=0)
    mean_sent_vector = np.mean(matrix, 0)
    return(mean_sent_vector)


# Adds labels to a token in a sentence
#############
def add_labels(token, sentence):
    sent_out = []

    # For MWEs (e.g. proper nouns)
    if ' ' in token:
        if token in sentence:
            sent_labeled = re.sub(token, '<b>%s</b>' %(token), sentence)

    else:
        for word in sentence.split(' '):
            if token == word:
                word = '<b>' + word + '</b>'
            elif token in word and re.search('[\.\?!,]$', word):
                word = '<b>' + token + '</b>' + word[-1]
            sent_out.append(word)

        sent_labeled = ' '.join(sent_out)

    sent_labeled = sent_labeled.replace('  ', ' ')
    sent_labeled = re.sub('^ ', '', sent_labeled)
    sent_labeled = re.sub(' $', '', sent_labeled)
    return(sent_labeled)

##########################
# Get vector of heads/deps
##########################
# Syntax mode
# Combined vector of the target word with head(s)/dep(s)
# [default: average]
def get_syntax_vector(target, sent, conllu, system, concat_layers, sum_layers, all_layers, tokens, map_tk, vec_mwe_concat, vec_mwe_sum, vecs_mwe, numb):
    
    '''
    Given the target word/MWEs and the parsed sentence (conllu),
    returns the combined vector of the target+head(s)/dep(s)
    '''
    
    heads, deps, tgt_tag, tgt_dep = get_heads_deps(target, conllu, 'form')

    words = []
    # Insert up to four words
    if len(heads)>0:
        words.append(heads[0][0])
        if len(deps)>0:
            words.append(deps[0][0])
        if len(heads)>1:
            words.append(heads[1][0])
        if len(deps)>1:
            words.append(deps[1][0])
    elif len(deps)>0:
        for w in deps:
            words.append(w[0])

    # Select
    if numb>0:
        combined_concat = []
        combined_sum = []
        combined_vectors = []
        if len(words)>0:
            token = get_word_in_sentence(words[0], sent)
            sent_token = add_labels(token, sent)
            tgt, sub_tgt, vec_concat, vec_sum, vecs_layers = get_mwe_vector(sent_token, system, concat_layers, sum_layers, all_layers, tokens, map_tk)
            combined_concat = get_sent_vector([vec_mwe_concat, vec_concat])
            combined_sum = get_sent_vector([vec_mwe_sum, vec_sum])
            for l in range(len(vecs_layers)):
                tmp = [vecs_layers[l].tolist(), vecs_mwe[l].tolist()]
                tmp_avg = np.average(tmp, axis=0)
                combined_vectors.append(tmp_avg)
        if numb == 1:
            if len(combined_vectors)>0:
                return(combined_vectors, combined_concat, combined_sum)
            else:
                return(vecs_mwe, vec_mwe_concat, vec_mwe_sum)

    if numb>1:
        if len(words)>1:
            token = get_word_in_sentence(words[1], sent)
            sent_token = add_labels(token, sent)
            tgt, sub_tgt, vec_concat, vec_sum, vecs_layers = get_mwe_vector(sent_token, system, concat_layers, sum_layers, all_layers, tokens, map_tk)
            combined_concat = get_sent_vector([combined_concat, vec_concat])
            combined_sum = get_sent_vector([vec_mwe_sum, vec_sum])
            for l in range(len(vecs_layers)):
                tmp = [vecs_layers[l].tolist(), combined_vectors[l]]
                tmp_avg = np.average(tmp, axis=0)
                combined_vectors[l] = tmp_avg
        if numb == 2:
            if len(combined_vectors)>0:
                return(combined_vectors, combined_concat, combined_sum)
            else:
                return(vecs_mwe, vec_mwe_concat, vec_mwe_sum)

    if numb>2:
        if len(words)>2:
            token = get_word_in_sentence(words[2], sent)
            sent_token = add_labels(token, sent)
            tgt, sub_tgt, vec_concat, vec_sum, vecs_layers = get_mwe_vector(sent_token, system, concat_layers, sum_layers, all_layers, tokens, map_tk)
            combined_concat = get_sent_vector([combined_concat, vec_concat])
            combined_sum = get_sent_vector([vec_mwe_sum, vec_sum])
            for l in range(len(vecs_layers)):
                tmp = [vecs_layers[l].tolist(), combined_vectors[l]]
                tmp_avg = np.average(tmp, axis=0)
                combined_vectors[l] = tmp_avg
        if numb == 3:
            if len(combined_vectors)>0:
                return(combined_vectors, combined_concat, combined_sum)
            else:
                return(vecs_mwe, vec_mwe_concat, vec_mwe_sum)

    if numb == 4:
        if len(words)>3:
            token = get_word_in_sentence(words[3], sent)
            sent_token = add_labels(token, sent)
            tgt, sub_tgt, vec_concat, vec_sum, vecs_layers = get_mwe_vector(sent_token, system, concat_layers, sum_layers, all_layers, tokens, map_tk)
            combined_concat = get_sent_vector([combined_concat, vec_concat])
            combined_sum = get_sent_vector([vec_mwe_sum, vec_sum])
            for l in range(len(vecs_layers)):
                tmp = [vecs_layers[l].tolist(), combined_vectors[l]]
                tmp_avg = np.average(tmp, axis=0)
                combined_vectors[l] = tmp_avg
        if len(combined_vectors)>0:
            return(combined_vectors, combined_concat, combined_sum)
        else:
            return(vecs_mwe, vec_mwe_concat, vec_mwe_sum)

    # No head(s)/dep(s)
    if len(words)==0:
        return(vecs_mwe, vec_mwe_concat, vec_mwe_sum)

