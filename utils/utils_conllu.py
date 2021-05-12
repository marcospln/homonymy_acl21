# Utils for parsing conllu

import sys
import os
import re
import subprocess
from subprocess import check_output
import conllu
from conllu import parse_incr
import edlib

# Reads conllu to dict
# Input: file.conllu with original sentences in metadata
# Output: dict key:sentence, value, parsed
def read_conllu(filename):
    sentences_conllu = {}
    file_conllu = open(filename, 'r', encoding='utf-8')
    for sentence in parse_incr(file_conllu):
        sentences_conllu[sentence.metadata['sent_csv']] = sentence.serialize()
    return(sentences_conllu)


# Generates .conllu file
########################
# Runs external tools (e.g., FreeLing, UDPipe)
def sentences_to_conllu(filename, lang):

    # Open csv
    sentences = []
    with open(filename, 'r') as inputdata:
        for i in inputdata:
            i = i.rstrip()
            i = re.sub('</?b>', '', i)
            i = re.sub('  +', ' ', i)
            inputinfo = i.split('\t')
            if inputinfo[0] != 'Target':
                for s in range(4, len(inputinfo)):
                    if inputinfo[s] not in sentences:
                        sentences.append(inputinfo[s])

    # Write sentences
    with open('.tmp.txt', 'w', encoding="utf-8") as ftmp:
        for s in sentences:
            ftmp.write(s)
            ftmp.write('\n')

    # Run [external] tools to generate .conllu (FreeLing/UDPipe, etc.)
    if lang == 'gl':
        saida = 'This should be a python list where each element is a line of a conllu file'
    elif lang == 'pt':
        saida = 'This should be a python list where each element is a line of a conllu file'
    elif lang == 'es':
        saida = 'This should be a python list where each element is a line of a conllu file'
    elif lang == 'en':
        saida = 'This should be a python list where each element is a line of a conllu file'

    # Read output
    outfile = re.sub('^.+/', '', filename)
    outfile = re.sub('\.[^.]+$', '', outfile) + '_sentences.conllu'
    data_folder = 'datasets'
    
    with open(os.path.join(data_folder, outfile), 'w') as out:
        count = 0
        for line in saida:
            if 'sent_id' in line:
                out.write(line)
                out.write('\n')
                texto = "# sent_csv = " + sentences[count]
                out.write(texto)
                out.write('\n')                
                count+=1
            else:
                out.write(line)
                out.write('\n')
    subprocess.run("rm .tmp.txt", shell=True)

    return(outfile)


# Get root: given a sentence, returns the root lemma
# Input: target, conll
def get_root(conll):
    from conllu import parse
    root_lemma = ''
    for token in parse(conll)[0]:
        if token['deprel'] == 'root':
            root_lemma = token['lemma']
    return(root_lemma)


# Get heads and deps from conllu
# For a given word/MWE in a sentence, gets up to two heads and deps
# WARNING: target is the first occurrence of the word in the sentence!
def get_heads_deps(target, conll, wform):

    """
    Selects (if found) up to two heads and dependents of the target word.
    Warning: the target word is not labeled in conllu, it selects its first occurrence.
    For MWEs, the first component is selected (and depX is not selected if it belongs to the MWE)
    For nouns, head1 is the head verb, and head2 is the other main core argument of the verb:
    If the target word is nsubj, head2 = obj/nmod/obl (and vice-versa).
    For nouns, dep1 is a dependent adjective (if any), and dep2 a dependent nmod (if any)
    For verbs, head1 is only selected if it is another verb or noun.
    For verbs, dep1 is the obj (if any), and dep2 is the nsubj/nmod/obl (or dep1 if not nsubj).
    """

    from conllu import parse
    
    head1 = () # lemma, deprel
    head2 = ()
    dep1 = ()
    dep2 = ()
    heads = []
    deps = []
    
    target_search = target
    # For MWEs: get first component
    if ' ' in target:
        target_search = target_search.split(' ')[0]

    # Target: first occurrence [dangerous]
    found = 0
    for token in parse(conll)[0]:
        if token['form'] == target_search:
            tgt_id = token['id']
            tgt_tag = token['upos']
            tgt_dep = token['deprel']
            tgt_head = token['head']
            if found == 1:
                print("\n\nWARNING\n\n")
            found = 1

    # NOUN
    if tgt_tag == 'NOUN' or tgt_tag == 'ADJ' or tgt_tag == 'NUM': # ADJ|NUM to prevent tagging/parsing errors
        # obj/obl/nmod/nsubj: get verb
        if tgt_dep == 'obj' or tgt_dep == 'obl' or tgt_dep == 'nmod' or tgt_dep == 'nsubj' or tgt_dep == 'csubj' or tgt_dep == 'amod' or tgt_dep == 'conj':
            for token in parse(conll)[0]:
                if token['id'] == tgt_head and (token['upos'] == 'VERB' or token['upos'] == 'AUX' or token['upos'] == 'PRON' or token['upos'] == 'NOUN' or token['upos'] == 'ADJ'): # For copulative [noun|pron as root]
                    head1 = (token[wform], tgt_dep)
                    head1_id = token['id']
            # Second 'head': for obj/obl/nmod -> nsubj (if exists). For nsubj -> obj/nmod/obl
            try:
                head1_id
                for token in parse(conll)[0]:
                    # Heads
                    if token['head'] == head1_id and token['upos'] == 'NOUN':
                        if tgt_dep == 'nsubj' or tgt_dep == 'csubj':
                            if token['deprel'] == 'obj':
                                head2 = (token[wform], tgt_dep)
                            elif token['deprel'] == 'nmod':
                                head2 = (token[wform], tgt_dep)
                            elif token['deprel'] == 'obl':
                                head2 = (token[wform], tgt_dep)
                        else:
                            if token['deprel'] == 'nsubj' or token['deprel'] == 'csubj':
                                head2 = (token[wform], tgt_dep)
                            elif token['deprel'] == 'obj' and tgt_dep != 'obj':
                                head2 = (token[wform], tgt_dep)
                            elif token['deprel'] == 'obl' and tgt_dep != 'obl':
                                head2 = (token[wform], tgt_dep)
                            elif token['deprel'] == 'nmod' and tgt_dep != 'nmod':
                                head2 = (token[wform], tgt_dep)
                    # Deps
                    if token['head'] == tgt_id:
                        #if token['upos'] == 'ADJ' or token['upos'] == 'NOUN':
                        if token['form'] not in target: # avoid MWE component as dep
                            if token['deprel'] == 'nmod':
                                if len(dep1)>0:
                                    dep2 = dep1
                                dep1 = (token[wform], token['deprel'])
                            elif token['upos'] == 'ADJ' or token['upos'] == 'NOUN' and len(dep2)==0:
                                dep2 = (token[wform], token['deprel'])

            except:
                None

        # Root NOUNs (e.g., copulative)
        elif tgt_dep == 'root':
            # Deps: first 
            for token in parse(conll)[0]:
                if token['head'] == tgt_id:
                    if token['deprel'] == 'obj':
                        if token[wform] in dep1:
                            dep2 = dep1
                        dep1 = (token[wform], token['deprel'])
                    elif token['deprel'] == 'nsubj' or token['deprel'] == 'csubj':
                        if len(dep1)==0:
                            dep1 = (token[wform], token['deprel'])
                        else:
                            dep2 = (token[wform], token['deprel'])
                    elif token['deprel'] == 'obl':
                        if len(dep1)==0:
                            dep1 = (token[wform], token['deprel'])
                        else:
                            dep2 = (token[wform], token['deprel'])
                    elif token['deprel'] == 'nmod':
                        if len(dep1)==0:
                            dep1 = (token[wform], token['deprel'])
                        else:
                            dep2 = (token[wform], token['deprel'])

    # VERB
    if tgt_tag == 'VERB' or tgt_tag == 'AUX':
        for token in parse(conll)[0]:
            # Heads
            if tgt_dep != 'root' and token['id'] == tgt_head:
                if len(head1)==0:
                    head1 = (token[wform], tgt_dep)
                else:
                    head2 = (token[wform], tgt_dep)
            # Deps obj and nsubj
            if token['head'] == tgt_id:
                if token['deprel'] == 'obj':
                    dep1 = (token[wform], token['deprel'])
                else:
                    if token['deprel'] == 'nsubj' or token['deprel'] == 'csubj':
                        dep2 = (token[wform], token['deprel'])
                    elif token['deprel'] == 'obl':
                        dep2 = (token[wform], token['deprel'])
                    elif token['deprel'] == 'nmod':
                        dep2 = (token[wform], token['deprel'])

    # If only one, prefer 1
    if len(dep1) == 0 and len(dep2) == 1:
        dep1 = dep2
    if len(head1) == 0 and len(head2) == 1:
        head1 = head2
    # If repeated, remove 2 or dep
    if dep2 == head1:
        dep2 = {}
    if dep2 == head2:
        dep2 = {}
    if dep1 == head2:
        head2 = {}
    if head1 == dep1:
        dep1 = {}
    
    if len(head1)>0:
        heads.append(head1)
    if len(head2)>0:
        heads.append(head2)
    if len(dep1)>0:
        deps.append(dep1)
    if len(dep2)>0:
        deps.append(dep2)
    return(heads, deps, tgt_tag, tgt_dep)


# Get word from splitted token
##############################
# Given a token from conllu (e.g., "Sentamos")
# returns the word in which it occurs in the text
# (e.g., "Sentámonos") by edit distance
def get_word_in_sentence(token, sent):

    sent = re.sub('([\.\?!,])', r' \1', sent)
    sent = re.sub('  +', ' ', sent)

    # For MWE tokens (e.g., proper nouns)
    if '_' in token and len(token)>2:
        token_mwe = token.replace('_', ' ')
        # First: check if it exists in sentence
        if token_mwe in sent:
            return(token_mwe)
        # else: get first token
        else:
            token = token.split('_')[0]
    
    words = sent.split(' ')
    values = []
    for w in words:
        values.append(edlib.align(w, token)['editDistance'])

    closest = values.index(min(values))
    return(words[closest])
