import pandas as pd
import os
import re
import time
import numpy as np
from operator import itemgetter
from typing import Dict, Tuple, List, Set
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import json
import io
import logging

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()
     


parser = argparse.ArgumentParser()

def load_text(textpath,reverse=True):
    corpus =[]
    with open(textpath,encoding="utf-8") as t:
        for l in t.readlines():
            corpus.append(l)
    if reverse:
        corpus = [x[::-1] for x in corpus]
    return corpus

def transform_text(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    sparse = vectorizer.transform(corpus)
    frequencies = sum(sparse).toarray()[0]
    vocab_df = pd.DataFrame(frequencies, index=vectorizer.get_feature_names(), columns=['frequency']).sort_values(by='frequency', ascending=False)
    vocab = vocab_df.to_dict()['frequency']
    vocab = {" ".join(x) + '</w>':y for x, y in vocab.items()}
    return vocab

def get_pair_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """Get counts of pairs of consecutive symbols."""

    pairs = {}
    for word, frequency in vocab.items():
        symbols = word.split()

        # count occurrences of pairs
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            current_frequency = pairs.get(pair, 0)
            pairs[pair] = current_frequency + frequency
    return pairs

def get_pair_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """Get counts of pairs of consecutive symbols."""

    pairs = {}
    for word, frequency in vocab.items():
        symbols = word.split()

        # count occurrences of pairs
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            current_frequency = pairs.get(pair, 0)
            pairs[pair] = current_frequency + frequency

    return pairs

def merge_vocab(best_pair: Tuple[str, str], vocab_in: Dict[str, int]) -> Dict[str, int]:
    """Step 3. Merge all occurrences of the most frequent pair"""

    vocab_out = {}

    # re.escape
    # ensures the characters of our input pair will be handled as is and
    # not get mistreated as special characters in the regular expression.
    pattern = re.escape(' '.join(best_pair))
    replacement = ''.join(best_pair)

    for word_in in vocab_in:
        # replace most frequent pair in all vocabulary
        word_out = re.sub(pattern, replacement, word_in)
        vocab_out[word_out] = vocab_in[word_in]

    return vocab_out

if __name__ =='__main__':
    parser.add_argument('-f', '--file', default='file.txt')
    parser.add_argument('-n', '--iter', default=60000,type=int)
    args = parser.parse_args()
    corpus = load_text(args.file)
    vocab =transform_text(corpus)
    bpe_codes = {}
    num_merges = args.iter  # hyperparameter
    items = list(range(0, num_merges))
    for i in progressBar(items, prefix = 'Progress:', suffix = 'Complete', length = 50):
        pair_stats = get_pair_stats(vocab)
        if not pair_stats:
            break

        best_pair = max(pair_stats, key=pair_stats.get)
        bpe_codes[str(best_pair)] = i

#        logging.warning('vocabulary: ', vocab)
#        logging.warning('best pair:', best_pair)
        vocab = merge_vocab(best_pair, vocab)
        
    with open('json_data.json', 'w',encoding='utf-8') as outfile:
        json.dump(bpe_codes, outfile,ensure_ascii = False)
