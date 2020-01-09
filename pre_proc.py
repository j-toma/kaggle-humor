import re
import pickle
import logging
import os
import sys

import pandas as pd
import numpy as np

from collections import defaultdict
from nltk.tokenize import TweetTokenizer

train = pd.read_csv("/home/jtoma/humor/humicroedit/data/task-1/train.csv")

def tokenizer(title):
    tokenizer = TweetTokenizer()
    ret = []
    tokens = tokenizer.tokenize(title)
    for token in tokens:
        token = token.lower()
        ret.append(token)
    return ret

def pre_proc(original_title, edit):
    edit_loc = 0
    split = original_title.split()
    l = len(split)
    for i in range(l):
        if split[i][0] == '<':
            edit_loc = i
    edit_rel_loc = round( edit_loc / l * 100 )
    orig = original_title.replace('<','').replace('/>','')
    orig = re.sub("[^a-zA-Z0-9' -]"," ", orig)
    orig = tokenizer(orig)
    edit = re.sub('<[^>]*>', edit, original_title)
    edit = re.sub("[^a-zA-Z0-9' -]"," ", edit)
    edit = tokenizer(edit)
    return orig, edit, edit_rel_loc

def build_data(df, train_ratio=0.8):
    data = []
    orig_titles = set()
    maxlen = 0
    vocab = defaultdict(float)
    for i in range(len(df)):
        # get titles 
        original_title = df['original'][i]
        edit_word = df['edit'][i]
        orig, edit, edit_loc = pre_proc(original_title, edit_word)

        # build vocab 
        words = set(edit)
        for word in words:
            vocab[word] += 1

        # prepare maxlen
        length = len(edit)
        if length > maxlen:
            maxlen = length

        # build datum
        train_datum = {'y': df['meanGrade'][i],
                       'orig': ' '.join(orig),
                       'edit': ' '.join(edit),
                       'edit_loc': edit_loc,
                       'split': int(np.random.rand() < train_ratio),
                       'num_words': length
                      }
        data.append(train_datum)
    return data, vocab, maxlen

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ''.join(sys.argv))

    data, vocab,  maxlen = build_data(train)
    logging.info('data built!')
    logging.info('num datum:' + str(len(data)))
    logging.info('first datum:' + str(data[0]))
    logging.info('maxlen:' + str(maxlen))

    STORE_PATH = '/home/jtoma/s1/patternRecognition/project1/pre_proc'
    file_name = 'preproc1.pickle'
    preproc_file = os.path.join(STORE_PATH, file_name)
    pickle.dump([data, vocab, maxlen], open(preproc_file, 'wb'))
