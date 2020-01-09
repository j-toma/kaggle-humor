import logging 
import os
import sys
import re
import pickle

import pandas as pd
import numpy as np

from pre_proc import pre_proc, tokenizer, build_data

from collections import defaultdict

from nltk.tokenize import TweetTokenizer

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

models = {
    'ATL': '300f_50mw_10c_ATL',
    'BB': '300f_50mw_10c_BB',
    'BI': '300f_50mw_10c_BI',
    'BUZ': '300f_50mw_10c_BUZ',
    'CNN': '300f_50mw_10c_CNN',
    'COR': '300f_50mw_10c_COR',
    'FOX': '300f_50mw_10c_FOX',
    'GND': '300f_50mw_10c_GDN',
    'NAT': '300f_50mw_10c_NAT',
    'NPR': '300f_50mw_10c_NPR',
    'NYP': '300f_50mw_10c_NYP',
    'NYT': '300f_50mw_10c_NYT',
    'REU': '300f_50mw_10c_REU',
    'TPM': '300f_50mw_10c_TPM',
    'VOX': '300f_50mw_10c_VOX',
    'WAPO': '300f_50mw_10c_WAPO'
}

def load_bin_vec(model, vocab):
    word_vecs = {}
    unknown_words_count = 0
    unknown_words = []
    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unknown_words.append(word)
            unknown_words_count += 1

    logging.info('unknown words: %d' % (unknown_words_count))
    #print('unkown words:', unknown_words)
    #print('word_vec["simple"]:',word_vecs['simple'])
    return word_vecs

def get_W(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_index_map = dict()

    W = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)
    W[0] = np.zeros((k,))
    W[1] = np.random.uniform(-0.25, 0.25,k)

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_index_map[word] = i
        i += 1
    return W, word_index_map

def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    for word in sent:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)
    return x


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ''.join(sys.argv))


    MODEL_PATH = '/home/jtoma/s1/patternRecognition/project1/models'
    EMBED_PATH = '/home/jtoma/s1/patternRecognition/project1/embeddings'
    for k,v in models.items():
        # load data
        PREPROC_PATH = '/home/jtoma/s1/patternRecognition/project1/pre_proc'
        file_name = 'preproc1.pickle'
        preproc_file = os.path.join(PREPROC_PATH, file_name)
        data, vocab, maxlen = pickle.load(open(preproc_file,'rb'))
        print('data pickle loaded!')

        # word2vec
        model_file = os.path.join(MODEL_PATH, v)
        model = Word2Vec.load(model_file)
        logging.info('model', v, 'loaded!')

        w2v = load_bin_vec(model, vocab)
        logging.info('word embeddings loaded!')
        logging.info('num words in embeddings: ' + str(len(w2v)))
        W, word_idx_map = get_W(w2v, k=model.vector_size)
        logging.info('extracted index from embeddings!')

        for i in range(len(data)):
            title_embedding = get_idx_from_sent(data[i]['edit'], word_idx_map)
            pickle_key = k+'_embed'
            data[i].update({pickle_key:title_embedding})

        logging.info('title embeddings for', v, 'created!')
        logging.info('first datum:' + str(data[0]))

        # store to pickle
        pickle_name = k+'.pickle'
        pickle_file = os.path.join(EMBED_PATH, pickle_name)
        pickle.dump([data, vocab, maxlen, W, word_idx_map], open(pickle_file, 'wb'))

    logging.info('dataset created!')
