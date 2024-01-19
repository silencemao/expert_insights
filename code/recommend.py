import pandas as pd
from gensim.models import KeyedVectors
import re
import os
import cfg
import numpy as np
from data_process import _get_title_key_words


weight_path = os.path.join(cfg.weight_path, 'sgns.wiki.bigram-char.bz2')
wv_from_text = KeyedVectors.load_word2vec_format(weight_path, binary=False, encoding="utf8", unicode_errors='ignore')
paper_key_word_df = pd.read_csv('../data/paper_key_word.csv')
vecs = []
for ind, row in paper_key_word_df.iterrows():
    vec0 = np.zeros(300)
    key_word = row['key_word'].split(',')
    for word in key_word:
        if word in wv_from_text:
            vec0 += wv_from_text[word]
    vecs.append(vec0)
vecs = np.array(vecs)


def _paper_recommend(key_word_vec):
    distances = np.linalg.norm(vecs - key_word_vec, axis=1)
    dis_df = pd.DataFrame({'dis': distances})
    res_df = pd.concat([paper_key_word_df, dis_df], axis=1)
    nearest_ids = res_df.nsmallest(10, 'dis')
    return nearest_ids


def _paper_recommend_by_title(paper_name):
    df = pd.DataFrame({'paper_id': '123', 'paper_name': paper_name})
    res_df = _get_title_key_words(df)
    key_words = res_df['key_word'].to_list()

    return _paper_recommend_by_key_word(key_words.split(','))


def _paper_recommend_by_key_word(key_word_list):
    vec = np.zeros(300)
    for word in key_word_list.split(','):
        if word in wv_from_text:
            vec += wv_from_text[word]
    return _paper_recommend(vec)


if __name__ == '__main__':
    pass
