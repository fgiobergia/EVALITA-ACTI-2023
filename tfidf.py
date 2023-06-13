import pandas as pd
import numpy as np
from glob import glob
from unidecode import unidecode
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode

def load_file(fname):
    with open(fname) as f:
        return [ unidecode(x.strip()) for x in f.readlines() ]

def load_stopwords():
    files = list(map(load_file, glob("stopwords-it/*.txt")))
    return set(list(chain(*files))) | { 'ie', 'th' }

def get_vec(token, vocabs, mat):
    return np.hstack([ mat[task][:, vocabs[task].get(token, mat[task].shape[1]-1)] for task in ["A","B"]])

from gensim.models.keyedvectors import KeyedVectors

def load_w2v(all_tokens):
    wv = KeyedVectors.load_word2vec_format("it_k_5_alpha_0.375.w2v")
    
    tok_vecs = []
    for t in all_tokens:
        if t in wv:
            tok_vecs.append(wv[t])
        else:
            tok_vecs.append(np.zeros(wv.vector_size))
    return np.vstack(tok_vecs)

def load_vocabs(dfA=None, dfB=None):
    tasks = {
        "A": {
            "key": "conspiratorial"
        },
        "B": {
            "key": "conspiracy"
        }
    }

    vocabs = {}
    mat = {}
    stopwords_it = load_stopwords()

    for task in ["A", "B"]:
        
        if task == "A" and dfA is not None:
            df = dfA
        elif task == "B" and dfB is not None:
            df = dfB
        else:
            df = pd.read_csv(f"subtask{task}_train.csv")

        print("Task", task, "len", len(df))
        tfidf = TfidfVectorizer(
            strip_accents=unidecode, #"unicode",
            lowercase=True,
            stop_words=stopwords_it,
            tokenizer = RegexpTokenizer(r'\w+').tokenize,

        )
        M = tfidf.fit_transform(df.groupby(tasks[task]["key"]).apply(lambda x: " ".join(x.comment_text.tolist())).tolist())
        # adding column of zeros for out-of-vocabulary words
        mat[task] = np.hstack([np.array(M.todense()), np.zeros((M.shape[0], 1))])

        # vocabs[task] = np.array([ a[0] for a in sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])])
        vocabs[task] = tfidf.vocabulary_
        
    return vocabs, mat