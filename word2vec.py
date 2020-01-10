from gensim.models import Word2Vec
import gensim
from utils import *
from pre_process import *
import pickle
import numpy as np


tags, docs = parse_csv_cluster('Data.csv')
processed = []
for doc in docs:
    tokens = process(doc['text'])
    processed.append(tokens)


print("**")
model = Word2Vec(processed, min_count = 1,  size = 100, window = 5)
#model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
print("*")
wvdata = []

i = 0
total = 0
notin = 0
for p in processed:
    if i %100 == 0:
        print(i)
    i += 1
    v = np.array([0.0 for i in range(100)])
    for word in p:
        total += 1
        if word not in model.wv.vocab:
            notin += 1
            continue

        v += model[word].copy()

    if len(p) is not 0:
        v = v / int(len(p))
    wvdata.append(v)

wvdata = np.array(wvdata)
print(total)
print(notin)
with open('word2vec2.data', 'wb') as f:
    pickle.dump(wvdata, f)